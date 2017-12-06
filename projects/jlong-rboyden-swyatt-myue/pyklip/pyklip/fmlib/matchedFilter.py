        #wv_index_list = [self.input_psfs_wvs.index(wv) for wv in wvs]
__author__ = 'jruffio'
import multiprocessing as mp
import ctypes

import numpy as np
import pyklip.spectra_management as spec
import os
import itertools

from pyklip.fmlib.nofm import NoFM
import pyklip.fm as fm

from scipy import interpolate
from copy import copy

import astropy.io.fits as pyfits

from time import time
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

debug = False

class MatchedFilter(NoFM):
    """
    Matched filter with forward modelling.
    """
    def __init__(self, inputs_shape,
                 numbasis,
                 input_psfs,
                 input_psfs_wvs,
                 spectrallib,
                 save_per_sector = None,
                 datatype="float",
                 fakes_sepPa_list = None,
                 disable_FM = None,
                 true_fakes_pos = None,
                 ref_center = None,
                 flipx = None,
                 rm_edge = None,
                 planet_radius = None,
                 background_width = None,
                 save_bbfm = None):
        '''
        Defining the forward model matched filter parameters

        Args:
            inputs_shape: shape of the inputs numpy array. Typically (N, y, x)
            numbasis: 1d numpy array consisting of the number of basis vectors to use
            input_psfs: the psf of the image. A numpy array with shape (wv, y, x)
            input_psfs_wvs: the wavelegnths that correspond to the input psfs
            spectrallib: if not None, a list of spectra in raw DN units. The spectra should:
                    - have the total flux of the star, ie correspond to a contrast of 1.
                    - represent the total flux of the PSF and not the simply peak value.
                    - be corrected for atmospheric and instrumental transmission.
                    - have the same size as the number of images in the dataset.
            save_per_sector: If not None, should be a filename where the fmout array will be saved after each sector.
                    (Caution: huge file!! easily tens of Gb.)
            datatype: datatype to be used for the numpy arrays: "double" or "float" (default).
            fakes_sepPa_list: [(sep_pix1,pa1),(sep_pix2,pa2),...].
                    List of separations and pas for the simulated planets in the data.
                    If not None, it will only calculated the matched filter at the position of the fakes and skip the rest.
            disable_FM: Disable the calculation of the forward model in the code.
                        The unchanged original PSF will be used instead. (Default False)
            true_fakes_pos: If True and fakes_only is True, calculate the forward model at the exact position of the
                    fakes and not at the center of the pixels. (Default False)
            ref_center: reference center to which all the images are aligned. It should be the same center as the one
                    used in fm.parallelized.
            flipx: Determines whether a relfection about the x axis is necessary to rotate image North-up East left.
                    Should match the same attribute in the instrument class.
            rm_edge: When True (default), remove image edges to avoid edge effect. When there is more than 25% of NaNs
                    in the projection of the FM model on the data, the result of the projection is set to NaNs right away.
            planet_radius: Radius of the aperture to be used for the matched filter (pick something of the order of the
                            2xFWHM)
            background_width: Half the width of the arc in which the local standard deviation will be calculated.
            save_bbfm: path of the file where to save the broadband forward models

        '''
        # allocate super class
        super(MatchedFilter, self).__init__(inputs_shape, np.array(numbasis))

        self.save_bbfm = save_bbfm

        if rm_edge is not None:
            self.rm_edge = rm_edge
        else:
            self.rm_edge = True

        if true_fakes_pos is None:
            self.true_fakes_pos = False
        else:
            self.true_fakes_pos = true_fakes_pos

        if datatype=="double":
            self.data_type = ctypes.c_double
        elif datatype=="float":
            self.data_type = ctypes.c_float

        if save_per_sector is not None:
            self.fmout_dir = save_per_sector
            self.save_raw_fmout = True
        else:
            self.save_raw_fmout = False

        self.N_numbasis =  np.size(numbasis)
        self.ny = self.inputs_shape[1]
        self.nx = self.inputs_shape[2]
        self.N_frames = self.inputs_shape[0]

        self.fakes_sepPa_list = fakes_sepPa_list
        if disable_FM is None:
            self.disable_FM = False
        else:
            self.disable_FM = disable_FM

        self.inputs_shape = self.inputs_shape

        self.input_psfs_wvs = list(np.array(input_psfs_wvs,dtype=self.data_type))

        # Make sure the total flux of each PSF is unity for all wavelengths
        # So the peak value won't be unity.
        self.input_psfs = input_psfs/np.nansum(input_psfs,axis=(1,2))[:,None,None]
        numwv_psf,ny_psf,nx_psf =  self.input_psfs.shape

        self.spectrallib = spectrallib
        self.N_spectra = len(self.spectrallib)


        # create bounds for PSF stamp size
        self.row_m = int(np.floor(ny_psf/2.0))    # row_minus
        self.row_p = int(np.ceil(ny_psf/2.0))     # row_plus
        self.col_m = int(np.floor(nx_psf/2.0))    # col_minus
        self.col_p = int(np.ceil(nx_psf/2.0))     # col_plus

        self.psf_centx_notscaled = {}
        self.psf_centy_notscaled = {}
        self.curr_pa_fk = {}
        self.curr_sep_fk = {}

        x_psf_grid, y_psf_grid = np.meshgrid(np.arange(nx_psf * 1.)-nx_psf//2,np.arange(ny_psf* 1.)-ny_psf//2)
        psfs_func_list = []
        self.input_psfs[np.where(np.isnan(self.input_psfs))] = 0
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for wv_index in range(numwv_psf):
                model_psf = self.input_psfs[wv_index, :, :]
                psf_func = interpolate.LSQBivariateSpline(x_psf_grid.ravel(),y_psf_grid.ravel(),model_psf.ravel(),x_psf_grid[0,0:nx_psf-1]+0.5,y_psf_grid[0:ny_psf-1,0]+0.5)
                psfs_func_list.append(psf_func)

        self.psfs_func_list = psfs_func_list

        ny_PSF,nx_PSF = input_psfs.shape[1:]
        if ny_PSF < 8 or nx_PSF < 8:
            raise Exception("PSF cube is too small. It needs a stamp width bigger than 8 pixels.")
        stamp_PSF_x_grid, stamp_PSF_y_grid = np.meshgrid(np.arange(0,nx_PSF,1)-nx_PSF//2,np.arange(0,ny_PSF,1)-ny_PSF//2)
        self.stamp_PSF_mask = np.ones((ny_PSF,nx_PSF))
        r_PSF_stamp = abs((stamp_PSF_x_grid) +(stamp_PSF_y_grid)*1j)
        if planet_radius is not None:
            self.planet_radius = planet_radius
        else:
            self.planet_radius =  int(np.round(np.min([ny_PSF,nx_PSF])*7./20.))
        print("self.planet_radius",self.planet_radius)
        self.stamp_PSF_mask[np.where(r_PSF_stamp < self.planet_radius)] = np.nan
        # self.stamp_PSF_mask[np.where(r_PSF_stamp < 4.)] = np.nan
        if background_width is not None:
            self.background_width = background_width
        else:
            self.background_width =  np.min([ny_PSF,nx_PSF])//2
        self.bbfm_mask = np.ones((self.planet_radius*2,self.planet_radius*2))
        stamp_bbfm_x_grid, stamp_bbfm_y_grid = np.meshgrid(np.arange(0,self.planet_radius*2,1)-self.planet_radius,
                                                           np.arange(0,self.planet_radius*2,1)-self.planet_radius)
        r_bbfm_stamp = abs((stamp_bbfm_x_grid) +(stamp_bbfm_y_grid)*1j)
        self.bbfm_mask[np.where(r_bbfm_stamp < planet_radius)] = np.nan

        if ref_center is not None:
            self.ref_center = ref_center
            # create some parameters for a blank canvas to draw psfs on
            nx = inputs_shape[2]
            ny = inputs_shape[1]
            self.x_grid, self.y_grid = np.meshgrid(np.arange(nx * 1.)-ref_center[0], np.arange(ny * 1.)-ref_center[1])
            self.r_grid = abs(self.x_grid +self.y_grid*1j)
            self.pa_grid = np.arctan2( -self.x_grid,self.y_grid) % (2.0 * np.pi)
            if flipx is not None:
                sign = -1.
                if flipx:
                    sign = 1.
                self.th0_grid = np.arctan2(sign*self.x_grid,self.y_grid)


    # def alloc_interm(self, max_sector_size, numsciframes):
    #     """Allocates shared memory array for intermediate step
    #
    #     Intermediate step is allocated for a sector by sector basis
    #
    #     Args:
    #         max_sector_size: number of pixels in this sector. Max because this can be variable. Stupid rotating sectors
    #
    #     Returns:
    #         interm: mp.array to store intermediate products from one sector in
    #         interm_shape:shape of interm array (used to convert to numpy arrays)
    #
    #     """
    #     if self.save_bbfm is not None:
    #         interm_size = self.planet_radius*self.planet_radius*self.ny*self.nx
    #         interm = mp.Array(ctypes.c_double, interm_size)
    #         interm_shape = [self.ny,self.nx,self.planet_radius,self.planet_radius]
    #
    #         return interm, interm_shape
    #     else:
    #         return None,None



    def alloc_fmout(self, output_img_shape):
        """
        Allocates shared memory for the output of the shared memory

        Args:
            output_img_shape: Not used

        Returns:
            fmout: mp.array to store auxilliary data in
            fmout_shape: shape of auxilliary array = (3,self.N_spectra,self.N_numbasis,self.N_frames,self.ny,self.nx)
                        The 3 is for saving the different term of the matched filter:
                            0: dot product
                            1: square of the norm of the model
                            2: Local estimated variance of the data
                            3: Number of pixels used in the matched filter

        """
        if not self.save_bbfm:
            fmout_size = 4*self.N_spectra*self.N_numbasis*self.N_frames*self.ny*self.nx
            fmout = mp.Array(self.data_type, fmout_size)
            fmout_shape = (4,self.N_spectra,self.N_numbasis,self.N_frames,self.ny,self.nx)
        else:
            fmout_size = 4*self.N_spectra*self.N_numbasis*self.N_frames*self.ny*self.nx + \
                         2*self.planet_radius*2*self.planet_radius*self.ny*self.nx
            fmout = mp.Array(self.data_type, fmout_size)
            fmout_shape = (4*self.N_spectra*self.N_numbasis*self.N_frames*self.ny*self.nx + \
                         2*self.planet_radius*2*self.planet_radius*self.ny*self.nx,1)


        return fmout, fmout_shape

    def skip_section(self, radstart, radend, phistart, phiend,flipx=True):
        """
        Returns a boolean indicating if the section defined by (radstart, radend, phistart, phiend) should be skipped.
        When True is returned the current section in the loop in klip_parallelized() is skipped.

        Args:
            radstart: minimum radial distance of sector [pixels]
            radend: maximum radial distance of sector [pixels]
            phistart: minimum azimuthal coordinate of sector [radians]
            phiend: maximum azimuthal coordinate of sector [radians]
            flipx: if True, flip x coordinate in final image

        Returns:
            Boolean: False so by default it never skips.
        """

        margin_sep = np.sqrt(2)/2.
        margin_phi = np.sqrt(2)/(2*radstart)
        if self.fakes_sepPa_list is not None:
            skipSectionBool = True
            for sep_it,pa_it in self.fakes_sepPa_list:
                if flipx:
                    paend= ((-phistart + np.pi/2.)% (2.0 * np.pi))
                    pastart = ((-phiend + np.pi/2.)% (2.0 * np.pi))
                else:
                    pastart = ((phistart - np.pi/2.)% (2.0 * np.pi))
                    paend= ((phiend - np.pi/2.)% (2.0 * np.pi))
                # Normal case when there are no 2pi wrap
                if pastart < paend:
                    if (radstart-margin_sep<=sep_it<=radend+margin_sep) and ((pa_it%360)/180.*np.pi >= pastart-margin_phi) & ((pa_it%360)/180.*np.pi < paend+margin_phi):
                        skipSectionBool = False
                        break
                # 2 pi wrap case
                else:
                    if (radstart-margin_sep<=sep_it<=radend+margin_sep) and (((pa_it%360)/180.*np.pi >= pastart-margin_phi) | ((pa_it%360)/180.*np.pi < paend+margin_phi)):
                        skipSectionBool = False
                        break
        else:
            skipSectionBool = False

        return skipSectionBool


    def fm_from_eigen(self, klmodes=None, evals=None, evecs=None, input_img_shape=None, input_img_num=None,
                      ref_psfs_indicies=None, section_ind=None,section_ind_nopadding=None, aligned_imgs=None, pas=None,
                     wvs=None, radstart=None, radend=None, phistart=None, phiend=None, padding=None,IOWA = None, ref_center=None,
                     parang=None, ref_wv=None, numbasis=None, fmout=None, perturbmag=None,klipped=None, flipx=True, **kwargs):
        """
        Calculate and project the FM at every pixel of the sector. Store the result in fmout.

        Args:
            klmodes: unpertrubed KL modes
            evals: eigenvalues of the covariance matrix that generated the KL modes in ascending order
                   (lambda_0 is the 0 index) (shape of [nummaxKL])
            evecs: corresponding eigenvectors (shape of [p, nummaxKL])
            input_img_shape: 2-D shape of inpt images ([ysize, xsize])
            input_img_num: index of sciece frame
            ref_psfs_indicies: array of indicies for each reference PSF
            section_ind: array indicies into the 2-D x-y image that correspond to this section.
                         Note needs be called as section_ind[0]
            pas: array of N parallactic angles corresponding to N reference images [degrees]
            wvs: array of N wavelengths of those referebce images
            radstart: radius of start of segment
            radend: radius of end of segment
            phistart: azimuthal start of segment [radians]
            phiend: azimuthal end of segment [radians]
            padding: amount of padding on each side of sector
            IOWA: tuple (IWA,OWA) where IWA = Inner working angle and OWA = Outer working angle both in pixels.
                It defines the separation interva in which klip will be run.
            ref_center: center of image
            numbasis: array of KL basis cutoffs
            parang: parallactic angle of input image [DEGREES]
            ref_wv: wavelength of science image
            fmout: numpy output array for FM output. Shape is (N, y, x, b)
            klipped: array of shape (p,b) that is the PSF subtracted data for each of the b KLIP basis
                     cutoffs. If numbasis was an int, then sub_img_row_selected is just an array of length p
            kwargs: any other variables that we don't use but are part of the input
        """
        if hasattr(self,"ref_center"):
            if (self.ref_center[0] != ref_center[0]) or (self.ref_center[1] != ref_center[1]):
                raise ValueError("ref_center needs to be the same when defining the matchedFilter class and calling parallelized")
        if np.size(numbasis) != 1:
            raise ValueError("Numbasis should only have a single element. e.g. numbasis = [30]. numbasis = [10,20,30] is not accepted.")

        if self.save_bbfm:
            fmout1 = fmout[:4*self.N_spectra*self.N_numbasis*self.N_frames*self.ny*self.nx]
            fmout1.shape = (4,self.N_spectra,self.N_numbasis,self.N_frames,self.ny,self.nx)
            fmout2 = fmout[4*self.N_spectra*self.N_numbasis*self.N_frames*self.ny*self.nx::]
            fmout2.shape = (self.ny,self.nx,2*self.planet_radius,2*self.planet_radius)
        else:
            fmout1 = fmout

        ref_wv = ref_wv.astype(self.data_type)

        sci = aligned_imgs[input_img_num, section_ind[0]]
        refs = aligned_imgs[ref_psfs_indicies, :]
        refs = refs[:, section_ind[0]]

        # Calculate the PA,sep 2D map

        if hasattr(self,"x_grid") and hasattr(self,"y_grid"):
            x_grid, y_grid = self.x_grid,self.y_grid
        else:
            x_grid, y_grid = np.meshgrid(np.arange(self.nx * 1.)-ref_center[0], np.arange(self.ny * 1.)-ref_center[1])
        x_grid=x_grid.astype(self.data_type)
        y_grid=y_grid.astype(self.data_type)
        # Define the masks for where the planet is and the background.
        if hasattr(self,"r_grid"):
            r_grid = self.r_grid
        else:
            r_grid = np.sqrt((x_grid)**2 + (y_grid)**2)
        if hasattr(self,"pa_grid"):
            pa_grid = self.pa_grid
        else:
            pa_grid = np.arctan2( -x_grid,y_grid) % (2.0 * np.pi)
        if flipx:
            paend= ((-phistart + np.pi/2.)% (2.0 * np.pi))
            pastart = ((-phiend + np.pi/2.)% (2.0 * np.pi))
        else:
            pastart = ((phistart - np.pi/2.)% (2.0 * np.pi))
            paend= ((phiend - np.pi/2.)% (2.0 * np.pi))
        # Normal case when there are no 2pi wrap
        if pastart < paend:
            where_section = np.where((r_grid >= radstart) & (r_grid < radend) & (pa_grid >= pastart) & (pa_grid < paend))
        # 2 pi wrap case
        else:
            where_section = np.where((r_grid >= radstart) & (r_grid < radend) & ((pa_grid >= pastart) | (pa_grid < paend)))

        # Get a list of the PAs and sep of the PA,sep map falling in the current section
        r_list = r_grid[where_section]
        pa_list = pa_grid[where_section]
        x_list = x_grid[where_section]
        y_list = y_grid[where_section]
        row_id_list = where_section[0]
        col_id_list = where_section[1]
        # Only select pixel with fakes if needed
        if self.fakes_sepPa_list is not None:
            r_list_tmp = []
            pa_list_tmp = []
            row_id_list_tmp = []
            col_id_list_tmp = []
            for sep_it,pa_it in self.fakes_sepPa_list:
                x_it = sep_it*np.cos(np.radians(90+pa_it))
                y_it = sep_it*np.sin(np.radians(90+pa_it))
                dist_list = np.sqrt((x_list-x_it)**2+(y_list-y_it)**2)
                min_id = np.nanargmin(dist_list)
                min_dist = dist_list[min_id]
                if min_dist < np.sqrt(2)/2.:
                    if self.true_fakes_pos:
                        r_list_tmp.append(sep_it)
                        pa_list_tmp.append(np.radians(pa_it))
                    else:
                        r_list_tmp.append(r_list[min_id])
                        pa_list_tmp.append(pa_list[min_id])
                    row_id_list_tmp.append(row_id_list[min_id])
                    col_id_list_tmp.append(col_id_list[min_id])
            r_list = r_list_tmp
            pa_list = pa_list_tmp
            row_id_list = row_id_list_tmp
            col_id_list = col_id_list_tmp

        greenboard = np.zeros((self.ny,self.nx))
        x_bbfm, y_bbfm = np.meshgrid(np.arange(self.ny, dtype=np.float32), np.arange(self.nx, dtype=np.float32))
        # flip x if needed to get East left of North
        if flipx is True:
            x_bbfm = ref_center[0] - (x_bbfm - ref_center[0])
        # do rotation. CW rotation formula to get a CCW of the image
        angle_rad = np.radians(parang)
        cosa = np.cos(angle_rad)
        sina = np.sin(angle_rad)
        xp = (x_bbfm-ref_center[0])*cosa + (y_bbfm-ref_center[1])*sina + ref_center[0]
        yp = -(x_bbfm-ref_center[0])*sina + (y_bbfm-ref_center[1])*cosa + ref_center[1]
        # create bounds for PSF stamp size
        self.bbfm_m = int(np.floor(self.planet_radius))
        self.bbfm_p = int(np.ceil(self.planet_radius))

        # Loop over the input template spectra and the number of KL modes in numbasis
        for spec_id,N_KL_id in itertools.product(range(self.N_spectra),range(self.N_numbasis)):
            # Calculate the projection of the FM and the klipped section for every pixel in the section.
            # 1/ Inject a fake at one pa and sep in the science image
            # 2/ Inject the corresponding planets at the same PA and sep in the reference images remembering that the
            # references rotate.
            # 3/ Calculate the perturbation of the KL modes
            # 4/ Calculate the FM
            # 5/ Calculate dot product (matched filter)
            for sep_fk,pa_fk,row_id,col_id in zip(r_list,np.rad2deg(pa_list),row_id_list,col_id_list):
                # print(sep_fk,pa_fk,row_id,col_id)
                # 1/ Inject a fake at one pa and sep in the science image
                model_sci,mask = self.generate_model_sci(input_img_shape, section_ind, parang, ref_wv,
                                                         radstart, radend, phistart, phiend, padding, ref_center,
                                                         parang, ref_wv,sep_fk,pa_fk, flipx)
                # Normalize the science image according to the spectrum. the model is normalize to unit contrast,
                model_sci = model_sci*self.spectrallib[spec_id][input_img_num]
                where_fk = np.where(mask==2)[0]
                where_background = np.where(mask>=1)[0] # Caution: it includes where the fake is...
                where_background_strict = np.where(mask==1)[0]

                if self.rm_edge and float(np.sum(np.isfinite(klipped[where_fk,N_KL_id])))/float(np.size(klipped[where_fk,N_KL_id]))<=0.75:
                    fmout1[0,spec_id,N_KL_id,input_img_num,row_id,col_id] = np.nan
                    fmout1[1,spec_id,N_KL_id,input_img_num,row_id,col_id] = np.nan
                    fmout1[2,spec_id,N_KL_id,input_img_num,row_id,col_id] = np.nan
                    fmout1[3,spec_id,N_KL_id,input_img_num,row_id,col_id] = np.nan
                    continue

                # 2/ Inject the corresponding planets at the same PA and sep in the reference images remembering that the
                # references rotate.
                if not self.disable_FM:
                    models_ref = self.generate_models(input_img_shape, section_ind, pas, wvs, radstart, radend,
                                                      phistart, phiend, padding, ref_center, parang, ref_wv,sep_fk,pa_fk, flipx)

                    # Normalize the models with the spectrum. the model is normalize to unit contrast,
                    input_spectrum = self.spectrallib[spec_id][ref_psfs_indicies]
                    models_ref = models_ref * input_spectrum[:, None]

                    # 3/ Calculate the perturbation of the KL modes
                    # using original Kl modes and reference models, compute the perturbed KL modes.
                    # Spectrum is already in the model, that's why we use perturb_specIncluded(). (Much faster)
                    delta_KL = fm.perturb_specIncluded(evals, evecs, klmodes, refs, models_ref)

                    # 4/ Calculate the FM: calculate postklip_psf using delta_KL
                    # postklip_psf has unit broadband contrast
                    model_sci_fk = model_sci[where_fk]
                    delta_KL_kl = delta_KL[:,where_fk]
                    klmodes_fk = klmodes[:,where_fk]
                    postklip_psf = calculate_fm_opti(delta_KL, klmodes,sci, model_sci_fk,delta_KL_kl,klmodes_fk)
                    # postklip_psf, oversubtraction, selfsubtraction = fm.calculate_fm(delta_KL, klmodes, numbasis,
                    #                                                                  sci, model_sci, inputflux=None)
                else:
                    #if one doesn't want the FM
                    if np.size(numbasis) == 1:
                        postklip_psf = model_sci[None,:]
                    else:
                        # Mh, is this actually working? shouldn't I use tile?
                        postklip_psf = model_sci

                # 5/ Calculate dot product (matched filter)
                # fmout_shape = (3,self.N_spectra,self.N_numbasis,self.N_frames,self.ny,self.nx)
                #         The 3 is for saving the different term of the matched filter:
                #             0: dot product
                #             1: square of the norm of the model
                #             2: Local estimated variance of the data
                sky = np.nanmean(klipped[where_background_strict,N_KL_id])
                # postklip_psf[N_KL_id,where_fk] = postklip_psf[N_KL_id,where_fk]-np.mean(postklip_psf[N_KL_id,where_background])
                # Subtract local sky background to the klipped image
                klipped_sub = klipped[where_fk,N_KL_id]-sky
                # klipped_sub_finite = np.where(np.isfinite(klipped_sub))
                # klipped_sub_nan = np.where(np.isnan(klipped_sub))
                postklip_psf[N_KL_id,np.where(np.isnan(klipped_sub))[0]] = np.nan
                postklip_psf_fk = postklip_psf[N_KL_id,:]
                dot_prod = np.nansum(klipped_sub*postklip_psf_fk)
                model_norm = np.nansum(postklip_psf_fk*postklip_psf_fk)
                klipped_rm_pl = copy(klipped[:,N_KL_id]) -sky
                klipped_rm_pl[where_fk] -=  (dot_prod/model_norm)*postklip_psf_fk
                klipped_rm_pl_bkg = klipped_rm_pl[where_background]
                if self.rm_edge and (float(np.sum(np.isfinite(klipped_rm_pl_bkg)))/float(np.size(klipped_rm_pl_bkg))<=0.75):
                    variance = np.nan
                    npix = np.nan
                else:
                    variance = np.nanvar(klipped_rm_pl_bkg)
                    npix = np.sum(np.isfinite(klipped_rm_pl_bkg))


                fmout1[0,spec_id,N_KL_id,input_img_num,row_id,col_id] = dot_prod
                fmout1[1,spec_id,N_KL_id,input_img_num,row_id,col_id] = model_norm
                fmout1[2,spec_id,N_KL_id,input_img_num,row_id,col_id] = variance
                fmout1[3,spec_id,N_KL_id,input_img_num,row_id,col_id] = npix

                if self.save_bbfm:
                    greenboard.shape = [input_img_shape[0] * input_img_shape[1]]
                    greenboard[section_ind[0][where_fk]] = postklip_psf[N_KL_id,:]
                    greenboard.shape = [input_img_shape[0],input_img_shape[1]]

                    rot_stamp = ndimage.map_coordinates(greenboard,
                                                        [yp[(row_id-self.bbfm_m):(row_id+self.bbfm_p),
                                                            (col_id-self.bbfm_m):(col_id+self.bbfm_p)].ravel(),
                                                         xp[(row_id-self.bbfm_m):(row_id+self.bbfm_p),
                                                            (col_id-self.bbfm_m):(col_id+self.bbfm_p)].ravel()],
                                                        cval=np.nan)
                    rot_stamp.shape = [2*self.planet_radius,2*self.planet_radius]

                    fmout2[row_id,col_id,:,:] = fmout2[row_id,col_id,:,:]+rot_stamp

                if 0:
                    print(dot_prod,model_norm,variance,npix)

                # Plot sector, klipped and FM model for debug only
                if 0: # and row_id>=10 and col_id > 5 # and np.nansum(klipped[where_fk,N_KL_id]) != 0:
                    # plt.figure(1)
                    # blackboard = np.zeros((self.ny,self.nx))
                    # blackboard.shape = [input_img_shape[0] * input_img_shape[1]]
                    # blackboard[section_ind] = klipped
                    # blackboard.shape = [input_img_shape[0],input_img_shape[1]]
                    # plt.imshow(blackboard,interpolation="nearest")
                    # plt.colorbar()
                    # plt.figure(2)
                    # for k in range(numbasis[0]):
                    #     blackboard = np.zeros((self.ny,self.nx))
                    #     blackboard.shape = [input_img_shape[0] * input_img_shape[1]]
                    #     blackboard[section_ind] = klmodes[k,:]
                    #     blackboard.shape = [input_img_shape[0],input_img_shape[1]]
                    #     plt.subplot(1,numbasis[0],k+1)
                    #     plt.imshow(blackboard[::-1,:],interpolation="nearest")
                    #     plt.title("KL{0}".format(k))
                    #     plt.colorbar()
                    # plt.show()


                    #if 0:
                    # print(klipped_sub)
                    # print(np.isfinite(klipped_sub))
                    # print(np.size(klipped_sub))
                    # print(float(np.sum(np.isfinite(klipped_sub)))/float(np.size(klipped_sub)))
                    # print(float(np.sum(np.isfinite(klipped[where_background,N_KL_id])))/float(np.size(klipped[where_background,N_KL_id])))
                    print(sep_fk,pa_fk,row_id,col_id)
                    print(dot_prod,model_norm,variance,npix)
                    # print(np.nanmean(klipped-sky),sky,dot_prod,model_norm,np.nanmean((dot_prod/model_norm)*postklip_psf[N_KL_id,:]))
                    # print(klipped.shape,postklip_psf[N_KL_id,:].shape)
                    # print(float(np.sum(np.isfinite(klipped_rm_pl[where_background]))),float(np.size(klipped_rm_pl[where_background])))
                    blackboard1 = np.zeros((self.ny,self.nx))
                    blackboard2 = np.zeros((self.ny,self.nx))
                    blackboard3 = np.zeros((self.ny,self.nx))
                    #print(section_ind)
                    plt.figure(1)
                    plt.subplot(1,3,1)
                    blackboard1.shape = [input_img_shape[0] * input_img_shape[1]]
                    blackboard1[section_ind] = mask
                    blackboard1[section_ind] = blackboard1[section_ind] + 1
                    blackboard1.shape = [input_img_shape[0],input_img_shape[1]]
                    plt.imshow(blackboard1,interpolation="nearest")
                    plt.colorbar()
                    plt.subplot(1,3,2)
                    blackboard2.shape = [input_img_shape[0] * input_img_shape[1]]
                    # blackboard2[section_ind[0][where_fk]] = klipped[where_fk,N_KL_id]
                    blackboard2[section_ind[0]] = klipped#klipped_rm_pl
                    blackboard2.shape = [input_img_shape[0],input_img_shape[1]]
                    plt.imshow(blackboard2,interpolation="nearest")
                    plt.colorbar()
                    plt.subplot(1,3,3)
                    blackboard3.shape = [input_img_shape[0] * input_img_shape[1]]
                    blackboard3[section_ind[0][where_fk]] = postklip_psf[N_KL_id,:]
                    blackboard3.shape = [input_img_shape[0],input_img_shape[1]]
                    plt.imshow(blackboard3,interpolation="nearest")
                    plt.colorbar()
                    #print(klipped[where_fk,N_KL_id])
                    #print(postklip_psf[N_KL_id,where_fk])
                    # print(np.sum(klipped[where_fk,N_KL_id]*postklip_psf[N_KL_id,where_fk]))
                    # print(np.sum(postklip_psf[N_KL_id,where_fk]*postklip_psf[N_KL_id,where_fk]))
                    # print(np.sum(klipped[where_fk,N_KL_id]*klipped[where_fk,N_KL_id]))
                    plt.show()
                # exit()


    def fm_end_sector(self, interm_data=None, fmout=None, sector_index=None,
                               section_indicies=None):
        """
        Save the fmout object at the end of each sector if save_per_sector was defined when initializing the class.
        """
        #fmout_shape = (3,self.N_spectra,self.N_numbasis,self.N_frames,self.ny,self.nx)
        if self.save_raw_fmout:
            if self.save_bbfm:
                fmout1 = fmout[:4*self.N_spectra*self.N_numbasis*self.N_frames*self.ny*self.nx]
                fmout1.shape = (4,self.N_spectra,self.N_numbasis,self.N_frames,self.ny,self.nx)
            else:
                fmout1 = fmout
            hdu = pyfits.PrimaryHDU(fmout)
            hdulist = pyfits.HDUList([hdu])
            hdulist.writeto(self.fmout_dir,clobber=True)


        return

    def save_fmout(self, dataset, fmout, outputdir, fileprefix, numbasis, klipparams=None, calibrate_flux=False,
                   spectrum=None):
        """
        Saves the fmout data to disk following the instrument's savedata function

        Args:
            dataset: Instruments.Data instance. Will use its dataset.savedata() function to save data
            fmout: the fmout data passed from fm.klip_parallelized which is passed as the output of cleanup_fmout
            outputdir: output directory
            fileprefix: the fileprefix to prepend the file name
            numbasis: KL mode cutoffs used
            klipparams: string with KLIP-FM parameters
            calibrate_flux: if True, flux calibrate the data (if applicable)
            spectrum: if not None, the spectrum to weight the data by. Length same as dataset.wvs
        """

        if self.save_bbfm:
            fmout1 = fmout[:4*self.N_spectra*self.N_numbasis*self.N_frames*self.ny*self.nx]
            fmout1.shape = (4,self.N_spectra,self.N_numbasis,self.N_frames,self.ny,self.nx)
            fmout2 = fmout[4*self.N_spectra*self.N_numbasis*self.N_frames*self.ny*self.nx::]
            fmout2.shape = (self.ny,self.nx,2*self.planet_radius,2*self.planet_radius)
        else:
            fmout1 = fmout

        # hdu = pyfits.PrimaryHDU(fmout1)
        # hdulist = pyfits.HDUList([hdu])
        # hdulist.writeto(outputdir+os.path.sep+'fmout1_test_before3.fits',clobber=True)

        #fmout_shape = (3,self.N_spectra,self.N_numbasis,self.N_frames,self.ny,self.nx)
        fmout1[np.where(fmout1==0)] = np.nan

        # hdu = pyfits.PrimaryHDU(fmout1)
        # hdulist = pyfits.HDUList([hdu])
        # hdulist.writeto(outputdir+os.path.sep+'fmout1_test3.fits',clobber=True)

        # The mf.MatchedFilter class calculate the projection of the FM on the data for each pixel and images.
        # The final combination to form the cross  cross correlation, matched filter and contrast maps is done right
        # here.
        FMCC_map = np.nansum(fmout1[0,:,:,:,:,:],axis=2) \
                        / np.sqrt(np.nansum(fmout1[1,:,:,:,:,:],axis=2))
        FMCC_map[np.where(FMCC_map==0)]=np.nan
        self.FMCC_map = FMCC_map

        FMMF_map = np.nansum(fmout1[0,:,:,:,:,:]/fmout1[2,:,:,:,:,:],axis=2) \
                        / np.sqrt(np.nansum(fmout1[1,:,:,:,:,:]/fmout1[2,:,:,:,:,:],axis=2))
        FMMF_map[np.where(FMMF_map==0)]=np.nan
        self.FMMF_map = FMMF_map

        contrast_map = np.nansum(fmout1[0,:,:,:,:,:]/fmout1[2,:,:,:,:,:],axis=2) \
                        / np.nansum(fmout1[1,:,:,:,:,:]/fmout1[2,:,:,:,:,:],axis=2)
        contrast_map[np.where(contrast_map==0)]=np.nan
        self.contrast_map = contrast_map

        self.FMNpix_map = np.nansum(fmout1[3,:,:,:,:,:],axis=2)

        self.metricMap = [self.FMMF_map,self.FMCC_map,self.contrast_map]


        for k in range(np.size(self.numbasis)):
            # Save the outputs (matched filter, shape map and klipped image) as fits files
            suffix = "FMMF-KL{0}".format(self.numbasis[k])
            dataset.savedata(outputdir+os.path.sep+fileprefix+'-'+suffix+'.fits',
                             self.FMMF_map[0,k,:,:],
                             filetype=suffix)

            suffix = "FMCont-KL{0}".format(self.numbasis[k])
            dataset.savedata(outputdir+os.path.sep+fileprefix+'-'+suffix+'.fits',
                             self.contrast_map[0,k,:,:],
                             filetype=suffix)

            suffix = "FMCC-KL{0}".format(self.numbasis[k])
            dataset.savedata(outputdir+os.path.sep+fileprefix+'-'+suffix+'.fits',
                             self.FMCC_map[0,k,:,:],
                             filetype=suffix)

            suffix = "FMN_pix-KL{0}".format(self.numbasis[k])
            dataset.savedata(outputdir+os.path.sep+fileprefix+'-'+suffix+'.fits',
                             self.FMNpix_map[0,k,:,:],
                             filetype=suffix)

            if self.save_bbfm:
                suffix = "BBFM-KL{0}".format(self.numbasis[k])
                dataset.savedata(outputdir+os.path.sep+fileprefix+'-'+suffix+'.fits',
                                 fmout2,
                                 filetype=suffix)

        return

    def generate_model_sci(self, input_img_shape, section_ind, pa, wv, radstart, radend, phistart, phiend, padding, ref_center, parang, ref_wv,sep_fk,pa_fk, flipx):
        """
        Generate model PSFs at the correct location of this segment of the science image denotated by its wv and
        parallactic angle.

        Args:
            input_img_shape: 2-D shape of inpt images ([ysize, xsize])
            section_ind: array indicies into the 2-D x-y image that correspond to this section.
                         Note needs be called as section_ind[0]
            pa: parallactic angle of the science image [degrees]
            wv: wavelength of the science image
            radstart: radius of start of segment (not used)
            radend: radius of end of segment (not used)
            phistart: azimuthal start of segment [radians] (not used)
            phiend: azimuthal end of segment [radians] (not used)
            padding: amount of padding on each side of sector
            ref_center: center of image
            parang: parallactic angle of input image [DEGREES] (not used)
            ref_wv: wavelength of science image
            sep_fk: separation of the planet to be injected.
            pa_fk: position angle of the planet to be injected.
            flipx: if True, flip x coordinate in final image

        Return: (models, mask)
            models: vector of size p where p is the number of pixels in the segment
            mask: vector of size p where p is the number of pixels in the segment
                    if pixel == 1: arc shape where to calculate the standard deviation
                    if pixel == 2: 7 pixels disk around the position of the planet.
        """
        # create some parameters for a blank canvas to draw psfs on
        nx = input_img_shape[1]
        ny = input_img_shape[0]

        if hasattr(self,"x_grid") and hasattr(self,"y_grid"):
            x_grid, y_grid = self.x_grid,self.y_grid
        else:
            x_grid, y_grid = np.meshgrid(np.arange(nx * 1.)-ref_center[0], np.arange(ny * 1.)-ref_center[1])

        numwv, ny_psf, nx_psf =  self.input_psfs.shape

        # a blank img array of write model PSFs into
        whiteboard = np.zeros((ny,nx))
        # grab PSF given wavelength
        wv_index = spec.find_nearest(self.input_psfs_wvs,wv)[1]

        sign = -1.
        if flipx:
            sign = 1.

        # The trigonometric calculation are save in a dictionary to avoid calculating them many times.
        recalculate_trig = False
        if pa not in self.psf_centx_notscaled:
            recalculate_trig = True
        else:
            if pa_fk != self.curr_pa_fk[pa] or sep_fk != self.curr_sep_fk[pa]:
                recalculate_trig = True
        if recalculate_trig: # we could actually store the values for the different pas too...
            # flipx requires the opposite rotation
            self.psf_centx_notscaled[pa] = sep_fk * np.cos(np.radians(90. - sign*pa_fk - pa))
            self.psf_centy_notscaled[pa] = sep_fk * np.sin(np.radians(90. - sign*pa_fk - pa))
            self.curr_pa_fk[pa] = pa_fk
            self.curr_sep_fk[pa] = sep_fk

        psf_centx = (ref_wv/wv) * self.psf_centx_notscaled[pa]
        psf_centy = (ref_wv/wv) * self.psf_centy_notscaled[pa]

        # create a coordinate system for the image that is with respect to the model PSF
        # round to nearest pixel and add offset for center
        l = int(round(psf_centx + ref_center[0]))
        k = int(round(psf_centy + ref_center[1]))
        # recenter coordinate system about the location of the planet
        # x_vec_stamp_centered = x_grid[0, (l-col_m):(l+col_p)]-psf_centx
        # y_vec_stamp_centered = y_grid[(k-row_m):(k+row_p), 0]-psf_centy
        x_vec_stamp_centered = x_grid[0, np.max([(l-self.col_m),0]):np.min([(l+self.col_p),nx])]-psf_centx
        y_vec_stamp_centered = y_grid[np.max([(k-self.row_m),0]):np.min([(k+self.row_p),ny]), 0]-psf_centy
        # rescale to account for the align and scaling of the refernce PSFs
        # e.g. for longer wvs, the PSF has shrunk, so we need to shrink the coordinate system
        x_vec_stamp_centered /= (ref_wv/wv)
        y_vec_stamp_centered /= (ref_wv/wv)

        # use intepolation spline to generate a model PSF and write to temp img
        # whiteboard[(k-row_m):(k+row_p), (l-col_m):(l+col_p)] = \
        #         self.psfs_func_list[wv_index[0]](x_vec_stamp_centered,y_vec_stamp_centered).transpose()
        whiteboard[np.max([(k-self.row_m),0]):np.min([(k+self.row_p),ny]), np.max([(l-self.col_m),0]):np.min([(l+self.col_p),nx])] = \
                self.psfs_func_list[wv_index](x_vec_stamp_centered,y_vec_stamp_centered).transpose()

        # write model img to output (segment is collapsed in x/y so need to reshape)
        whiteboard.shape = [input_img_shape[0] * input_img_shape[1]]
        segment_with_model = copy(whiteboard[section_ind])
        whiteboard.shape = [input_img_shape[0],input_img_shape[1]]

        # Define the masks for where the planet is and the background.
        if hasattr(self,"th0_grid") and hasattr(self,"r_grid"):
            r_grid = self.r_grid
            th_grid = (self.th0_grid-sign*np.radians(pa))% (2.0 * np.pi)
        else:
            r_grid = abs(x_grid +y_grid*1j)
            th_grid = (np.arctan2(sign*x_grid,y_grid)-sign*np.radians(pa))% (2.0 * np.pi)
        w = self.background_width
        thstart = (np.radians(pa_fk)- float(w)/sep_fk) % (2.0 * np.pi) # -(2*np.pi-np.radians(pa))
        thend = (np.radians(pa_fk) + float(w)/sep_fk) % (2.0 * np.pi) # -(2*np.pi-np.radians(pa))
        # thstart = (np.radians(pa_fk)- 2*float(w)/sep_fk) % (2.0 * np.pi) # -(2*np.pi-np.radians(pa))
        #thend = (np.radians(pa_fk) + 2*float(w)/sep_fk) % (2.0 * np.pi) # -(2*np.pi-np.radians(pa))
        if thstart < thend:
            where_mask = np.where((r_grid>=(sep_fk-w)) & (r_grid<(sep_fk+w)) & (th_grid >= thstart) & (th_grid < thend))
        else:
            where_mask = np.where((r_grid>=(sep_fk-w)) & (r_grid<(sep_fk+w)) & ((th_grid >= thstart) | (th_grid < thend)))
        whiteboard[where_mask] = 1
        #TODO check the modification I did to these lines
        whiteboard = np.pad(whiteboard,((self.row_m,self.row_p),(self.col_m,self.col_p)),mode="constant",constant_values=0)
        # whiteboard[(k-row_m):(k+row_p), (l-col_m):(l+col_p)][np.where(np.isnan(self.stamp_PSF_mask))]=2
        whiteboard[(k):(k+self.row_m+self.row_p), (l):(l+self.col_m+self.col_p)][np.where(np.isnan(self.stamp_PSF_mask))]=2
        whiteboard = np.ascontiguousarray(whiteboard[self.row_m:self.row_m+input_img_shape[0],self.col_m:self.col_m+input_img_shape[1]])
        whiteboard.shape = [input_img_shape[0] * input_img_shape[1]]
        mask = whiteboard[section_ind]

        # create a canvas to place the new PSF in the sector on
        if 0:#np.size(np.where(mask==2)[0])==0: 296
            print(pa,pa_fk)
            print(thstart,thend)
            whiteboard[section_ind] = whiteboard[section_ind] + 0.5
            whiteboard.shape = (input_img_shape[0], input_img_shape[1])
            blackboard = np.zeros((ny,nx))
            blackboard.shape = [input_img_shape[0] * input_img_shape[1]]
            blackboard[section_ind] = segment_with_model
            blackboard.shape = [input_img_shape[0],input_img_shape[1]]
            plt.figure(1)
            plt.subplot(1,3,1)
            im = plt.imshow(whiteboard)
            plt.colorbar(im)
            plt.subplot(1,3,2)
            im = plt.imshow(blackboard)
            plt.colorbar(im)
            plt.subplot(1,3,3)
            im = plt.imshow(np.degrees(th_grid))
            plt.colorbar(im)
            plt.show()

        return segment_with_model,mask

    def generate_models(self, input_img_shape, section_ind, pas, wvs, radstart, radend, phistart, phiend, padding, ref_center, parang, ref_wv,sep_fk,pa_fk,flipx):
        """
        Generate model PSFs at the correct location of this segment for each image denotated by its wv and parallactic
        angle.

        Args:
            input_img_shape: 2-D shape of inpt images ([ysize, xsize])
            section_ind: array indicies into the 2-D x-y image that correspond to this section.
                         Note needs be called as section_ind[0]
            pas: array of N parallactic angles corresponding to N images [degrees]
            wvs: array of N wavelengths of those images
            radstart: radius of start of segment (not used)
            radend: radius of end of segment (not used)
            phistart: azimuthal start of segment [radians] (not used)
            phiend: azimuthal end of segment [radians] (not used)
            padding: amount of padding on each side of sector
            ref_center: center of image
            parang: parallactic angle of input image [DEGREES] (not used)
            ref_wv: wavelength of science image
            sep_fk: separation of the planet to be injected.
            pa_fk: position angle of the planet to be injected.
            flipx: if True, flip x coordinate in final image

        Return:
            models: array of size (N, p) where p is the number of pixels in the segment
        """
        # create some parameters for a blank canvas to draw psfs on
        nx = input_img_shape[1]
        ny = input_img_shape[0]
        try:
            x_grid, y_grid = self.x_grid,self.y_grid
        except:
            x_grid, y_grid = np.meshgrid(np.arange(nx * 1.)-ref_center[0], np.arange(ny * 1.)-ref_center[1])

        sign = -1.
        if flipx:
            sign = 1.

        # a blank img array of write model PSFs into
        whiteboard = np.zeros((ny,nx))
        models = []
        #print(self.input_psfs.shape)
        for pa, wv in zip(pas, wvs):
            # grab PSF given wavelength
            wv_index = [spec.find_nearest(self.input_psfs_wvs,wv)[1]]

            # find center of psf
            # to reduce calculation of sin and cos, see if it has already been calculated before
            recalculate_trig = False
            if pa not in self.psf_centx_notscaled:
                recalculate_trig = True
            else:
                #print(self.psf_centx_notscaled[pa],pa)
                if pa_fk != self.curr_pa_fk[pa] or sep_fk != self.curr_sep_fk[pa]:
                    recalculate_trig = True
            if recalculate_trig: # we could actually store the values for the different pas too...
                self.psf_centx_notscaled[pa] = sep_fk * np.cos(np.radians(90. - sign*pa_fk - pa))
                self.psf_centy_notscaled[pa] = sep_fk * np.sin(np.radians(90. - sign*pa_fk - pa))
                self.curr_pa_fk[pa] = pa_fk
                self.curr_sep_fk[pa] = sep_fk

            psf_centx = (ref_wv/wv) * self.psf_centx_notscaled[pa]
            psf_centy = (ref_wv/wv) * self.psf_centy_notscaled[pa]

            # create a coordinate system for the image that is with respect to the model PSF
            # round to nearest pixel and add offset for center
            l = int(round(psf_centx + ref_center[0]))
            k = int(round(psf_centy + ref_center[1]))
            # recenter coordinate system about the location of the planet
            x_vec_stamp_centered = x_grid[0, (l-self.col_m):(l+self.col_p)]-psf_centx
            y_vec_stamp_centered = y_grid[(k-self.row_m):(k+self.row_p), 0]-psf_centy
            # rescale to account for the align and scaling of the refernce PSFs
            # e.g. for longer wvs, the PSF has shrunk, so we need to shrink the coordinate system
            x_vec_stamp_centered /= (ref_wv/wv)
            y_vec_stamp_centered /= (ref_wv/wv)

            # use intepolation spline to generate a model PSF and write to temp img
            whiteboard[(k-self.row_m):(k+self.row_p), (l-self.col_m):(l+self.col_p)] = \
                    self.psfs_func_list[wv_index[0]](x_vec_stamp_centered,y_vec_stamp_centered).transpose()

            # write model img to output (segment is collapsed in x/y so need to reshape)
            whiteboard.shape = [input_img_shape[0] * input_img_shape[1]]
            segment_with_model = copy(whiteboard[section_ind])
            whiteboard.shape = [input_img_shape[0],input_img_shape[1]]

            models.append(segment_with_model)

            # create a canvas to place the new PSF in the sector on
            if 0:
                blackboard = np.zeros((ny,nx))
                blackboard.shape = [input_img_shape[0] * input_img_shape[1]]
                blackboard[section_ind] = segment_with_model
                blackboard.shape = [input_img_shape[0],input_img_shape[1]]
                plt.figure(1)
                plt.subplot(1,2,1)
                im = plt.imshow(whiteboard)
                plt.colorbar(im)
                plt.subplot(1,2,2)
                im = plt.imshow(blackboard)
                plt.colorbar(im)
                plt.show()

            whiteboard[(k-self.row_m):(k+self.row_p), (l-self.col_m):(l+self.col_p)] = 0.0

        return np.array(models)



def calculate_fm_opti(delta_KL, original_KL, sci, model_sci_fk,delta_KL_fk, original_KL_fk):
    r"""
    Optimized version for calculate_fm() (if numbas) for a single numbasis.

    Calculate what the PSF looks up post-KLIP using knowledge of the input PSF, assumed spectrum of the science target,
    and the partially calculated KL modes (\Delta Z_k^\lambda in Laurent's paper). If inputflux is None,
    the spectral dependence has already been folded into delta_KL_nospec (treat it as delta_KL).

    Note: if inputflux is None and delta_KL_nospec has three dimensions (ie delta_KL_nospec was calculated using
    pertrurb_nospec() or perturb_nospec_modelsBased()) then only klipped_oversub and klipped_selfsub are returned.
    Besides they will have an extra first spectral dimension.

    Args:
        delta_KL_nospec: perturbed KL modes but without the spectral info. delta_KL = spectrum x delta_Kl_nospec.
                         Shape is (numKL, wv, pix). If inputflux is None, delta_KL_nospec = delta_KL
        orignal_KL: unpertrubed KL modes (array of size [numbasis, numpix])
        sci: array of size p representing the science data
        model_sci: array of size p corresponding to the PSF of the science frame
        input_spectrum: array of size wv with the assumed spectrum of the model

    If delta_KL_nospec does NOT include a spectral dimension or if inputflux is not None:
    Returns:
        fm_psf: array of shape (b,p) showing the forward modelled PSF
                Skipped if inputflux = None, and delta_KL_nospec has 3 dimensions.
        klipped_oversub: array of shape (b, p) showing the effect of oversubtraction as a function of KL modes
        klipped_selfsub: array of shape (b, p) showing the effect of selfsubtraction as a function of KL modes
        Note: psf_FM = model_sci - klipped_oversub - klipped_selfsub to get the FM psf as a function of K Lmodes
              (shape of b,p)

    If inputflux = None and if delta_KL_nospec include a spectral dimension:
    Returns:
        klipped_oversub: Sum(<S|KL>KL) with klipped_oversub.shape = (1,Npix)
        klipped_selfsub: Sum(<N|DKL>KL) + Sum(<N|KL>DKL) with klipped_selfsub.shape = (1,N_lambda or N_ref,N_pix)

    """
    # remove means and nans from science image
    sci_mean_sub = (sci - np.nanmean(sci))[None,:]
    sci_mean_sub[np.where(np.isnan(sci_mean_sub))] =0


    # science PSF models, ready for FM
    # /!\ JB: If subtracting the mean. It should be done here. not in klip_math since we don't use model_sci there.
    model_sci_mean_sub = model_sci_fk[None,:] # should be subtracting off the mean?
    model_sci_mean_sub[np.where(np.isnan(model_sci_mean_sub))] =0


    # Forward model the PSF
    # 3 terms: 1 for oversubtracton (planet attenauted by speckle KL modes),
    # and 2 terms for self subtraction (planet signal leaks in KL modes which get projected onto speckles)
    #
    # Klipped = N-Sum(<N|KL>KL) + S-Sum(<S|KL>KL) - Sum(<N|DKL>KL) - Sum(<N|KL>DKL)
    # With  N = noise/speckles (science image)
    #       S = signal/planet model
    #       KL = KL modes
    #       DKL = perturbation of the KL modes/Delta_KL
    #
    # sci_mean_sub_rows.shape = (1,N_pix)
    # model_sci_mean_sub_rows.shape = (1,N_pix)
    # original_KL.shape = (max_basis,N_pix)
    # delta_KL.shape = (max_basis,N_pix)
    oversubtraction_inner_products = np.dot(model_sci_mean_sub, original_KL_fk.T)
    selfsubtraction_1_inner_products = np.dot(sci_mean_sub, delta_KL.T)
    selfsubtraction_2_inner_products = np.dot(sci_mean_sub, original_KL.T)

    # oversubtraction_inner_products = (1,numbasis)
    klipped_oversub = np.dot(oversubtraction_inner_products, original_KL_fk)

    # selfsubtraction_1_inner_products = (1,numbasis)
    # selfsubtraction_2_inner_products = (1,numbasis)
    klipped_selfsub = np.dot(selfsubtraction_1_inner_products, original_KL_fk) + \
                      np.dot(selfsubtraction_2_inner_products, delta_KL_fk)

    return model_sci_fk[None,:] - klipped_oversub - klipped_selfsub