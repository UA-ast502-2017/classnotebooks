__author__ = 'jruffio'
import multiprocessing as mp
import ctypes

import numpy as np
import pyklip.spectra_management as spec
import os

from pyklip.fmlib.nofm import NoFM
import pyklip.fm as fm
import pyklip.fakes as fakes

from scipy import interpolate, linalg
from copy import copy

#import matplotlib.pyplot as plt
debug = False


class ExtractSpec(NoFM):
    """
    Planet Characterization class. Goal to characterize the astrometry and photometry of a planet
    """
    def __init__(self, inputs_shape,
                 numbasis,
                 sep, pa,
                 input_psfs,
                 input_psfs_wvs,
                 datatype="float",
                 stamp_size = None):
        """
        Defining the planet to characterizae

        Args:
            inputs_shape: shape of the inputs numpy array. Typically (N, y, x)
            numbasis: 1d numpy array consisting of the number of basis vectors to use
            sep: separation of the planet
            pa: position angle of the planet
            input_psfs: the psf of the image. A numpy array with shape (wv, y, x)
            input_psfs_wvs: the wavelegnths that correspond to the input psfs
            flux_conversion: an array of length N to convert from contrast to DN for each frame. Units of DN/contrast
            wavelengths: wavelengths of data. Can just be a string like 'H' for H-band
            spectrallib: if not None, a list of spectra
            star_spt: star spectral type, if None default to some random one
            refine_fit: refine the separation and pa supplied
        """
        # allocate super class
        super(ExtractSpec, self).__init__(inputs_shape, np.array(numbasis))

        if stamp_size is None:
            self.stamp_size = 10
        else:
            self.stamp_size = stamp_size

        if datatype=="double":
            self.data_type = ctypes.c_double
        elif datatype=="float":
            self.data_type = ctypes.c_float

        self.N_numbasis =  np.size(numbasis)
        self.ny = self.inputs_shape[1]
        self.nx = self.inputs_shape[2]
        self.N_frames = self.inputs_shape[0]

        self.inputs_shape = inputs_shape
        self.numbasis = numbasis
        self.sep = sep
        self.pa = pa


        self.input_psfs = input_psfs
        self.input_psfs_wvs = list(np.array(input_psfs_wvs,dtype=self.data_type))
        self.nl = np.size(input_psfs_wvs)
        #self.flux_conversion = flux_conversion
        # Make sure the peak value is unity for all wavelengths
        self.sat_spot_spec = np.nanmax(self.input_psfs,axis=(1,2))
        self.aper_over_peak_ratio = np.zeros(np.size(self.input_psfs_wvs))
        for l_id in range(self.input_psfs.shape[0]):
            self.aper_over_peak_ratio[l_id] = np.nansum(self.input_psfs[l_id,:,:])/self.sat_spot_spec[l_id]
            self.input_psfs[l_id,:,:] = self.input_psfs[l_id,:,:]/np.nansum(self.input_psfs[l_id,:,:])

        self.nl, self.ny_psf, self.nx_psf =  self.input_psfs.shape

        self.psf_centx_notscaled = {}
        self.psf_centy_notscaled = {}

        numwv,ny_psf,nx_psf =  self.input_psfs.shape
        x_psf_grid, y_psf_grid = np.meshgrid(np.arange(nx_psf * 1.)-nx_psf/2,np.arange(ny_psf* 1.)-ny_psf/2)
        psfs_func_list = []
        for wv_index in range(numwv):
            model_psf = self.input_psfs[wv_index, :, :] #* self.flux_conversion * self.spectrallib[0][wv_index] * self.dflux
            psfs_func_list.append(interpolate.LSQBivariateSpline(x_psf_grid.ravel(),y_psf_grid.ravel(),model_psf.ravel(),x_psf_grid[0,0:nx_psf-1]+0.5,y_psf_grid[0:ny_psf-1,0]+0.5))

        self.psfs_func_list = psfs_func_list


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
    #
    #     interm_size = max_sector_size * np.size(self.numbasis) * numsciframes * len(self.spectrallib)
    #
    #     interm = mp.Array(ctypes.c_double, interm_size)
    #     interm_shape = [numsciframes, len(self.spectrallib), max_sector_size, np.size(self.numbasis)]
    #
    #     return interm, interm_shape


    def alloc_fmout(self, output_img_shape):
        """
        Allocates shared memory for the output of the shared memory

        Args:
            output_img_shape: shape of output image (usually N,y,x,b)

        Returns:
            fmout: mp.array to store FM data in
            fmout_shape: shape of FM data array

        """

        # The 3rd dimension (self.N_frames corresponds to the spectrum)
        # The +1 in (self.N_frames+1) is for the klipped image
        fmout_size = self.N_numbasis*self.N_frames*(self.N_frames+1)*self.stamp_size*self.stamp_size
        fmout = mp.Array(self.data_type, fmout_size)
        # fmout shape is defined as:
        #   (self.N_numbasis,self.N_frames,(self.N_frames+1),self.stamp_size*self.stamp_size)
        # 1st dim: The size of the numbasis input. numasis gives the list of the number of KL modes we want to try out
        #           e.g. numbasis = [10,20,50].
        # 2nd dim: It is the Forward model dimension. It contains the forard model for each frame in the dataset.
        #           N_frames = N_cubes*(Number of spectral channel=37)
        # 3nd dim: It contains both the "spectral dimension" and the klipped image.
        #           The regular klipped data is fmout[:,:, -1,:]
        #           The regular forward model is fmout[:,:, 0:self.N_frames,:]
        #           Multiply a vector of fluxes to this dimension of fmout[:,:, 0:self.N_frames,:] and you should get
        #           forward model for that given spectrum.
        # 4th dim: pixels value. It has the size of the number of pixels in the stamp self.stamp_size*self.stamp_size.
        fmout_shape = (self.N_numbasis,self.N_frames,(self.N_frames+1),self.stamp_size*self.stamp_size )

        return fmout, fmout_shape


    # def alloc_perturbmag(self, output_img_shape, numbasis):
    #     """
    #     Allocates shared memory to store the fractional magnitude of the linear KLIP perturbation
    #     Stores a number for each frame = max(oversub + selfsub)/std(PCA(image))
    #
    #     Args:
    #         output_img_shape: shape of output image (usually N,y,x,b)
    #         numbasis: array/list of number of KL basis cutoffs requested
    #
    #     Returns:
    #         perturbmag: mp.array to store linaer perturbation magnitude
    #         perturbmag_shape: shape of linear perturbation magnitude
    #
    #     """
    #     perturbmag_shape = (output_img_shape[0], np.size(numbasis))
    #     perturbmag = mp.Array(ctypes.c_double, np.prod(perturbmag_shape))
    #
    #     return perturbmag, perturbmag_shape


    def generate_models(self, input_img_shape, section_ind, pas, wvs, radstart, radend, phistart, phiend, padding, ref_center, parang, ref_wv,stamp_size = None):
        """
        Generate model PSFs at the correct location of this segment for each image denoated by its wv and parallactic angle

        Args:
            pas: array of N parallactic angles corresponding to N images [degrees]
            wvs: array of N wavelengths of those images
            radstart: radius of start of segment
            radend: radius of end of segment
            phistart: azimuthal start of segment [radians]
            phiend: azimuthal end of segment [radians]
            padding: amount of padding on each side of sector
            ref_center: center of image
            parang: parallactic angle of input image [DEGREES]
            ref_wv: wavelength of science image
            stamp_size: size of the stamp for spectral extraction

        Return:
            models: array of size (N, p) where p is the number of pixels in the segment
        """
        # create some parameters for a blank canvas to draw psfs on
        nx = input_img_shape[1]
        ny = input_img_shape[0]
        x_grid, y_grid = np.meshgrid(np.arange(nx * 1.)-ref_center[0], np.arange(ny * 1.)-ref_center[1])


        numwv, ny_psf, nx_psf =  self.input_psfs.shape

        # create bounds for PSF stamp size
        row_m = np.floor(ny_psf/2.0)    # row_minus
        row_p = np.ceil(ny_psf/2.0)     # row_plus
        col_m = np.floor(nx_psf/2.0)    # col_minus
        col_p = np.ceil(nx_psf/2.0)     # col_plus

        if stamp_size is not None:
            stamp_mask = np.zeros((ny,nx))
            # create bounds for spectral extraction stamp size
            row_m_stamp = np.floor(stamp_size/2.0)    # row_minus
            row_p_stamp = np.ceil(stamp_size/2.0)     # row_plus
            col_m_stamp = np.floor(stamp_size/2.0)    # col_minus
            col_p_stamp = np.ceil(stamp_size/2.0)     # col_plus
            stamp_indices=[]

        # a blank img array of write model PSFs into
        whiteboard = np.zeros((ny,nx))
        if debug:
            canvases = []
        models = []
        #print(self.input_psfs.shape)
        for pa, wv in zip(pas, wvs):
            #print(self.pa,self.sep)
            #print(pa,wv)
            # grab PSF given wavelength
            wv_index = spec.find_nearest(self.input_psfs_wvs,wv)[1]
            #model_psf = self.input_psfs[wv_index[0], :, :] #* self.flux_conversion * self.spectrallib[0][wv_index] * self.dflux

            # find center of psf
            # to reduce calculation of sin and cos, see if it has already been calculated before
            if pa not in self.psf_centx_notscaled:
                self.psf_centx_notscaled[pa] = self.sep * np.cos(np.radians(90. - self.pa - pa))
                self.psf_centy_notscaled[pa] = self.sep * np.sin(np.radians(90. - self.pa - pa))
            psf_centx = (ref_wv/wv) * self.psf_centx_notscaled[pa]
            psf_centy = (ref_wv/wv) * self.psf_centy_notscaled[pa]

            # create a coordinate system for the image that is with respect to the model PSF
            # round to nearest pixel and add offset for center
            l = round(psf_centx + ref_center[0])
            k = round(psf_centy + ref_center[1])
            # recenter coordinate system about the location of the planet
            x_vec_stamp_centered = x_grid[0, int(l-col_m):int(l+col_p)]-psf_centx
            y_vec_stamp_centered = y_grid[int(k-row_m):int(k+row_p), 0]-psf_centy
            # rescale to account for the align and scaling of the refernce PSFs
            # e.g. for longer wvs, the PSF has shrunk, so we need to shrink the coordinate system
            x_vec_stamp_centered /= (ref_wv/wv)
            y_vec_stamp_centered /= (ref_wv/wv)

            # use intepolation spline to generate a model PSF and write to temp img
            whiteboard[int(k-row_m):int(k+row_p), int(l-col_m):int(l+col_p)] = \
                    self.psfs_func_list[int(wv_index)](x_vec_stamp_centered,y_vec_stamp_centered).transpose()

            # write model img to output (segment is collapsed in x/y so need to reshape)
            whiteboard.shape = [input_img_shape[0] * input_img_shape[1]]
            segment_with_model = copy(whiteboard[section_ind])
            whiteboard.shape = [input_img_shape[0],input_img_shape[1]]

            models.append(segment_with_model)

            if stamp_size is not None:
                # These are actually indices of indices. they indicate which indices correspond to the stamp in section_ind
                stamp_mask[int(k-row_m_stamp):int(k+row_p_stamp), int(l-col_m_stamp):int(l+col_p_stamp)] = 1
                stamp_mask.shape = [nx*ny]
                stamp_indices.append(np.where(stamp_mask[section_ind] == 1)[0])
                stamp_mask.shape = [ny,nx]
                stamp_mask[int(k-row_m_stamp):int(k+row_p_stamp), int(l-col_m_stamp):int(l+col_p_stamp)] = 0

        if stamp_size is not None:
            return np.array(models),stamp_indices
        else:
            return np.array(models)




    def fm_from_eigen(self, klmodes=None, evals=None, evecs=None, input_img_shape=None, input_img_num=None, ref_psfs_indicies=None, section_ind=None,section_ind_nopadding=None, aligned_imgs=None, pas=None,
                     wvs=None, radstart=None, radend=None, phistart=None, phiend=None, padding=None,IOWA = None, ref_center=None,
                     parang=None, ref_wv=None, numbasis=None, fmout=None, perturbmag=None, klipped=None, **kwargs):
        """
        Generate forward models using the KL modes, eigenvectors, and eigenvectors from KLIP. Calls fm.py functions to
        perform the forward modelling

        Args:
            klmodes: unpertrubed KL modes
            evals: eigenvalues of the covariance matrix that generated the KL modes in ascending order
                   (lambda_0 is the 0 index) (shape of [nummaxKL])
            evecs: corresponding eigenvectors (shape of [p, nummaxKL])
            input_image_shape: 2-D shape of inpt images ([ysize, xsize])
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
            perturbmag: numpy output for size of linear perturbation. Shape is (N, b)
            klipped: PSF subtracted image. Shape of ( size(section), b)
            kwargs: any other variables that we don't use but are part of the input
        """
        sci = aligned_imgs[input_img_num, section_ind[0]]
        refs = aligned_imgs[ref_psfs_indicies, :]
        refs = refs[:, section_ind[0]]


        # generate models for the PSF of the science image
        model_sci, stamp_indices = self.generate_models(input_img_shape, section_ind, [parang], [ref_wv], radstart, radend, phistart, phiend, padding, ref_center, parang, ref_wv,stamp_size=self.stamp_size)
        model_sci = model_sci[0]
        stamp_indices = stamp_indices[0]

        # generate models of the PSF for each reference segments. Output is of shape (N, pix_in_segment)
        models_ref = self.generate_models(input_img_shape, section_ind, pas, wvs, radstart, radend, phistart, phiend, padding, ref_center, parang, ref_wv)

        # using original Kl modes and reference models, compute the perturbed KL modes (spectra is already in models)
        #delta_KL = fm.perturb_specIncluded(evals, evecs, klmodes, refs, models_ref)
        delta_KL_nospec = fm.pertrurb_nospec(evals, evecs, klmodes, refs, models_ref)

        # calculate postklip_psf using delta_KL
        oversubtraction, selfsubtraction = fm.calculate_fm(delta_KL_nospec, klmodes, numbasis, sci, model_sci, inputflux=None)
        # klipped_oversub.shape = (size(numbasis),Npix)
        # klipped_selfsub.shape = (size(numbasis),N_lambda or N_ref,N_pix)
        # klipped_oversub = Sum(<S|KL>KL)
        # klipped_selfsub = Sum(<N|DKL>KL) + Sum(<N|KL>DKL)


        # Note: The following could be used if we want to derotate the image but JB doesn't think we have to.
        # # write forward modelled PSF to fmout (as output)
        # # need to derotate the image in this step
        # for thisnumbasisindex in range(np.size(numbasis)):
        #         fm._save_rotated_section(input_img_shape, postklip_psf[thisnumbasisindex], section_ind,
        #                          fmout[input_img_num, :, :,thisnumbasisindex], None, parang,
        #                          radstart, radend, phistart, phiend, padding,IOWA, ref_center, flipx=True)


        # fmout shape is defined as:
        #   (self.N_numbasis,self.N_frames,(self.N_frames+1),self.stamp_size*self.stamp_size)
        # 1st dim: The size of the numbasis input. numasis gives the list of the number of KL modes we want to try out
        #           e.g. numbasis = [10,20,50].
        # 2nd dim: It is the Forward model dimension. It contains the forard model for each frame in the dataset.
        #           N_frames = N_cubes*(Number of spectral channel=37)
        # 3nd dim: It contains both the "spectral dimension" and the klipped image.
        #           The regular klipped data is fmout[:,:, -1,:]
        #           The regular forward model is fmout[:,:, 0:self.N_frames,:]
        #           Multiply a vector of fluxes to this dimension of fmout[:,:, 0:self.N_frames,:] and you should get
        #           forward model for that given spectrum.
        # 4th dim: pixels value. It has the size of the number of pixels in the stamp self.stamp_size*self.stamp_size.
        for k in range(self.N_numbasis):
            fmout[k,input_img_num, input_img_num,:] = fmout[k,input_img_num, input_img_num,:]+model_sci[stamp_indices]
        fmout[:,input_img_num, input_img_num,:] = fmout[:,input_img_num, input_img_num,:]-oversubtraction[:,stamp_indices]
        fmout[:,input_img_num, ref_psfs_indicies,:] = fmout[:,input_img_num, ref_psfs_indicies,:]-selfsubtraction[:,:,stamp_indices]
        fmout[:,input_img_num, -1,:] = klipped.T[:,stamp_indices]




    def cleanup_fmout(self, fmout):
        """
        After running KLIP-FM, we need to reshape fmout so that the numKL dimension is the first one and not the last

        Args:
            fmout: numpy array of ouput of FM

        Return:
            fmout: same but cleaned up if necessary
        """
        # Here we actually extract the spectrum


        return fmout

def gen_fm(dataset, pars, numbasis = 20, mv = 2.0, stamp=10, numthreads=4,
           maxnumbasis = 100, spectra_template=None, manual_psfs=None, aligned_center=None):
    """
    inputs: 
    - pars              - tuple of planet position (sep (pixels), pa (deg)).
    - numbasis          - can be a list or a single number
    - mv                - klip movement (pixels)
    - stamp             - size of box around companion for FM
    - numthreads        (default=4)
    - spectra_template  - Can provide a template, default is None
    - manual_psfs       - If dataset does not have attribute "psfs" will look for
                        manual input of psf model.
    - aligned_center    - pass to klip_dataset
    """

    maxnumbasis = maxnumbasis
    movement = mv
    stamp_size=stamp
    N_frames = len(dataset.input)
    N_cubes = len(dataset.exthdrs)
    nl = N_frames // N_cubes

    print("====================================")
    print("planet separation, pa: {0}".format(pars))
    print("numbasis: {0}".format(numbasis))
    print("movement: {0}".format(mv))
    print("====================================")
    print("Generating forward model...")

    planet_sep, planet_pa = pars

    # If 'dataset' does not already have psf model, check if manual_psfs not None.
    if hasattr(dataset, "psfs"):
        print("Using dataset PSF model.")
        # What is this normalization? Not sure it matters (see line 82).
        radial_psfs = dataset.psfs / \
            (np.mean(dataset.spot_flux.reshape([dataset.spot_flux.shape[0]//nl, nl]),\
             axis=0)[:, None, None])
    elif manual_psfs is not None:
        radial_psfs = manual_psfs
    else:
        raise AttributeError("dataset has no psfs attribute. \n"+\
              "Either run dataset.generate_psfs before gen_fm or"+\
              "provide psf models in keyword manual_psfs. \n"+\
              "examples/FM_spectral_extraction_tutorial.py for example.")

    # The forward model class
    fm_class = ExtractSpec(dataset.input.shape,
                           numbasis,
                           planet_sep,
                           planet_pa,
                           radial_psfs,
                           np.unique(dataset.wvs),
                           stamp_size = stamp_size)

    # Now run KLIP!
    fm.klip_dataset(dataset, fm_class,
                    fileprefix="fmspect",
                    annuli=[[planet_sep-stamp,planet_sep+stamp]],
                    subsections=[[(planet_pa-stamp)/180.*np.pi,\
                                  (planet_pa+stamp)/180.*np.pi]],
                    movement=movement,
                    numbasis = numbasis, 
                    maxnumbasis=maxnumbasis,
                    numthreads=numthreads,
                    spectrum=spectra_template,
                    save_klipped=False, highpass=True,
                    aligned_center=aligned_center)

    return dataset.fmout

def invert_spect_fmodel(fmout, dataset, method = "JB", units = "DN"):
    """
    A. Greenbaum Nov 2016
    
    Args:
        fmout: the forward model matrix which has structure:
               [numbasis, n_frames, n_frames+1, npix]
        dataset: from GPI.GPIData(filelist) -- typically set highpass=True also
        method: "JB" or "LP" to try the 2 different inversion methods (JB's or Laurent's)
        units: "DN" means raw data number units (not converted to contrast)
               "CONTRAST" is normalized to contrast units. You can only use this if
                          spot fluxes are saved in dataset.spot_flux
               default is 'DN' require user to do their own calibration for contrast.
    Returns:
        A tuple containing the spectrum and the forward model
        (spectrum, forwardmodel)
        The spectrum:
            default units=DN unless kwarg unit="CONTRAST"
            spectrum shape:(len(numbasis), nwav)
        The forward model:
        
    """
    N_frames = fmout.shape[2] - 1 # The last element in this axis contains klipped image
    N_cubes = len(dataset.exthdrs) # 
    nl = N_frames // N_cubes
    stamp_size_squared = fmout.shape[-1]
    stamp_size = np.sqrt(stamp_size_squared)

    # Selection matrix (N_cubes, 1) shape
    spec_identity = np.identity(nl)
    selec = np.tile(spec_identity,(N_frames//nl, 1))

    # set up array for klipped image for each numbasis, n_frames x npix
    klipped = np.zeros((fmout.shape[0], fmout.shape[1], fmout.shape[3]))
    estim_spec = np.zeros((fmout.shape[0], nl))

    # The first dimension in fmout is numbasis, and there can be multiple of these,
    # Especially if you want to see how the spectrum behaves when you change parameters.
    # We'll also set aside an array to store the forward model matrix
    fm_coadd_mat = np.zeros((len(fmout), nl*stamp_size_squared, nl))
    for ii in range(len(fmout)):
        klipped[ii, ...] = fmout[ii,:, -1,:]
        # klipped_coadd will be coadded over N_cubes
        klipped_coadd = np.zeros((int(nl),int(stamp_size_squared)))
        for k in range(N_cubes):
            klipped_coadd = klipped_coadd + klipped[ii, k*nl:(k+1)*nl,:]
        print(klipped_coadd.shape)
        klipped_coadd.shape = [int(nl),int(stamp_size),int(stamp_size)]
        # This is the 'raw' forward model, need to rearrange to solve FM*spec = klipped
        FM_noSpec = fmout[ii, :,:N_frames, :]

        # Move spectral dimension to the end (Effectively move pixel dimension to the middle)
        # [nframes, nframes, npix] -> [nframes, npix, nframes]
        FM_noSpec = np.rollaxis(FM_noSpec, 2, 1)

        # S^T . FM[npix, nframes, nframes] . S
        # essentially coadds over N_cubes via selection matrix
        # reduces to [nwav, npix, nwav]
        fm_noSpec_coadd = np.dot(selec.T,np.dot(np.rollaxis(FM_noSpec,1,0),selec))
        if method == "JB":
            #
            #JBR's matrix inversion adds up over all exposures, then inverts
            #
            #Back to a 2D array pixel array in the middle
            fm_noSpec_coadd.shape = [int(nl),int(stamp_size),int(stamp_size),int(nl)]
            # Flatten over first 3 dims for the FM matrix to solve FM*spect = klipped
            fm_noSpec_coadd_mat = np.reshape(fm_noSpec_coadd,(int(nl*stamp_size_squared),int(nl)))
            # Invert the FM matrix
            pinv_fm_coadd_mat = np.linalg.pinv(fm_noSpec_coadd_mat)
            # solve via FM^-1 . klipped_PSF (flattened) << both are coadded over N_cubes
            estim_spec[ii,:]=np.dot(pinv_fm_coadd_mat,np.reshape(klipped_coadd,(int(nl*stamp_size_squared),)))
            fm_coadd_mat[ii,:, :] = fm_noSpec_coadd_mat
        elif method == "LP":
            #
            #LP's matrix inversion adds over frames and one wavelength axis, then inverts
            #
            A = np.zeros((nl, nl))
            b = np.zeros(nl)
            fm = fm_noSpec_coadd.reshape(int(nl), int(stamp_size*stamp_size),int(nl))
            fm_coadd_mat[ii,:, :] = \
                fm_noSpec_coadd.reshape(int(nl*stamp_size_squared), int(nl))
            fm = np.rollaxis(fm, 2,0)
            fm = np.rollaxis(fm, 2,1)
            data = klipped_coadd.reshape(int(nl), int(stamp_size*stamp_size))
            for q in range(nl):
                A[q,:] = np.dot(fm[q,:].T,fm[q,:])[q,:]
                b[q] = np.dot(fm[q,:].T,data[q])[q]
            estim_spec[ii,:] = np.dot(np.linalg.inv(A), b)
        elif method == "leastsq":
            # MF's suggestion of solving using a least sq function 
            # instead of matrix inversions
            
            #Back to a 2D array pixel array in the middle
            fm_noSpec_coadd.shape = [int(nl),int(stamp_size),int(stamp_size),int(nl)]
            # Flatten over first 3 dims for the FM matrix to solve FM*spect = klipped
            fm_noSpec_coadd_mat = np.reshape(fm_noSpec_coadd,(int(nl*stamp_size_squared),int(nl)))
            # Saving the coadded FM
            fm_coadd_mat[ii,:, :] = fm_noSpec_coadd_mat

            # properly flatten
            flat_klipped_coadd = np.reshape(klipped_coadd, (int(nl*stamp_size_squared),))
            # used leastsq solver
            results = linalg.lstsq(fm_noSpec_coadd_mat, flat_klipped_coadd)
            # grab the spectrum, not using the other parts for now.
            estim_spec[ii,:], res, rank, s = results

        else:
            print("method not understood. Choose either JB, LP or leastsq.")

    # We can do a contrast conversion here if kwarg units = "CONTRAST"
    # But the default is in raw data units, and it's probably best to stick to that.
    if units == 'CONTRAST':
        # From JB's code, normalize by sum / peak ratio:
        # First set up the PSF model and sums
        sat_spot_sum = np.sum(dataset.psfs, axis=(1,2))
        PSF_cube = dataset.psfs / sat_spot_sum[:,None,None]
        sat_spot_spec = np.nanmax(PSF_cube, axis=(1,2))
        # Now devide the sum by the peak for each wavelength slice
        aper_over_peak_ratio = np.zeros(nl)
        for l_id in range(PSF_cube.shape[0]):
            aper_over_peak_ratio[l_id] = \
                np.nansum(PSF_cube[l_id,:,:]) / sat_spot_spec[l_id]

        # Alex's normalization terms:
        # Avg spot ratio
        band = dataset.prihdrs[0]['APODIZER'].split('_')[1]
        # CANNOT USE dataset.band !!! (always returns K1 for some reason)
        spot_flux_spectrum = \
            np.median(dataset.spot_flux.reshape(len(dataset.spot_flux)//nl, nl), axis=0)
        spot_to_star_ratio = dataset.spot_ratio[band]
        normfactor = aper_over_peak_ratio*spot_flux_spectrum / spot_to_star_ratio
        spec_unit = "CONTRAST"

        return estim_spec / normfactor, fm_coadd_mat
    else:
        spec_unit = "DN"
        return estim_spec, fm_coadd_mat
