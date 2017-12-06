import multiprocessing as mp
import ctypes

import numpy as np
import pyklip.spectra_management as spec
import os

from pyklip.fmlib.nofm import NoFM
import pyklip.fm as fm

from scipy import interpolate
from copy import copy

debug = False


class FMPlanetPSF(NoFM):
    """
    Forward models the PSF of the planet through KLIP. Returns the forward modelled planet PSF
    """
    def __init__(self, inputs_shape, numbasis, sep, pa, dflux, input_psfs, input_wvs, flux_conversion=None, spectrallib=None, spectrallib_units="flux", star_spt=None, refine_fit=False):
        """
        Defining the planet to characterizae

        Args:
            inputs_shape: shape of the inputs numpy array. Typically (N, y, x)
            numbasis: 1d numpy array consisting of the number of basis vectors to use
            sep: separation of the planet
            pa: position angle of the planet
            dflux: guess for contrast of planet averaged across band w.r.t star
            input_psfs: the psf of the image. A numpy array with shape (wv, y, x)
            input_wvs: the wavelegnths that correspond to the input psfs
                (doesn't need to be tiled to match the dimension of the input data of the instrument class)
            flux_conversion: an array of length N to convert from contrast to DN for each frame. Units of DN/contrast. 
                             If None, assumes dflux is the ratio between the planet flux and tthe input_psfs flux
            spectrallib: if not None, a list of spectra based on input_wvs
            spectrallib_units: can be either "flux"" or "contrast". Flux units requires dividing by the flux of the star to get contrast 
            star_spt: star spectral type, if None default to some random one
            refine_fit: (NOT implemented) refine the separation and pa supplied
        """
        # allocate super class
        super(FMPlanetPSF, self).__init__(inputs_shape, numbasis)

        self.inputs_shape = inputs_shape
        self.numbasis = numbasis
        self.sep = sep
        self.pa = pa
        self.dflux = dflux

        if spectrallib_units.lower() != "flux" and spectrallib_units.lower() != "contrast":
            raise ValueError("spectrallib_units needs to be either 'flux' or 'contrast', not {0}".format(spectrallib_units))

        # only need spectral info if not broadband
        numwvs = np.size(input_wvs)
        if numwvs  > 1:
            if spectrallib is not None:
                self.spectrallib = spectrallib
            else:
                spectra_folder = os.path.dirname(os.path.abspath(spec.__file__)) + os.sep + "spectra" + os.sep
                spectra_files = [spectra_folder + "t650g18nc.flx"]
                self.spectrallib = [spec.get_planet_spectrum(filename, input_wvs)[1] for filename in spectra_files]

            # TODO: calibrate to contrast units
            # calibrate spectra to DN
            if spectrallib_units.lower() == "flux":
                # need to divide by flux of the star to get contrast units
                self.spectrallib = [spectrum/(spec.get_star_spectrum(input_wvs, star_type=star_spt)[1]) for spectrum in self.spectrallib]
            self.spectrallib = [spectrum/np.mean(spectrum) for spectrum in self.spectrallib]
        else:
            self.spectrallib = [np.array([1])]

        self.input_psfs = input_psfs
        self.input_psfs_wvs = input_wvs

        if flux_conversion is None:
            flux_conversion = np.ones(inputs_shape[0])
        self.flux_conversion = flux_conversion

        self.psf_centx_notscaled = {}
        self.psf_centy_notscaled = {}

        numwv,ny_psf,nx_psf =  self.input_psfs.shape
        x_psf_grid, y_psf_grid = np.meshgrid(np.arange(nx_psf * 1.) - nx_psf//2, np.arange(ny_psf * 1.) - ny_psf//2)
        psfs_func_list = []
        for wv_index in range(numwv):
            model_psf = self.input_psfs[wv_index, :, :] #* self.flux_conversion * self.spectrallib[0][wv_index] * self.dflux
            #psfs_interp_model_list.append(interpolate.bisplrep(x_psf_grid,y_psf_grid,model_psf))
            #psfs_interp_model_list.append(interpolate.SmoothBivariateSpline(x_psf_grid.ravel(),y_psf_grid.ravel(),model_psf.ravel()))
            psfs_func_list.append(interpolate.LSQBivariateSpline(x_psf_grid.ravel(),y_psf_grid.ravel(),model_psf.ravel(),x_psf_grid[0,0:nx_psf-1]+0.5,y_psf_grid[0:ny_psf-1,0]+0.5))
            #psfs_interp_model_list.append(interpolate.interp2d(x_psf_grid,y_psf_grid,model_psf,kind="cubic",bounds_error=False,fill_value=0.0))
            #psfs_interp_model_list.append(interpolate.Rbf(x_psf_grid,y_psf_grid,model_psf,function="gaussian"))

            if 0:
                import matplotlib.pylab as plt
                #print(x_psf_grid.shape)
                #print(psfs_interp_model_list[wv_index](x_psf_grid.ravel(),y_psf_grid.ravel()).shape)
                a = psfs_func_list[wv_index](x_psf_grid[0,:],y_psf_grid[:,0]).transpose()
                plt.figure()
                plt.subplot(1,3,1)
                plt.imshow(a,interpolation="nearest")
                plt.colorbar()
                ##plt.imshow(psfs_interp_model_list[wv_index](np.linspace(-10,10,500),np.linspace(-10,10,500)),interpolation="nearest")
                plt.subplot(1,3,2)
                plt.imshow(self.input_psfs[wv_index, :, :],interpolation="nearest")
                plt.colorbar()
                plt.subplot(1,3,3)
                plt.imshow(abs(self.input_psfs[wv_index, :, :]-a),interpolation="nearest")
                plt.colorbar()
                plt.show()

        self.psfs_func_list = psfs_func_list


    def alloc_fmout(self, output_img_shape):
        """Allocates shared memory for the output of the shared memory


        Args:
            output_img_shape: shape of output image (usually N,y,x,b)

        Returns:
            fmout: mp.array to store FM data in
            fmout_shape: shape of FM data array

        """
        fmout_size = int(np.prod(output_img_shape))
        fmout = mp.Array(self.data_type, fmout_size)
        fmout_shape = output_img_shape

        return fmout, fmout_shape


    def alloc_perturbmag(self, output_img_shape, numbasis):
        """
        Allocates shared memory to store the fractional magnitude of the linear KLIP perturbation
        Stores a number for each frame = max(oversub + selfsub)/std(PCA(image))

        Args:
            output_img_shape: shape of output image (usually N,y,x,b)
            numbasis: array/list of number of KL basis cutoffs requested

        Returns:
            perturbmag: mp.array to store linaer perturbation magnitude
            perturbmag_shape: shape of linear perturbation magnitude

        """
        perturbmag_shape = (output_img_shape[0], np.size(numbasis))
        perturbmag = mp.Array(self.data_type, int(np.prod(perturbmag_shape)))

        return perturbmag, perturbmag_shape


    def generate_models(self, input_img_shape, section_ind, pas, wvs, radstart, radend, phistart, phiend, padding, ref_center, parang, ref_wv, flipx):
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
            flipx: if True, flip x coordinate in final image

        Return:
            models: array of size (N, p) where p is the number of pixels in the segment
        """
        # create some parameters for a blank canvas to draw psfs on
        nx = input_img_shape[1]
        ny = input_img_shape[0]
        x_grid, y_grid = np.meshgrid(np.arange(nx * 1.)-ref_center[0], np.arange(ny * 1.)-ref_center[1])

        numwv, ny_psf, nx_psf =  self.input_psfs.shape

        # create bounds for PSF stamp size
        row_m = int(np.floor(ny_psf/2.0))    # row_minus
        row_p = int(np.ceil(ny_psf/2.0))     # row_plus
        col_m = int(np.floor(nx_psf/2.0))   # col_minus
        col_p = int(np.ceil(nx_psf/2.0))    # col_plus

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
                # flipx requires the opposite rotation
                sign = -1.
                if flipx:
                    sign = 1.
                self.psf_centx_notscaled[pa] = self.sep * np.cos(np.radians(90. - sign*self.pa - pa))
                self.psf_centy_notscaled[pa] = self.sep * np.sin(np.radians(90. - sign*self.pa - pa))
            psf_centx = (ref_wv/wv) * self.psf_centx_notscaled[pa]
            psf_centy = (ref_wv/wv) * self.psf_centy_notscaled[pa]

            # create a coordinate system for the image that is with respect to the model PSF
            # round to nearest pixel and add offset for center
            l = int(round(psf_centx + ref_center[0]))
            k = int(round(psf_centy + ref_center[1]))
            # recenter coordinate system about the location of the planet
            x_vec_stamp_centered = np.copy(x_grid[0, (l-col_m):(l+col_p)]) - psf_centx
            y_vec_stamp_centered = np.copy(y_grid[(k-row_m):(k+row_p), 0]) - psf_centy
            # rescale to account for the align and scaling of the refernce PSFs
            # e.g. for longer wvs, the PSF has shrunk, so we need to shrink the coordinate system
            x_vec_stamp_centered /= (ref_wv/wv)
            y_vec_stamp_centered /= (ref_wv/wv)

            # use intepolation spline to generate a model PSF and write to temp img
            whiteboard[(k-row_m):(k+row_p), (l-col_m):(l+col_p)] = \
                    self.psfs_func_list[wv_index](x_vec_stamp_centered,y_vec_stamp_centered).transpose()

            # write model img to output (segment is collapsed in x/y so need to reshape)
            whiteboard.shape = [input_img_shape[0] * input_img_shape[1]]
            segment_with_model = copy(whiteboard[section_ind])
            whiteboard.shape = [input_img_shape[0],input_img_shape[1]]

            models.append(segment_with_model)

            # clean whiteboard
            whiteboard[(k-row_m):(k+row_p), (l-col_m):(l+col_p)] = 0


        return np.array(models)




    def fm_from_eigen(self, klmodes=None, evals=None, evecs=None, input_img_shape=None, input_img_num=None, ref_psfs_indicies=None, section_ind=None,section_ind_nopadding=None, aligned_imgs=None, pas=None,
                     wvs=None, radstart=None, radend=None, phistart=None, phiend=None, padding=None,IOWA = None, ref_center=None,
                     parang=None, ref_wv=None, numbasis=None, fmout=None, perturbmag=None, klipped=None, covar_files=None, flipx=True, **kwargs):
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
        model_sci = self.generate_models(input_img_shape, section_ind, [parang], [ref_wv], radstart, radend, phistart, phiend, padding, ref_center, parang, ref_wv, flipx)[0]
        model_sci *= self.flux_conversion[input_img_num] * self.spectrallib[0][np.where(self.input_psfs_wvs == ref_wv)] * self.dflux

        # generate models of the PSF for each reference segments. Output is of shape (N, pix_in_segment)
        models_ref = self.generate_models(input_img_shape, section_ind, pas, wvs, radstart, radend, phistart, phiend, padding, ref_center, parang, ref_wv, flipx)

        # Calculate the spectra to determine the flux of each model reference PSF
        total_imgs = np.size(self.flux_conversion)
        num_wvs = self.spectrallib[0].shape[0]
        input_spectrum = self.flux_conversion[:num_wvs] * self.spectrallib[0] * self.dflux
        input_spectrum = np.ravel(np.tile(input_spectrum,(1, total_imgs//num_wvs)))
        input_spectrum = input_spectrum[ref_psfs_indicies]
        models_ref = models_ref * input_spectrum[:, None]

        # using original Kl modes and reference models, compute the perturbed KL modes (spectra is already in models)
        # also grab the pertrubed (C_AS) covariance matrix to compute its eigenvalues
        delta_KL, covar_perturb = fm.perturb_specIncluded(evals, evecs, klmodes, refs, models_ref, return_perturb_covar=True)

        # calculate postklip_psf using delta_KL
        postklip_psf, oversubtraction, selfsubtraction = fm.calculate_fm(delta_KL, klmodes, numbasis, sci, model_sci, inputflux=None)

        # calculate validity of linear perturbation on KLIP modes
        pca_img = (sci - np.nanmean(sci))[:, None] - klipped # shape of ( size(section), b)
        perturb_frac = np.nanmax(np.abs(oversubtraction + selfsubtraction), axis=1)/np.nanstd(pca_img, axis=0) # array of b
        this_validity = fm.calculate_validity(covar_perturb, models_ref, numbasis, evals, covar_files, evecs, klmodes, delta_KL) # array of b
        #this_validity = fm.calculate_validity2(evals, models_ref, numbasis) # array of b
        perturbmag[input_img_num] = this_validity

        fmout_shape = fmout.shape

        # nan the same pixels as the klipped image
        klipped_nans = np.where(np.isnan(klipped))
        postklip_psf[:, klipped_nans[0]] = np.nan

        # write forward modelled PSF to fmout (as output)
        # need to derotate the image in this step
        for thisnumbasisindex in range(np.size(numbasis)):
                fm._save_rotated_section(input_img_shape, postklip_psf[thisnumbasisindex], section_ind,
                                 fmout[input_img_num, :, :,thisnumbasisindex], None, parang,
                                 radstart, radend, phistart, phiend, padding,IOWA, ref_center, flipx=flipx)



    def cleanup_fmout(self, fmout):
        """
        After running KLIP-FM, we need to reshape fmout so that the numKL dimension is the first one and not the last

        Args:
            fmout: numpy array of ouput of FM

        Return:
            fmout: same but cleaned up if necessary
        """

        # Let's reshape the output images
        # move number of KLIP modes as leading axis (i.e. move from shape (N,y,x,b) to (b,N,y,x)
        dims = fmout.shape
        fmout = np.rollaxis(fmout.reshape((dims[0], dims[1], dims[2], dims[3])), 3)
        return fmout

    def save_fmout(self, dataset, fmout, outputdir, fileprefix, numbasis, klipparams=None, calibrate_flux=False,
                   spectrum=None):
        """
        Saves the FM planet PSFs to disk. Saves both a KL mode cube and spectral cubes

        Args:
            dataset: Instruments.Data instance. Will use its dataset.savedata() function to save data
            fmout: the fmout data passed from fm.klip_parallelized which is passed as the output of cleanup_fmout
            outputdir: output directory
            fileprefix: the fileprefix to prepend the file name
            numbasis: KL mode cutoffs used
            klipparams: string with KLIP-FM parameters
            calibrate_flux: if True, flux calibrate the data in the same way as the klipped data
            spectrum: if not None, spectrum to weight the data by. Length same as dataset.wvs
        """
        # collapse in time and wavelength to examine KL modes
        if spectrum is None:
            KLmode_cube = np.nanmean(fmout, axis=1)
        else:
            #do the mean combine by weighting by the spectrum
            KLmode_cube = np.nanmean(fmout * spectrum[None,:,None,None], axis=1)\
                          / np.mean(spectrum)

        # save FM location into header
        more_keywords = {'fm_sep': self.sep, 'fm_pa': self.pa}

        # broadband flux calibration for KL mode cube
        if calibrate_flux:
            KLmode_cube = dataset.calibrate_output(KLmode_cube, spectral=False)
        dataset.savedata(outputdir + '/' + fileprefix + "-fmpsf-KLmodes-all.fits", KLmode_cube,
                         klipparams=klipparams.format(numbasis=str(numbasis)), filetype="KL Mode Cube",
                         zaxis=numbasis, more_keywords=more_keywords)

        # if there is more than one wavelength, save spectral cubes
        if np.size(np.unique(dataset.wvs)) > 1:
            numwvs = np.size(np.unique(dataset.wvs))
            klipped_spec = fmout.reshape([fmout.shape[0], fmout.shape[1]//numwvs, numwvs,
                                            fmout.shape[2], fmout.shape[3]]) # (b, N_cube, wvs, y, x) 5-D cube

            # for each KL mode, collapse in time to examine spectra
            KLmode_spectral_cubes = np.nanmean(klipped_spec, axis=1)
            for KLcutoff, spectral_cube in zip(numbasis, KLmode_spectral_cubes):
                # calibrate spectral cube if needed
                if calibrate_flux:
                    spectral_cube = dataset.calibrate_output(spectral_cube, spectral=True)
                dataset.savedata(outputdir + '/' + fileprefix + "-fmpsf-KL{0}-speccube.fits".format(KLcutoff),
                                 spectral_cube, klipparams=klipparams.format(numbasis=KLcutoff),
                                 filetype="PSF Subtracted Spectral Cube", more_keywords=more_keywords)


