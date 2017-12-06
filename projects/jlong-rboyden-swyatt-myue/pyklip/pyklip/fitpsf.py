import warnings
import pickle
import math

import numpy as np
import scipy.linalg as linalg
import scipy.ndimage as ndi
import scipy.ndimage.interpolation as sinterp

import pyklip.covars as covars

# emcee more MCMC sampling
import emcee



class FMAstrometry(object):
    """
    Base class to perform astrometry on direct imaging data_stamp using MCMC and GP regression

    Args:
        guess_sep: the guessed separation (pixels)
        guess_pa: the guessed position angle (degrees)
        fitboxsize: fitting box side length (pixels)

    Attributes:
        guess_sep (float): (initialization) guess separation for planet [pixels]
        guess_pa (float): (initialization) guess PA for planet [degrees]
        guess_RA_offset (float): (initialization) guess RA offset [pixels]
        guess_Dec_offset (float): (initialization) guess Dec offset [pixels]
        raw_RA_offset (:py:class:`pyklip.fitpsf.ParamRange`): (result) the raw result from the MCMC fit for the planet's location [pixels]
        raw_Dec_offset (:py:class:`pyklip.fitpsf.ParamRange`): (result) the raw result from the MCMC fit for the planet's location [pixels]
        raw_flux (:py:class:`pyklip.fitpsf.ParamRange`): (result) factor to scale the FM to match the flux of the data
        covar_params (list of :py:class:`pyklip.fitpsf.ParamRange`): (result) hyperparameters for the Gaussian process
        raw_sep(:py:class:`pyklip.fitpsf.ParamRange`): (result) the inferred raw result from the MCMC fit for the planet's location [pixels]
        raw_PA(:py:class:`pyklip.fitpsf.ParamRange`): (result) the inferred raw result from the MCMC fit for the planet's location [degrees]
        RA_offset(:py:class:`pyklip.fitpsf.ParamRange`): (result) the RA offset of the planet that includes all astrometric errors [pixels or mas]
        Dec_offset(:py:class:`pyklip.fitpsf.ParamRange`): (result) the Dec offset of the planet that includes all astrometric errors [pixels or mas]
        sep(:py:class:`pyklip.fitpsf.ParamRange`): (result) the separation of the planet that includes all astrometric errors [pixels or mas]
        PA(:py:class:`pyklip.fitpsf.ParamRange`): (result) the PA of the planet that includes all astrometric errors [degrees]
        fm_stamp (np.array): (fitting) The 2-D stamp of the forward model (centered at the nearest pixel to the guessed location)
        data_stamp (np.array): (fitting) The 2-D stamp of the data (centered at the nearest pixel to the guessed location)
        noise_map (np.array): (fitting) The 2-D stamp of the noise for each pixel the data computed assuming azimuthally similar noise
        padding (int): amount of pixels on one side to pad the data/forward model stamp
        sampler (emcee.EnsembleSampler): an instance of the emcee EnsambleSampler. See emcee docs for more details. 
    

    """
    def __init__(self, guess_sep, guess_pa, fitboxsize):
        """
        Initilaizes the FMAstrometry class
        """
        # store initailization
        self.guess_sep = guess_sep
        self.guess_pa = guess_pa
        self.fitboxsize = fitboxsize

        # derive delta RA and delta Dec
        # in pixels
        self.guess_RA_offset = self.guess_sep * np.sin(np.radians(self.guess_pa))
        self.guess_Dec_offset = self.guess_sep * np.cos(np.radians(self.guess_pa))

        # stuff that isn't generated yet
        # stamps of the data_stamp and the forward model
        self.fm_stamp = None # Forward Model
        self.padding = 0 # padding for FM. You kinda need this to shift the FM around
        self.data_stamp = None # Data
        self.noise_map = None # same shape as self.data_stamp
        self.data_stamp_RA_offset = None # RA offset of data_stamp (in pixels)
        self.data_stamp_Dec_offset = None # Dec offset (in pixels)
        self.data_stamp_RA_offset_center = None # RA offset of center pixel (stampsize // 2)
        self.data_stamp_Dec_offset_center = None # Dec offset of center pixel (stampsize // 2)

        # guess flux (a hyperparameter)
        self.guess_flux = None

        # covariance paramters. Use the covariance initilizer function to initilize them
        self.covar = None
        self.covar_param_guesses = None
        self.covar_param_labels = None
        self.include_readnoise = False

        # MCMC fit params
        self.bounds = None
        self.sampler = None

        # best fit
        self.raw_RA_offset = None
        self.raw_Dec_offset = None
        self.raw_flux = None
        self.covar_params = None
        # best fit infered parameters
        self.raw_sep = None
        self.raw_PA = None
        self.RA_offset = None
        self.Dec_offset = None
        self.sep = None
        self.PA = None


    def generate_fm_stamp(self, fm_image, fm_center=None, fm_wcs=None, extract=True, padding=5):
        """
        Generates a stamp of the forward model and stores it in self.fm_stamp
        Args:
            fm_image: full imgae containing the fm_stamp
            fm_center: [x,y] center of image (assuing fm_stamp is located at sep/PA) corresponding to guess_sep and guess_pa
            fm_wcs: if not None, specifies the sky angles in the image. If None, assume image is North up East left
            extract: if True, need to extract the forward model from the image. Otherwise, assume the fm_stamp is already
                    centered in the frame (fm_image.shape // 2)
            padding: number of pixels on each side in addition to the fitboxsize to extract to pad the fm_stamp
                        (should be >= 1)

        Returns:

        """
        # cheeck the padding to make sure it's valid
        if not isinstance(padding, int):
            raise TypeError("padding must be an integer")
        if padding < 1:
            warnings.warn("Padding really should be >= 1 pixel so we can shift the FM around", RuntimeWarning)
        self.padding = padding


        if extract:
            if fm_wcs is not None:
                raise NotImplementedError("Have not implemented rotation using WCS")

            # image is now rotated North up east left
            # find the location of the FM
            thistheta = np.radians(self.guess_pa + 90)
            psf_xpos = self.guess_sep * np.cos(thistheta) + fm_center[0]
            psf_ypos = self.guess_sep * np.sin(thistheta) + fm_center[1]

        else:
            # PSf is already cenetered
            psf_xpos = fm_image.shape[1]//2
            psf_ypos = fm_image.shape[0]//2

        # now we found the FM in the image, extract out a centered stamp of it
        # grab the coordinates of the image
        stampsize = 2 * self.padding + self.fitboxsize # full stamp needs padding around all sides
        x_stamp, y_stamp = np.meshgrid(np.arange(stampsize * 1.) - stampsize //2,
                                       np.arange(stampsize * 1.) - stampsize// 2)

        x_stamp += psf_xpos
        y_stamp += psf_ypos

        # zero nans because it messes with interpolation
        fm_image[np.where(np.isnan(fm_image))] = 0

        fm_stamp = ndi.map_coordinates(fm_image, [y_stamp, x_stamp])
        self.fm_stamp = fm_stamp



    def generate_data_stamp(self, data, data_center, data_wcs=None, noise_map=None, dr=4, exclusion_radius=10):
        """
        Generate a stamp of the data_stamp ~centered on planet and also corresponding noise map
        Args:
            data: the final collapsed data_stamp (2-D)
            data_center: location of star in the data_stamp
            data_wcs: sky angles WCS object. To rotate the image properly [NOT YET IMPLMETNED]
                      if None, data_stamp is already rotated North up East left
            noise_map: if not None, noise map for each pixel in the data_stamp (2-D).
                        if None, one will be generated assuming azimuthal noise using an annulus widthh of dr
            dr: width of annulus in pixels from which the noise map will be generated
            exclusion_radius: radius around the guess planet location which doens't get factored into noise estimate

        Returns:

        """
        # rotate image North up east left if necessary
        if data_wcs is not None:
            # rotate
            raise NotImplementedError("Rotating based on WCS is not currently implemented yet")

        xguess = -self.guess_RA_offset + data_center[0]
        yguess = self.guess_Dec_offset + data_center[1]

        # round to nearest pixel
        xguess_round = int(np.round(xguess))
        yguess_round = int(np.round(yguess))

        # get index bounds for grabbing pixels from data_stamp
        ymin = yguess_round - self.fitboxsize//2
        xmin = xguess_round - self.fitboxsize//2
        ymax = yguess_round + self.fitboxsize//2 + 1
        xmax = xguess_round + self.fitboxsize//2 + 1
        if self.fitboxsize % 2 == 0:
            # for even fitbox sizes, need to truncate ymax/xmax by 1
            ymax -= 1
            xmax -= 1

        data_stamp = data[ymin:ymax, xmin:xmax]
        self.data_stamp = data_stamp

        # store coordinates of stamp also
        dy_img, dx_img = np.indices(data.shape, dtype=float)
        dy_img -= data_center[1]
        dx_img -= data_center[0]

        dx_data_stamp = dx_img[ymin:ymax, xmin:xmax]
        dy_data_stamp = dy_img[ymin:ymax, xmin:xmax]
        self.data_stamp_RA_offset = -dx_data_stamp
        self.data_stamp_Dec_offset = dy_data_stamp
        self.data_stamp_RA_offset_center = self.data_stamp_RA_offset[0, self.fitboxsize // 2]
        self.data_stamp_Dec_offset_center = self.data_stamp_Dec_offset[self.fitboxsize // 2, 0]

        # generate noise map if necessary
        if noise_map is None:
            # blank map
            noise_stamp = np.zeros(data_stamp.shape)

            # define exclusion around planet.
            distance_from_planet = np.sqrt((dx_img - (xguess - data_center[0]))**2 +
                                           (dy_img - (yguess - data_center[1]))**2)
            # define radial coordinate
            rimg = np.sqrt(dx_img**2 + dy_img**2)

            # calculate noise for each pixel in the data_stamp stamp
            for y_index, x_index in np.ndindex(data_stamp.shape):
                r_pix = np.sqrt(dy_data_stamp[y_index, x_index]**2 + dx_data_stamp[y_index, x_index]**2)
                pixels_for_noise = np.where((np.abs(rimg - r_pix) <= dr/2.) & (distance_from_planet > exclusion_radius))
                noise_stamp[y_index, x_index] = np.nanstd(data[pixels_for_noise])

        else:
            noise_stamp = noise_map[ymin:ymax, xmin:xmax]

        self.noise_map = noise_stamp


    def set_kernel(self, covar, covar_param_guesses, covar_param_labels, include_readnoise=False,
                   read_noise_fraction=0.01):
        """
        Set the Gaussian process kernel used in our astrometric fit

        Args:
            covar: Covariance kernel for GP regression. If string, can be "matern32" or "sqexp"
                    Can also be a function: cov = cov_function(x_indices, y_indices, sigmas, cov_params)
            covar_param_guesses: a list of guesses on the hyperparmeteres (size of N_hyperparams)
            covar_param_labels: a list of strings labelling each covariance parameter
            include_readnoise: if True, part of the noise is a purely diagonal term (i.e. read/photon noise)
            read_noise_fraction: fraction of the total measured noise is read noise (between 0 and 1)

        Returns:

        """
        if isinstance(covar, str):
            if covar.lower() == "matern32":
                self.covar = covars.matern32
            elif covar.lower() == "sqexp":
                self.covar = covars.sq_exp
            else:
                raise ValueError("Covariance matricies currently supported are 'matern32' and 'sqexp'")
        else:
            # this better be a covariance function. We're trusting you
            self.covar = covar

        self.covar_param_guesses = covar_param_guesses
        self.covar_param_labels = covar_param_labels

        if include_readnoise:
            self.include_readnoise = True
            self.covar_param_guesses.append(read_noise_fraction)
            self.covar_param_labels.append(r"K_{\delta}")


    def set_bounds(self, dRA, dDec, df, covar_param_bounds, read_noise_bounds=None):
        """
        Set bounds on Bayesian priors. All paramters can be a 2 element tuple/list/array that specifies
        the lower and upper bounds x_min < x < x_max. Or a single value whose interpretation is specified below
        If you are passing in both lower and upper bounds, both should be in linear scale!
        Args:
            dRA: Distance from initial guess position in pixels. For a single value, this specifies the largest distance
                form the initial guess (i.e. RA_guess - dRA < x < RA_guess + dRA)
            dDec: Same as dRA except with Dec
            df: Flux range. If single value, specifies how many orders of 10 the flux factor can span in one direction
                (i.e. log_10(guess_flux) - df < log_10(guess_flux) < log_10(guess_flux) + df
            covar_param_bounds: Params for covariance matrix. Like df, single value specifies how many orders of
                                magnitude parameter can span. Otherwise, should be a list of 2-elem touples
            read_noise_bounds: Param for read noise term. If single value, specifies how close to 0 it can go
                                based on powers of 10 (i.e. log_10(-read_noise_bound) < read_noise < 1 )

        Returns:

        """
        self.bounds = []

        # x/RA bounds
        if np.size(dRA) == 2:
            self.bounds.append(dRA)
        else:
            self.bounds.append([self.guess_RA_offset - dRA, self.guess_RA_offset + dRA])

        # y/Dec bounds
        if np.size(dDec) == 2:
            self.bounds.append(dDec)
        else:
            self.bounds.append([self.guess_Dec_offset - dDec, self.guess_Dec_offset + dDec])

        # flux bounds
        # need to guess flux if None
        if self.guess_flux is None:
            if self.fm_stamp is not None and self.data_stamp is not None:
                # use the two to scale it and put it in log scale
                self.guess_flux = np.max(self.data_stamp) / np.max(self.fm_stamp)
            else:
                # we haven't read in the data_stamp and FM yet. Assume they're on the same scale
                # should be in log scale
                self.guess_flux = 1
        if np.size(df) == 2:
            self.bounds.append(df)
        else:
            self.bounds.append([self.guess_flux / (10.**df), self.guess_flux * (10**df)])

        # hyperparam bounds
        if np.ndim(covar_param_bounds) == 2:
            for covar_param_bound in covar_param_bounds:
                self.bounds.append(covar_param_bound)
        else:
            # this is a 1-D list, with each param specified by one paramter
            for covar_param_bound, covar_param_guess in zip(covar_param_bounds, self.covar_param_guesses):
                self.bounds.append([covar_param_guess / (10.**covar_param_bound),
                                    covar_param_guess * (10**covar_param_bound)])

        if read_noise_bounds is not None:
        # read noise
            if np.size(read_noise_bounds) == 2:
                self.bounds.append(read_noise_bounds)
            else:
                self.bounds.append([self.covar_param_guesses[-1]/10**read_noise_bounds, 1])


    def fit_astrometry(self, nwalkers=100, nburn=200, nsteps=800, save_chain=True, chain_output="bka-chain.pkl",
                       numthreads=None):
        """
        Run a Bayesian fit of the astrometry using MCMC
        Saves to self.chian

        Args:
            nwalkers: number of walkers
            nburn: numbe of samples of burn-in for each walker
            nsteps: number of samples each walker takes
            save_chain: if True, save the output in a pickled file
            chain_output: filename to output the chain to
            numthreads: number of threads to use

        Returns:

        """
        # create array of initial guesses
        # array of guess RA, Dec, and flux
        # for everything that's not RA/Dec offset, should be converted to log space for MCMC sampling
        init_guess = np.array([self.guess_RA_offset, self.guess_Dec_offset, math.log(self.guess_flux)])
        # append hyperparams for covariance matrix, which also need to be converted to log space
        init_guess = np.append(init_guess, np.log(self.covar_param_guesses))
        # number of dimensions of MCMC fit
        ndim = np.size(init_guess)

        # initialize walkers in a ball around the best fit value
        pos = [init_guess + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]

        # prior bounds also need to be put in log space
        sampler_bounds = np.copy(self.bounds)
        sampler_bounds[2:] = np.log(sampler_bounds[2:])

        global lnprob
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(self, sampler_bounds, self.covar),
                                        kwargs={'readnoise' : self.include_readnoise}, threads=numthreads)

        # burn in
        print("Running burn in")
        pos, _, _ = sampler.run_mcmc(pos, nburn)
        # reset sampler
        sampler.reset()
         
        # chains should hopefulyl have converged. Now run MCMC
        print("Burn in finished. Now sampling posterior")
        sampler.run_mcmc(pos, nsteps)
        print("MCMC sampler has finished")

        # convert chains in log space back in linear space
        sampler.chain[:,:,2:] = np.exp(sampler.chain[:,:,2:])

        # save state
        self.sampler = sampler

        # save best fit values
        # percentiles has shape [ndims, 3]
        percentiles = np.swapaxes(np.percentile(sampler.flatchain, [16, 50, 84], axis=0), 0, 1)
        self.raw_RA_offset = ParamRange(percentiles[0][1], np.array([percentiles[0][2], percentiles[0][0]]) - percentiles[0][1])
        self.raw_Dec_offset = ParamRange(percentiles[1][1], np.array([percentiles[1][2], percentiles[1][0]]) - percentiles[1][1])
        self.raw_flux =  ParamRange(percentiles[2][1], np.array([percentiles[2][2], percentiles[2][0]]) -  percentiles[2][1])
        self.covar_params = [ParamRange(thispercentile[1], np.array([thispercentile[2], thispercentile[0]]) - thispercentile[1] ) for thispercentile in percentiles[3:]]

        if save_chain:
            pickle_file = open(chain_output, 'wb')
            pickle.dump(sampler.chain, pickle_file)
            pickle.dump(sampler.lnprobability, pickle_file)
            pickle.dump(sampler.acceptance_fraction, pickle_file)
            #pickle.dump(sampler.acor, pickle_file)
            pickle_file.close()


    def make_corner_plot(self, fig=None):
        """
        Generate a corner plot of the posteriors from the MCMC
        Args:
            fig: if not None, a matplotlib Figure object

        Returns:
            fig: the Figure object. If input fig is None, function will make a new one

        """
        import corner

        all_labels = [r"x", r"y", r"$\alpha$"]
        all_labels = np.append(all_labels, self.covar_param_labels)

        fig = corner.corner(self.sampler.flatchain, labels=all_labels, quantiles=[0.16, 0.5, 0.84], fig=fig)

        return fig


    def best_fit_and_residuals(self, fig=None):
        """
        Generate a plot of the best fit FM compared with the data_stamp and also the residuals
        Args:
            fig (matplotlib.Figure): if not None, a matplotlib Figure object

        Returns:
            fig (matplotlib.Figure): the Figure object. If input fig is None, function will make a new one

        """
        import matplotlib
        import matplotlib.pylab as plt

        if fig is None:
            fig = plt.figure(figsize=(12, 4))

        # create best fit FM
        dx = -(self.raw_RA_offset.bestfit - self.data_stamp_RA_offset_center)
        dy = self.raw_Dec_offset.bestfit - self.data_stamp_Dec_offset_center

        fm_bestfit = self.raw_flux.bestfit * sinterp.shift(self.fm_stamp, [dy, dx])
        if self.padding > 0:
            fm_bestfit = fm_bestfit[self.padding:-self.padding, self.padding:-self.padding]

        # make residual map
        residual_map = self.data_stamp - fm_bestfit

        # normalize all images to same scale
        colornorm = matplotlib.colors.Normalize(vmin=np.percentile(self.data_stamp, 0.03),
                                                vmax=np.percentile(self.data_stamp, 99.7))

        # plot the data_stamp
        ax1 = fig.add_subplot(131)
        im1 = ax1.imshow(self.data_stamp, interpolation='nearest', cmap='cubehelix', norm=colornorm)
        ax1.invert_yaxis()
        ax1.set_title("Data")
        ax1.set_xlabel("X (pixels)")
        ax1.set_ylabel("Y (pixels)")

        ax2 = fig.add_subplot(132)
        im2 = ax2.imshow(fm_bestfit, interpolation='nearest', cmap='cubehelix', norm=colornorm)
        ax2.invert_yaxis()
        ax2.set_title("Best-fit Model")
        ax2.set_xlabel("X (pixels)")

        ax3 = fig.add_subplot(133)
        im3 = ax3.imshow(residual_map, interpolation='nearest', cmap='cubehelix', norm=colornorm)
        ax3.invert_yaxis()
        ax3.set_title("Residuals")
        ax3.set_xlabel("X (pixels)")

        fig.subplots_adjust(right=0.82)
        fig.subplots_adjust(hspace=0.4)
        ax_pos = ax3.get_position()

        cbar_ax = fig.add_axes([0.84, ax_pos.y0, 0.02, ax_pos.height])
        cb = fig.colorbar(im1, cax=cbar_ax)
        cb.set_label("Counts (DN)")

        return fig

    def propogate_errs(self, star_center_err=None, platescale=None, platescale_err=None, pa_offset=None, pa_uncertainty=None):
        """
        Propogate astrometric error. Stores results in its own fields

        Args:
            star_center_err (float): uncertainity of the star location (pixels)
            platescale (float): mas/pix conversion to angular coordinates 
            platescale_err (float): mas/pix error on the platescale
            pa_offset (float): Offset, in the same direction as position angle, to set North up (degrees)
            pa_uncertainity (float): Error on position angle/true North calibration (Degrees)
        """
        # ensure numpy arrays
        x_mcmc = self.sampler.chain[:,:,0].flatten()
        y_mcmc = self.sampler.chain[:,:,1].flatten()

        # calcualte statistial errors in x and y
        x_best = np.median(x_mcmc)
        y_best = np.median(y_mcmc)
        x_1sigma_raw = (np.percentile(x_mcmc, [84,16]) - x_best)
        y_1sigma_raw = (np.percentile(y_mcmc, [84,16]) - y_best)

        print("Raw X/Y Centroid = ({0}, {1}) with statistical error of {2} pix in X and {3} pix in Y".format(x_best, y_best, x_1sigma_raw, y_1sigma_raw))

        # calculate sep and pa from x/y separation
        sep_mcmc = np.sqrt((x_mcmc)**2 + (y_mcmc)**2)
        pa_mcmc = (np.degrees(np.arctan2(y_mcmc, -x_mcmc)) - 90) % 360

        # calculate sep and pa statistical errors
        sep_best = np.median(sep_mcmc)
        pa_best = np.median(pa_mcmc)
        sep_1sigma_raw = (np.percentile(sep_mcmc, [84,16]) - sep_best)
        pa_1sigma_raw = (np.percentile(pa_mcmc, [84,16]) - pa_best)

        print("Raw Sep/PA Centroid = ({0}, {1}) with statistical error of {2} pix in Sep and {3} pix in PA".format(sep_best, pa_best, sep_1sigma_raw, pa_1sigma_raw))
        
        # store the raw sep and PA values
        self.raw_sep = ParamRange(sep_best, sep_1sigma_raw)
        self.raw_PA = ParamRange(pa_best, pa_1sigma_raw)

        # Now let's start propogating error terms if they are supplied. 
        # We do them in Sep/PA space first since it's more natural here

        # star center error
        if star_center_err is None:
            print("Skipping star center uncertainity...")
            star_center_err = 0
        else:
            print("Adding in star center uncertainity")

        sep_err_pix = (sep_1sigma_raw**2) + star_center_err**2
        sep_err_pix = np.sqrt(sep_err_pix)

        # plate scale error
        if platescale is not None:
            print("Converting pixels to milliarcseconds")
            if platescale_err is None:
                print("Skipping plate scale uncertainity...")
                platescale_err = 0
            else:
                print("Adding in plate scale error")
            sep_err_mas = np.sqrt((sep_err_pix * platescale)**2 + (platescale_err * sep_best)**2)

        # PA Offset
        if pa_offset is not None:
            print("Adding in a PA/North angle offset")
            pa_mcmc = (pa_mcmc + pa_offset) % 360
        
        # PA Uncertainity
        if pa_uncertainty is None:
            print("Skipping PA/North uncertainity...")
            pa_uncertainty = 0
        else:
            print("Adding in PA uncertainity")

        pa_err = np.radians(pa_1sigma_raw)**2 + (star_center_err/sep_best)**2 + np.radians(pa_uncertainty)**2
        pa_err = np.sqrt(pa_err)
        pa_err_deg = np.degrees(pa_err)

        sep_err_pix_avg = np.mean(np.abs(sep_err_pix))
        pa_err_deg_avg = np.mean(np.abs(pa_err_deg))

        print("Sep = {0} +/- {1} ({2}) pix, PA = {3} +/- {4} ({5}) degrees".format(sep_best, sep_err_pix_avg, sep_err_pix, pa_best, pa_err_deg_avg, pa_err_deg))

        # Store sep/PA (excluding platescale) values
        self.sep = ParamRange(sep_best, sep_err_pix)
        self.PA = ParamRange(pa_best, pa_err_deg)

        if platescale is not None:
            sep_err_mas_avg = np.mean(np.abs(sep_err_mas))
            print("Sep = {0} +/- {1} ({2}) mas, PA = {3} +/- {4} ({5}) degrees".format(sep_best*platescale, sep_err_mas_avg, sep_err_mas, pa_best, pa_err_deg_avg, pa_err_deg))
            # overwrite sep values with values converted to milliarcseconds
            self.sep = ParamRange(sep_best*platescale, sep_err_mas)

        # convert PA errors back into x y (RA/Dec)
        ra_mcmc = -sep_mcmc * np.cos(np.radians(pa_mcmc+90))
        dec_mcmc = sep_mcmc * np.sin(np.radians(pa_mcmc+90))

        # ra/dec statistical errors
        ra_best = np.median(ra_mcmc)
        dec_best = np.median(dec_mcmc)
        ra_1sigma_raw = np.percentile(ra_mcmc, [84,16]) - ra_best
        dec_1sigma_raw = np.percentile(dec_mcmc, [84,16]) - dec_best

        ra_err_full_pix = np.sqrt((ra_1sigma_raw**2)  + (star_center_err)**2 + (dec_best * np.radians(pa_uncertainty))**2 )
        dec_err_full_pix = np.sqrt((dec_1sigma_raw**2)  + (star_center_err)**2 + (ra_best * np.radians(pa_uncertainty))**2 )

        # Store error propgoated RA/Dec values (excluding platescale)
        self.RA_offset = ParamRange(ra_best, ra_err_full_pix)
        self.Dec_offset = ParamRange(dec_best, dec_err_full_pix)

        print("RA offset = {0} +/- {1} ({2}) pix".format(self.RA_offset.bestfit, self.RA_offset.error, self.RA_offset.error_2sided))
        print("Dec offset = {0} +/- {1} ({2}) pix".format(self.Dec_offset.bestfit, self.Dec_offset.error, self.Dec_offset.error_2sided))

        if platescale is not None:
            ra_err_full_mas = np.sqrt((ra_err_full_pix*platescale)**2 + (platescale_err * ra_best)**2)
            dec_err_full_mas = np.sqrt((dec_err_full_pix*platescale)**2 + (platescale_err * dec_best)**2)
            
            # Overwrite with calibrated RA/Dec converted to milliarcsecs
            self.RA_offset = ParamRange(ra_best*platescale, ra_err_full_mas)
            self.Dec_offset = ParamRange(dec_best*platescale, dec_err_full_mas)

            print("RA offset = {0} +/- {1} ({2}) mas".format(self.RA_offset.bestfit, self.RA_offset.error, self.RA_offset.error_2sided))
            print("Dec offset = {0} +/- {1} ({2}) mas".format(self.Dec_offset.bestfit, self.Dec_offset.error, self.Dec_offset.error_2sided))




def lnprior(fitparams, bounds, readnoise=False):
    """
    Bayesian prior

    Args:
        fitparams: array of params (size N)

        bounds: array of (N,2) with corresponding lower and upper bound of params
                bounds[i,0] <= fitparams[i] < bounds[i,1]
        readnoise: boolean. If True, the last fitparam fits for diagonal noise

    Returns:
        prior: 0 if inside bound ranges, -inf if outside

    """
    prior = 0.0

    for param, bound in zip(fitparams, bounds):
        if (param >= bound[1]) | (param < bound[0]):
            prior *= -np.inf
            break

    return prior


def lnlike(fitparams, fma, cov_func, readnoise=False):
    """
    Likelihood function
    Args:
        fitparams: array of params (size N). First three are [dRA,dDec,f]. Additional parameters are GP hyperparams
                    dRA,dDec: RA,Dec offsets from star. Also coordianaes in self.data_{RA,Dec}_offset
                    f: flux scale factor to normalizae the flux of the data_stamp to the model
        fma (FMAstrometry): a FMAstrometry object that has been fully set up to run
        cov_func (function): function that given an input [x,y] coordinate array returns the covariance matrix
                  e.g. cov = cov_function(x_indices, y_indices, sigmas, cov_params)
        readnoise: boolean. If True, the last fitparam fits for diagonal noise

    Returns:
        likeli: log of likelihood function (minus a constant factor)
    """
    dRA_trial = fitparams[0]
    dDec_trial = fitparams[1]
    f_trial = fitparams[2]
    hyperparms_trial = fitparams[3:]

    if readnoise:
        # last hyperparameter is a diagonal noise term. Separate it out
        readnoise_amp = np.exp(hyperparms_trial[-1])
        hyperparms_trial = hyperparms_trial[:-1]

    # get trial parameters out of log space
    f_trial = math.exp(f_trial)
    hyperparms_trial = np.exp(hyperparms_trial)

    dx = -(dRA_trial - fma.data_stamp_RA_offset_center)
    dy = dDec_trial - fma.data_stamp_Dec_offset_center

    fm_shifted = sinterp.shift(fma.fm_stamp, [dy, dx])

    if fma.padding > 0:
        fm_shifted = fm_shifted[fma.padding:-fma.padding, fma.padding:-fma.padding]

    diff_ravel = fma.data_stamp.ravel() - f_trial * fm_shifted.ravel()

    cov = cov_func(fma.data_stamp_RA_offset.ravel(), fma.data_stamp_Dec_offset.ravel(), fma.noise_map.ravel(),
                   hyperparms_trial)

    if readnoise:
        # add a diagonal term
        cov = (1 - readnoise_amp) * cov + readnoise_amp * np.diagflat(fma.noise_map.ravel()**2 )

    # solve Cov * x = diff for x = Cov^-1 diff. Numerically more stable than inverse
    # to make it faster, we comptue the Cholesky factor and use it to also compute the determinent
    try:
        (L_cov, lower_cov) = linalg.cho_factor(cov)
        cov_inv_dot_diff = linalg.cho_solve((L_cov, lower_cov), diff_ravel) # solve Cov x = diff for x
    except: 
        cov_inv = np.linalg.inv(cov)
        cov_inv_dot_diff = np.dot(cov_inv, diff_ravel)
    residuals = diff_ravel.dot(cov_inv_dot_diff)

    # compute log(det(Cov))
    logdet = 2*np.sum(np.log(np.diag(L_cov)))
    constant = logdet

    return -0.5 * (residuals + constant)


def lnprob(fitparams, fma, bounds, cov_func, readnoise=False):
    """
    Function to compute the relative posterior probabiltiy. Product of likelihood and prior
    Args:
        fitparams: array of params (size N). First three are [dRA,dDec,f]. Additional parameters are GP hyperparams
                    dRA,dDec: RA,Dec offsets from star. Also coordianaes in self.data_{RA,Dec}_offset
                    f: flux scale factor to normalizae the flux of the data_stamp to the model
        fma: a FMAstrometry object that has been fully set up to run
        bounds: array of (N,2) with corresponding lower and upper bound of params
                bounds[i,0] <= fitparams[i] < bounds[i,1]
        cov_func: function that given an input [x,y] coordinate array returns the covariance matrix
                  e.g. cov = cov_function(x_indices, y_indices, sigmas, cov_params)
        readnoise: boolean. If True, the last fitparam fits for diagonal noise

    Returns:

    """
    lp = lnprior(fitparams, bounds, readnoise=readnoise)

    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(fitparams, fma, cov_func, readnoise=readnoise)


class ParamRange(object):
    """
    Stores the best fit value and uncertainities for a parameter in a neat fasion

    Args:
        bestfit (float): the bestfit value
        err_range: either a float or a 2-element tuple (+val1, -val2) and gives the 1-sigma range

    Attributes:
        bestfit (float): the bestfit value
        error (float): the average 1-sigma error
        error_2sided (np.array): [+error1, -error2] 2-element array with asymmetric errors
    """
    def __init__(self, bestfit, err_range):
        self.bestfit = bestfit

        if isinstance(err_range, (int, float)):
            self.error = err_range
            self.error_2sided = np.array([err_range, -err_range])
        elif len(err_range) == 2:
            self.error_2sided = np.array(err_range)
            self.error = np.mean(np.abs(err_range))