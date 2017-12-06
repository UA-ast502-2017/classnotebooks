.. _bka-label:

Bayesian KLIP-FM Astrometry (BKA)
==================================

This tutorial will provide the necessary steps to run the Bayesian KLIP-FM Astorometry technique (BKA)
that is described in `Wang et al. (2016) <https://arxiv.org/abs/1607.05272>`_ to obtain one milliarcsecond
astrometry on β Pictoris b.

Why BKA?
---------

Astrometry of directly imaged exoplanets is challenging since PSF subtraction algorithms (like pyKLIP)
distort the PSF of the planet. `Pueyo (2016) <http://arxiv.org/abs/1604.06097>`_ provide a technique to
forward model the PSF of a planet through KLIP. Taking this forward model, you could fit it to the data
with MCMC, but you would underestimate your errors because the noise in direct imaging data is correlated
(i.e. each pixel is not independent). To account for the correlated nature of the noise, we use Gaussian
process regression to model and account for the correlated nature of the noise. This allows us to obtain
accurate astrometry and accurate uncertainties on the astrometry.

BKA Requirements
-----------------
To run BKA, you need the additional packages installed, which should be available readily:

* `emcee <http://dan.iel.fm/emcee/current/>`_
* `corner <https://github.com/dfm/corner.py>`_

You also need the following pieces of data to forward model the data.

* Data to run PSF subtraction on
* A model or data of the instrumental PSF
* A good guess of the position of the planet (a center of light centroid routine should get the astrometry to a pixel)
* For IFS data, an estimate of the spectrum of the planet (it does not need to be very accurate, and 20% errors are fine)

Generating instrumental PSFs for GPI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A quick aside for GPI spectral mode data, here is how to generate the instrumental PSF from the satellite spots.

.. code-block:: python

    import glob
    import numpy as np
    import pyklip.instruments.GPI as GPI

    # read in the data into a dataset
    filelist = glob.glob("path/to/dataset/*.fits")
    dataset = GPI.GPIData(filelist)

    # generate instrumental PSF
    boxsize = 17 # we want a 17x17 pixel box centered on the instrumental PSF
    dataset.generate_psfs(boxrad=boxsize//2) # this function extracts the satellite spots from the data
    # now dataset.psfs contains a 37x25x25 spectral cube with the instrumental PSF
    # normalize the instrumental PSF so the peak flux is unity
    dataset.psfs /= (np.mean(dataset.spot_flux.reshape([dataset.spot_flux.shape[0] // 37, 37]),
                             axis=0)[:, None, None])


Here is an exmaple using three datacubes from the publicly available GPI data on beta Pic.
Note that the wings of the PSF are somewhat noisy, due to the fact the speckle noise
in J-band is high near the satellite spots. However, this should still give us an acceptable instrumental PSF.

.. image:: imgs/betpic_j_instrumental_psf.png

Forward Modelling the PSF with KLIP-FM
---------------------------------------
With an estimate of the planet position, the instrumental PSF, and, if applicable, an estimate of the spectrum,
we can use the :py:mod:`pyklip.fm` implementation of KLIP-FM and :py:class:`pyklip.fmlib.fmpsf.FMPlanetPSF` extension to
forward model the PSF of a planet through KLIP.

First, let us initalize :py:class:`pyklip.fmlib.fmpsf.FMPlanetPSF` to forward model the planet in our data.

For GPI, we are using normalized copies of the satellite spots as our input PSFs, and because of that, we need to pass in
a flux conversion value, ``dn_per_contrast``, that allows us to scale our ``guessflux`` in contrast units to data units. If
you are not using normalized PSFs, ``dn_per_contrast`` should be the factor that scales your input PSF to the flux of the 
unocculted star. If your input PSF is already scaled to the flux of the stellar PSF, ``dn_per_contrast`` is optional 
and should not actually be passed into the function.

.. code-block:: python

    # setup FM guesses
    # You should change these to be suited to your data!
    numbasis = np.array([1, 7, 100]) # KL basis cutoffs you want to try
    guesssep = 30.1 # estimate of separation in pixels
    guesspa = 212.2 # estimate of position angle, in degrees
    guessflux = 5e-5 # estimated contrast
    dn_per_contrast = your_flux_conversion # factor to scale PSF to star PSF. For GPI, this is dataset.dn_per_contrast
    guessspec = your_spectrum # should be 1-D array with number of elements = np.size(np.unique(dataset.wvs))

    # initialize the FM Planet PSF class
    import pyklip.fmlib.fmpsf as fmpsf
    fm_class = fmpsf.FMPlanetPSF(dataset.input.shape, numbasis, guesssep, guesspa, guessflux, dataset.psfs,
                                 np.unique(dataset.wvs), dn_per_contrast, star_spt='A6',
                                 spectrallib=[guessspec])

.. note::
   When executing the initializing of FMPlanetPSF, you will get a warning along the lines of
   "The coefficients of the spline returned have been computed as the minimal norm least-squares solution of a
   (numerically) rank deficient system." This is completeness normal and expected, and should not be an issue.

Next we will run KLIP-FM with the :py:mod:`pyklip.fm` module. Before we run it, we will need to pick our
PSF subtraction parameters (see the :ref:`basic-tutorial-label` for more details on picking KLIP parameters).
For our zones, we will run KLIP only on one zone: an annulus centered on the guessed location of the planet with
a width of 30 pixels. The width just needs to be big enough that you see the entire planet PSF.

.. code-block:: python

    # PSF subtraction parameters
    # You should change these to be suited to your data!
    outputdir = "." # where to write the output files
    prefix = "betpic-131210-j-fmpsf" # fileprefix for the output files
    annulus_bounds = [[guesssep-15, guesssep+15]] # one annulus centered on the planet
    subsections = 1 # we are not breaking up the annulus
    padding = 0 # we are not padding our zones
    movement = 4 # we are using an conservative exclusion criteria of 4 pixels

    # run KLIP-FM
    import pyklip.fm as fm
    fm.klip_dataset(dataset, fm_class, outputdir=outputdir, fileprefix=prefix, numbasis=numbasis,
                    annuli=annulus_bounds, subsections=subsections, padding=padding, movement=movement)


This will now run KLIP-FM, producing both a PSF subtracted image of the data and a forward-modelled PSF of the planet
at the gussed location of the planet. The PSF subtracted image as the "-klipped-" string in its filename, while the
forward-modelled planet PSF has the "-fmpsf-" string in its filename.

Fitting the Astrometry using MCMC and Gaussian Processes
--------------------------------------------------------
Now that we have the forward-modeled PSF and the data, we can fit them in a Bayesian framework
using Gaussian processes to model the correlated noise and MCMC to sample the posterior distribution.

First, let's read in the data from our previous forward modelling. We will take the collapsed
KL mode cubes, and select the KL mode cutoff we want to use. For the example, we will use
7 KL modes to model and subtract off the stellar PSF.

.. code-block:: python

    import os
    import astropy.io.fits as fits
    # read in outputs
    output_prefix = os.path.join(outputdir, prefix)
    fm_hdu = fits.open(output_prefix + "-fmpsf-KLmodes-all.fits")
    data_hdu = fits.open(output_prefix + "-klipped-KLmodes-all.fits")

    # get FM frame, use KL=7
    fm_frame = fm_hdu[1].data[1]
    fm_centx = fm_hdu[1].header['PSFCENTX']
    fm_centy = fm_hdu[1].header['PSFCENTY']

    # get data_stamp frame, use KL=7
    data_frame = data_hdu[1].data[1]
    data_centx = data_hdu[1].header["PSFCENTX"]
    data_centy = data_hdu[1].header["PSFCENTY"]

    # get initial guesses
    guesssep = fm_hdu[0].header['FM_SEP']
    guesspa = fm_hdu[0].header['FM_PA']

We will generate a :py:class:`pyklip.fitpsf.FMAstrometry` object that we handle all of the fitting processes.
The first thing we will do is create this object, and feed it in the data and forward model. It will use them to
generate stamps of the data and forward model which can be accessed using ``fma.data_stmap`` and ``fma.fm_stamp``
respectively. When reading in the data, it will also generate a noise map for the data stamp by computing the standard
deviation around an annulus, with the planet masked out

.. code-block:: python

    import pyklip.fitpsf as fitpsf
    # create FM Astrometry object
    fma = fitpsf.FMAstrometry(guesssep, guesspa, 13)

    # generate FM stamp
    # padding should be greater than 0 so we don't run into interpolation problems
    fma.generate_fm_stamp(fm_frame, [fm_centx, fm_centy], padding=5)

    # generate data_stamp stamp
    # not that dr=4 means we are using a 4 pixel wide annulus to sample the noise for each pixel
    # exclusion_radius excludes all pixels less than that distance from the estimated location of the planet
    fma.generate_data_stamp(data_frame, [data_centx, data_centy], dr=4, exclusion_radius=10)

Next we need to choose the Gaussian process kernel. We currently only support the Matern (ν=3/2)
and square exponential kernel, so we will pick the Matern kernel here. Note that there is the option
to add a diagonal (i.e. read/photon noise) term to the kernel, but we have chosen not to use it in this
example. If you are not dominated by speckle noise (i.e. around fainter stars or further out from the star),
you should enable the read noies term.

.. code-block:: python

    # set kernel, no read noise
    corr_len_guess = 3.
    corr_len_label = r"$l$"
    fma.set_kernel("matern32", [corr_len_guess], [corr_len_label])

Now we need to set bounds on our priors for our MCMC. We are going to be simple and use uninformative priors.
The priors in the x/y posible will be flat in linear space, and the priors on the flux scaling and kernel parameters
will be flat in log space, since they are scale paramters. In the following function below, we will set the boundaries
of the priors. The first two values are for x/y and they basically say how far away (in pixels) from the
guessed position of the planet can the chains wander. For the rest of the parameters, the values say how many ordres
of magnitude can the chains go from the guessed value (e.g. a value of 1 means we allow a factor of 10 variation
in the value).

.. code-block:: python

    # set bounds
    x_range = 1.5 # pixels
    y_range = 1.5 # pixels
    flux_range = 1. # flux can vary by an order of magnitude
    corr_len_range = 1. # between 0.3 and 30
    fma.set_bounds(x_range, y_range, flux_range, [corr_len_range])

Finally, we are set to run the MCMC sampler (using the emcee package). Here we have provided a wrapper to already
set up the likelihood and prior. All we want to do is specify the number of walkers, number of steps each walker takes,
and the number of production steps the walkers take. We also can specify the number of threads to use.
If you have not turned BLAS and MKL off, you probably only want one or a few threads, as MKL/BLAS automatically
parallelizes the likelihood calculation, and trying to parallelize on top of that just creates extra overhead.

.. code-block:: python

    # run MCMC fit
    fma.fit_astrometry(nwalkers=100, nburn=200, nsteps=800, numthreads=1)


When the MCMC finishes running, we have our answer for the location of the planet in the data.
Here are some fields to access this information:

* ``fma.RA_offset``: RA offset of the planet from the star as determined by the median of the marginalized posterior
* ``fma.Dec_offset``: Dec offset of the plnaet from the star as determined by the median of the marginalized posterior
* ``fma.RA_offset_1sigma``: 16th and 84th percentile values for the RA offset of the planet
* ``fma.Dec_offset_1sigma``: 16th and 84th percentile values for the Dec offset of the planet
* ``fma.flux``, ``fma.flux_1sigma``: same thing except for the flux of the planet
* ``fma.covar_param_bestfits``, ``fma.covar_param_1sigma``: same thing for the hyperparameters on the Gaussian process kernel. These are both kept in a list with length equal to the number of hyperparameters.
* ``fma.sampler``: this is the ``emcee.EnsembleSampler`` object which contains the full chains and other MCMC fitting information

The RA offset and Dec offset are what we are interested in for the purposes of astrometry. The flux scaling
paramter (α) and the correlation length (l) are hyperparameters we marginalize over. First,
we want to check to make sure all of our chains have converged by plotting them. As long as they have
settled down (no large scale movements), then the chains have probably converged.

.. code-block:: python

    import matplotlib.pylab as plt
    fig = plt.figure(figsize=(10,8))

    # grab the chains from the sampler
    chain = fma.sampler.chain

    # plot RA offset
    ax1 = fig.add_subplot(411)
    ax1.plot(chain[:,:,0].T, '-', color='k', alpha=0.3)
    ax1.set_xlabel("Steps")
    ax1.set_ylabel(r"$\Delta$ RA")

    # plot Dec offset
    ax2 = fig.add_subplot(412)
    ax2.plot(chain[:,:,1].T, '-', color='k', alpha=0.3)
    ax2.set_xlabel("Steps")
    ax2.set_ylabel(r"$\Delta$ Dec")

    # plot flux scaling
    ax3 = fig.add_subplot(413)
    ax3.plot(chain[:,:,2].T, '-', color='k', alpha=0.3)
    ax3.set_xlabel("Steps")
    ax3.set_ylabel(r"$\alpha$")

    # plot hyperparameters.. we only have one for this example: the correlation length
    ax4 = fig.add_subplot(414)
    ax4.plot(chain[:,:,3].T, '-', color='k', alpha=0.3)
    ax4.set_xlabel("Steps")
    ax4.set_ylabel(r"$l$")

Here is an example using three cubes of public GPI data on beta Pic.

.. image:: imgs/betpic_j_bka_chains.png

We can also plot the corner plot to look at our posterior distribution and correlation between parameters.

.. code-block:: python

    fig = plt.figure()
    fig = fma.make_corner_plot(fig=fig)

.. image:: imgs/betpic_j_bka_corner.png

Hopefully the corner plot does not contain too much structure (the posteriors should be roughly Gaussian).
In the example figure from three cubes of GPI data on beta Pic, the residual speckle noise has not been
very whitened, so there is some asymmetry in the posterior, which represents the local strucutre of
the speckle noise. These posteriors should become more Gaussian as we add more data and whiten the speckle noise.
And finally, we can plot the visual comparison of our data, best fitting model, and residuals to the fit.

.. code-block:: python

    fig = plt.figure()
    fig = fma.best_fit_and_residuals(fig=fig)

And here is the example from the three frames of beta Pic b J-band GPI data:

.. image:: imgs/betpic_j_bka_comparison.png

The data and best fit model should look pretty close, and the residuals hopefully do not show any obvious strcuture that
was missed in the fit. The residual ampltidue should also be consistent with noise. If that is the case, we can use the
best fit values for the astrometry of this epoch. 

The best fit values from the MCMC give us the raw RA and Dec offsets for the planet. We will still need to fold in uncertainties
in the star location and calibration uncertainties. To do this, we use :py:meth:`pyklip.fitpsf.FMAstrometry.propogate_errs` to 
include these terms and obtain our final astrometric values. All of the infered parameters and the raw fit parameters are fields 
that can be accessed (see :py:class:`pyklip.fitpsf.FMAstrometry`) and each field is a :py:class:`pyklip.fitpsf.ParamRange` object
that stores the best fit value and 1-sigma range (both average error, and 2-sided uncertainites are included). 

.. code-block:: python

    fma.propogate_errs(star_center_err=0.05, platescale=GPI.GPIData.lenslet_scale*1000, platescale_err=0.007, pa_offset=-0.1, pa_uncertainty=0.13)


    # show what the raw uncertainites are on the location of the planet
    print("\nPlanet Raw RA offset is {0} +/- {1}, Raw Dec offset is {2} +/- {3}".format(fma.raw_RA_offset.bestfit, fma.raw_RA_offset.error,
                                                                                        fma.raw_Dec_offset.bestfit, fma.raw_Dec_offset.error)) 
    
    # Full error budget included
    print("Planet RA offset is at {0} with a 1-sigma uncertainity of {1}".format(fma.RA_offset.bestfit, fma.RA_offset.error))
    print("Planet Dec offset is at {0} with a 1-sigma uncertainity of {1}".format(fma.Dec_offset.bestfit, fma.Dec_offset.error))

    # Propogate errors into separation and PA space
    print("Planet separation is at {0} with a 1-sigma uncertainity of {1}".format(fma.sep.bestfit, fma.sep.error))
    print("Planet PA at {0} with a 1-sigma uncertainity of {1}".format(fma.PA.bestfit, fma.PA.error))


