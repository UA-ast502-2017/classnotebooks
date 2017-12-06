.. _fmmf-label:

Forward Model Matched Filter (FMMF)
==================================

This tutorial will provide the necessary steps to run the Forward Model Matched Filter (FMMF)
that is described in `Ruffio et al. (2016) <https://arxiv.org/pdf/1705.05477.pdf>`_. This example uses fm.klip_dataset() wrapper.

.. note::
    The Klip POst Processing (KPOP) framework also includes a wrapper to run FMMF. KPOP was designed to help the
    reduction of an entire survey by abstracting some tasks and providing a consistent class based architecture to
    calculate matched filter maps, SNR maps, detect candidate...

Why FMMF?
---------

In signal processing, a matched filter is the linear filter maximizing the Signal to Noise Ratio (SNR) of a known signal
 in the presence of additive noise.

Matched filters are used in Direct imaging to detect point sources using the expected shape of the planet Point Spread
Function (PSF) as a template.

Detection of directly imaged exoplanets is challenging since PSF subtraction algorithms (like pyKLIP)
distort the PSF of the planet. `Pueyo (2016) <http://arxiv.org/abs/1604.06097>`_ provide a technique to
forward model the PSF of a planet through KLIP.

FMMF uses this forward model as the template of the matched filter therefore improving the planet sensitivity of the
algorithm compared to a conventional approach.

FMMF Requirements
-----------------

FMMF is computationally very intensive. It will take up to a few days to run a typical GPI dataset for example on a
basic cluster node (but it's worth it! ;)).

You also need the following pieces of data to forward model the data.

* Data to run PSF subtraction on
* A model or data of the instrumental PSF
* For IFS data, an estimate of the spectrum of the planet that one wants to find.

Running FMMF
-----------------
There are 3 steps in running FMMF:

* Read the data using an instrument class
* Define the MatchedFilter object
* Call klip_dataset to run the reduction ()

We provide 2 examples using GPI and SPHERE data and the generaliztion to other instruments should be
straightforward.

GPI example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The script below uses the 3 GPI fits cubes of beta Pictoris available when downloading the pyklip repository.

.. code-block:: python

    import pyklip.parallelized as parallelized
    pykliproot = os.path.dirname(os.path.realpath(parallelized.__file__))
    inputDir = os.path.join(pykliproot,"..","tests","data")
    outputDir = os.path.join(inputDir,"fmmf_test")
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    ####################
    ## Define the instrument object
    from pyklip.instruments import GPI
    import glob
    dataset = GPI.GPIData(glob.glob(os.path.join(inputDir,"S*distorcorr.fits")), highpass=True)

    ####################
    ## Generate PSF cube for GPI from the satellite spots
    dataset.generate_psf_cube(20,same_wv_only=True)

    ####################
    ## Define the fmlib object
    # flat spectrum here, make sure to define your own and correct it for transmission
    spectrum_vec = np.ones((dataset.input.shape[0],))
    # Number KL modes used for KLIP
    numbasis = [5]
    # Number of images in the reference library
    maxnumbasis = [10]
    # Definition of the planet PSF
    PSF_cube_arr = dataset.psfs
    PSF_cube_wvs = np.unique(dataset.wvs)

    # Build the FM class to do matched filter
    import pyklip.fmlib.matchedFilter as mf
    fm_class = mf.MatchedFilter(dataset.input.shape,numbasis,
                                     PSF_cube_arr, PSF_cube_wvs,
                                     [spectrum_vec])
    # run KLIP-FM
    import pyklip.fm as fm
    prefix = "betpic-131210-J_GPI"
    annulus_bounds = 5
    N_pix_sector = 200
    padding = PSF_cube_arr.shape[1]//2
    movement = 2.0
    fm.klip_dataset(dataset, fm_class, outputdir=outputDir, fileprefix=prefix, numbasis=numbasis,
                    annuli=annulus_bounds, N_pix_sector=N_pix_sector, padding=padding, movement=movement)

SPHERE example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example show how to process SPHERE data that have been processed based on `Vigan et al. (2015) <http:astro.vigan.fr/tools.html>`_.

.. code-block:: python

    inputDir = "my/path/to/the/data"
    outputDir = "where/it/should/save/the/processed/data"

    ####################
    ## Define the instrument object
    data_cube = glob.glob(os.path.join(inputDir,"*_cube_coro.fits"))[0]
    psf_cube = glob.glob(os.path.join(inputDir,"*_cube_psf.fits"))[0]
    info_fits = glob.glob(os.path.join(inputDir,"*_info.fits"))[0]
    wavelength_info = glob.glob(os.path.join(inputDir,"*_wavelength.fits"))[0]
    import pyklip.instruments.SPHERE as sph
    dataset = sph.Ifs(data_cube,psf_cube,info_fits,wavelength_info)

    ####################
    ## Define the fmlib object
    # flat spectrum here, make sure to define your own and correct it for transmission
    spectrum_vec = np.ones((dataset.input.shape[0],))
    # Number KL modes used for KLIP
    numbasis = [5]
    # Number of images in the reference library
    maxnumbasis = [10]
    # Definition of the planet PSF
    PSF_cube_arr = dataset.psfs
    PSF_cube_wvs = dataset.psfs_wvs

    # Build the FM class to do matched filter
    import pyklip.fmlib.matchedFilter as mf
    fm_class = mf.MatchedFilter(dataset.input.shape,numbasis,
                                     PSF_cube_arr, PSF_cube_wvs,
                                     [spectrum_vec])

    # run KLIP-FM
    import pyklip.fm as fm
    prefix = "HD131399A_SPHERE_2015-06-12_IFS"
    annulus_bounds = [[35,40]]
    N_pix_sector = 50
    padding = PSF_cube_arr.shape[1]//2
    movement = 2.0
    fm.klip_dataset(dataset, fm_class, outputdir=outputDir, fileprefix=prefix, numbasis=numbasis,
                    annuli=annulus_bounds, N_pix_sector=N_pix_sector, padding=padding, movement=movement)
