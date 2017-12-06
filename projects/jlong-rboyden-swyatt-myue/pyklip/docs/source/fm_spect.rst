.. _fmspect-label:

Spectrum Extraction using extractSpec FM
========================================

This document describes how to use KLIP-FM to extract a spectrum,
described in 
`Pueyo et al. (2016) <http://adsabs.harvard.edu/abs/2016ApJ...824..117P>`_ 
to account the effect of the companion signal in the reference library
when measuring its spectrum.

Running gen_fm and invert_spect_fmodel
--------------------------------------
gen_fm and invert_spect_fm are modules in pyklip/fmlib/extractSpec

gen_fm generates the forward model array given a pyklip instrument 
dataset and invert_spect_fm returns a spectrum in contrast units 
given the forward model array.

gen_fm usage::
 
    import glob
    import pyklip.instruments.GPI as GPI
    import pyklip.fmlib.extractSpec as es

    files = glob.glob("path/to/dataset/*.fits")
    dataset = GPI.GPIData(files, highpass=True)
    dataset.generate_psf_cube(20)

    pars = (45, 222) # separation and pa
    # other optional parameters shown below w/ default values
    fmarr = es.gen_fm(dataset, pars, numbasis=20, mv=2.0, stamp=10, numthreads=4,
                      spectra_template=None)
    # numbasis is K_klip
    # mv is movement parameter for reference library selection
    # stamp is postage stamp size
    # numthreads is specific to your machine
    # spectra_template is default None

    # You may want to hold on to the klipped psf for later
    klipped = fmout[:,:,-1,:]
    klipped_coadd = np.zeros((num_k_klip, nl, stamp*stamp))
    dataset.klipped = klipped_coadd

    spectrum, fm_matrix = es.invert_spect_fmodel(fmarr, dataset, method="JB")
    # method indicates which matrix inversion method to use,
    # "JB" matrix inversion adds up over all exposures, then inverts
    # "leastsq" uses a leastsq solver.
    # "LP" inversion adds over frames and one wavelength axis, then inverts
    # (LP is not recommended)

The units of the spectrum, FM matrix, and klipped data are all in raw data units
in this example. Calibration of instrument and atmospheric transmmission and 
stellar spectrum are left to the user

An important note about GPI data: There is a keyword to convert the spectrum
to contrast units by setting unit="CONTRAST" however, the FM matrix and the 
klipped data remain in raw data units. We recommend calculating the contrast
conversion separately and explain the steps in the following section.

Converting from raw DN to contrast for GPI data
-----------------------------------------------
Converting to contrast for GPI data is done using the flux of the satellite spots.
The dataset object has attribute spot_flux that represent the average peak flux of
the four spots. The normalization factor is computed by dividing the spot flux 
spectrum by the ratio between the stellar flux and the spot flux (stored in 
spot_ratio) and adjusting for the ratio between the peak and the sum of the spot
PSF.

Example::

    # First set up a PSF model and sums
    sat_spot_sum = np.sum(dataset.psfs, axis=(1,2))
    PSF_cube = dataset.psfs / sat_spot_sum[:,None,None]
    sat_spot_spec = np.nanmax(PSF_cube, axis=(1,2))
    # Now divide the sum by the peak for each wavelength slice
    aper_over_peak_ratio = np.zeros(nl)
    for l_id in range(PSF_cube.shape[0]):
        aper_over_peak_ratio[l_id] = \
            np.nansum(PSF_cube[l_id,:,:]) / sat_spot_spec[l_id]

    # Avg spot ratio
    band = dataset.prihdrs[0]['APODIZER'].split('_')[1]
    # DO NOT USE dataset.band !!! (always returns K1)
    spot_flux_spectrum = \
        np.median(dataset.spot_flux.reshape(len(dataset.spot_flux)/nl, nl), axis=0)
    spot_to_star_ratio = dataset.spot_ratio[band]
    normfactor = aper_over_peak_ratio*spot_flux_spectrum / spot_to_star_ratio

Divide your spectrum in DN by this normalization factor.::

    spectrum_contrast = spectrum / normfactor



Calculating Errobars
--------------------
One way to calculate a spectrum with errorbars after running the above::

    # This will take a long time - it is running the fm for multiple fakes
    N_frames = len(dataset.input)
    N_cubes = len(dataset.exthdrs)
    nl = N_frames/N_cubes

    # generate a psf model
    sat_spot_sum = np.sum(dataset.psfs, axis=(1,2))
    PSF_cube = dataset.psfs / sat_spot_sum[:,None,None]
    inputpsfs = np.tile(PSF_cube, (N_cubes, 1, 1))

    # Define a set of PAs to put in fake sources
    npas = 11
    pas = (np.linspace(loc[1], loc[1]+360, num=npas+2)%360)[1:-1]

    # array to store fake source fluxes
    flux = copy(dataset.spot_flux)
    error = np.zeros(fmout.shape[0])
    # Loop through number of numbasis
    for ii in range(fmout.shape[0]):
        # store the extracted spectrum into the fake flux array
        for k in range(N_cubes):
            flux[k*nl:(k+1)*nl] = estim_spec[ii, ...]
        fake_spectra= np.zeros((len(pas), estim_spec.shape[1]))
        for p, pa in enumerate(pas):
            print "Fake # ({0}, {1}/{2}".format(ii, p+1, len(pas))
            print "flux:",fluxjb
            psf_inject = inputpsfs*(flux)[:,None,None]
            # Create a temporary dataset so things don't get reset
            tmp_dataset = setup_data(filelist)
            fakes.inject_planet(tmp_dataset.input, tmp_dataset.centers, psf_inject,\
                                tmp_dataset.wcs, loc[0], pa)
            fmtmp = es.gen_fm(tmp_dataset, (loc[0], pa), numbasis=numbasis[ii],\
                              mv=args.movement, stamp=10, numthreads=8)
            fake_spectra[p,:], fakefm = es.invert_spect_fmodel(fmtmp, tmp_dataset, method="JB")
            del tmp_dataset
            del fmtmp

        # Get the error of your fakes (here just taking standard deviation)
        error[ii] = np.std(fake_spectra, axis=0)

You may also want to look at the "bias" -- 
are your fake spectra evenly distributed around the recovered spectrum?::

    offset[ii] = estim_spec[ii] - np.median(fake_spectra, axis=0)

how does this offset change with numbasis & movement?
    
Some diagnostics you can run to check the FM
--------------------------------------------
1) First step is to look at the postage stamp of the klipped data and make sure
the companion signal is in there.::

    
    # useful values
    N_frames = len(dataset.input)
    N_cubes = len(dataset.exthdrs)
    nl = int(N_frames / N_cubes)
    num_k_klip = len(numbasis)

    fmarr = es.gen_fm(dataset, pars, numbasis=20, mv=2.0, stamp=10, numthreads=4,
                      spectra_template=None)
    klipped_data = fmarr[:,:,-1, :]
    klipped_coadd = np.zeros((num_k_klip, nl, stamp*stamp))
    for k in range(N_cubes):
        klipped_coadd = klipped_coadd + klipped[:, k*nl:(k+1)*nl, :]
    klipped_coadd.shape = [num_k_klip, nl, int(stamp), int(stamp)]
    # you can save this as an attribute of dataset...
    dataset.klipped = klipped_coadd

    import matplotlib.pyplot as plt
    plt.figure()
    # pick a wavelength slice slc
    plt.imshow(dataset.klipped[slc], interpolation="nearest")
    plt.show()

2) You can compare the klipped PSF to the forward model::

    spectrum, fm_matrix = es.invert_spect_fmodel(fmarr, dataset, method="JB")
    # fm_matrix has shape (n_k_klip, npix, nwav)
    # spectrum has shape (n_k_klip, nwav)
    # To get the FM for kth element of numbasis:
    fm_image_k = np.dot(fm_matrix[k,:,:], spectrum[k].transpose()).reshape(nl, stamp, stamp)
    fm_image_combined = np.zeros((stamp, stamp))

    plt.figure()
    # compared the same wavelength slice slc
    plt.imshow(fm_image_combined[slc], interpolation="nearest")
    plt.show()

Do the two look the same? If yes -- this is a good sign. If not, something went wrong.



