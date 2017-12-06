import os
import glob
import numpy as np
import astropy.io.fits as fits
import pyklip.instruments.Instrument as Instrument
import pyklip.instruments.GPI as GPI
import pyklip.parallelized as parallelized
import pyklip.fakes as fakes

testdir = os.path.dirname(os.path.abspath(__file__)) + os.path.sep

def test_generic_dataset():
    """
    Tests the generic dataset interface into pyklip using some GPI data

    Just makes sure it doesn't crash
    """

    filelist = glob.glob(testdir + os.path.join("data", "S20131210*distorcorr.fits"))
    filename = filelist[0]
    
    numfiles = 1

    hdulist = fits.open(filename)
    inputdata = hdulist[1].data

    fakewvs = np.arange(37*numfiles)
    fakepas = np.arange(37*numfiles)
    fakecenters = np.array([[140,140] for _ in fakewvs])
    filenames = np.repeat([filename], 37)

    dataset = Instrument.GenericData(inputdata, fakecenters, parangs=fakepas, wvs=fakewvs, filenames=filenames)

    dataset.savedata(os.path.join(testdir, "generic_dataset.fits"), dataset.input)
    # it didn't crash? Good enough

def test_gpi_dataset():
    """
    Tests the GPI data interface, mostly on some edge cases since the general case is tested in test_parallelized_klip
    """
    # this shouldn't crash
    dataset = GPI.GPIData()

    # empty filelist should raise an error
    error_raised = False
    filelist = []
    try:
        dataset = GPI.GPIData(filelist)
    except ValueError:
        error_raised = True
    
    assert error_raised


def test_spectral_collapse():
    """
    Tests the spectral collpase feature
    """
     # grab the files
    filelist = glob.glob(testdir + os.path.join("data", "S20131210*distorcorr.fits"))
    # hopefully there is still 3 filelists
    assert(len(filelist) == 3)

    # create the dataset object
    dataset = GPI.GPIData(filelist, highpass=False)

    # collapse into 2 channels
    dataset.spectral_collapse(collapse_channels=2)

    assert(dataset.input.shape[0] == len(filelist)*2)
    assert(np.size(dataset.spot_flux) == len(filelist)*2)

    # collapse again, now into broadband
    dataset.spectral_collapse()

    assert(dataset.input.shape[0] == len(filelist))

    # run a broadband reduction
    outputdir = testdir
    prefix = "broadbandcollapse-betapic-j-k100a9s4m1-fakes50pa50"
    parallelized.klip_dataset(dataset, outputdir=outputdir, fileprefix=prefix,
                          annuli=9, subsections=4, movement=1, numbasis=[1],
                          calibrate_flux=True, mode="ADI", lite=False, highpass=True)


    # look at the output data. Validate the KL mode cube
    kl_hdulist = fits.open("{out}/{pre}-KLmodes-all.fits".format(out=outputdir, pre=prefix))
    klframe = kl_hdulist[1].data[0]

    # check beta pic b is where we think it is
    true_sep = 426.6 / 1e3 / GPI.GPIData.lenslet_scale # in pixels
    true_pa = 212.2 # degrees

    # find planet in collapsed cube
    flux_meas, x_meas, y_meas, fwhm_meas = fakes.retrieve_planet(klframe, dataset.output_centers[0], dataset.output_wcs[0],
                                                                 true_sep, true_pa, searchrad=4, guesspeak=2.e-5,
                                                                 guessfwhm=2)
    print(flux_meas, x_meas, y_meas, fwhm_meas)

    # positonal error
    theta = fakes.convert_pa_to_image_polar(true_pa, dataset.output_wcs[0])
    true_x = true_sep * np.cos(np.radians(theta)) + dataset.output_centers[0, 0]
    true_y = true_sep * np.sin(np.radians(theta)) + dataset.output_centers[0, 1]
    assert np.abs(true_x - x_meas) < 0.4
    assert np.abs(true_y - y_meas) < 0.4


if __name__ == "__main__":
    test_spectral_collapse()