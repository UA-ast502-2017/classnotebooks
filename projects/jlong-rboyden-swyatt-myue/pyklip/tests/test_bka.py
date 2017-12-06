import glob
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import scipy.interpolate as sinterp
import astropy.io.fits as fits

import pyklip.instruments.GPI as GPI
import pyklip.fmlib.fmpsf as fmpsf
import pyklip.fm as fm
import pyklip.fitpsf as fitpsf

testdir = os.path.dirname(os.path.abspath(__file__)) + os.path.sep

def test_fmastrometry():
    """
    Tests FM astrometry using MCMC + GP Regression

    """
    # time it
    t1 = time.time()

    # # open up already generated FM and data_stamp
    # fm_hdu = fits.open("/home/jwang/GPI/betapic/fm_models/final_altpsf/pyklip-131118-h-k100m4-dIWA8-nohp-klipfm-KL7cube.fits")
    # data_hdu = fits.open("/home/jwang/GPI/betapic/klipped/final_altpsf/pyklip-131118-h-k100m4-dIWA8-nohp-onezone-KL7cube.fits")

    ########### generate FM ############
    # grab the files
    filelist = glob.glob(testdir + os.path.join("data", "S20131210*distorcorr.fits"))
    filelist.sort()

    # hopefully there is still 3 filelists
    assert(len(filelist) == 3)

    # only read in one spectral channel
    skipslices = [i for i in range(37) if i != 7 and i != 33]
    # read in data
    dataset = GPI.GPIData(filelist, highpass=9, skipslices=skipslices)

    numwvs = np.size(np.unique(dataset.wvs))
    assert(numwvs == 2)

    # save old centesr for later
    oldcenters = np.copy(dataset.centers)

    # generate PSF
    dataset.generate_psfs(boxrad=25//2)
    dataset.psfs /= (np.mean(dataset.spot_flux.reshape([dataset.spot_flux.shape[0] // numwvs, numwvs]), axis=0)[:, None, None])

    # read in model spectrum
    model_file = os.path.join(testdir, "..", "pyklip", "spectra", "cloudy", "t1600g100f2.flx")
    spec_dat = np.loadtxt(model_file)
    spec_wvs = spec_dat[1]
    spec_f = spec_dat[3]
    spec_interp = sinterp.interp1d(spec_wvs, spec_f, kind='nearest')
    inputspec = spec_interp(np.unique(dataset.wvs))

    # setup FM guesses
    numbasis = np.array([1, 7, 100])
    guesssep = 0.4267 / GPI.GPIData.lenslet_scale
    guesspa = 212.15
    guessflux = 5e-5
    print(guesssep, guesspa)
    fm_class = fmpsf.FMPlanetPSF(dataset.input.shape, numbasis, guesssep, guesspa, guessflux, dataset.psfs,
                                 np.unique(dataset.wvs), dataset.dn_per_contrast, star_spt='A6',
                                 spectrallib=[inputspec])
    # run KLIP-FM
    prefix = "betpic-131210-j-fmpsf"
    fm.klip_dataset(dataset, fm_class, outputdir=testdir, fileprefix=prefix, numbasis=numbasis,
                    annuli=[[guesssep-15, guesssep+15]], subsections=1, padding=0, movement=2)

    # before we do anything else, check that dataset.centers remains unchanged
    assert(dataset.centers[0][0] == oldcenters[0][0])

    # read in outputs
    output_prefix = os.path.join(testdir, prefix)
    fm_hdu = fits.open(output_prefix + "-fmpsf-KLmodes-all.fits")
    data_hdu = fits.open(output_prefix + "-klipped-KLmodes-all.fits")


    # get FM frame
    fm_frame = np.nanmean(fm_hdu[1].data, axis=0)
    fm_centx = fm_hdu[1].header['PSFCENTX']
    fm_centy = fm_hdu[1].header['PSFCENTY']

    # get data_stamp frame
    data_frame = np.nanmean(data_hdu[1].data, axis=0)
    data_centx = data_hdu[1].header["PSFCENTX"]
    data_centy = data_hdu[1].header["PSFCENTY"]

    # get initial guesses
    guesssep = fm_hdu[0].header['FM_SEP']
    guesspa = fm_hdu[0].header['FM_PA']

    # create FM Astrometry object
    fma = fitpsf.FMAstrometry(guesssep, guesspa, 9)

    # generate FM stamp
    fma.generate_fm_stamp(fm_frame, [fm_centx, fm_centy], padding=5)

    # generate data_stamp stamp
    fma.generate_data_stamp(data_frame, [data_centx, data_centy], dr=6)

    # set kernel, with read noise
    fma.set_kernel("matern32", [3.], [r"$l$"], True, 0.05)

    # set bounds
    fma.set_bounds(1.5, 1.5, 1, [1.], 1)

    print(fma.guess_RA_offset, fma.guess_Dec_offset)
    print(fma.bounds)
    # test likelihood function
    mod_bounds = np.copy(fma.bounds)
    mod_bounds[2:] = np.log(mod_bounds[2:])
    print(mod_bounds)
    lnpos = fitpsf.lnprob((-16, -25.7, np.log(0.8), np.log(3.3), np.log(0.05)), fma, mod_bounds, fma.covar, readnoise=True)
    print(lnpos, np.nanmean(data_frame), np.nanmean(fm_frame), np.nanmean(fma.data_stamp), np.nanmean(fma.fm_stamp))
    assert lnpos > -np.inf

    # run MCMC fit
    fma.fit_astrometry(nburn=150, nsteps=25, nwalkers=50, numthreads=1)

    print("{0} seconds to run".format(time.time()-t1))

    fma.propogate_errs(star_center_err=0.05, platescale=GPI.GPIData.lenslet_scale*1000, platescale_err=0.007, pa_offset=-0.1, pa_uncertainty=0.13)

    assert(np.abs(fma.RA_offset.bestfit - -227.2) < 5.)
    assert(np.abs(fma.Dec_offset.bestfit - -361.1) < 5.)


    fma.best_fit_and_residuals()
    plt.savefig("tests/bka2.png")


if __name__ == "__main__":
    test_fmastrometry()