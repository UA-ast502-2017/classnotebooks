import os
import re
import subprocess

import astropy.io.fits as fits
from astropy import wcs
from astropy.modeling import models, fitting
import numpy as np
import scipy.ndimage as ndimage
import scipy.stats

#different imports depending on if python2.7 or python3
import sys
from copy import copy
if sys.version_info < (3,0):
    #python 2.7 behavior
    import ConfigParser
    from pyklip.instruments.Instrument import Data
    from pyklip.instruments.utils.nair import nMathar
else:
    import configparser as ConfigParser
    from pyklip.instruments.Instrument import Data
    from pyklip.instruments.utils.nair import nMathar

from scipy.interpolate import interp1d
from pyklip.parallelized import high_pass_filter_imgs
from pyklip.fakes import gaussfit2d
from pyklip.fakes import gaussfit2dLSQ

class NIRC2Data(Data):
    """
    A sequence of Keck NIRC2 ADI Data. Each NIRC2Data object has the following fields and functions

    Args:
        filepaths: list of filepaths to files

    Attributes:
        input: Array of shape (N,y,x) for N images of shape (y,x)
        centers: Array of shape (N,2) for N centers in the format [x_cent, y_cent]
        filenums: Array of size N for the numerical index to map data to file that was passed in
        filenames: Array of size N for the actual filepath of the file that corresponds to the data
        PAs: Array of N for the parallactic angle rotation of the target (used for ADI) [in degrees]
        wvs: Array of N wavelengths of the images (used for SDI) [in microns]. For polarization data, defaults to "None"
        wcs: Array of N wcs astormetry headers for each image.
        IWA: a floating point scalar (not array). Specifies to inner working angle in pixels
        output: Array of shape (b, len(files), len(uniq_wvs), y, x) where b is the number of different KL basis cutoffs
        creator: string for creator of the data (used to identify pipelines that call pyklip)
        klipparams: a string that saves the most recent KLIP parameters

    Methods:
        readdata(): reread in the dadta
        savedata(): save a specified data in the GPI datacube format (in the 1st extension header)
        calibrate_output(): flux calibrate the output data
    """
    ##########################
    ###Class Initilization ###
    ##########################
    #some static variables to define the GPI instrument
    centralwave = {}  # in microns
    fpm_diam = {}  # in pixels
    flux_zeropt = {}
    #spot_ratio = {} #w.r.t. central star
    lenslet_scale = 1.0 # arcseconds per pixel (pixel scale)
    #ifs_rotation = 0.0  # degrees CCW from +x axis to zenith

    observatory_latitude = 0.0

    ## read in GPI configuration file and set these static variables
    package_directory = os.path.dirname(os.path.abspath(__file__))
    configfile = package_directory + "/" + "NIRC2.ini"
    config = ConfigParser.ConfigParser()
    try:
        config.read(configfile)
        #get pixel scale
        lenslet_scale = float(config.get("instrument", "pixel_scale_narrow"))  # arcsecond/pix
        #get IFS rotation
        #ifs_rotation = float(config.get("instrument", "ifs_rotation")) #degrees
        #get some information specific to each band
        bands = ['Ks', 'Lp', 'Ms']
        for band in bands:
            centralwave[band] = float(config.get("instrument", "cen_wave_{0}".format(band)))
            fpm_diam[band] = float(config.get("instrument", "fpm_diam_{0}".format(band))) / lenslet_scale  # pixels
            flux_zeropt[band] = float(config.get("instrument", "zero_pt_flux_{0}".format(band)))
            #spot_ratio[band] = float(config.get("instrument", "APOD_{0}".format(band)))
        observatory_latitude = float(config.get("observatory", "observatory_lat"))
    except ConfigParser.Error as e:
        print("Error reading GPI configuration file: {0}".format(e.message))
        raise e

    ####################
    ### Constructors ###
    ####################
    def __init__(self, filepaths=None):
        """
        Initialization code for NIRC2Data

        Note:
            see class docstring for argument details
        """
        super(NIRC2Data, self).__init__()
        self._output = None
        if filepaths is None:
            self._input = None
            self._centers = None
            self._filenums = None
            self._filenames = None
            self._PAs = None
            self._wvs = None
            self._wcs = None
            self._IWA = None
            self.spot_flux = None
            self.star_flux = None
            self.contrast_scaling = None
            self.prihdrs = None
            self.exthdrs = None
        else:
            self.readdata(filepaths)

    ################################
    ### Instance Required Fields ###
    ################################
    @property
    def input(self):
        return self._input
    @input.setter
    def input(self, newval):
        self._input = newval

    @property
    def centers(self):
        return self._centers
    @centers.setter
    def centers(self, newval):
        self._centers = newval

    @property
    def filenums(self):
        return self._filenums
    @filenums.setter
    def filenums(self, newval):
        self._filenums = newval

    @property
    def filenames(self):
        return self._filenames
    @filenames.setter
    def filenames(self, newval):
        self._filenames = newval

    @property
    def PAs(self):
        return self._PAs
    @PAs.setter
    def PAs(self, newval):
        self._PAs = newval

    @property
    def wvs(self):
        return self._wvs
    @wvs.setter
    def wvs(self, newval):
        self._wvs = newval

    @property
    def wcs(self):
        return self._wcs
    @wcs.setter
    def wcs(self, newval):
        self._wcs = newval

    @property
    def IWA(self):
        return self._IWA
    @IWA.setter
    def IWA(self, newval):
        self._IWA = newval

    @property
    def output(self):
        return self._output
    @output.setter
    def output(self, newval):
        self._output = newval


    ###############
    ### Methods ###
    ###############
    def readdata(self, filepaths):
        """
        Method to open and read a list of NIRC2 data

        Args:
            filespaths: a list of filepaths

        Returns:
            Technically none. It saves things to fields of the NIRC2Data object. See object doc string
        """
        #check to see if user just inputted a single filename string
        if isinstance(filepaths, str):
            filepaths = [filepaths]

        #make some lists for quick appending
        data = []
        filenums = []
        filenames = []
        rot_angles = []
        wvs = []
        centers = []
        wcs_hdrs = []
        star_fluxes = []
        spot_fluxes = []
        prihdrs = []

        #extract data from each file
        for index, filepath in enumerate(filepaths):
            cube, center, pa, wv, astr_hdrs, filt_band, fpm_band, ppm_band, star_flux, spot_flux, prihdr, exthdr = _nirc2_process_file(filepath)

            data.append(cube)
            centers.append(center)
            star_fluxes.append(star_flux)
            spot_fluxes.append(spot_flux)
            rot_angles.append(pa)
            wvs.append(wv)
            filenums.append(np.ones(pa.shape[0]) * index)
            wcs_hdrs.append(astr_hdrs)
            prihdrs.append(prihdr)

            #filename = np.chararray(pa.shape[0])
            #filename[:] = filepath
            filenames.append([filepath for i in range(pa.shape[0])])
        #convert everything into numpy arrays
        #reshape arrays so that we collapse all the files together (i.e. don't care about distinguishing files)
        data = np.array(data)
        dims = data.shape
        data = data.reshape([dims[0] * dims[1], dims[2], dims[3]])
        filenums = np.array(filenums).reshape([dims[0] * dims[1]])
        filenames = np.array(filenames).reshape([dims[0] * dims[1]])
        rot_angles = np.array(rot_angles).reshape([dims[0] * dims[1]])  # want North Up
        wvs = np.array(wvs).reshape([dims[0] * dims[1]])
        wcs_hdrs = np.array(wcs_hdrs).reshape([dims[0] * dims[1]])
        centers = np.array(centers).reshape([dims[0] * dims[1], 2])
        star_fluxes = np.array(star_fluxes).reshape([dims[0] * dims[1]])
        spot_fluxes = np.array(spot_fluxes).reshape([dims[0] * dims[1]])

        #set these as the fields for the GPIData object
        self._input = data
        self._centers = centers
        self._filenums = filenums
        self._filenames = filenames
        self._PAs = rot_angles
        self._wvs = wvs
        self._wcs = None#wcs_hdrs
        self.spot_flux = spot_fluxes
        self._IWA = NIRC2Data.fpm_diam[fpm_band]/2.0
        self.star_flux = star_fluxes
        self.contrast_scaling = 1./star_fluxes #GPIData.spot_ratio[ppm_band]/np.tile(np.mean(spot_fluxes.reshape(dims[0], dims[1]), axis=0), dims[0])
        self.prihdrs = prihdrs
        #self.exthdrs = exthdrs

    def savedata(self, filepath, data, klipparams = None, filetype = None, zaxis = None, center=None, astr_hdr=None,
                 fakePlparams = None, more_keywords=None):
        """
        Save data in a GPI-like fashion. Aka, data and header are in the first extension header

        Inputs:
            filepath: path to file to output
            data: 2D or 3D data to save
            klipparams: a string of klip parameters
            filetype: filetype of the object (e.g. "KL Mode Cube", "PSF Subtracted Spectral Cube")
            zaxis: a list of values for the zaxis of the datacub (for KL mode cubes currently)
            astr_hdr: wcs astrometry header (None for NIRC2)
            center: center of the image to be saved in the header as the keywords PSFCENTX and PSFCENTY in pixels.
                The first pixel has coordinates (0,0)
            fakePlparams: fake planet params
            more_keywords (dictionary) : a dictionary {key: value, key:value} of header keywords and values which will
                            written into the primary header

        """
        hdulist = fits.HDUList()
        hdulist.append(fits.PrimaryHDU(header=self.prihdrs[0]))
        hdulist.append(fits.ImageHDU(data=data, name="Sci"))

        # save all the files we used in the reduction
        # we'll assume you used all the input files
        # remove duplicates from list
        filenames = np.unique(self.filenames)
        nfiles = np.size(filenames)
        hdulist[0].header["DRPNFILE"] = nfiles
        for i, thispath in enumerate(filenames):
            thispath = thispath.replace("\\", '/')
            splited = thispath.split("/")
            fname = splited[-1]
#            matches = re.search('S20[0-9]{6}[SE][0-9]{4}', fname)
            filename = fname#matches.group(0)
            hdulist[0].header["FILE_{0}".format(i)] = filename

        # write out psf subtraction parameters
        # get pyKLIP revision number
        pykliproot = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        # the universal_newline argument is just so python3 returns a string instead of bytes
        # this will probably come to bite me later
        try:
            pyklipver = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=pykliproot, universal_newlines=True).strip()
        except:
            pyklipver = "unknown"
        hdulist[0].header['PSFSUB'] = "pyKLIP"
        hdulist[0].header.add_history("Reduced with pyKLIP using commit {0}".format(pyklipver))
        #if self.creator is None:
        #    hdulist[0].header['CREATOR'] = "pyKLIP-{0}".format(pyklipver)
        #else:
        #    hdulist[0].header['CREATOR'] = self.creator
        #    hdulist[0].header.add_history("Reduced by {0}".self.creator)

        # store commit number for pyklip
        hdulist[0].header['pyklipv'] = pyklipver

        if klipparams is not None:
            hdulist[0].header['PSFPARAM'] = klipparams
            hdulist[0].header.add_history("pyKLIP reduction with parameters {0}".format(klipparams))

        if fakePlparams is not None:
            hdulist[0].header['FAKPLPAR'] = fakePlparams
            hdulist[0].header.add_history("pyKLIP reduction with fake planet injection parameters {0}".format(fakePlparams))

        if filetype is not None:
            hdulist[0].header['FILETYPE'] = filetype

        if zaxis is not None:
            #Writing a KL mode Cube
            if "KL Mode" in filetype:
                hdulist[0].header['CTYPE3'] = 'KLMODES'
                #write them individually
                for i, klmode in enumerate(zaxis):
                    hdulist[0].header['KLMODE{0}'.format(i)] = klmode

        #use the dataset astr hdr if none was passed in
        #if astr_hdr is None:
        #    print self.wcs[0]
        #    astr_hdr = self.wcs[0]
        if astr_hdr is not None:
            #update astro header
            #I don't have a better way doing this so we'll just inject all the values by hand
            astroheader = astr_hdr.to_header()
            exthdr = hdulist[0].header
            exthdr['PC1_1'] = astroheader['PC1_1']
            exthdr['PC2_2'] = astroheader['PC2_2']
            try:
                exthdr['PC1_2'] = astroheader['PC1_2']
                exthdr['PC2_1'] = astroheader['PC2_1']
            except KeyError:
                exthdr['PC1_2'] = 0.0
                exthdr['PC2_1'] = 0.0
            #remove CD values as those are confusing
            exthdr.remove('CD1_1')
            exthdr.remove('CD1_2')
            exthdr.remove('CD2_1')
            exthdr.remove('CD2_2')
            exthdr['CDELT1'] = 1
            exthdr['CDELT2'] = 1

        #use the dataset center if none was passed in
        if center is None:
            center = self.output_centers[0]
        if center is not None:
            hdulist[0].header.update({'PSFCENTX':center[0],'PSFCENTY':center[1]})
            hdulist[0].header.update({'CRPIX1':center[0],'CRPIX2':center[1]})
            hdulist[0].header.add_history("Image recentered to {0}".format(str(center)))

        # store extra keywords in header
        if more_keywords is not None:
            for hdr_key in more_keywords:
                hdulist[0].header[hdr_key] = more_keywords[hdr_key]

        try:
            hdulist.writeto(filepath, overwrite=True)
        except TypeError:
            hdulist.writeto(filepath, clobber=True)
        hdulist.close()

    def calibrate_data(self, units="contrast"):
        """
        Calibrates the flux of the output of PSF subtracted data.

        Args:
            img: unclaibrated image.
                 If spectral is not set, this can either be a 2-D or 3-D broadband image
                 where the last two dimensions are [y,x]
                 If specetral is True, this is a 3-D spectral cube with shape [wv,y,x]
            spectral: if True, this is a spectral datacube. Otherwise, it is a broadband image.
            units: currently only support "contrast" w.r.t central star

        Return:
            img: calibrated image of the same shape (this is the same object as the input!!!)
        """
        if units == "contrast":
            for i in range(self.output.shape[0]):
                self.output[i] *= self.contrast_scaling[:, None, None]

    def calibrate_output(self, img, spectral=False, units="contrast"):
        """
        Calibrates the flux of the output of PSF subtracted data.

        Assumes the broadband flux calibration is just multiplication by a single scalar number whereas spectral
        datacubes may have a separate calibration value for each wavelength

        Args:
            img: unclaibrated image.
                 If spectral is not set, this can either be a 2-D or 3-D broadband image
                 where the last two dimensions are [y,x]
                 If specetral is True, this is a 3-D spectral cube with shape [wv,y,x]
            spectral: if True, this is a spectral datacube. Otherwise, it is a broadband image.
            units: currently only support "contrast" w.r.t central star

        Return:
            img: calibrated image of the same shape (this is the same object as the input!!!)
        """
        if units == "contrast":
            if spectral:
                # spectral cube, each slice needs it's own calibration
                numwvs = img.shape[0]
                img *= self.contrast_scaling[:numwvs, None, None]
            else:
                # broadband image
                img *= np.nanmean(self.contrast_scaling)

        return img

######################
## Static Functions ##
######################

def _nirc2_process_file(filepath):
    """
    Method to open and parse a NIRC2 file

    Args:
        filepath: the file to open

    Returns: (using z as size of 3rd dimension, z=1 for NIRC2)
        cube: 3D data cube from the file. Shape is (z,256,256)
        center: array of shape (z,2) giving each datacube slice a [xcenter,ycenter] in that order
        parang: array of z of the parallactic angle of the target (same value just repeated z times)
        wvs: array of z of the wavelength of each datacube slice. (For NIRC2, wvs = [None])
        astr_hdrs: array of z of the WCS header for each datacube slice. (For NIRC2, wcs = [None])
        filt_band: the band (Ks, Lp, Ms) used in the data (string)
        fpm_band: For NIRC2, fpm_band = [None]
        ppm_band: For NIRC2, ppm_band = [None]
        spot_fluxes: For NIRC2, array of z containing 1.0 for each image
        prihdr: primary header of the FITS file
        exthdr: For NIRC2, None
    """
    print("Reading File: {0}".format(filepath))
    hdulist = fits.open(filepath)
    try:
        #grab the data and headers
        cube = hdulist[0].data
        exthdr = None #hdulist[1].header
        prihdr = hdulist[0].header

        #get some instrument configuration from the primary header
        filt_band = prihdr['FILTER'].split('+')[0].strip()
        fpm_band = filt_band
        ppm_band = None #prihdr['APODIZER'].split('_')[1] #to determine sat spot ratios

        #for NIRC2, we only have broadband but want to keep the GPI array shape to make processing easier
        if prihdr['CURRINST'].strip() == 'NIRC2':
            wvs = [1.0]
            center = [[128,128]]#[[prihdr['PSFCENTX'], prihdr['PSFCENTY']]]

            # Flipping x-axis to enable use of GPI data rotation code without modification
            dims = cube.shape
            x, y = np.meshgrid(np.arange(dims[1], dtype=np.float32), np.arange(dims[0], dtype=np.float32))
            nx = center[0][0] - (x - center[0][0])
            minval = np.min([np.nanmin(cube), 0.0])
            flipped_cube = ndimage.map_coordinates(np.copy(cube), [y, nx], cval=minval * 5.0)

            star_flux = calc_starflux(flipped_cube, center)
            cube = flipped_cube.reshape([1, flipped_cube.shape[0], flipped_cube.shape[1]])  #maintain 3d-ness
            parang = prihdr['ROTNORTH']*np.ones(1)
            astr_hdrs = np.repeat(None, 1)
            spot_fluxes = [[1]] #not suported currently
    finally:
        hdulist.close()

    return cube, center, parang, wvs, astr_hdrs, filt_band, fpm_band, ppm_band, star_flux, spot_fluxes, prihdr, exthdr

def calc_starflux(cube, center):
    """
    Fits a 2D Gaussian to an image to calculate the peak pixel value of
    the central star. The code assumes an unobscurated PSF.

    Args:
        cube: 2D image array. Shape is (256,256)
        center: star center in image in (x,y)

    Returns:
        Amplitude: Best fit amplitude of the 2D Gaussian.
    """

    dims = cube.shape
    y, x = np.meshgrid( np.arange(dims[0]), np.arange(dims[1]) )

    # Initializing Model. Fixing the rotation and the X, Y location of the star.
    g_init = models.Gaussian2D(cube.max(), x_mean=center[0][0], y_mean=center[0][1], x_stddev=5, y_stddev=5, \
        fixed={'x_mean':True,'y_mean':True,'theta':True})

    # Initializing Levenburg-Marquart Least-Squares fitting routine.
    fit_g = fitting.LevMarLSQFitter()

    # Fitting the amplitude, x_stddev and y_stddev
    g = fit_g(g_init, y, x, cube)

    return [[g.amplitude]]

def measure_star_flux(img, star_x, star_y):
    """
    Measure star peak fluxes using a Gaussian matched filter

    Args:
        img: 2D frame with unobscured, unsaturated PSF
        star_x, star_y: coordinates of the star
    Return:
        star_f: star flux
    """

    flux, fwhm, xfit, yfit = gaussfit2d(img, star_x, star_y, refinefit=False)
    if flux == np.inf: flux == np.nan
    print(flux, fwhm, xfit, yfit)

    return flux

