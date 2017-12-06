import os
import re
import glob
import subprocess
from copy import deepcopy

import astropy.io.fits as fits
from astropy import wcs
import astropy.stats as astrostats
import numpy as np
import scipy.ndimage as ndimage
import scipy.stats
import random as rd

import pyklip.spectra_management as spec
import pyklip.fakes as fakes

import multiprocessing as mp


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
from pyklip.fakes import PSFcubefit
import pyklip.spectra_management as spec


class GPIData(Data):
    """
    A sequence of GPI Data. Each GPIData object has the following fields and functions

    Args:
        filepaths: list of filepaths to files
        skipslices: a list of datacube slices to skip (supply index numbers e.g. [0,1,2,3])
        highpass: if True, run a Gaussian high pass filter (default size is sigma=imgsize/10)
                  can also be a number specifying FWHM of box in pixel units
        meas_satspot_flux: if True, remeasure the satellite spot fluxes (would be down after hp filter)
        numthreads: Number of threads to be used. Default -1 sequential sat spot flux calc.
                    If None, numthreads = mp.cpu_count().
        PSF_cube: 3D array (nl,ny,nx) with the PSF cube to be used in the flux calculation.
        recalc_wvs: if True, uses sat spot positions and the central wavelength to recalculate wavelength solution
        recalc_centers: if True, uses a least squares fit and the satellite spots to recalculate the img centers


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
        wv_indices: Array of N indicies specifying the slice of datacube this frame comes frame (accounts of skipslices)
                You can use this to index into the header to grab info for the respective slice
        spot_flux: Array of N of average satellite spot flux for each frame
        dn_per_contrast: Flux calibration factor in units of DN/contrast (divide by image to "calibrate" flux)
                Can also be thought of as the DN of the unocculted star
        flux_units: units of output data [DN, contrast]
        prihdrs: Array of N primary GPI headers (these are written by Gemini Observatory + GPI DRP Pipeline)
        exthdrs: Array of N extension GPI headers (these are written by GPI DRP Pipeline)
        bad_sat_spots: a list of up to 4 elements indicating if a sat spot is systematically bad. Indexing is based on
            sat spot x location. Possible values are 0,1,2,3. [0,3] would mark upper left and lower right sat spots bad

    Methods:
        readdata(): reread in the data
        savedata(): save a specified data in the GPI datacube format (in the 1st extension header)
        calibrate_output(): calibrates flux of self.output
    """
    ##########################
    ###Class Initilization ###
    ##########################
    # some static variables to define the GPI instrument
    centralwave = {}  # in microns
    fpm_diam = {}  # in pixels
    flux_zeropt = {}
    spot_ratio = {} #w.r.t. central star
    lenslet_scale = 1.0 # arcseconds per pixel (pixel scale)
    ifs_rotation = 0.0  # degrees CCW from +x axis to zenith

    observatory_latitude = 0.0

    ## read in GPI configuration file and set these static variables
    package_directory = os.path.dirname(os.path.abspath(__file__))
    configfile = package_directory + "/" + "GPI.ini"
    config = ConfigParser.ConfigParser()
    try:
        config.read(configfile)
        #get pixel scale
        lenslet_scale = float(config.get("instrument", "ifs_lenslet_scale"))  # arcsecond/pix
        #get IFS rotation
        ifs_rotation = float(config.get("instrument", "ifs_rotation")) #degrees
        #get some information specific to each band
        bands = ['Y', 'J', 'H', 'K1', 'K2']
        for band in bands:
            centralwave[band] = float(config.get("instrument", "cen_wave_{0}".format(band)))
            fpm_diam[band] = float(config.get("instrument", "fpm_diam_{0}".format(band))) / lenslet_scale  # pixels
            flux_zeropt[band] = float(config.get("instrument", "zero_pt_flux_{0}".format(band)))
            spot_ratio[band] = float(config.get("instrument", "APOD_{0}".format(band)))
        observatory_latitude = float(config.get("observatory", "observatory_lat"))
    except ConfigParser.Error as e:
        print("Error reading GPI configuration file: {0}".format(e.message))
        raise e


    ####################
    ### Constructors ###
    ####################
    def __init__(self, filepaths=None, skipslices=None, highpass=False, meas_satspot_flux=False, numthreads=-1,
                 PSF_cube=None, recalc_wvs=True, recalc_centers=True, bad_sat_spots=None, quiet=False):
        """
        Initialization code for GPIData

        Note:
            Argument information is in the GPIData class definition docstring
        """
        super(GPIData, self).__init__()
        self._output = None
        self.bad_sat_spots = bad_sat_spots
        if filepaths is None:
            print("Creating a blank GPI data instance with all fields set to None. Did you want to do this?")
            self._input = None
            self._centers = None
            self._filenums = None
            self._filenames = None
            self._PAs = None
            self._wvs = None
            self._wcs = None
            self._IWA = None
            self.spot_flux = None
            self.dn_per_contrast = None
            self.prihdrs = None
            self.exthdrs = None
            self.flux_units = None
            self.wv_indices = None
        else:
            self.readdata(filepaths, skipslices=skipslices, highpass=highpass,meas_satspot_flux=meas_satspot_flux,
                          numthreads=numthreads,PSF_cube=PSF_cube, recalc_wvs=recalc_wvs, recalc_centers=recalc_centers,
                          bad_sat_spots=bad_sat_spots, quiet=quiet)

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

    def readdata(self, filepaths, skipslices=None, highpass=False, meas_satspot_flux=False,numthreads = -1,
                 PSF_cube=None, recalc_wvs=True, recalc_centers=True, bad_sat_spots=None, quiet=False):
        """
        Method to open and read a list of GPI data

        Args:
            filespaths: a list of filepaths
            skipslices: a list of wavelenegth slices to skip for each datacube (supply index numbers e.g. [0,1,2,3])
            highpass: if True, run a Gaussian high pass filter (default size is sigma=imgsize/10)
                      can also be a number specifying FWHM of box in pixel units
            meas_satspot_flux: if True, remeasure the satellite spot fluxes (would be done after hp filter)
            numthreads: Number of threads to be used. Default -1 sequential sat spot flux calc.
                        If None, numthreads = mp.cpu_count().
            PSF_cube: 3D array (nl,ny,nx) with the PSF cube to be used in the flux calculation.
            recalc_wvs: if True, uses sat spot positions and the central wavelength to recalculate wavelength solution
            recalc_centers: if True, uses a least squares fit and the satellite spots to recalculate the img centers
            bad_sat_spots: a list of up to 4 elements indicating if a sat spot is systematically bad.

        Returns:
            Technically none. It saves things to fields of the GPIData object. See object doc string
        """
        # check to see if user just inputted a single filename string
        if isinstance(filepaths, str):
            filepaths = [filepaths]

        # check that the list of files actually contains something
        if len(filepaths) == 0:
            raise ValueError("An empty filelist was passed in to GPIData")

        # check bad sat spots to make sure they are reasonable
        if bad_sat_spots is not None:
            for bad_sat_index in bad_sat_spots:
                if not 0 <= bad_sat_index < 4:
                    raise ValueError("Sat spots can only be labelled 0 to 3")

        #make some lists for quick appending
        data = []
        filenums = []
        filenames = []
        rot_angles = []
        wvs = []
        wv_indices = []
        centers = []
        wcs_hdrs = []
        spot_fluxes = []
        inttimes = []
        prihdrs = []
        exthdrs = []

        if PSF_cube is not None:
            if isinstance(PSF_cube, np.ndarray):
                PSF_cube_arr = PSF_cube
            else: # Read PSF cube from memory if a string given
                if os.path.isabs(PSF_cube):
                    PSF_cube_path = os.path.abspath(glob.glob(os.path.join(PSF_cube))[0])
                else:
                    base_path = os.path.dirname(filepaths[0])
                    PSF_cube_path = os.path.abspath(glob.glob(os.path.join(base_path,PSF_cube))[0])
                hdulist = fits.open(PSF_cube_path)
                PSF_cube_arr = hdulist[1].data
            numwv,ny_psf,nx_psf =  PSF_cube_arr.shape
            x_psf_grid, y_psf_grid = np.meshgrid(np.arange(nx_psf * 1.)-nx_psf//2,np.arange(ny_psf* 1.)-ny_psf//2)
            psfs_func_list = []
            from scipy import interpolate
            for wv_index in range(numwv):
                model_psf = PSF_cube_arr[wv_index, :, :]
                psfs_func_list.append(interpolate.LSQBivariateSpline(x_psf_grid.ravel(),y_psf_grid.ravel(),model_psf.ravel(),x_psf_grid[0,0:nx_psf-1]+0.5,y_psf_grid[0:ny_psf-1,0]+0.5))
        else:
            psfs_func_list = None

        #extract data from each file
        for index, filepath in enumerate(filepaths):
            cube, center, pa, wv, cube_wv_indices, astr_hdrs, filt_band, fpm_band, ppm_band, spot_flux, inttime, prihdr, exthdr = \
                _gpi_process_file(filepath, skipslices=skipslices, highpass=highpass,
                                  meas_satspot_flux=meas_satspot_flux, numthreads=numthreads,
                                  psfs_func_list=psfs_func_list, bad_sat_spots=bad_sat_spots, quiet=quiet)

            # import matplotlib.pyplot as plt
            # print(filepath)
            # plt.plot(spot_flux,'r')
            # plt.show()

            data.append(cube)
            centers.append(center)
            spot_fluxes.append(spot_flux)
            rot_angles.append(pa)
            wvs.append(wv)
            wv_indices.append(cube_wv_indices)
            filenums.append(np.ones(pa.shape[0], dtype=int) * index)
            wcs_hdrs.append(astr_hdrs)
            inttimes.append(inttime)
            prihdrs.append(prihdr)
            exthdrs.append(exthdr)

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
        rot_angles = -(np.array(rot_angles).reshape([dims[0] * dims[1]])) + (90 - self.ifs_rotation)  # want North Up
        wvs = np.array(wvs).reshape([dims[0] * dims[1]])
        wv_indices = np.array(wv_indices).reshape([dims[0] * dims[1]])
        wcs_hdrs = np.array(wcs_hdrs).reshape([dims[0] * dims[1]])
        centers = np.array(centers).reshape([dims[0] * dims[1], 2])
        spot_fluxes = np.array(spot_fluxes).reshape([dims[0] * dims[1]])
        inttimes = np.array(inttimes).reshape([dims[0] * dims[1]])

        # if there is more than 1 integration time, normalize all data to the first integration time
        if np.size(np.unique(inttimes)) > 1:
            inttime0 = inttime[0]
            # normalize integration times
            data = data * inttime0/inttimes[:, None, None]
            spot_fluxes *= inttime0/inttimes

        # only do the wavelength solution and center recalculation if it isn't broadband imaging
        if np.size(np.unique(wvs)) > 1:
            # recalculate wavelegnths from satellite spots
            if recalc_wvs:
                wvs = rescale_wvs(exthdrs, wvs, skipslices=skipslices, bad_sat_spots=bad_sat_spots)

            # recaclulate centers from satellite spots and new wavelegnth solution
            if recalc_centers:
                wvs_bycube = wvs.reshape([dims[0], dims[1]])
                centers_bycube = centers.reshape([dims[0], dims[1], 2])
                for i, cubewvs in enumerate(wvs_bycube):
                    try:
                        centers_bycube[i] = calc_center(prihdrs[i], exthdrs[i], cubewvs, skipslices=skipslices,
                                                        bad_sat_spots=bad_sat_spots)
                    except KeyError:
                        print("Unable to recenter the data using a least squraes fit due to not enough header info for file "
                              "{0}".format(filenames[i*dims[1]]))

        # contrast_scaling = np.zeros(dims[1])
        # spot_fluxes_wvs = np.reshape(spot_fluxes, (dims[0], dims[1]))
        # for wv_i in range(dims[1]):
        #     spot_fluxes_wv = spot_fluxes_wvs[:,wv_i]
        #     spot_fluxes_wv_filt = astrostats.sigma_clip(spot_fluxes_wv, sig=5, iters=2)
        #     contrast_scaling[i] = GPIData.spot_ratio[ppm_band]/np.nanmean(spot_fluxes_wv_filt)

        #set these as the fields for the GPIData object
        self._input = data
        self._centers = centers
        self._filenums = filenums
        self._filenames = filenames
        self._PAs = rot_angles
        self._wvs = wvs
        self._wcs = wcs_hdrs
        self._IWA = GPIData.fpm_diam[fpm_band]/2.0
        self.wv_indices = wv_indices
        self.spot_flux = spot_fluxes
        self.flux_units = "DN"
        self.dn_per_contrast = np.tile(np.nanmean(spot_fluxes.reshape(dims[0], dims[1]), axis=0), dims[0]) / GPIData.spot_ratio[ppm_band]
        # self.contrast_scaling = np.tile(contrast_scaling, dims[0])
        self.prihdrs = prihdrs
        self.exthdrs = exthdrs

        # Required for automatically querying Simbad for the spectral type of the star.
        self.object_name = self.prihdrs[0]["OBJECT"]


    def savedata(self, filepath, data, klipparams = None, filetype = None, zaxis = None, more_keywords=None,
                 center=None, astr_hdr=None, fakePlparams = None,user_prihdr = None, user_exthdr = None,
                 extra_exthdr_keywords = None, extra_prihdr_keywords = None,pyklip_output = True):
        """
        Save data in a GPI-like fashion. Aka, data and header are in the first extension header

        Args:
            filepath: path to file to output
            data: 2D or 3D data to save
            klipparams: a string of klip parameters
            filetype: filetype of the object (e.g. "KL Mode Cube", "PSF Subtracted Spectral Cube")
            zaxis: a list of values for the zaxis of the datacub (for KL mode cubes currently)
            more_keywords (dictionary) : a dictionary {key: value, key:value} of header keywords and values which will
                                         written into the primary header
            astr_hdr: wcs astrometry header
            center: center of the image to be saved in the header as the keywords PSFCENTX and PSFCENTY in pixels.
                The first pixel has coordinates (0,0)
            fakePlparams: fake planet params
            user_prihdr: User defined primary headers to be used instead
            user_exthdr: User defined extension headers to be used instead
            extra_exthdr_keywords: Fits keywords to be added to the extension header before saving the file
            extra_prihdr_keywords: Fits keywords to be added to the primary header before saving the file
            pyklip_output: (default True) If True, indicates that the attributes self.output_wcs and self.output_centers
                            have been defined.

        """
        hdulist = fits.HDUList()
        if user_prihdr is None:
            hdulist.append(fits.PrimaryHDU(header=self.prihdrs[0]))
        else:
            hdulist.append(fits.PrimaryHDU(header=user_prihdr))
        if user_exthdr is None:
            hdulist.append(fits.ImageHDU(header=self.exthdrs[0], data=data, name="Sci"))
        else:
            hdulist.append(fits.ImageHDU(header=user_exthdr, data=data, name="Sci"))

        # save all the files we used in the reduction
        # we'll assume you used all the input files
        # remove duplicates from list
        filenames = np.unique(self.filenames)
        nfiles = np.size(filenames)
        # The following paragraph is only valid when reading raw GPI cube.
        try:
            hdulist[0].header["DRPNFILE"] = (nfiles, "Num raw files used in pyKLIP")
            for i, thispath in enumerate(filenames):
                thispath = thispath.replace("\\", '/')
                splited = thispath.split("/")
                fname = splited[-1]
                matches = re.search('S20[0-9]{6}[SE][0-9]{4}(_fixed)?', fname)
                filename = matches.group(0)
                hdulist[0].header["FILE_{0}".format(i)] = filename + '.fits'
        except:
            pass

        # write out psf subtraction parameters
        # get pyKLIP revision number
        pykliproot = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        # the universal_newline argument is just so python3 returns a string instead of bytes
        # this will probably come to bite me later
        try:
            pyklipver = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=pykliproot, universal_newlines=True).strip()
        except:
            pyklipver = "unknown"
        hdulist[0].header['PSFSUB'] = ("pyKLIP", "PSF Subtraction Algo")
        if user_prihdr is None:
            hdulist[0].header.add_history("Reduced with pyKLIP using commit {0}".format(pyklipver))
        if self.creator is None:
            hdulist[0].header['CREATOR'] = "pyKLIP-{0}".format(pyklipver)
        else:
            hdulist[0].header['CREATOR'] = self.creator
            hdulist[0].header.add_history("Reduced by {0}".format(self.creator))

        # store commit number for pyklip
        hdulist[0].header['pyklipv'] = (pyklipver, "pyKLIP version that was used")

        if klipparams is not None:
            hdulist[0].header['PSFPARAM'] = (klipparams, "KLIP parameters")
            hdulist[0].header.add_history("pyKLIP reduction with parameters {0}".format(klipparams))

        if fakePlparams is not None:
            hdulist[0].header['FAKPLPAR'] = (fakePlparams, "Fake planet parameters")
            hdulist[0].header.add_history("pyKLIP reduction with fake planet injection parameters {0}".format(fakePlparams))
        # store file type
        if filetype is not None:
            hdulist[0].header['FILETYPE'] = (filetype, "GPI File type")

        #write flux units/conversion
        hdulist[1].header['FUNIT'] = (self.flux_units, "Flux units of data")
        if self.flux_units.upper() == 'CONTRAST':
            if "spec" in filetype.lower():
                # individual contrast scalings for spectral cube
                for wv_i in range(data.shape[0]):
                    hdulist[1].header['DN2CON{0}'.format(wv_i)] = (self.dn_per_contrast[wv_i], "DN/Contrast for slice {0}".format(wv_i))
                hdulist[0].header.add_history("Converted to contrast units using CON2DN scaling for each wv slice")
            else:
                # broadband cube so only have one scaling
                broadband_contrast_scaling = np.nanmean(self.dn_per_contrast)
                hdulist[1].header['DN2CON'] = (broadband_contrast_scaling, "Broadband DN/Contrast")
                hdulist[0].header.add_history("Converted to contrast units using {0} DN/Contrast".format(broadband_contrast_scaling))

        # store extra keywords in header
        if more_keywords is not None:
            for hdr_key in more_keywords:
                hdulist[0].header[hdr_key] = more_keywords[hdr_key]

        # JB's code to store keywords
        if extra_prihdr_keywords is not None:
            for name,value in extra_prihdr_keywords:
                hdulist[0].header[name] = value
        if extra_exthdr_keywords is not None:
            for name,value in extra_exthdr_keywords:
                hdulist[1].header[name] = value

        # write z axis units if necessary
        if zaxis is not None:
            #Writing a KL mode Cube
            if "KL Mode" in filetype:
                hdulist[1].header['CTYPE3'] = 'KLMODES'
                #write them individually
                for i, klmode in enumerate(zaxis):
                    hdulist[1].header['KLMODE{0}'.format(i)] = (klmode, "KL Mode of slice {0}".format(i))
            elif "spec" in filetype.lower():
                hdulist[1].header['CTYPE3'] = 'WAVE'
            else:
                hdulist[1].header['CTYPE3'] = 'NONE'

        if np.ndim(data) == 2:
            if 'CTYPE3' in  hdulist[1].header.keys():
                hdulist[1].header['CTYPE3'] = 'NONE'

        if user_exthdr is None:
            #use the dataset astr hdr if none was passed in
            if astr_hdr is None:
                if not pyklip_output:
                    astr_hdr = self.wcs[0].deepcopy()
                else:
                    astr_hdr = self.output_wcs[0]
            if astr_hdr is not None:
                #update astro header
                #I don't have a better way doing this so we'll just inject all the values by hand
                astroheader = astr_hdr.to_header()
                exthdr = hdulist[1].header
                exthdr['PC1_1'] = astroheader['PC1_1']
                exthdr['PC2_2'] = astroheader['PC2_2']
                try:
                    exthdr['PC1_2'] = astroheader['PC1_2']
                    exthdr['PC2_1'] = astroheader['PC2_1']
                except KeyError:
                    exthdr['PC1_2'] = 0.0
                    exthdr['PC2_1'] = 0.0
                #remove CD values as those are confusing
                try:
                    exthdr.remove('CD1_1')
                    exthdr.remove('CD1_2')
                    exthdr.remove('CD2_1')
                    exthdr.remove('CD2_2')
                except:
                    pass # nothing to do if they were removed already
                exthdr['CDELT1'] = 1
                exthdr['CDELT2'] = 1

            #use the dataset center if none was passed in
            if center is None:
                if not pyklip_output:
                    center = self.centers[0]
                else:
                    center = self.output_centers[0]
            if center is not None:
                hdulist[1].header.update({'PSFCENTX':center[0],'PSFCENTY':center[1]})
                hdulist[1].header.update({'CRPIX1':center[0],'CRPIX2':center[1]})
                hdulist[0].header.add_history("Image recentered to {0}".format(str(center)))

        try:
            hdulist.writeto(filepath, overwrite=True)
        except TypeError:
            hdulist.writeto(filepath, clobber=True)
        hdulist.close()

    def calibrate_output(self, img, spectral=False, units="contrast"):
        """
        Calibrates the flux of an output image. Can either be a broadband image or a spectral cube depending
        on if the spectral flag is set.

        Assumes the broadband flux calibration is just multiplication by a single scalar number whereas spectral
        datacubes may have a separate calibration value for each wavelength

        Args:
            img: unclaibrated image.
                 If spectral is not set, this can either be a 2-D or 3-D broadband image
                 where the last two dimensions are [y,x]
                 If specetral is True, this is a 3-D spectral cube with shape [wv,y,x]
            spectral: if True, this is a spectral datacube. Otherwise, it is a broadband image.
            units: currently only support "contrast" w.r.t central star

        Returns:
            img: calibrated image of the same shape (this is the same object as the input!!!)
        """
        if units == "contrast":
            if spectral:
                # spectral cube, each slice needs it's own calibration
                numwvs = img.shape[0]
                img /= self.dn_per_contrast[:numwvs, None, None]
            else:
                # broadband image
                img /= np.nanmean(self.dn_per_contrast)
            self.flux_units = "contrast"

        return img


    def generate_psfs(self, boxrad=7):
        """
        Generates PSF for each frame of input data. Only works on spectral mode data.

        Args:
            boxrad: the halflength of the size of the extracted PSF (in pixels)

        Returns:
            saves PSFs to self.psfs as an array of size(N,psfy,psfx) where psfy=psfx=2*boxrad + 1
        """
        self.psfs = []

        for i,frame in enumerate(self.input):
            # figure out which header and which wavelength slice
            numwaves = np.size(np.unique(self.wvs))
            # hdrindex = int(i//numwaves)
            # slice = i % numwaves

            hdrindex = self.filenums[i]
            slice = self.wv_indices[i]

            # now grab the values from them by parsing the header
            hdr = self.exthdrs[hdrindex]
            spot0 = hdr['SATS{wave}_0'.format(wave=slice)].split()
            spot1 = hdr['SATS{wave}_1'.format(wave=slice)].split()
            spot2 = hdr['SATS{wave}_2'.format(wave=slice)].split()
            spot3 = hdr['SATS{wave}_3'.format(wave=slice)].split()

            # put all the sat spot info together
            spots = []
            for j, spot in enumerate([spot0, spot1, spot2, spot3]):
                # ignore bad sat spots
                if self.bad_sat_spots is not None:
                    if j in self.bad_sat_spots:
                        continue
                spots.append([float(spot[0]), float(spot[1])])

            #now make a psf
            spotpsf = generate_psf(frame, spots, boxrad=boxrad)
            self.psfs.append(spotpsf)

        self.psfs = np.array(self.psfs)

        # collapse in time dimension
        numwvs = np.size(np.unique(self.wvs))
        self.psfs = np.reshape(self.psfs, (self.psfs.shape[0]//numwvs, numwvs, self.psfs.shape[1], self.psfs.shape[2]))
        self.psfs = np.mean(self.psfs, axis=0)

    def generate_psf_cube(self, boxw=20, threshold=0.01, tapersize=0, zero_neg=False, same_wv_only = True):
        """
        Generates an average PSF from all frames of input data. Only works on spectral mode data.
        Overall cube is normalized to have the average sat spot spectrum in DN units.
        The spectrum is built by combining all the estimated sat spot fluxes.
        It can take a while as this function is not parallelized...

        The center of the PSF is exactly on the central pixel of the PSF.
        The center pixel index is always (nx/2,nx/2) assuming integer division.

        The output PSF cube shape doesn't depend on the underlying sat spot flux calculation.
        The sat spot fluxes are only used to set the spectrum of the PSF at the very end.


        //!\\ CAUTION 1: I think same_wv_only = False has a bug even in the rescaling of the coordinates
        //!\\ CAUTION 2: Currently hard coded assuming 37 spectral channels!!!
                        This function is not compatible with skipslices.

        Args:
            boxw: the width the extracted PSF (in pixels). Should be bigger than 20 because there is an interpolation
                of the background by a plane which is then subtracted to remove linear biases.
            threshold: fractional pixel value of max pixel value below which everything gets set to 0's
            tapersize: if > 0, apply a hann window on the edges with size equal to this argument
            zero_neg: if True, set all negative values to zero instead
            same_wv_only: If true (default), it only combines sat spot from the same wavelength.
                        Otherwise it rescales them to each wavelengths.
                        CAUTION: I think same_wv_only = False has a bug even in the rescaling of the coordinates

        Returns:
            A cube of shape 37*boxw*boxw. Each slice [k,:,:] is the PSF for a given wavelength.
        """

        n_frames,ny,nx = self.input.shape
        unique_wvs = np.unique(self.wvs)
        numwaves = np.size(np.unique(self.wvs))

        # Array containing all the individual measured sat spots form the dataset
        # - 0th dim: The wavelength of the final PSF cube
        # - 1st dim: spatial x-axis
        # - 2nd dim: spatial y-axis
        # - 3rd dim: All the slices in dataset
        # - 4th dim: The 4 spots per slice
        psfs = np.zeros((numwaves,boxw,boxw,n_frames,4)) + np.nan

        # Loop over the wavelength of the final PSF cube
        for lambda_ref_id, lambda_ref in enumerate(unique_wvs):
            # Loop over all the all slices (cubes and wavelengths). Note that each slice has 4 sat spots.
            if same_wv_only:
                frames_iter = [(k,self.input[k,:,:]) for k in range(lambda_ref_id,n_frames,37)]
            else:
                frames_iter = enumerate(self.input)

            for i,frame in frames_iter:
                #figure out which header and which wavelength slice
                hdrindex = int(i)//int(numwaves)
                slice = i % numwaves
                lambda_curr = unique_wvs[slice]
                #now grab the values from them by parsing the header
                hdr = self.exthdrs[hdrindex]
                # Each 'SATS{wave}_i' is a tuple and corresponds to the (x,y) coordinates of the given sat spot.
                spot0 = hdr['SATS{wave}_0'.format(wave=slice)].split()
                spot1 = hdr['SATS{wave}_1'.format(wave=slice)].split()
                spot2 = hdr['SATS{wave}_2'.format(wave=slice)].split()
                spot3 = hdr['SATS{wave}_3'.format(wave=slice)].split()

                #put all the sat spot coordinates together
                spots = []
                for j, spot in enumerate([spot0, spot1, spot2, spot3]):
                    # ignore bad sat spots
                    if self.bad_sat_spots is not None:
                        if j in self.bad_sat_spots:
                            continue
                    spots.append([float(spot[0]), float(spot[1])])

                #mask nans
                cleaned = np.copy(frame)
                cleaned[np.where(np.isnan(cleaned))] = 0

                # Loop over the 4 spots in the current slice
                for loc_id, loc in enumerate(spots):
                    #grab current satellite spot positions
                    spotx = loc[0]
                    spoty = loc[1]
                    # Get the closest pixel
                    xarr_spot = int(np.round(spotx))
                    yarr_spot = int(np.round(spoty))
                    # Extract a stamp around the sat spot

                    stamp = cleaned[(yarr_spot-int(np.floor(boxw/2.0))):(yarr_spot+int(np.ceil(boxw/2.0))),\
                                    (xarr_spot-int(np.floor(boxw/2.0))):(xarr_spot+int(np.ceil(boxw/2.0)))]
                    # Define coordinates grids for the stamp
                    stamp_x, stamp_y = np.meshgrid(np.arange(boxw, dtype=np.float32), np.arange(boxw, dtype=np.float32))
                    # Calculate the shift of the sat spot centroid relative to the closest pixel.
                    dx = spotx-xarr_spot
                    dy = spoty-yarr_spot


                    # The goal of the following section is to remove the local background (or sky) around the sat spot.
                    # The plane is defined by 3 constants (a,b,c) such that z = a*x+b*y+c
                    # In order to do so we fit a 2D plane to the stamp after having masked the sat spot (centered disk)
                    stamp_r = np.sqrt((stamp_x-dx-boxw//2)**2+(stamp_y-dy-boxw//2)**2)
                    stamp_masked = copy(stamp)
                    stamp_x_masked = stamp_x-dx
                    stamp_y_masked = stamp_y-dy
                    stamp_center = np.where(stamp_r<7)
                    stamp_masked[stamp_center] = np.nan
                    stamp_x_masked[stamp_center] = np.nan
                    stamp_y_masked[stamp_center] = np.nan
                    background_med =  np.nanmedian(stamp_masked)
                    stamp_masked = stamp_masked - background_med
                    #Solve 2d linear fit to remove background
                    xx = np.nansum(stamp_x_masked**2)
                    yy = np.nansum(stamp_y_masked**2)
                    xy = np.nansum(stamp_y_masked*stamp_x_masked)
                    xz = np.nansum(stamp_masked*stamp_x_masked)
                    yz = np.nansum(stamp_y_masked*stamp_masked)
                    #Cramer's rule
                    a = (xz*yy-yz*xy)/(xx*yy-xy*xy)
                    b = (xx*yz-xy*xz)/(xx*yy-xy*xy)
                    stamp = stamp - (a*(stamp_x-dx)+b*(stamp_y-dy) + background_med)

                    if not same_wv_only:
                        # The next section rescale the grid to take into account wavelength widening
                        # For example if lambda_ref < lambda_curr the grid values need to increase because the current stamp
                        #  is bigger than the reference.
                        # The next 2 lines convert cartesion coordinates to cylindrical.
                        stamp_r = np.sqrt((stamp_x-dx-boxw//2)**2+(stamp_y-dy-boxw//2)**2)
                        stamp_th = np.arctan2(stamp_y-dy-boxw//2,stamp_x-dx-boxw//2)
                        #stamp_th = np.arctan2(stamp_x-dx-boxw/2,stamp_y-dy-boxw/2)
                        # Rescale radius grid
                        stamp_r /= lambda_ref/lambda_curr
                        # Converting cylindrical back to cartesian.
                        stamp_x = stamp_r*np.cos(stamp_th)+boxw//2 + dx
                        stamp_y = stamp_r*np.sin(stamp_th)+boxw//2 + dy
                        # At this point stamp_x/y is centered on the center pixel but properly scaled wrt wavelength.

                    # Because map_coordinates wants the coordinate of the new grid relative to the old we need to shift
                    # it in the opposite direction as before (+dx/dy instead of -dx/dy)
                    stamp = ndimage.map_coordinates(stamp, [stamp_y+dy, stamp_x+dx])
                    #stamp = ndimage.map_coordinates(stamp, [stamp_y+dx, stamp_x+dy])

                    # apply a hann window on the edges with size equal to this argument
                    if tapersize > 0:
                        tapery, taperx = np.indices(stamp.shape)
                        taperr = np.sqrt((taperx-boxw//2)**2 + (tapery-boxw//2)**2)
                        stamp[np.where(taperr > boxw//2)] = 0
                        hann_window = 0.5  - 0.5 * np.cos(np.pi * (boxw//2 - taperr) / tapersize)
                        taper_region = np.where(taperr > boxw//2 - tapersize)
                        stamp[taper_region] *= hann_window[taper_region]

                    # Set to zero negative values if requested
                    if zero_neg:
                        stamp[np.where(stamp < 0)] = 0

                    # Store the rescaled PSF in the big array
                    psfs[lambda_ref_id,:,:,i,loc_id] = stamp

        # Collapse big array over the dimensions corresponding to:
        # - 3rd dim: All the slices in dataset
        # - 4th dim: The 4 spots per slice
        PSF_cube = np.nanmean(psfs,axis=(3,4))

        #Build the average spectrum of the sat spots
        # Number of cubes in dataset
        N_cubes = int(self.input.shape[0])//int(numwaves)
        all_sat_spot_spec = np.zeros((37,N_cubes))
        for k in range(N_cubes):
            all_sat_spot_spec[:,k] = self.spot_flux[37*k:37*(k+1)]
        sat_spot_spec = np.nanmean(all_sat_spot_spec,axis=1)

        # Include sat spot spectrum PSF_cube and apply threshold.
        PSF_cube /= np.sqrt(np.nansum(PSF_cube**2))
        for l in range(numwaves):
            PSF_cube[l,:,:] *= sat_spot_spec[l]/np.nanmax(PSF_cube[l,:,:])
            PSF_cube[l,:,:][np.where(abs(PSF_cube[l,:,:])/np.nanmax(abs(PSF_cube[l,:,:]))< threshold)] = 0.0

        self.psfs = PSF_cube


    def get_radial_psf(self,save = None):
        """
        Return a pure radial PSF by averaging the original psf. The new PSF is invariant by rotation.
        A call to generate_psf_cube() is required prior to calling this function.
        The center pixel index is always (nx/2,nx/2) assuming integer division.

        Args:
            save: Optionally automatically save the radial psf cube as a fits file with filename:
                    save+"-original_radial_PSF_cube.fits"

        Returns:
            rad_psf_cube: a (37,nx,nx) cube with the radial psf.

        """
        if np.size(np.shape(self.psfs)) == 3 and np.shape(self.psfs)[0] == 37:
            nl,ny,nx = self.psfs.shape
            # We should have nx = ny

            sat_spot_spec = np.nanmax(self.psfs,axis=(1,2))

            k_hd=4 # should be even
            nx_hd = k_hd*(nx-1) + 1
            hd_psf = np.zeros((nl,nx_hd,nx_hd))

            rad_psf_cube = np.zeros((nl,nx,nx))
            #current_slice = np.zeros((nx,nx))

            stamp_x, stamp_y = np.meshgrid(np.arange(nx, dtype=np.float32), np.arange(nx, dtype=np.float32))
            stamp_r = np.sqrt((stamp_x - nx//2)**2+(stamp_y - nx//2)**2)
            stamp_x_hd, stamp_y_hd = np.meshgrid(np.arange(nx_hd, dtype=np.float32)/(nx_hd-1)*(nx-1), np.arange(nx_hd, dtype=np.float32)/(nx_hd-1)*(nx-1))
            for l in range(nl):
                hd_psf[l,:,:] = ndimage.map_coordinates(self.psfs[l,:,:], [stamp_y_hd, stamp_x_hd])
                #hd_psf[l,nx/2*k_hd,nx/2*k_hd] = 0. # center
            stamp_r_hd = np.sqrt((stamp_x_hd-stamp_x_hd[nx//2*k_hd,nx//2*k_hd])**2+(stamp_y_hd-stamp_y_hd[nx//2*k_hd,nx//2*k_hd])**2)

            dr = 1.0/k_hd
            Dr = 2.0/k_hd
            r_samp = np.arange(0,np.max(stamp_r_hd)+dr,dr)

            radial_val = np.zeros((nl,np.size(r_samp)))

            for r_id, r_it in enumerate(r_samp):
                selec_pix = np.where( ((r_it-Dr/2.0) < stamp_r_hd) * (stamp_r_hd < (r_it+Dr/2.0)) )
                selec_y, selec_x = selec_pix
                radial_val[:,r_id] = np.nanmean(hd_psf[:,selec_y, selec_x],1)

            for l_id in np.arange(nl):
                f = interp1d(r_samp, radial_val[l_id,:], kind='cubic',bounds_error=False, fill_value=np.nan)
                rad_psf_cube[l_id,:,:] = f(stamp_r.reshape(nx*nx)).reshape(nx,nx)
                rad_psf_cube[l_id,:,:] *= sat_spot_spec[l_id]/np.nanmax(rad_psf_cube[l_id,:,:])

                if 0:
                    import matplotlib.pyplot as plt
                    print(rad_psf_cube[l_id,0,0])
                    plt.figure(1)
                    plt.imshow(rad_psf_cube[l_id,:,:],interpolation = 'nearest')
                    plt.figure(2)
                    plt.plot(np.nanmax(self.psfs,axis=(1,2)))
                    plt.show()



            if save is not None:
                self.savedata(save+"-original_radial_PSF_cube.fits", rad_psf_cube)

            return rad_psf_cube

        else:
            print("Wrong size of the PSFs stored in gpi dataset structure when calling get_radial_psf. Return 0")
            return 0
        

    def spectral_collapse(self, collapse_channels=1, align_frames=True, numthreads=None):
        """
        GPI wrapper of spectral_collapse(). Adds GPI values to collapse
        
        Collapses the dataset spectrally, bining the data into the desired number of output wavelengths. 
        This bins each cube individually; it does not bin the data tempoarally. 
        If number of wavelengths / output channels is not a whole number, some output channels will have more frames
        that went into the collapse

        Args:
            collapse_channels (int): number of output channels to evenly-ish collapse the dataset into. Default is 1 (broadband)
            align_frames (bool): if True, aligns each channel before collapse so that they are centered properly
            numthreads (bool,int): number of threads to parallelize align and scale. If None, use default which is all of them
        """
        gpi_params = ["spot_flux", "dn_per_contrast"]

        super(GPIData, self).spectral_collapse(collapse_channels=collapse_channels, align_frames=align_frames, numthreads=numthreads,
                                                additional_params=gpi_params)

######################
## Static Functions ##
######################

def _gpi_process_file(filepath, skipslices=None, highpass=False, meas_satspot_flux=False, numthreads=-1,
                      psfs_func_list=None, bad_sat_spots=None, quiet=False):
    """
    Method to open and parse a GPI file

    Args:
        filepath: the file to open
        skipslices: a list of datacube slices to skip (supply index numbers e.g. [0,1,2,3])
        highpass: if True, run a Gaussian high pass filter (default size is sigma=imgsize/10)
                  can also be a number specifying FWHM of box in pixel units
        meas_satspot_flux: if True, measure sat spot fluxes. Will be down after high pass filter
        numthreads: Number of threads to be used. Default -1 sequential sat spot flux calc.
                    If None, numthreads = mp.cpu_count().
        psfs_func_list: List of spline fit function for the PSF_cube.
        bad_sat_spots: a list of which 4 sat spots are systematically bad

    Returns: (using z as size of 3rd dimension, z=37 for spec, z=1 for pol (collapsed to total intensity))
        cube: 3D data cube from the file. Shape is (z,281,281)
        center: array of shape (z,2) giving each datacube slice a [xcenter,ycenter] in that order
        parang: array of z of the parallactic angle of the target (same value just repeated z times)
        wvs: array of z of the wavelength of each datacube slice. (For pol mode, wvs = [None])
        astr_hdrs: array of z of the WCS header for each datacube slice
        filt_band: the band (Y, J, H, K1, K2) used in the IFS Filter (string)
        fpm_band: which coronagrpah was used (string)
        ppm_band: which apodizer was used (string)
        spot_fluxes: array of z containing average satellite spot fluxes for each image
        inttime: array of z of total integration time (accounting for co-adds by multipling data and sat spot fluxes by number of co-adds)
        prihdr: primary header of the FITS file
        exthdr: 1st extention header of the FITS file
    """
    if not quiet:
        print("Reading File: {0}".format(filepath))
    hdulist = fits.open(filepath)
    try:

        #grab the data and headers
        cube = hdulist[1].data
        exthdr = hdulist[1].header
        prihdr = hdulist[0].header

        #get some instrument configuration from the primary header
        filt_band = prihdr['IFSFILT'].split('_')[1]
        fpm_band = prihdr['OCCULTER'].split('_')[1]
        ppm_band = prihdr['APODIZER'].split('_')[1] #to determine sat spot ratios

        #grab the astro header
        w = wcs.WCS(header=exthdr, naxis=[1,2])
        #turns out WCS data can be wrong. Let's recalculate it using avparang
        parang = exthdr['AVPARANG']
        vert_angle = -(360-parang) + GPIData.ifs_rotation - 90
        vert_angle = np.radians(vert_angle)
        pc = np.array([[np.cos(vert_angle), np.sin(vert_angle)],[-np.sin(vert_angle), np.cos(vert_angle)]])
        cdmatrix = pc * GPIData.lenslet_scale /3600.
        w.wcs.cd[0,0] = cdmatrix[0,0]
        w.wcs.cd[0,1] = cdmatrix[0,1]
        w.wcs.cd[1,0] = cdmatrix[1,0]
        w.wcs.cd[1,1] = cdmatrix[1,1]

        # get number of co-adds
        coadds = exthdr['COADDS0']

        #for spectral mode we need to treat each wavelegnth slice separately (unprocessed data)
        if exthdr['CTYPE3'].strip() == 'WAVE':
            channels = exthdr['NAXIS3']
            wvs = exthdr['CRVAL3'] + exthdr['CD3_3'] * np.arange(channels) #get wavelength solution
            wv_indices = np.arange(channels, dtype=int)
            center = []
            spot_fluxes = []
            spots_xloc = []
            spots_yloc = []
            # Try to retrieve the spot fluxes from DN2CON# in the case where we are reading an already processed cube.
            if 'DN2CON0' in exthdr.keys():
                for i in range(channels):
                    spot_fluxes.append(float(exthdr['DN2CON{0}'.format(i)])*GPIData.spot_ratio[ppm_band])
            else:
                for i in range(channels):
                    #grab sat spot fluxes if they're there
                    try:
                        spot0flux = float(exthdr['SATF{wave}_0'.format(wave=i)])
                        spot1flux = float(exthdr['SATF{wave}_1'.format(wave=i)])
                        spot2flux = float(exthdr['SATF{wave}_2'.format(wave=i)])
                        spot3flux = float(exthdr['SATF{wave}_3'.format(wave=i)])
                    except KeyError:
                        spot0flux = 1
                        spot1flux = 1
                        spot2flux = 1
                        spot3flux = 1
                    this_frame_spot_fluxes = [spot0flux, spot1flux, spot2flux, spot3flux]
                    # delete bad data
                    if bad_sat_spots is not None:
                        bad_sat_spots.sort(reverse=True)
                        # delete fom highest index first to not mess up indexing
                        for bad_sat_index in bad_sat_spots:
                            del(this_frame_spot_fluxes[bad_sat_index])
                    spot_fluxes.append(np.nanmean(this_frame_spot_fluxes))
            #calculate centers from satellite spots
            for i in range(channels):
                #grab satellite spot positions
                spot0 = exthdr['SATS{wave}_0'.format(wave=i)].split()
                spot1 = exthdr['SATS{wave}_1'.format(wave=i)].split()
                spot2 = exthdr['SATS{wave}_2'.format(wave=i)].split()
                spot3 = exthdr['SATS{wave}_3'.format(wave=i)].split()
                centx = np.nanmean([float(spot0[0]), float(spot1[0]), float(spot2[0]), float(spot3[0])])
                centy = np.nanmean([float(spot0[1]), float(spot1[1]), float(spot2[1]), float(spot3[1])])
                center.append([centx, centy])

                # for the rest, compile the list of sat spot data, ignoring bad sat spots
                this_frame_spot_x_locs = [float(spot0[0]), float(spot1[0]), float(spot2[0]), float(spot3[0])]
                this_frame_spot_y_locs = [float(spot0[1]), float(spot1[1]), float(spot2[1]), float(spot3[1])]
                this_frame_spot_indices = [0, 1, 2, 3]
                # delete bad data
                if bad_sat_spots is not None:
                    bad_sat_spots.sort(reverse=True)
                    # delete fom highest index first to not mess up indexing
                    for bad_sat_index in bad_sat_spots:
                        del(this_frame_spot_x_locs[bad_sat_index])
                        del(this_frame_spot_y_locs[bad_sat_index])
                        del(this_frame_spot_indices[bad_sat_index])

                spots_xloc.append(this_frame_spot_x_locs)
                spots_yloc.append(this_frame_spot_y_locs)

            # if the data is a pyklip reduced spectral cube, PSFCENTX/Y should be used to define the center of the image
            if "PSFSUB" in prihdr:
                if prihdr["PSFSUB"].strip() == "pyKLIP":
                    center = [[exthdr['PSFCENTX'], exthdr['PSFCENTY']],]*len(center)


            parang = np.repeat(exthdr['AVPARANG'], channels) #populate PA for each wavelength slice (the same)
            inttime = np.repeat(exthdr['ITIME0'] / 1.e6, channels)
            astr_hdrs = [w.deepcopy() for i in range(channels)] #repeat astrom header for each wavelength slice
        #for pol mode, we consider only total intensity but want to keep the same array shape to make processing easier
        elif exthdr['CTYPE3'].strip() == 'STOKES':
            wvs = [1.0]
            wv_indices = np.array([0])
            cube = np.sum(cube, axis=0)  #sum to total intensity
            cube = cube.reshape([1, cube.shape[0], cube.shape[1]])  #maintain 3d-ness
            center = [[exthdr['PSFCENTX'], exthdr['PSFCENTY']]]
            parang = exthdr['AVPARANG']*np.ones(1)
            inttime = np.repeat(exthdr['ITIME0'] / 1.e6, 1)
            astr_hdrs = np.repeat(w, 1)
            try:
                polspot_fluxes = []
                for i in [0,1]:
                    spot0flux = float(exthdr['SATF{wave}_0'.format(wave=i)])
                    spot1flux = float(exthdr['SATF{wave}_1'.format(wave=i)])
                    spot2flux = float(exthdr['SATF{wave}_2'.format(wave=i)])
                    spot3flux = float(exthdr['SATF{wave}_3'.format(wave=i)])
                    polspot_fluxes.append(np.nanmean([spot0flux, spot1flux, spot2flux, spot3flux]))
                spot_fluxes = [np.sum(polspot_fluxes)]
            except KeyError:
                spot_fluxes = [1]
        else:# exthdr['CTYPE3'].strip() == 'KLMODES':
            try:
                # If datacube
                if 'NAXIS3' in exthdr.keys():
                    channels = exthdr['NAXIS3']
                # If single slice
                else:
                    cube = cube.reshape([1, cube.shape[0], cube.shape[1]])  #maintain 3d-ness
                    channels = 1
                wvs = [0,]*channels
                wv_indices = [0,]*channels
                spot_fluxes = [0,]*channels
                center = [[exthdr['PSFCENTX'], exthdr['PSFCENTY']],]*channels
                spots_xloc = [0,]*channels
                spots_yloc = [0,]*channels

                parang = np.repeat(exthdr['AVPARANG'], channels) #populate PA for each wavelength slice (the same)
                inttime = np.repeat(exthdr['ITIME0'] / 1.e6, channels)
                astr_hdrs = [w.deepcopy() for i in range(channels)] #repeat astrom header for each wavelength slice
            except:
                raise AttributeError("Unrecognized GPI Mode: %{mode}".format(mode=exthdr['CTYPE3']))
    finally:
        hdulist.close()

    # normalize data to be for a single co-add (e.g. add co-adds together)
    if coadds > 1:
        # multiply each frame and sat spot fluxes by number of coadds
        cube *= float(coadds)
        spot_fluxes = [x * float(coadds) for x in spot_fluxes]
        # also multiply integration time by coadds
        inttime *= float(coadds)

    #remove undesirable slices of the datacube if necessary
    if skipslices is not None:
        cube = np.delete(cube, skipslices, axis=0)
        center = np.delete(center, skipslices, axis=0)
        parang = np.delete(parang, skipslices)
        wvs = np.delete(wvs, skipslices)
        wv_indices = np.delete(wv_indices, skipslices)
        astr_hdrs = np.delete(astr_hdrs, skipslices)
        spot_fluxes = np.delete(spot_fluxes, skipslices)
        spots_xloc = np.delete(spots_xloc, skipslices)
        spots_yloc = np.delete(spots_yloc, skipslices)
        inttime = np.delete(inttime, skipslices)

    #high pass and remeasure the satellite spot fluxes if necessary
    highpassed = False
    if isinstance(highpass, bool):
        if highpass:
            cube = high_pass_filter_imgs(cube)
            highpassed = True
    else:
        # should be a number
        if isinstance(highpass, (float, int)):
            highpass = float(highpass)
            fourier_sigma_size = (cube.shape[1]/(highpass)) / (2*np.sqrt(2*np.log(2)))
            cube = high_pass_filter_imgs(cube, filtersize=fourier_sigma_size)
            highpassed = True


    # remeasure satellite spot fluxes
    if meas_satspot_flux:

        # only do for spec mode, because I don't have the pol mode photometry tool implemented here
        if exthdr['CTYPE3'].strip() == 'WAVE':
            spot_fluxes = []

            wv_unique = np.unique(wvs)

            if numthreads == -1:
                # default sat spot measuring code
                for slice, spots_xs, spots_ys, wv, wv_index in zip(cube, spots_xloc, spots_yloc, wvs, wv_indices):
                    new_spotfluxes = measure_sat_spot_fluxes(slice, spots_xs, spots_ys,psfs_func_list=psfs_func_list,wave_index=wv_index)
                    if np.sum(np.isfinite(new_spotfluxes)) == 0:
                        print("Infite satellite spot fluxes", (slice, spots_xs, spots_ys))
                    spot_fluxes.append(np.nanmean(new_spotfluxes))

            else:
                # JB: On going test..
                if numthreads is None:
                    numthreads = mp.cpu_count()
                tpool = mp.Pool(processes=numthreads, maxtasksperchild=50)
                tpool_outputs = [tpool.apply_async(measure_sat_spot_fluxes,
                                                   args=(slice, spots_xs, spots_ys,psfs_func_list,wv_indices))
                                 for id,(slice, spots_xs, spots_ys, wv, wv_index) in enumerate(zip(cube, spots_xloc, spots_yloc, wvs, wv_indices))]

                for out in tpool_outputs:
                    out.wait()
                    new_spotfluxes = out.get()
                    spot_fluxes.append(np.nanmean(new_spotfluxes))
                tpool.close()
        #print(spot_fluxes)

    return cube, center, parang, wvs, wv_indices, astr_hdrs, filt_band, fpm_band, ppm_band, spot_fluxes, inttime, prihdr, exthdr


def measure_sat_spot_fluxes(img, spots_x, spots_y,psfs_func_list=None,wave_index=None, residuals = False):
    """
    Measure satellite spot peak fluxes using a Gaussian matched filter

    Args:
        img: 2D frame with 4 sat spots
        spots_x: list of 4 satellite spot x coordinates
        spots_y: list of 4 satellite spot y coordinates
        psfs_func_list: List of spline fit function for the PSF_cube. If None (default) a gaussian fit is used.
        wave_index: Index of the current wavelength. In [0,36] for GPI. Only used when psfs_func_list is not None.
        residuals: If True (Default = False) then calculate the residuals of the sat spot fit (gaussian or PSF cube).
    Returns:
        spots_f: list of 4 satellite spot fluxes
    """
    spots_f = []
    residual_map_list=[]
    for spotx, spoty in zip(spots_x, spots_y):
        # flux, fwhm, xfit, yfit = gaussfit2d(img, spotx, spoty, refinefit=False)
        if psfs_func_list is None:
            if residuals:
                flux,residual_map = gaussfit2dLSQ(img, spotx, spoty,residuals=residuals)
                residual_map_list.append(residual_map)
            else:
                flux = gaussfit2dLSQ(img, spotx, spoty,residuals=residuals)
        else:
            if residuals:
                flux,residual_map = PSFcubefit(img, spotx, spoty,psfs_func_list=psfs_func_list,wave_index=wave_index,residuals=residuals)
                residual_map_list.append(residual_map)
            else:
                flux = PSFcubefit(img, spotx, spoty,psfs_func_list=psfs_func_list,wave_index=wave_index,residuals=residuals)
        fwhm = 3

        if flux == np.inf:
            flux = np.nan

        if (fwhm < 1) | (fwhm > 10):
            # most definitely bogus measurements
            flux = np.nan

        spots_f.append(flux)

    if residuals:
        return spots_f,residual_map_list
    else:
        return spots_f

def recalculate_sat_spot_fluxes(dataset, skipslices=None, numthreads=-1, PSF_cube=None, residuals = False):
    """
    Recalculate the satellite spots fluxes.

    Args:
        dataset: GPIData object.
        skipslices: a list of datacube slices to skip (supply index numbers e.g. [0,1,2,3])
                WARNING! SKIPSLICES features hasn't been tested with this function.
        numthreads: Number of threads to be used. Default -1 sequential sat spot flux calc.
                    If None, numthreads = mp.cpu_count().
        PSF_cube: 3D array (nl,ny,nx) with the PSF cube to be used in the flux calculation.
        residuals: If True (Default = False) then calculate the residuals of the sat spot fit (gaussian or PSF cube).

    Returns:
        spot_fluxes: The list of sat spot fluxes. Can be used to redefine dataset.spot_flux.
    """

    if PSF_cube is not None:
        numwv,ny_psf,nx_psf =  PSF_cube.shape
        x_psf_grid, y_psf_grid = np.meshgrid(np.arange(nx_psf * 1.)-nx_psf//2,np.arange(ny_psf* 1.)-ny_psf//2)
        psfs_func_list = []
        from scipy import interpolate
        for wv_index in range(numwv):
            model_psf = PSF_cube[wv_index, :, :]
            psfs_func_list.append(interpolate.LSQBivariateSpline(x_psf_grid.ravel(),y_psf_grid.ravel(),model_psf.ravel(),x_psf_grid[0,0:nx_psf-1]+0.5,y_psf_grid[0:ny_psf-1,0]+0.5))
    else:
        psfs_func_list = None

    N_cubes = len(dataset.exthdrs)
    wv_unique = np.unique(dataset.wvs)
    nl = np.size(wv_unique)

    spot_fluxes = []
    residuals_map_list = []
    for cube_id,(prihdr,exthdr) in enumerate(zip(dataset.prihdrs,dataset.exthdrs)):
        #for spectral mode we need to treat each wavelegnth slice separately
        if exthdr['CTYPE3'].strip() == 'WAVE':
            channels = exthdr['NAXIS3']
            spots_xloc = []
            spots_yloc = []
            cube = []

            #calculate centers from satellite spots
            for i in range(channels):
                slice_id = nl*cube_id + i
                if skipslices is not None:
                    if not (slice_id in skipslices):
                        cube.append(dataset.input[slice_id])

                        #grab satellite spot positions
                        spot0 = exthdr['SATS{wave}_0'.format(wave=i)].split()
                        spot1 = exthdr['SATS{wave}_1'.format(wave=i)].split()
                        spot2 = exthdr['SATS{wave}_2'.format(wave=i)].split()
                        spot3 = exthdr['SATS{wave}_3'.format(wave=i)].split()
                        spots_xloc.append([float(spot0[0]), float(spot1[0]), float(spot2[0]), float(spot3[0])])
                        spots_yloc.append([float(spot0[1]), float(spot1[1]), float(spot2[1]), float(spot3[1])])
                else:
                    cube.append(dataset.input[slice_id])

                    #grab satellite spot positions
                    spot0 = exthdr['SATS{wave}_0'.format(wave=i)].split()
                    spot1 = exthdr['SATS{wave}_1'.format(wave=i)].split()
                    spot2 = exthdr['SATS{wave}_2'.format(wave=i)].split()
                    spot3 = exthdr['SATS{wave}_3'.format(wave=i)].split()
                    spots_xloc.append([float(spot0[0]), float(spot1[0]), float(spot2[0]), float(spot3[0])])
                    spots_yloc.append([float(spot0[1]), float(spot1[1]), float(spot2[1]), float(spot3[1])])

            if numthreads == -1:
                # default sat spot measuring code
                for slice, spots_xs, spots_ys, wv in zip(cube, spots_xloc, spots_yloc, dataset.wvs):
                    meas_sat_spot_out = measure_sat_spot_fluxes(slice, spots_xs, spots_ys,psfs_func_list=psfs_func_list,wave_index=np.where(wv_unique == wv)[0],residuals=residuals)
                    if residuals:
                        new_spotfluxes,residuals_map = meas_sat_spot_out
                        residuals_map_list.extend(residuals_map)
                    else:
                        new_spotfluxes = meas_sat_spot_out
                    if np.sum(np.isfinite(new_spotfluxes)) == 0:
                        print("Infite satellite spot fluxes", (slice, spots_xs, spots_ys))
                    spot_fluxes.append(np.nanmean(new_spotfluxes))
            else:
                # JB: On going test..
                if numthreads is None:
                    numthreads = mp.cpu_count()
                tpool = mp.Pool(processes=numthreads, maxtasksperchild=50)
                tpool_outputs = [tpool.apply_async(measure_sat_spot_fluxes,
                                                   args=(slice, spots_xs, spots_ys,psfs_func_list,np.where(wv_unique == wv)[0],residuals))
                                 for id,(slice, spots_xs, spots_ys,wv) in enumerate(zip(cube, spots_xloc, spots_yloc,dataset.wvs))]

                for out in tpool_outputs:
                    out.wait()
                    if residuals:
                        new_spotfluxes,residuals_map = out.get()
                        residuals_map_list.extend(residuals_map)
                    else:
                        new_spotfluxes = out.get()
                    spot_fluxes.append(np.nanmean(new_spotfluxes))
                tpool.close()

    if residuals:
        return np.array(spot_fluxes),residuals_map_list
    else:
        return np.array(spot_fluxes)

def generate_psf(frame, locations, boxrad=5, medianboxsize=30):
    """
    Generates a GPI PSF for the frame based on the satellite spots

    Args:
        frame: 2d frame of data
        location: array of (N,2) containing [x,y] coordinates of all N satellite spots
        boxrad: half length of box to use to pull out PSF
        medianboxsize: size in pixels of box for median filter

    Returns:
        genpsf: 2d frame of size (2*boxrad+1, 2*boxrad+1) with average PSF of satellite spots
    """
    genpsf = []
    #mask nans
    cleaned = np.copy(frame)
    cleaned[np.where(np.isnan(cleaned))] = 0

    #highpass filter to remove background
    #mask source for median filter
    # masked = np.copy(cleaned)
    # for loc in locations:
    #     spotx = np.round(loc[0])
    #     spoty = np.round(loc[1])
    #     masked[spotx-boxrad:spotx+boxrad+1, spoty-boxrad:spoty+boxrad+1] = scipy.stats.nanmedian(
    #         masked.reshape(masked.shape[0]*masked.shape[1]))
    #subtract out median filtered image

    #cleaned -= ndimage.median_filter(masked, size=(medianboxsize,medianboxsize))

    for loc in locations:
        #grab satellite spot positions
        spotx = loc[0]
        spoty = loc[1]

        #interpolate image to grab satellite psf with it centered
        #add .1 to make sure we get 2*boxrad+1 but can't add +1 due to floating point precision (might cause us to
        #create arrays of size 2*boxrad+2)
        x,y = np.meshgrid(np.arange(spotx-boxrad, spotx+boxrad+0.1, 1), np.arange(spoty-boxrad, spoty+boxrad+0.1, 1))
        spotpsf = ndimage.map_coordinates(cleaned, [y,x])

        # if applicable, do a background subtraction
        if boxrad >= 7:
            y_img, x_img = np.indices(frame.shape, dtype=float)
            r_img = np.sqrt((x_img - spotx)**2 + (y_img - spoty)**2)
            noise_annulus = np.where((r_img > 9) & (r_img <= 12))
            background_mean = np.nanmean(cleaned[noise_annulus])
            spotpsf -= background_mean

        genpsf.append(spotpsf)

    genpsf = np.array(genpsf)
    genpsf = np.mean(genpsf, axis=0) #average the different psfs together    

    return genpsf


def rescale_wvs(exthdrs, wvs, refwv=None, skipslices=None, bad_sat_spots=None):
    """
    Hack to try to fix wavelength scaling issue. This will calculate the scaling between channels,
    and adjust the wavelength solution such that the scaling comes out linear in scaling vs wavelength.
    Finicky - requires that all images in the dataset have the same number of wavelength channels
    Args:
        exthdrs: a list of extension headers, from a pyklip.instrument dataset
        wvs: a list of wvs (can repeat. This function will only look at the first cube's wavelenghts)
        refwv (optional): integer index of the channel to normalize the scaling
        skipslices: list of skipped wavelength slices (needs to be consistent with the ones skipped by wv)
    Returns:
        scaled_wvs: Nlambda*Nexthdrs array of wavelengths that produce a linear plot of wavelength vs scaling
    """
    #wvs_mean = wvs.reshape(len(exthdrs), len(wvs)/len(exthdrs)).mean(axis=0)
    if refwv is None:
        refwv = np.size(np.unique(wvs)) // 2
    wv_indicies = range(0, exthdrs[0]['NAXIS3'])
    if skipslices is not None:
        wv_indicies = np.delete(wv_indicies, skipslices)
    sats = np.array([[[h['SATS{0}_{1}'.format(i,j)].split() for i in wv_indicies]
                      for j in range(0,4)] 
                     for h in exthdrs], dtype=np.float)
    sats = sats.mean(axis=0)
    pairs = [(0,3), (1,2)]
    separations = np.mean([0.5*np.sqrt(np.diff(sats[p,:,0], axis=0)[0]**2 + np.diff(sats[p,:,1], axis=0)[0]**2) 
                           for p in pairs], 
                          axis=0) # average over each pair, the first axis
    scaling_factors = separations/separations[refwv]
    scaled_wvs = scaling_factors*wvs[refwv]
    return np.tile(scaled_wvs, len(exthdrs))


def calc_center_least_squares(xpos, ypos, wvs, orderx, ordery, displacement):
    """
    calcualte the center position, linear least squares fit to 4 parameters

    Args:
        xpos: array of length n of x positions of satellite spots
        ypos: array of length n of y positions of satellite spots
        wvs: the wavelength of each pair of positoins
        orderx: the x order (can be -1 or 1 in this case. -1 is under the center, 1 is above the center)
        ordery: the y order (e.g. pos0 is at pox=-1, posy=1).
        displacment: the displacement from zenith
    Returns:
        four fit parameters (xcenter, ycenter, adrx, adry). xcenters = xcenter + ardx * displacement
    """

    pos_x = np.matrix(xpos).T
    pos_y = np.matrix(ypos).T

    #create the B matrix for the transform. See email from James on how to make this
    Bx = np.append(np.matrix(np.ones(np.size(pos_x))).T, np.matrix(orderx*wvs).T,1)
    Bx = np.append(Bx,np.matrix(-ordery*wvs).T, 1)
    Bx = np.append(Bx, np.matrix(displacement).T , 1)
    Bx = np.append(Bx, np.matrix(np.zeros(np.size(pos_x))).T, 1)
    Bx = np.append(Bx, np.matrix(np.zeros(np.size(pos_x))).T, 1)
    By = np.append(np.matrix(np.zeros(np.size(pos_y))).T, np.matrix(ordery*wvs).T, 1)
    By = np.append(By, np.matrix(orderx*wvs).T, 1)
    By = np.append(By, np.matrix(np.zeros(np.size(pos_y))).T, 1)
    By = np.append(By, np.matrix(displacement).T , 1)
    By = np.append(By, np.matrix(np.ones(np.size(pos_y))).T, 1)

    B = np.append(Bx,By,0)

    #the measured inputs
    X = np.append(pos_x, pos_y, 0)

    #fit outputs
    Q = (B.T*B).I * B.T* X

    xcenter = float(Q[0])
    ycenter = float(Q[5])
    shift1 = float(Q[1])
    shift2 = float(Q[2])
    adrx = float(Q[3])
    adry = float(Q[4])


    return xcenter, ycenter, adrx, adry


def calc_center(prihdr, exthdr, wvs, ignoreslices=None, skipslices=None, bad_sat_spots=None):
    """
    calcualte the center position of a spectral data cube

    Args:
        prihdr: primary GPI header
        exthdr: extention GPI header
        wvs: wvs of the datacube
        ignoreslices: slices to ignore in the fit. A list of wavelength slice indicies to ignore
                        if none, ignores slices 0,1, len-2, len-1 (first and last two)
        skipslices: slices that were already skipped in processing
        bad_sat_stots: of the 4 sat spots, which are bad and should be ignored. Indexed 0-3 based on x coordinate

    Returns:
        centx, centy: star center
    """
    maxwvs = exthdr['NAXIS3']
    if ignoreslices is None:
        ignoreslices = np.array([0,1,maxwvs-2,maxwvs-1])
    ignoreslices %= np.size(wvs)

    utstart = prihdr['UTSTART']
    utstart = float(utstart[0:2]) + float(utstart[3:5])/60.+float(utstart[6:])/3600. #covert to decimal

    #Grab info for ADR correction
    #try to get environment parameters but sometimes we need to default
    #Get HA
    HA = prihdr['HA']
    HA_sgn = HA[0]
    if HA_sgn == '+':
        HA_sgn = 1
    else:
        HA_sgn = -1
    HA = float(HA[0:3]) + HA_sgn*float(HA[4:6])/60. + HA_sgn*float(HA[7:])/3600.
    HA *= 15*np.pi/180. # rad
    #Get Temp
    Temp = prihdr['TAMBIENT'] + 273.15 #Kelvin
    #Get pressure
    Pressure = prihdr['PRESSUR2'] #Pascal
    #Get declination from header and convert to radians
    dec = exthdr['CRVAL2'] * np.pi/ 180. #rad

    #Calculate angle from zenith, need this for ADR corrections
    zenith = np.arccos(np.sin(GPIData.observatory_latitude)*np.sin(dec)
                       + np.cos(GPIData.observatory_latitude)*np.cos(dec)*np.cos(HA))

    spots_posx = []
    spots_posy = []
    order_x = []
    order_y = []
    displacement = []
    spot_wvs = []
    spots_wvs_index = []

    #calculate reference wavelegnth
    refwv = np.mean(wvs)
    n0 = nMathar(refwv, Pressure, Temp) #reference index of refrraction

    #get centers from header values inputted by GPI pipeline
    #mask = bin(int(pcenthdr['SATSMASK'],16)) #assume all spot locations are placed in header
    #iterate over headers in cube
    i = 0
    for wv in wvs:
        thisfour = []
        n = nMathar(wv, Pressure, Temp) #index of refraction

        # increiment loop index if we need to skip
        if skipslices is not None:
            while i in skipslices:
                i += 1
                # sanity check in case we get stuck in an infinite loop (hopefully won't)
                if i >= maxwvs:
                    print("oops.. infinite loop in skipping wavelenghts")
                    break

        for j in range(4):
            if bad_sat_spots is not None:
                if j in bad_sat_spots:
                    continue
            hdr_str = "sats{0}_{1}".format(i, j)
            cents = exthdr[hdr_str]
            args = cents.split()

            #append this data to the list
            #calcuate deltaZ effect of ADR
            displacement.append( (n-n0)/n0 * np.tan(zenith)) #deltaZ calculation
            spots_posx.append(float(args[0]))
            spots_posy.append(float(args[1]))
            spot_wvs.append(wv)
            spots_wvs_index.append(i)

            #this better account for all cases or this for loop is messed up
            if j == 0:
                order_x.append(-1)
                order_y.append(1)
            elif j == 1:
                order_x.append(-1)
                order_y.append(-1)
            elif j == 2:
                order_x.append(1)
                order_y.append(1)
            elif j ==3:
                order_x.append(1)
                order_y.append(-1)

        i += 1

    spots_posx = np.array(spots_posx)
    spots_posy = np.array(spots_posy)
    order_x = np.array(order_x)
    order_y = np.array(order_y)
    displacement = np.array(displacement)
    spot_wvs = np.array(spot_wvs)
    spots_wvs_index = np.array(spots_wvs_index)

    good = np.where(~np.in1d(spots_wvs_index, ignoreslices))

    x0, y0, adrx, adry = calc_center_least_squares(spots_posx[good], spots_posy[good], spot_wvs[good], order_x[good],
                                                   order_y[good], displacement[good])
    centers_x = x0 + adrx*displacement
    centers_y = y0 + adry*displacement
    centers = np.array([centers_x, centers_y])
    # centers are duplicated 4 times (for each sat spot) and the dimensions are flipped. need to remove this...
    if bad_sat_spots is None:
        num_sat_spots = 4
    else:
        num_sat_spots = 4 - np.size(bad_sat_spots)
    centers = np.swapaxes(centers, 0, 1)
    centers = centers.reshape([centers.shape[0]//num_sat_spots, num_sat_spots, 2])
    centers = centers[:,0,:]
    return centers


def get_gpi_wavelength_sampling(filter_name):
    """
    Return GPI wavelength sampling for a given band.

    Args:
        filter_name: 'H', 'J', 'K1', 'K2', 'Y'.
                    Wavelength samples are linearly spaced between the first and the last wavelength of the band.

    Returns:
        wavelengths: is the gpi sampling of the considered band in micrometer.
    """
    # First and last wavelength of each band
    band_sampling = {'Z' : (0.9444, 1.1448, 37),
                     'Y' : (0.9444, 1.1448, 37),
                     'J' : (1.1108, 1.353, 37),
                     'H' : (1.4904, 1.8016, 37),
                     'K1' : (1.8818, 2.1994, 37),
                     'K2' : (2.1034, 2.4004, 37)}

    w_start, w_end, N_sample = band_sampling[filter_name]
    sampling_pip = np.linspace(w_start,w_end,N_sample,endpoint=True)

    return sampling_pip

def as2pix(sep_as):
    return np.array(sep_as)/GPIData.lenslet_scale

def pix2as(sep_pix):
    return np.array(sep_pix)*GPIData.lenslet_scale
