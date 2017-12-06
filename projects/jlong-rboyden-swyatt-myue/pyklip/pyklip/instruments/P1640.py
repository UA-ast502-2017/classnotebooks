import os
import re
import subprocess
import glob
from copy import deepcopy

import sys
if sys.version_info.major >= 3:
    from functools import reduce
import astropy.io.fits as fits
from astropy import wcs
import numpy as np
import scipy.ndimage as ndimage
import scipy.stats
import random as rd


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

from pyklip.instruments.P1640_support import P1640spots
from pyklip.instruments.P1640_support import P1640utils
# from pyklip.instruments.P1640_support import P1640_cube_checker

from scipy.interpolate import interp1d
from pyklip.parallelized import high_pass_filter_imgs

class P1640Data(Data):
    """
    Note: update object string when output is decided
    A sequence of P1640 Data. Each P1640Data object has the following fields and functions

    Args:
        filepaths: list of filepaths to occulted files
        skipslices: a list of datacube slices to skip (supply index numbers e.g. [0,1,2,3])
        corefilepaths: a list of filepaths to core (i.e. unocculted) files, for contrast calc
        spot_directory: (None) path to the directory where the spot positions are stored. Defaults to P1640.ini val
        verbose: [False] if True, print more stuff
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
        spot_flux: Array of N of average satellite spot flux for each frame
        contrast_scaling: Flux calibration factors (multiply by image to "calibrate" flux)
        flux_units: units of output data [DN, contrast]
        prihdrs: not used for P1640, set to None
        exthdrs: Array of N P1640 headers (these are written by the P1640 cube extraction pipeline)

    Methods:
        readdata(): reread in the data
        savedata(): save a specified data in the P1640 datacube format (in the 1st extension header)
        calibrate_output(): calibrates flux of self.output
    """
    ##########################
    ###Class Initilization ###
    ##########################
    #some static variables to define the P1640 instrument
    centralwave = {}  # in microns
    fpm_diam = {}  # in pixels
    flux_zeropt = {}
    spot_ratio = {} #w.r.t. central star
    lenslet_scale = 1.0 # arcseconds per pixel (pixel scale)
    ifs_rotation = 0.0  # degrees CCW from +x axis to zenith
    nchannels_all = None
    nchannels_used = None
    observatory_latitude = 0.0

    ## read in P1640 configuration file and set these static variables
    package_directory = os.path.dirname(os.path.abspath(__file__))
    configfile = package_directory + "/" + "P1640.ini"
    config = ConfigParser.ConfigParser()
    try:
        config.read(configfile)
        #get pixel scale
        lenslet_scale = float(config.get("instrument", "ifs_lenslet_scale"))  # arcsecond/pix
        #get IFS rotation
        ifs_rotation = float(config.get("instrument", "ifs_rotation")) #degrees
        #get some information specific to each band
        bands = ['H']
        for band in bands:
            centralwave[band] = float(config.get("instrument", "cen_wave_{0}".format(band)))
            fpm_diam[band] = float(config.get("instrument", "fpm_diam")) / lenslet_scale  # pixels
        flux_zeropt = float(config.get("instrument", "zero_pt_flux"))
        observatory_latitude = float(config.get("observatory", "observatory_lat"))
        nchannels_all = int(config.get("instrument", "cube_channels"))
    except ConfigParser.Error as e:
        print("Error reading P1640 configuration file: {0}".format(e.message))
        raise e


    ####################
    ### Constructors ###
    ####################
    def __init__(self, filepaths=None, skipslices=None, corefilepaths=None, spot_directory=None, highpass=True,
                 numthreads=-1, PSF_cube=None, verbose=False):
        """
        Initialization code for P1640Data

        Note:
            Information on arguments are available in the class docstring
        """
        super(P1640Data, self).__init__()
        self._output = None

        # P1640 stuff
        self.corefilenames = corefilepaths
        self.spot_directory = spot_directory
        self.spot_flux = None # Currently not implemented, may be in future
        self.scale_factors = None # scaling between wavelength slices
        self.spot_positions = None # Ncube x Nchan x 4 x 2 spot positions [row, col]
        #self.spot_scaling = None # scaling factor between wavelengths

        self.channels_all = list(range(P1640Data.nchannels_all))
        try:
            self.channels_used = [i for i in self.channels_all if i not in skipslices]
        except:
            self.channels_used = self.channels_all[:]
        self.nchannels_used = len(self.channels_used)

        if filepaths is None:
            # general stuff
            self._input = None
            self._centers = None
            self._filenums = None
            self._filenames = None
            self._PAs = None
            self._wvs = None
            self._wcs = None
            self._IWA = None
            self.corefilenames = corefilepaths
            self.spot_flux = None # Currently not implemented, may be in future
            self.scale_factors = None # scaling factor between wavelengths
            self.contrast_scaling = None # Currently not implemented, may be in future
            self.prihdrs = None # Not used by P1640
            self.exthdrs = None # for P1640 this is the prihdrs; exthdrs used for compatibility with P1640 class
            self.flux_units = None # Currently not implemented, may be in future
        else:
            self.readdata(filepaths, skipslices=skipslices, highpass=highpass, numthreads=numthreads, PSF_cube=PSF_cube)


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
    def configfile(self):
        return self._configfile
    @configfile.setter
    def configfile(self, newval):
        self._configfile = newval

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

    # P1640 stuff
    @property
    def corefilenames(self):
        return self._corefilenames
    @corefilenames.setter
    def corefilenames(self, newval):
        self._corefilenames = newval

    @property
    def scale_factors(self):
        return self._scale_factors
    @scale_factors.setter
    def scale_factors(self, newval):
        self._scale_factors = newval

    @property
    def spot_locations(self):
        return self._spot_locations
    @spot_locations.setter
    def spot_locations(self, newval):
        self._spot_locations = newval

    @property
    def spot_directory(self):
        return self._spot_directory
    @spot_directory.setter
    def spot_directory(self, newval):
        #if newval is None:
        #    newval = P1640Data.config("spots","spot_file_path")
        self._spot_directory = newval
        print("Spot file directory set to {0}".format(self.spot_directory))

    ###############
    ### Methods ###
    ###############
    def get_scaling_and_centering(self, filepaths, spot_directory=None, skipslices=None):
        """
        Method to get star position and wavelength scaling factors
        Args:
            filepaths: a list of filepaths to datacubes
            spot_path: (None) directory where the spot files are stored. Defaults to P1640.ini value
            skipslices: a list of wavelength slices to skip for each datacube (supply index numbers e.g. [0,1,2,3])
        Returns:
            Nothing. Sets centers and scale_factors fields of the P1640Data object
        """
        if isinstance(filepaths, str):
            filepaths = [filepaths]

        scale_factors = []
        centers = []
        spot_photometry = []

        for filepath in filepaths:
            hdulist = fits.open(filepath)
            cube = hdulist[0].data

            try:
                if spot_directory is not None:
                    spot_filedir = spot_directory
                    print(spot_filedir)
                else: # use the default set in P1640.ini
                    spot_filedir = P1640Data.config.get("spots","spot_filepath")
                spot_filebasename = os.path.splitext(os.path.basename(filepath))[0]
                spot_fullpath = os.path.join(spot_filedir, spot_filebasename)
                spot_filepaths = glob.glob(spot_fullpath+'*')

                # check if all the spot files exist, if so, read them in
                exist = np.all([os.path.isfile(f) for f in spot_filepaths])
                assert(exist is not False)
                # print("Reading spots from files: {0}".format(os.path.commonprefix(spot_filepaths)))
                spot_locations = np.array([np.genfromtxt(f, delimiter=',') 
                                           for f in spot_filepaths])
                # return spot_locations
            except AssertionError:
                # if they haven't already been written to file, calculate them
                spot_locations = P1640spots.get_single_cube_spot_positions(cube)
                # write them to disk so they don't have to be recalculated
                write_p1640_spots_to_file(P1640Data.config, filepath, spot_locations)

            cube_spot_fluxes = P1640spots.get_single_cube_spot_photometry(cube, spot_locations)
            cube_scale, cube_center = P1640spots.get_scaling_and_centering_from_spots(spot_locations)

            hdulist.close()

            scale_factors.append(cube_scale)
            centers.append(cube_center)
            spot_photometry.append(cube_spot_fluxes)
        return centers
        #self.scale_factors = scale_factors
        #self.centers = centers
        #self.spot_flux = cube_spot_fluxes


    def readdata(self, filepaths, skipslices=None, corefilepaths=None, highpass=True, numthreads=-1, PSF_cube=None, verbose=False):
        """
        Method to open and read a list of P1640 data. Handles everything that can be done by
        reading directly from the P1640 header or cubes, no calculations. Scaling and Centering handled elsewhere

        Args:
            filepaths: a list of filepaths
            skipslices: a list of wavelenegth slices to skip for each datacube (supply index numbers e.g. [0,1,2,3])
            highpass: if True, run a Gaussian high pass filter (default size is sigma=imgsize/10)
                      can also be a number specifying FWHM of box in pixel units
            numthreads: Number of threads to be used. Default -1 sequential sat spot flux calc.
                        If None, numthreads = mp.cpu_count().
            PSF_cube: 3D array (nl,ny,nx) with the PSF cube to be used in the flux calculation.
        Returns:
            Technically none. It saves things to fields of the P1640Data object. See object doc string
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
        spot_scalings = []
        spot_locations = []
        wcs_hdrs = []
        spot_fluxes = []
        prihdrs = []
        exthdrs = []

        #extract data from each file
        for index, filepath in enumerate(filepaths):
            cube, center, spot_scaling_single_cube, spot_locations_single_cube, pa, wv, \
            astr_hdrs, filt_band, fpm_band, ppm_band, spot_flux, prihdr, exthdr = \
                _p1640_process_file(filepath, spot_directory=self.spot_directory, skipslices=skipslices,
                                    highpass=highpass, numthreads=numthreads, verbose=verbose)

            data.append(cube)
            centers.append(center)
            spot_scalings.append(spot_scaling_single_cube)
            spot_locations.append(spot_locations_single_cube)
            spot_fluxes.append(spot_flux)
            rot_angles.append(pa)
            wvs.append(wv)
            filenums.append(np.ones(pa.shape[0]) * index)
            wcs_hdrs.append(astr_hdrs)
            prihdrs.append(prihdr)
            exthdrs.append(exthdr)
            filenames.append([filepath for i in range(pa.shape[0])])
        if verbose == False:
            print("{N} files processed from {p} (verbose = False).".format(N=len(filepaths),
                                                                           p=os.path.commonprefix(filepaths)))


        #convert everything into numpy arrays
        #reshape arrays so that we collapse all the files together (i.e. don't care about distinguishing files)
        data = np.array(data)
        data = P1640utils.set_zeros_to_nan(data)
        dims = data.shape
        data = data.reshape([dims[0] * dims[1], dims[2], dims[3]])
        filenums = np.array(filenums).reshape([dims[0] * dims[1]])
        filenames = np.array(filenames).reshape([dims[0] * dims[1]])
        rot_angles = -(np.array(rot_angles).reshape([dims[0] * dims[1]])) + (90 - self.ifs_rotation) # want North Up
        wvs = np.array(wvs).reshape([dims[0] * dims[1]])
        wcs_hdrs = np.array(wcs_hdrs).reshape([dims[0] * dims[1]])
        centers = np.array(centers).reshape([dims[0] * dims[1], 2])
        spot_fluxes = np.array(spot_fluxes).reshape([dims[0] * dims[1]])
        spot_locations = np.rollaxis(np.array(spot_locations), -2, 1).reshape([dims[0]*dims[1], 4, 2])[..., ::-1]
        # for the spot scalings factors, take the mean for each wavelength
        spot_scalings = np.array(spot_scalings)#.reshape([dims[0]*dims[1]])
        # Not used by P1640
        '''
        #only do the wavelength solution and center recalculation if it isn't broadband imaging
        if np.size(np.unique(wvs)) > 1:
            # recalculate wavelengths from satellite spots
            wvs = rescale_wvs(exthdrs, wvs, skipslices=skipslices)
            # recaclulate centers from satellite spots and new wavelength solution
            wvs_bycube = wvs.reshape([dims[0], dims[1]])
            centers_bycube = centers.reshape([dims[0], dims[1], 2])
            for i, cubewvs in enumerate(wvs_bycube):
                try:
                    centers_bycube[i] = calc_center(prihdrs[i], exthdrs[i], cubewvs, skipslices=skipslices)
                except KeyError:
                    print("Unable to recenter the data using a least squraes fit due to not enough header info for file "
                          "{0}".format(filenames[i*dims[1]]))
        '''
        
        #set these as the fields for the P1640Data object
        self._input = data
        self._centers = centers
        self._filenums = filenums
        self._filenames = filenames
        self._corefilenames = corefilepaths
        self._PAs = rot_angles
        # use the spot scalings instead of the wavelength solution; that comes separately anyway
        #self._wvs =  wvs
        self._wvs = np.tile(np.mean(spot_scalings, axis=0), spot_scalings.shape[0])
        self._wcs = wcs_hdrs 
        self._IWA = P1640Data.fpm_diam[fpm_band]/2.0
        self.spot_flux = spot_fluxes
        self.scale_factors = spot_scalings
        self.spot_locations = spot_locations
        self.contrast_scaling = None #P1640Data.spot_ratio[ppm_band]/np.mean(spot_fluxes)
        self.flux_units = "DN"
        self.prihdrs = prihdrs
        self.exthdrs = exthdrs

    def savedata(self, filepath, data, klipparams = None, filetype = 'PSF Subtracted Spectral Cube',
                 zaxis = None, center=None, astr_hdr=None, fakePlparams = None, more_keywords=None,
                 user_prihdr = None, user_exthdr = None, extra_exthdr_keywords = None,
                 extra_prihdr_keywords = None ):
        """
        Save data in a fits file in a GPI-like fashion. Aka, data and header are in the extension HDU.
        For now, the Primary HDU contains the KLIP parameters, scaling, and centering. This may later change 
        storing the data in the Primary HDU, all the headers from the input files in the extension.
        Inputs:
            filepath: path to file to output
            data: 2D or 3D data to save
            klipparams: a string of klip parameters
            filetype: filetype of the object (e.g. "KL Mode Cube", "PSF Subtracted Spectral Cube")
            zaxis: a list of values for the zaxis of the datacube (for KL mode cubes currently)
            astr_hdr: wcs astrometry header
            center: center of the image to be saved in the header as the keywords PSFCENTX and PSFCENTY in pixels.
                The first pixel has coordinates (0,0)
            fakePlparams: fake planet params

        """
        hdulist = fits.HDUList()
        hdulist.append(fits.PrimaryHDU(header=None, data=data))
        # save all the files we used in the reduction
        # we'll assume you used all the input files
        # remove duplicates from list
        filenames = np.unique(self.filenames)
        nfiles = np.size(filenames)
        hdulist[0].header["DRPNFILE"] = (nfiles, "Num raw files used in pyKLIP")
        for i, thispath in enumerate(filenames):
            thispath = thispath.replace("\\", '/')
            splited = thispath.split("/")
            fname = splited[-1]
            matches = re.search('20[0-9]{2}-[0-9]{2}-[0-9]{2}_[0-9]{3}', fname)
            filename = matches.group(0)
            hdulist[0].header["FILE_{0}".format(i)] = filename + '.fits'

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
            hdulist[0].header['FILETYPE'] = (filetype, "P1640 File type")

        if self.channels_used is not None:
            hdulist[0].header['channels'] = ','.join(['{0}'.format(i) for i in self.channels_used])

        #write flux units/conversion
        hdulist[0].header['FUNIT'] = (self.flux_units, "Flux units of data")
        if self.flux_units.upper() == 'CONTRAST':
            hdulist[0].header['DN2CON'] = (self.contrast_scaling, "Contrast/DN")
            hdulist[0].header.add_history("Converted to contrast units using {0} Contrast/DN".format(self.contrast_scaling))
            core_header = fits.Header()
            for i,cf in enumerate(self._corefilenames):
                core_header['Core_{0:02d}'.format(i)] = cf
            core_header['Method'] = 'median'
            core_hdu = fits.ImageHDU(self._core_psf, name='Core')
            core_hdu.header = core_header
            hdulist.append(fits.ImageHDU(self._core_psf))

        # write z axis units if necessary
        if zaxis is not None:
            #Writing a KL mode Cube
            if "KL Mode" in filetype:
                hdulist[0].header['CTYPE3'] = 'KLMODES'
                #write them individually
                for i, klmode in enumerate(zaxis):
                    hdulist[0].header['KLMODE{0}'.format(i)] = (klmode, "KL Mode of slice {0}".format(i))

        #use the dataset astr hdr if none was passed in
        if astr_hdr is None:
            astr_hdr = self.output_wcs[0]
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
            # remove CD values as those are confusing
            # currently not found in P1640 headers
            try:
                exthdr.remove('CD1_1')
                exthdr.remove('CD1_2')
                exthdr.remove('CD2_1')
                exthdr.remove('CD2_2')
            except:
                pass
            exthdr['CDELT1'] = 1
            exthdr['CDELT2'] = 1

        #use the dataset center if none was passed in
        if center is None:
            center = self.output_centers[0]
        if center is not None:
            hdulist[0].header.update({'PSFCENTX':center[0],'PSFCENTY':center[1]})
            hdulist[0].header.update({'CRPIX1':center[0],'CRPIX2':center[1]})
            hdulist[0].header.add_history("Image recentered to {0}".format(str(center)))
        #if scaling is None:
        #    scaling = self.scaling

        # store the wavelength solution in a TableHDU
        wvs_hdu = fits.TableHDU.from_columns([fits.Column(name='wavelength',
                                                          format='E',
                                                          array=self.wvs)],
                                             name='wavelengths')

        hdulist.append(wvs_hdu)
        # store all headers from the source files
        for i, ext in enumerate(self.exthdrs):
            hdulist.append(fits.ImageHDU(header=ext, name="Data_{0:02d}".format(i)))


        try:
            hdulist.writeto(filepath, overwrite=True)
        except TypeError:
            hdulist.writeto(filepath, clobber=True)
        hdulist.close()

    def calibrate_output(self, units="contrast"):
        """
        Calibrates the flux of the output of PSF subtracted data.

        Assumes self.output exists and has shape (b,N,y,x) for N is the number of images and b is
        number of KL modes used.

        Args:
            units: currently only support "contrast" w.r.t central star

        Returns:
            stores calibrated data in self.output
        """
        if units == "contrast":
            # assemble median core
            try:
                assert(self._corefilenames is not None)
            except AssertionError:
                "No core files = no calibration possible"
                return
            if len(self._corefilenames) == 1:
                core_files = [self._corefilenames]
            core_hdus = [fits.open(f) for f in self._corefilenames]
            core_cubes = np.array([core_hdus.data for hdu in core_hdus])
            star_psf = P1640cores.make_median_core(core_cubes)
            for hdu in core_hdus:
                hdu.close()

            self.contrast_scaling = star_flux = np.nansum(axis=0)


            self.output *= self.contrast_scaling
            self.flux_units = "contrast"
            self._corefilenames = core_files
            self._core_psf = star_psf


    def generate_psfs(self, boxrad=7, gauss=True, mirror_adjust=1.0):
        """
        Generates PSF for each frame of input data. Only works on spectral mode data.
        Args:
            boxrad: the halflength of the size of the extracted PSF (in pixels)
            # spotyx: Ncube x Nchan x 4 x 2 array of spot (row, col) positions
            gauss [False]: if True, use a gaussian PSF
            mirror_adjust [1.0]: multiply mirror diameter by this factor to adjust for Lyot stop
        Returns:
            saves PSFs to self.psfs as an array of size(N,psfy,psfx) where psfy=psfx=2*boxrad + 1
        """
        self.psfs = []

        # spots initially have shape (Ncube x Nchan x Nspots x 2)
        nframes = reduce(lambda x,y: x*y, self.spot_locations.shape[:-2])

        spot0 = self.spot_locations[..., 0,:].reshape(nframes, 2)
        spot1 = self.spot_locations[..., 1,:].reshape(nframes, 2)
        spot2 = self.spot_locations[..., 2,:].reshape(nframes, 2)
        spot3 = self.spot_locations[..., 3,:].reshape(nframes, 2)

        if gauss == True: # simulate the PSF as a gaussian
            stamp = 2*boxrad + 1
            uniqwvs = np.unique(self.wvs)
            nl = uniqwvs.size
            radial_psfs = np.zeros((nl, stamp, stamp))
            telD = self.config.getfloat('observatory','primary_diam')
            telD *= mirror_adjust # undersize by 10% for Lyot stop
            for wv, lam in enumerate(uniqwvs):
                # Calculate lam/D in pixels - first convert wavelength to [m]
                # lam[m] / D[m] is in radians -- convert to arcsec
                fwhm_arcsec = 1.22*((lam*1.0e-6)/telD) * (3600*180/np.pi)
                # convert to pixels with ifs_lenslet_scale
                fwhm = fwhm_arcsec/self.config.getfloat('instrument','ifs_lenslet_scale')
                # Gaussian standard deviation - from another routine
                sigma = fwhm/(2.*np.sqrt(2*np.log(2)))
                #centered in the array
                y,x = np.indices([stamp, stamp])
                y -= stamp // 2
                x -= stamp // 2
                radial_psfs[wv,...] = np.exp(-(x**2. + y**2.) / (2. * sigma**2))
            self.psfs = radial_psfs / np.mean(radial_psfs.sum(axis=0))
        else: # use the spots to build the psf
            for i,frame in enumerate(self.input):
                spots = [[float(spot0[i, 0]), float(spot0[i, 1])], [float(spot1[i, 0]), float(spot1[i, 1])],
                         [float(spot2[i, 0]), float(spot2[i, 1])], [float(spot3[i, 0]), float(spot3[i, 1])]]
                #now make a psf
                spotpsf = generate_psf(frame, spots, boxrad=boxrad)
                self.psfs.append(spotpsf)
            self.psfs = np.array(self.psfs)
            self.psfs = np.reshape(self.psfs, (self.psfs.shape[0]//self.nchannels_used, self.nchannels_used,
                                               self.psfs.shape[1], self.psfs.shape[2]))
            self.psfs = np.mean(self.psfs, axis=0)

    def generate_psf_cube(self, boxw=14):
        return generate_psfs(int(np.floor(boxw//2)))
    """
    def generate_psf_cube(self, boxw=14):
        \"""
        # P1640 - use unocculted files
        Generates an average PSF from all frames of input data. Only works on spectral mode data.
        Overall cube normalized to unity with norm 2.
        Currently hard coded assuming 37 spectral channels!!!

        The center of the PSF is exactly on the central pixel of the PSF.
        (If even width of the array it is the middle pixel with the highest row and column index.)
        The center pixel index is always (nx/2,nx/2) assuming integer division.

        Args:
            boxw: the width the extracted PSF (in pixels). Should be bigger than 12 because there is an interpolation
                of the background by a plane which is then subtracted to remove linear biases.

        Returns:
            A cube of shape Nchan*boxw*boxw. Each slice [k,:,:] is the PSF for a given wavelength.
        \"""

        n_frames,ny,nx = self.input.shape
        x_grid, y_grid = np.meshgrid(np.arange(ny), np.arange(nx))
        unique_wvs = np.unique(self.wvs)
        numwaves = np.size(self.channels_used)  # np.unique(self.wvs))

        psfs = np.zeros((numwaves,boxw,boxw,n_frames,4))

        # spots initially have shape (Ncube x Nchan x Nspots x 2)
        nframes = reduce(lambda x,y: x*y, self.spot_locations.shape[:-2])
        spot0 = self.spot_locations[..., 0,:].reshape(nframes, 2)
        spot1 = self.spot_locations[..., 1,:].reshape(nframes, 2)
        spot2 = self.spot_locations[..., 2,:].reshape(nframes, 2)
        spot3 = self.spot_locations[..., 3,:].reshape(nframes, 2)

        scale_factors = self.scale_factors.ravel()
        for lambda_ref_id, lambda_ref in enumerate(self.channels_used):
            for i,frame in enumerate(self.input):
                #figure out which header and which wavelength slice
                #put all the sat spot info together
                spots = [[float(spot0[i, 0]), float(spot0[i, 1])], [float(spot1[i, 0]), float(spot1[i, 1])],
                         [float(spot2[i, 0]), float(spot2[i, 1])], [float(spot3[i, 0]), float(spot3[i, 1])]]
                #mask nans
                cleaned = np.copy(frame)
                cleaned[np.where(np.isnan(cleaned))] = 0

                for loc_id, loc in enumerate(spots):
                    #grab satellite spot positions
                    spotx = loc[0]
                    spoty = loc[1]
                    xarr_spot = np.round(spotx)
                    yarr_spot = np.round(spoty)
                    stamp = cleaned[(yarr_spot-np.floor(boxw/2.0)):(yarr_spot+np.ceil(boxw/2.0)),(xarr_spot-np.floor(boxw/2.0)):(xarr_spot+np.ceil(boxw/2.0))]
                    #x_stamp = x_grid[(yarr_spot-boxw/2):(yarr_spot+boxw/2),(xarr_spot-boxw/2):(xarr_spot+boxw/2)]
                    #y_stamp = y_grid[(yarr_spot-boxw/2):(yarr_spot+boxw/2),(xarr_spot-boxw/2):(xarr_spot+boxw/2)]
                    #print(spotx,spoty)
                    #print(stamp_x+ spotx-xarr_spot,stamp_y+spoty-yarr_spot)
                    stamp_x, stamp_y = np.meshgrid(np.arange(boxw, dtype=np.float32), np.arange(boxw, dtype=np.float32))
                    dx = spotx-xarr_spot
                    dy = spoty-yarr_spot
                    #stamp_x += spotx-xarr_spot
                    #stamp_y += spoty-yarr_spot
                    #stamp_x -= spotx-xarr_spot
                    #stamp_y -= spoty-yarr_spot
                    #print(spotx-xarr_spot,spoty-yarr_spot)


                    #mask the central blob to calculate background median
                    stamp_r = np.sqrt((stamp_x-dx-boxw/2)**2+(stamp_y-dy-boxw/2)**2)
                    stamp_masked = copy(stamp)
                    stamp_x_masked = stamp_x-dx
                    stamp_y_masked = stamp_y-dy
                    stamp_center = np.where(stamp_r<4)
                    stamp_masked[stamp_center] = np.nan
                    stamp_x_masked[stamp_center] = np.nan
                    stamp_y_masked[stamp_center] = np.nan
                    background_med =  np.nanmedian(stamp_masked)
                    stamp_masked -= background_med
                    #Solve 2d linear fit to remove background
                    xx = np.nansum(stamp_x_masked**2)
                    yy = np.nansum(stamp_y_masked**2)
                    xy = np.nansum(stamp_y_masked*stamp_x_masked)
                    xz = np.nansum(stamp_masked*stamp_x_masked)
                    yz = np.nansum(stamp_y_masked*stamp_masked)
                    #Cramer's rule
                    a = (xz*yy-yz*xy)/(xx*yy-xy*xy)
                    b = (xx*yz-xy*xz)/(xx*yy-xy*xy)
                    stamp -= a*(stamp_x-dx)+b*(stamp_y-dy) + background_med
                    #stamp -= background_med

                    #rescale to take into account wavelength widening
                    if 1:
                        stamp_r = np.sqrt((stamp_x-dx-boxw/2)**2+(stamp_y-dy-boxw/2)**2)
                        stamp_th = np.arctan2(stamp_x-dx-boxw/2,stamp_y-dy-boxw/2)
                        stamp_r /= scale_factors[i]  # lambda_ref/lambda_curr
                        stamp_x = stamp_r*np.cos(stamp_th)+boxw/2
                        stamp_y = stamp_r*np.sin(stamp_th)+boxw/2
                        #print(stamp_x,stamp_y)

                    stamp = ndimage.map_coordinates(stamp, [stamp_y+dx, stamp_x+dy])
                    #print(stamp)
                    psfs[lambda_ref_id,:,:,i,loc_id] = stamp


        #PSF_cube = np.mean(psfs[:,:,:,:,0],axis=(3))
        PSF_cube = np.mean(psfs,axis=(3,4))

        #Build the spectrum of the sat spots
        # Number of cubes in dataset
        N_cubes = int(self.input.shape[0])/int(numwaves)
        N_chans = np.size(self.channels_all)
        all_sat_spot_spec = np.zeros((N_chans, N_cubes))
        for k in range(N_cubes):
            all_sat_spot_spec[:,k] = self.spot_flux[N_chans*k: N_chans*(k+1)]
        sat_spot_spec = np.nanmean(all_sat_spot_spec,axis=1)

        PSF_cube /= np.sqrt(np.nansum(PSF_cube**2))
        for l in range(numwaves):
            #PSF_cube[l,:,:] -= np.nanmedian(PSF_cube[l,:,:][stamp_center])
            PSF_cube[l,:,:] *= sat_spot_spec[l]/np.nanmax(PSF_cube[l,:,:])
            PSF_cube[l,:,:][np.where(abs(PSF_cube[l,:,:])/np.nanmax(abs(PSF_cube[l,:,:]))<0.05)] = 0.0

        if 0:
            import matplotlib.pyplot as plt
            plt.figure(1)
            plt.plot(sat_spot_spec,'or')
            plt.plot(np.nanmax(PSF_cube,axis=(1,2)),"--b")
            plt.show()
        if 0: # for debugging purposes
            import matplotlib.pyplot as plt
            plt.figure(1)
            plt.imshow(PSF_cube[0,:,:],interpolation = 'nearest')
            plt.figure(2)
            plt.imshow(PSF_cube[36,:,:],interpolation = 'nearest')
            plt.show()

        self.psfs = PSF_cube
    """
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
            stamp_r = np.sqrt((stamp_x - nx/2)**2+(stamp_y - nx/2)**2)
            stamp_x_hd, stamp_y_hd = np.meshgrid(np.arange(nx_hd, dtype=np.float32)/(nx_hd-1)*(nx-1), np.arange(nx_hd, dtype=np.float32)/(nx_hd-1)*(nx-1))
            for l in range(nl):
                hd_psf[l,:,:] = ndimage.map_coordinates(self.psfs[l,:,:], [stamp_y_hd, stamp_x_hd])
                #hd_psf[l,nx/2*k_hd,nx/2*k_hd] = 0. # center
            stamp_r_hd = np.sqrt((stamp_x_hd-stamp_x_hd[nx/2*k_hd,nx/2*k_hd])**2+(stamp_y_hd-stamp_y_hd[nx/2*k_hd,nx/2*k_hd])**2)

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
            print("Wrong size of the PSFs stored in p1640 dataset structure when calling get_radial_psf. Return 0")
            return 0

######################
## Static Functions ##
######################
def get_p1640_spot_filepaths(config, data_filepath):
    """
    Look to see if the spot positions have already been written to file.
    If the files exist, return their paths; otherwise return None.
    Input:
        config: a ConfigParser object with the file path information
        data_filepath: the name of the P1640 data file whose spots you want to locate
    Returns:
        filepath: a path to the files (None if they don't exist)
    """
    try:
        spot_filedir = config.get("spots","spot_file_path")
        spot_filepostfix = config.get("spots","spot_file_postfix")
        spot_fileext = config.get("spots", "spot_file_ext")
    except ConfigParser.Error as e:
        print("Spot file path not found in P1640 config file: {0}".format(e.message))
        raise e

    spot_filebasename = os.path.splitext(os.path.basename(data_filepath))[0] + spot_filepostfix
    spot_fullpath = os.path.join(spot_filedir, spot_filebasename)
    spot_filepaths = [spot_fullpath + "{0}".format(i)+spot_fileext for i in range(4)]
    return spot_filepaths

def write_p1640_spots_to_file(config, data_filepath, spot_positions,
                              spot_directory=None, overwrite=True):
    """
    EDIT: NOW THIS IS JUST A WRAPPER FOR THE P1640spots.write_spots_to_file() METHOD

    Write the spot (row, col) positions to 4 files (1 per spot) in the directory specified
    in the config file.
    Input:
        config: a ConfigParser object with the file path information
        data_filepath: source file, spot files will have same prefix
        spot_positions: a Nspot x Nchan x 2 array of (row, col) spot positions
        spot_directory: (None) Directory to save the spot files
        overwrite: True -> overwrite (default), False -> don't overwrite existing files
    Output:
        None
    """
    if spot_directory is None:
        spot_directory = config.get("spots","spot_file_path")
    spot_filepostfix = config.get("spots","spot_file_postfix")
    spot_fileext = config.get("spots", "spot_file_ext")
    P1640spots.write_spots_to_file(data_filepath, spot_positions,
                                   spot_directory, overwrite,
                                   spot_filepostfix, spot_fileext)
    return

def _p1640_process_file(filepath, spot_directory=None, skipslices=None, highpass=True,
                        numthreads=-1, psfs_func_list=None, verbose=False):
    """
    Method to open and parse a P1640 file

    Args:
        filepath: the file to open
        spot_directory: path to folder were spot positions are stored (defaults to P1640.ini)
        skipslices: a list of datacube slices to skip (supply index numbers e.g. [0,1,2,3])
        highpass: if True, run a Gaussian high pass filter (default size is sigma = imgsize/10)
                  can also be a number specifying FWHM of box in pixel units
        numthreads: : Number of threads to be used. Default -1 sequential sat spot flux calc.
                        If None, numthreads = mp.cpu_count().
        PSF_cube: 3D array (nl,ny,nx) with the PSF cube to be used in the flux calculation.
        verbose [False]: if True, print more stuff
    Returns: (using z as size of 3rd dimension, z=32 for spec including all wavelengths)
        cube: 3D data cube from the file. Shape is (z,281,281)
        center: array of shape (z,2) giving each datacube slice a [xcenter,ycenter] in that order
        parang: array of z of the parallactic angle of the target (same value just repeated z times b/c SDI)
        wvs: array of z of the wavelength of each datacube slice. (For pol mode, wvs = [None])
        astr_hdrs: array of z of the WCS header for each datacube slice
        filt_band: the band (Y, J, H, K1, K2) used in the IFS Filter (string) NOT USED
        fpm_band: which coronagrpah was used (string) NOT USED
        ppm_band: which apodizer was used (string) NOT USED
        spot_fluxes: array of z containing average satellite spot fluxes for each image NOT USED
        spot_positions: array of (row, col) spot positions for each slice
        prihdr: primary header of the FITS file NOT USED
        exthdr: 1st extention header of the FITS file
    """
    if verbose == True:
        print("Reading File: {0}".format(filepath))
    hdulist = fits.open(filepath)
    try:
        #grab the data and headers
        cube = hdulist[0].data
        exthdr = hdulist[0].header
        prihdr = None #hdulist[0].header

        #get some instrument configuration from the primary header
        # GPI relics, kept for compatibility
        filt_band ='H' # None #prihdr['IFSFILT'].split('_')[1]
        fpm_band = 'H' #None #prihdr['OCCULTER'].split('_')[1]
        ppm_band = 'H' #None #prihdr['APODIZER'].split('_')[1] #to determine sat spot ratios

        #grab the astro header
        w = wcs.WCS(header=exthdr, naxis=[1,2])
        #turns out WCS data can be wrong. Let's recalculate it using avparang
        parang = 0 # for P1640 #exthdr['AVPARANG']
        # WCS information not saved by P1640
        vert_angle = -(360-parang) + P1640Data.ifs_rotation - 90
        vert_angle = np.radians(vert_angle)
        pc = np.array([[np.cos(vert_angle), np.sin(vert_angle)],[-np.sin(vert_angle), np.cos(vert_angle)]])
        cdmatrix = pc * P1640Data.lenslet_scale /3600.
        # P1640 workaround
        w.wcs.cd = cdmatrix
        """ old version
        w.wcs.cd[0,0] = cdmatrix[0,0]
        w.wcs.cd[0,1] = cdmatrix[0,1]
        w.wcs.cd[1,0] = cdmatrix[1,0]
        w.wcs.cd[1,1] = cdmatrix[1,1]
        """

        channels = exthdr['NAXIS3']
        wvs = P1640spots.P1640params.wlsol #get wavelength solution
        # calculate centers from satellite spots
        # first, check if spot positions have been stored on disk
        # build the path
        spot_locations = []
        try:
            if spot_directory is not None:
                spot_filedir = spot_directory
                if verbose == True:
                    print("Spot directory: {s}".format(s=spot_filedir))
            else: # use the default set in P1640.ini
                spot_filedir = P1640Data.config.get("spots","spot_file_path")
                #spot_filepaths = get_p1640_spot_filepaths(P1640Data.config, filepath)
            spot_filebasename = os.path.splitext(os.path.basename(filepath))[0]
            spot_fullpath = os.path.join(spot_filedir, spot_filebasename)
            spot_filepaths = sorted(glob.glob(spot_fullpath+'*'))

            # check if all the spot files exist, if so, read them in
            exist = np.all([os.path.isfile(f) for f in spot_filepaths])
            assert(exist is not False)
            if verbose == True:
                print("Reading spots from files: {0}".format(os.path.commonprefix(spot_filepaths)))
            spot_locations = np.array([np.genfromtxt(f, delimiter=',') 
                                       for f in spot_filepaths])
        except AssertionError:
            # if they haven't already been written to file, calculate them
            spot_locations = P1640spots.get_single_cube_spot_positions(cube)
            # write them to disk so they don't have to be recalculated
            write_p1640_spots_to_file(P1640Data.config, filepath, spot_locations)
        spot_fluxes = P1640spots.get_single_cube_spot_photometry(cube, spot_locations)
        scale_factors, center = P1640spots.get_scaling_and_centering_from_spots(spot_locations,
                                                                                mean_scaling=False)

        #parang = np.repeat(exthdr['AVPARANG'], channels) #populate PA for each wavelength slice (the same)
        parang = np.repeat(0, channels) #populate PA for each wavelength slice (the same)
        astr_hdrs = [w.deepcopy() for i in range(channels)] #repeat astrom header for each wavelength slice
    finally:
        hdulist.close()

    scale_factors = np.squeeze(scale_factors).mean(axis=-2)
    spot_fluxes = np.mean(spot_fluxes, axis=-2)
    
    #remove undesirable slices of the datacube if necessary
    if skipslices is not None:
        cube = np.delete(cube, skipslices, axis=0)
        center = np.delete(center, skipslices, axis=0)
        parang = np.delete(parang, skipslices)
        wvs = np.delete(wvs, skipslices)
        astr_hdrs = np.delete(astr_hdrs, skipslices)
        spot_fluxes = np.delete(spot_fluxes, skipslices, axis=0)
        spot_locations = np.delete(spot_locations, skipslices, axis=1)
        scale_factors = np.delete(scale_factors, skipslices, axis=0)
    highpassed = False
    if isinstance(highpass, bool):
        if highpass:
            cube = high_pass_filter_imgs(cube)
            highpassed = True
    else:
        # should be a number
        if isinstance(highpass, (float, int)):
            fourier_sigma_size = (cube.shape[1]/(highpass)) / (2*np.sqrt(2*np.log(2)))
            cube = high_pass_filter_imgs(cube, filtersize=fourier_sigma_size)
            highpassed = True

    # pyklip centers need to be [x,y] instead of (row, col)
    #print(center.shape)
    center = np.fliplr(center) # [0] because of the way P1640spots works
    return cube, center, scale_factors, spot_locations, parang, wvs, astr_hdrs, filt_band, fpm_band, ppm_band, spot_fluxes, prihdr, exthdr


def generate_psf(frame, locations, boxrad=5, medianboxsize=30):
    """
    Generates a P1640 PSF for the frame based on the satellite spots

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
    #mask source for median filter
    masked = np.copy(cleaned)
    for loc in locations:
        spotx = np.int(np.round(loc[0]))
        spoty = np.int(np.round(loc[1]))
        masked[spotx-boxrad:spotx+boxrad+1, spoty-boxrad:spoty+boxrad+1] = np.nanmedian(
            masked.reshape(masked.shape[0]*masked.shape[1]))
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
        genpsf.append(spotpsf)

    genpsf = np.array(genpsf)
    genpsf = np.mean(genpsf, axis=0) #average the different psfs together
    return genpsf


def rescale_wvs(exthdrs, wvs, refwv=18, skipslices=None):
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
    wv_indicies = range(0, exthdrs[0]['NAXIS3'])
    if skipslices is not None:
        wv_indicies = np.delete(wv_indicies, skipslices)
    sats = np.array([[[h['SATS{0}_{1}'.format(i,j)].split() for i in wv_indicies]
                          for j in range(0,4)] for h in exthdrs], dtype=np.float)
    sats = sats.mean(axis=0)
    pairs = [(0,3), (1,2)]
    separations = np.mean([0.5*np.sqrt(np.diff(sats[p,:,0], axis=0)[0]**2 + np.diff(sats[p,:,1], axis=0)[0]**2) 
                           for p in pairs], 
                          axis=0) # average over each pair, the first axis
    scaling_factors = separations/separations[refwv]
    scaled_wvs = scaling_factors*wvs[refwv]
    return np.tile(scaled_wvs, len(exthdrs))


