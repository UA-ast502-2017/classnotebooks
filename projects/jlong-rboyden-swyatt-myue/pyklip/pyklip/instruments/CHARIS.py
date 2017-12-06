import os
import numpy as np
import scipy.ndimage as ndimage
import astropy.io.fits as fits
import datetime
import astropy.time as time
import astropy.coordinates as coord
import astropy.units as u
import pyklip.klip as klip
from pyklip.instruments.Instrument import Data
import pyklip.fakes as fakes


class CHARISData(Data):
    """
    A sequence of GPI Data. Each GPIData object has the following fields and functions

    Args:
        filepaths: list of filepaths to files
        skipslices: a list of datacube slices to skip (supply index numbers e.g. [0,1,2,3])
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
        prihdrs: Array of N primary headers
        exthdrs: Array of N extension headers
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
    # some static variables to define the CHARIS instrument
    centralwave = {}  # in microns
    fpm_diam = {}  # in pixels
    flux_zeropt = {}
    spot_ratio = {} #w.r.t. central star
    lenslet_scale = 1.0 # arcseconds per pixel (pixel scale)
    ifs_rotation = 0.0  # degrees CCW from +x axis to zenith

    obs_latitude = 19 + 49./60 + 43./3600 # radians
    obs_longitude = -(155 + 28./60 + 50./3600) # radians

    ####################
    ### Constructors ###
    ####################
    def __init__(self, filepaths, guess_spot_index, guess_spot_locs, skipslices=None, 
                 PSF_cube=None, recalc_wvs=True, recalc_centers=True):
        """
        Initialization code for CHARISData

        Note:
            Argument information is in the GPIData class definition docstring
        """
        super(CHARISData, self).__init__()
        self._output = None
        self.flipx = False
        self.readdata(filepaths, guess_spot_index, guess_spot_locs, 
                      skipslices=skipslices, PSF_cube=PSF_cube, 
                      recalc_wvs=recalc_wvs, recalc_centers=recalc_centers)

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

    def readdata(self, filepaths, guess_spot_index, guess_spot_loc, skipslices=None, 
                 PSF_cube=None, recalc_wvs=True, recalc_centers=True):
        """
        Method to open and read a list of GPI data

        Args:
            filespaths: a list of filepaths
            guess_spot_index:
            guess_spot_loc:
            skipslices: a list of wavelenegth slices to skip for each datacube (supply index numbers e.g. [0,1,2,3])
            PSF_cube: 3D array (nl,ny,nx) with the PSF cube to be used in the flux calculation.
            recalc_wvs: if True, uses sat spot positions and the central wavelength to recalculate wavelength solution
            recalc_centers: if True, uses a least squares fit and the satellite spots to recalculate the img centers

        Returns:
            Technically none. It saves things to fields of the GPIData object. See object doc string
        """
        # check to see if user just inputted a single filename string
        if isinstance(filepaths, str):
            filepaths = [filepaths]

        # check that the list of files actually contains something
        if len(filepaths) == 0:
            raise ValueError("An empty filelist was passed in")

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
        spot_locs = []
        inttimes = []
        prihdrs = []
        exthdrs = []

        if PSF_cube is not None:
            dataset.psfs = PSF_cube

        #extract data from each file
        for index, filepath in enumerate(filepaths):
            with fits.open(filepath, lazy_load_hdus=False) as hdulist:
                cube = hdulist[1].data
                prihdr = hdulist[0].header
                exthdr = hdulist[0].header

            # mask pixels that receive no light as nans. Includ masking a 1 pix boundary around NaNs
            input_minfilter = ndimage.minimum_filter(cube, (0, 1, 1))
            cube[np.where(input_minfilter == 0)] = np.nan

            # recalculate parang if necessary
            parang = prihdr['PARANG']

            # compute weavelengths
            cube_wv_indices = np.arange(cube.shape[0])
            thiswvs = prihdr['LAM_MIN'] * np.exp(cube_wv_indices * prihdr['DLOGLAM'])


            #remove undesirable slices of the datacube if necessary
            if skipslices is not None:
                cube = np.delete(cube, skipslices, axis=0)
                thiswvs = np.delete(thiswvs, skipslices)
                wv_indices = np.delete(wv_indices, skipslices)


            print("Finding satellite spots for cube {0}".format(index))
            spot_loc, spot_flux, spot_fwhm = _measure_sat_spots(cube, thiswvs, guess_spot_index, guess_spot_loc)

            # simple mean for center for now
            center = np.mean(spot_loc, axis=1)

            data.append(cube)
            centers.append(center)
            spot_fluxes.append(spot_flux)
            spot_locs.append(spot_loc)
            rot_angles.append(np.ones(cube.shape[0], dtype=int) * parang)
            wvs.append(thiswvs)
            wv_indices.append(cube_wv_indices)
            filenums.append(np.ones(cube.shape[0], dtype=int) * index)
            wcs_hdrs.append([None for _ in range(cube.shape[0])])
            inttimes.append(np.ones(cube.shape[0], dtype=int) * prihdr['EXPTIME'])
            prihdrs.append(prihdr)
            exthdrs.append(exthdr)
            filenames.append([filepath for i in range(cube.shape[0])])



        #convert everything into numpy arrays
        #reshape arrays so that we collapse all the files together (i.e. don't care about distinguishing files)
        data = np.array(data)
        dims = data.shape
        data = data.reshape([dims[0] * dims[1], dims[2], dims[3]])
        filenums = np.array(filenums).reshape([dims[0] * dims[1]])
        filenames = np.array(filenames).reshape([dims[0] * dims[1]])
        rot_angles = (np.array(rot_angles).reshape([dims[0] * dims[1]])) 
        wvs = np.array(wvs).reshape([dims[0] * dims[1]])
        wv_indices = np.array(wv_indices).reshape([dims[0] * dims[1]])
        wcs_hdrs = np.array(wcs_hdrs).reshape([dims[0] * dims[1]])
        centers = np.array(centers).reshape([dims[0] * dims[1], 2])
        spot_fluxes = np.array(spot_fluxes).reshape([dims[0] * dims[1]])
        spot_locs = np.array(spot_locs).reshape([dims[0] * dims[1], 4, 2])
        inttimes = np.array(inttimes).reshape([dims[0] * dims[1]])

        # if there is more than 1 integration time, normalize all data to the first integration time
        if np.size(np.unique(inttimes)) > 1:
            inttime0 = inttime[0]
            # normalize integration times
            data = data * inttime0/inttimes[:, None, None]
            spot_fluxes *= inttime0/inttimes


        #set these as the fields for the GPIData object
        self._input = data
        self._centers = centers
        self._filenums = filenums
        self._filenames = filenames
        self._PAs = rot_angles
        self._wvs = wvs
        self._wcs = wcs_hdrs
        self._IWA = 5
        self.prihdrs = prihdrs
        self.exthdrs = exthdrs

        self.spot_fluxes = spot_fluxes
        self.spot_locs = spot_locs

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
            hdulist[0].header['FILETYPE'] = (filetype, "CHARIS File type")

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
        numwaves = np.size(np.unique(self.wvs))

        for i,frame in enumerate(self.input):
            this_spot_locs = self.spot_locs[i]

            # now grab the values from them by parsing the header
            spot0 = this_spot_locs[0]
            spot1 = this_spot_locs[1]
            spot2 = this_spot_locs[2]
            spot3 = this_spot_locs[3]

            # put all the sat spot info together
            spots = [spot0, spot1, spot2, spot3]

            #now make a psf
            spotpsf = generate_psf(frame, spots, boxrad=boxrad)
            self.psfs.append(spotpsf)

        self.psfs = np.array(self.psfs)

        # collapse in time dimension
        numwvs = np.size(np.unique(self.wvs))
        self.psfs = np.reshape(self.psfs, (self.psfs.shape[0]//numwvs, numwvs, self.psfs.shape[1], self.psfs.shape[2]))
        self.psfs = np.mean(self.psfs, axis=0)


def _measure_sat_spots(cube, wvs, guess_spot_index, guess_spot_locs, highpass=True):
    """
    Find sat spots in a datacube. TODO: return sat spot psf cube also
    """
    # use dictionary to store list of locs/fluxes for each slice
    spot_locs = {}
    spot_fluxes = {}
    spot_fwhms = {}

    # start with guess center
    start_frame = cube[guess_spot_index]
    if highpass:
        start_frame = klip.high_pass_filter(start_frame, 10)

    start_spot_locs = []
    start_spot_fluxes = []
    start_spot_fwhms = []
    for guess_spot_loc in guess_spot_locs:
        xguess, yguess = guess_spot_loc
        fitargs = fakes.gaussfit2d(start_frame, xguess, yguess, refinefit=True, searchrad=6)
        fitflux, fitfwhm, fitx, fity = fitargs
        start_spot_locs.append([fitx, fity])
        start_spot_fluxes.append(fitflux)
        start_spot_fwhms.append(fitfwhm)

    spot_locs[guess_spot_index] = start_spot_locs
    spot_fluxes[guess_spot_index] = np.nanmean(start_spot_fluxes)
    spot_fwhms[guess_spot_index] = np.nanmean(start_spot_fwhms)

    # set this reference center to use for finding the spots at other wavelengths
    ref_wv = wvs[guess_spot_index]
    ref_center = np.mean(start_spot_locs, axis=0)
    ref_spot_locs_deltas = np.array(start_spot_locs) - ref_center[None, :] # offset from center

    for i, (frame, wv) in enumerate(zip(cube, wvs)):
        # we already did the inital index
        if i == guess_spot_index:
            continue

        if highpass:
            frame = klip.high_pass_filter(frame, 10)

        # guess where the sat spots are based on the wavelength
        wv_scaling = wv/ref_wv # shorter wavelengths closer in
        thiswv_guess_spot_locs = ref_spot_locs_deltas * wv_scaling + ref_center

        # fit each sat spot now
        thiswv_spot_locs = []
        thiswv_spot_fluxes = []
        thiswv_spot_fwhms = []
        for guess_spot_loc in thiswv_guess_spot_locs:
            xguess, yguess = guess_spot_loc
            fitargs = fakes.gaussfit2d(frame, xguess, yguess, refinefit=True, searchrad=6)
            fitflux, fitfwhm, fitx, fity = fitargs
            thiswv_spot_locs.append([fitx, fity])
            thiswv_spot_fluxes.append(fitflux)
            thiswv_spot_fwhms.append(fitfwhm)

        spot_locs[i] = thiswv_spot_locs
        spot_fluxes[i] = np.nanmean(thiswv_spot_fluxes)
        spot_fwhms[i] = np.nanmean(thiswv_spot_fwhms)

    # turn them into numpy arrays
    locs = []
    fluxes = []
    fwhms = []
    for i in range(cube.shape[0]):
        locs.append(spot_locs[i])
        fluxes.append(spot_fluxes[i])
        fwhms.append(spot_fwhms[i])

    return np.array(locs), np.array(fluxes), np.array(fwhms)

def generate_psf(frame, locations, boxrad=5):
    """
    Generates a GPI PSF for the frame based on the satellite spots

    Args:
        frame: 2d frame of data
        location: array of (N,2) containing [x,y] coordinates of all N satellite spots
        boxrad: half length of box to use to pull out PSF

    Returns:
        genpsf: 2d frame of size (2*boxrad+1, 2*boxrad+1) with average PSF of satellite spots
    """
    genpsf = []
    #mask nans
    cleaned = np.copy(frame)
    cleaned[np.where(np.isnan(cleaned))] = 0


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
