import abc
import os
import subprocess
import multiprocessing as mp
import numpy as np
import astropy.io.fits as fits
import pyklip.klip as klip

class Data(object):
    """
    Abstract Class with the required fields and methods that need to be implemented

    Attributes:
        input: Array of shape (N,y,x) for N images of shape (y,x)
        centers: Array of shape (N,2) for N input centers in the format [x_cent, y_cent]
        filenums: Array of size N for the numerical index to map data to file that was passed in
        filenames: Array of size N for the actual filepath of the file that corresponds to the data
        PAs: Array of N for the parallactic angle rotation of the target (used for ADI) [in degrees]
        wvs: Array of N wavelengths of the images (used for SDI) [in microns]. For polarization data, defaults to "None"
        wcs: Array of N wcs astormetry headers for each input image.
        IWA: a floating point scalar (not array). Specifies to inner working angle in pixels
        OWA: (optional) specifies outer working angle in pixels
        output: Array of shape (b, len(files), len(uniq_wvs), y, x) where b is the number of different KL basis cutoffs
        output_centers: Array of shape (N,2) for N output centers. Also coresponds to FM centers (does not need to be implemented)
        output_wcs: Array of N wcs astrometry headers for each output image (does not need to be implemneted)
        creator: (optional) string for creator of the data (used to identify pipelines that call pyklip)
        klipparams: (optional) a string that saves the most recent KLIP parameters
        flipx: (optional) True by default. Determines whether a relfection about the x axis is necessary to rotate image North-up East left

    Methods:
        readdata(): reread in the dadta
        savedata(): save a specified data in the GPI datacube format (in the 1st extension header)
        calibrate_output(): flux calibrate the output data
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        # set field for the creator of the data (used for pipeline work)
        self.creator = None
        # set field for klip parameters
        self.klipparams = None
        # set the outer working angle (optional parameter)
        self.OWA = None
        # determine whether a reflection is needed for North-up East-left (optional)
        self.flipx = True
        # self output centers and wcs to None until after running KLIP
        self.output_centers = None
        self.output_wcs = None


    ###################################
    ### Required Instance Variances ###
    ###################################

    #Note that each field has a getter and setter method so by default they are all read/write

    @abc.abstractproperty
    def input(self):
        """
        Input Data. Shape of (N, y, x)
        """
        return
    @input.setter
    def input(self, newval):
        return

    @abc.abstractproperty
    def centers(self):
        """
        Image centers. Shape of (N, 2) where the 2nd dimension is [x,y] pixel coordinate (in that order)
        """
        return
    @centers.setter
    def centers(self, newval):
        return

    @abc.abstractproperty
    def filenums(self):
        """
        Array of size N for the numerical index to map data to file that was passed in
        """
        return
    @filenums.setter
    def filenums(self, newval):
        return

    @abc.abstractproperty
    def filenames(self):
        """
        Array of size N for the actual filepath of the file that corresponds to the data
        """
        return
    @filenames.setter
    def filenames(self, newval):
        return


    @abc.abstractproperty
    def PAs(self):
        """
        Array of N for the parallactic angle rotation of the target (used for ADI) [in degrees]
        """
        return
    @PAs.setter
    def PAs(self, newval):
        return


    @abc.abstractproperty
    def wvs(self):
        """
        Array of N wavelengths (used for SDI) [in microns]. For polarization data, defaults to "None"
        """
        return
    @wvs.setter
    def wvs(self, newval):
        return


    @abc.abstractproperty
    def wcs(self):
        """
        Array of N wcs astormetry headers for each image.
        """
        return
    @wcs.setter
    def wcs(self, newval):
        return


    @abc.abstractproperty
    def IWA(self):
        """
        a floating point scalar (not array). Specifies to inner working angle in pixels
        """
        return
    @IWA.setter
    def IWA(self, newval):
        return


    @abc.abstractproperty
    def output(self):
        """
        Array of shape (b, len(files), len(uniq_wvs), y, x) where b is the number of different KL basis cutoffs
        """
        return
    @output.setter
    def output(self, newval):
        return


    # not an abstract property
    @property
    def numwvs(self):
        if not hasattr(self, "_numwvs"):
            self._numwvs = int(np.size(np.unique(self.wvs)))
        return self._numwvs


    ########################
    ### Required Methods ###
    ########################
    @abc.abstractmethod
    def readdata(self, filepaths):
        """
        Reads in the data from the files in the filelist and writes them to fields
        """
        return NotImplementedError("Subclass needs to implement this!")

    @staticmethod
    @abc.abstractmethod
    def savedata(self, filepath, data, klipparams=None, filetype="", zaxis=None, more_keywords=None):
        """
        Saves data for this instrument

        Args:
            filepath: filepath to save to
            data: data to save
            klipparams: a string of KLIP parameters. Write it to the 'PSFPARAM' keyword
            filtype: type of file (e.g. "KL Mode Cube", "PSF Subtracted Spectral Cube"). Wrriten to 'FILETYPE' keyword
            zaxis: a list of values for the zaxis of the datacub (for KL mode cubes currently)
            more_keywords (dictionary) : a dictionary {key: value, key:value} of header keywords and values which will
                                         written into the primary header
        """
        return NotImplementedError("Subclass needs to implement this!")

    @abc.abstractmethod
    def calibrate_output(self, img, spectral=False):
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

        Return:
            calib_img: calibrated image of the same shape
        """
        return NotImplementedError("Subclass needs to implement this!")

    def spectral_collapse(self, collapse_channels=1, align_frames=True, numthreads=None, additional_params=None):
        """
        Collapses the dataset spectrally, bining the data into the desired number of output wavelengths. 
        This bins each cube individually; it does not bin the data tempoarally. 
        If number of wavelengths / output channels is not a whole number, some output channels will have more frames
        that went into the collapse

        Args:
            collapse_channels (int): number of output channels to evenly-ish collapse the dataset into. Default is 1 (broadband)
            align_frames (bool): if True, aligns each channel before collapse so that they are centered properly
            numthreads (bool,int): number of threads to parallelize align and scale. If None, use default which is all of them
            additional_params (list of str): other dataset parameters to collapse. Assume each variable has first dimension of Nframes
        """
        # reshpae input into 4D cube
        Ncubes = self.input.shape[0] // self.numwvs
        input_4d = self.input.reshape([Ncubes, self.numwvs, self.input.shape[1], self.input.shape[2]])

        slices_per_group = self.numwvs // collapse_channels # how many wavelengths per each output channel
        leftover_slices = self.numwvs % collapse_channels

        collapsed_4d = np.zeros([Ncubes, collapse_channels, self.input.shape[1], self.input.shape[2]])
        wvs_collapsed = np.zeros([Ncubes, collapse_channels])
        pas_collapsed = np.zeros([Ncubes, collapse_channels])
        centers_collapsed = np.zeros([Ncubes, collapse_channels, 2])
        # appending following as lists
        wcs_collapsed = [] 
        filenums_collapsed = []
        filenames_collapsed = []
        # additional params, if needed
        if additional_params is not None:
            additional_collapsed = []
            for param_field in additional_params:
                param_orig = getattr(self, param_field)
                reshaped_shape = (Ncubes, collapse_channels) + param_orig.shape[1:]
                additional_collapsed.append(np.zeros(reshaped_shape))
        # populate the output image
        next_start_channel = 0 # initialize which channel to start with for the input images
        for i in range(collapse_channels):
            # figure out which slices to pick
            slices_this_group = slices_per_group
            if leftover_slices > 0:
                # take one extra slice, yummy
                slices_this_group += 1
                leftover_slices -= 1

            i_start = next_start_channel
            i_end = next_start_channel + slices_this_group # this is the index after the last one in this group

            if align_frames:
                tpool = mp.Pool(processes=numthreads)

                # for this range of wvs, one (x,y) center per cube
                centers_4d = self.centers.reshape([Ncubes, self.numwvs, 2])
                mean_centers =  np.mean(centers_4d[:,i_start:i_end,:], axis=1)

                tasks = [tpool.apply_async(klip.align_and_scale, args=(img, old_center, new_center))        
                         for cube_j, new_center in enumerate(mean_centers)
                          for img, old_center in zip(input_4d[cube_j, i_start:i_end], centers_4d[cube_j, i_start:i_end])
                        ]

                # reform back into a giant array
                derotated = np.array([task.get() for task in tasks])
                derotated.shape = (Ncubes, slices_this_group, self.input.shape[1], self.input.shape[2])
                input_4d[:, i_start:i_end, :, :] = derotated


            collapsed_4d[:,i,:,:] = np.nanmean(input_4d[:,i_start:i_end,:,:], axis=1)
            wvs_collapsed[:, i] = np.mean(self.wvs.reshape([Ncubes, self.numwvs])[:,i_start:i_end], axis=1)
            pas_collapsed[:, i] = np.mean(self.PAs.reshape([Ncubes, self.numwvs])[:,i_start:i_end], axis=1)
            centers_collapsed[:,i,:] = np.mean(self.centers.reshape([Ncubes, self.numwvs, 2])[:,i_start:i_end,:], axis=1)
            # append arrays, we'll reshape them later
            # these variables are all the same for a single cube, so we can just select one
            wcs_collapsed.append(self.wcs.reshape([Ncubes, self.numwvs])[:,i_start]) 
            filenums_collapsed.append(self.filenums.reshape([Ncubes, self.numwvs])[:,i_start]) 
            filenames_collapsed.append(self.filenames.reshape([Ncubes, self.numwvs])[:,i_start])

            if additional_params is not None:
                for param_collapsed, param_field in zip(additional_collapsed, additional_params):
                    param_orig = getattr(self, param_field)
                    reshaped_shape = (Ncubes, self.numwvs) + param_orig.shape[1:]
                    param_collapsed[:, i] = np.nanmean(param_orig.reshape(reshaped_shape)[:, i_start:i_end], axis=1)

            next_start_channel = i_end

        # unravel the wavelength information
        collapsed_4d.shape = [Ncubes * collapse_channels, self.input.shape[1], self.input.shape[2]]
        wvs_collapsed.shape = [Ncubes * collapse_channels]
        pas_collapsed.shape = [Ncubes * collapse_channels]
        centers_collapsed.shape = [Ncubes * collapse_channels, 2]

        # unfold the lists, need to flip the dimensions, so they are ordered properly
        wcs_collapsed = np.array(wcs_collapsed).T.ravel()
        filenums_collapsed = np.array(filenums_collapsed).T.ravel()
        filenames_collapsed = np.array(filenames_collapsed).T.ravel()

        # ok time to set all the variables correctly
        self._numwvs = collapse_channels
        self.input = collapsed_4d
        self.wvs = wvs_collapsed
        self.PAs = pas_collapsed
        self.centers = centers_collapsed
        self.wcs = wcs_collapsed
        self.filenums = filenums_collapsed
        self.filenames = filenames_collapsed

        if additional_params is not None:
            for param_field, param_collapsed in zip(additional_params, additional_collapsed):
                param_collapsed.shape = (Ncubes * collapse_channels, ) + param_collapsed.shape[2:]
                setattr(self, param_field, param_collapsed)

class GenericData(Data):
    """
    Basic class to interface with a basic direct imaging dataset

    Args:
        input_data: either a 1-D list of filenames to read in, or a 3-D cube of all data (N, y, x)
        centers: array of shape (N,2) for N centers in the format [x_cent, y_cent]
        parangs: Array of N for the parallactic angle rotation of the target (used for ADI) [in degrees]
        wvs: Array of N wavelengths of the images (used for SDI) [in microns]. For polarization data, defaults to "None"
        IWA: a floating point scalar (not array). Specifies to inner working angle in pixels
        filenames: Array of size N for the actual filepath of the file that corresponds to the data

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
    """
    # Constructor
    def __init__(self, input_data, centers, parangs=None, wvs=None, IWA=0, filenames=None):
        super(GenericData, self).__init__()
        # read in the data
        if np.array(input_data).ndim == 1:
            self._input = self.readdata(input_data)
        else:
            # assume this is a 3-D cube
            self._input = np.array(input_data)
        
        nfiles = self.input.shape[0]

        self.centers = np.array(centers)

        if self.centers.shape [0] != nfiles:
            raise ValueError("Input data has shape {0} but centers has shape {1}".format(self.input.shape,
                                                                                         self.centers.shape))

        if parangs is not None:
            self._PAs = parangs
        else:
            self._PAs = np.zeros(nfiles)

        if wvs is not None:
            self._wvs = wvs
        else:
            self._wvs = np.ones(nfiles)

        self.IWA = IWA

        if filenames is not None:
            self._filenames = filenames
            unique_filenames = np.unique(filenames)                                                                                 
            self._filenums = np.array([np.argwhere(filename == unique_filenames).ravel()[0] for filename in filenames])
        else:
            self._filenums = np.arange(nfiles)
            self._filenames = np.array(["{0}".format(i) for i in self.filenums])

        self._wcs = np.array([None for _ in range(nfiles)])

        self._output = None


         
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
 

    def readdata(self, filepaths):
        """
        Reads in the data from the files in the filelist and writes them to fields.
        """
        input_data = []
        for filename in filepaths:
            with fits.open(filename) as hdulist:
                # assume the data is in the primary header
                data = hdulist[0].data
                # if this data has more than 2-D, collapse the Data
                dims = data.shape
                if np.size(dims) > 2:
                    nframes = np.prod(dims[:-2])
                    # collapse in all dimensions except y and x
                    data.shape = (nframes, dims[-2], dims[-1])

                input_data.append(data)

        # collapse data again
        input_data = np.array(input_data)
        dims = input_data.shape
        if np.szie(dims) > 3:
            nframes = np.prod(dims[:-2])
            # collapse in all dimensions except y and x
            input_data.shape = (nframes, dims[-2], dims[-1])


    def savedata(self, filepath, data, klipparams=None, filetype="", zaxis=None, more_keywords=None):
        """
        Saves data for this instrument

        Args:
            filepath: filepath to save to
            data: data to save
            klipparams: a string of KLIP parameters. Write it to the 'PSFPARAM' keyword
            filtype: type of file (e.g. "KL Mode Cube", "PSF Subtracted Spectral Cube"). Wrriten to 'FILETYPE' keyword
            zaxis: a list of values for the zaxis of the datacub (for KL mode cubes currently)
            more_keywords (dictionary) : a dictionary {key: value, key:value} of header keywords and values which will
                                         written into the primary header
        """
        hdulist = fits.HDUList()
        hdulist.append(fits.PrimaryHDU(data=data))

        # save all the files we used in the reduction
        # we'll assume you used all the input files
        # remove duplicates from list
        filenames = np.unique(self.filenames)
        nfiles = np.size(filenames)
        hdulist[0].header["DRPNFILE"] = (nfiles, "Num raw files used in pyKLIP")
        for i, filename in enumerate(filenames):
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
        hdulist[0].header['CREATOR'] = "pyKLIP-{0}".format(pyklipver)

        # store commit number for pyklip
        hdulist[0].header['pyklipv'] = (pyklipver, "pyKLIP version that was used")

        if klipparams is not None:
            hdulist[0].header['PSFPARAM'] = (klipparams, "KLIP parameters")
            hdulist[0].header.add_history("pyKLIP reduction with parameters {0}".format(klipparams))


        # write z axis units if necessary
        if zaxis is not None:
            # Writing a KL mode Cube
            if "KL Mode" in filetype:
                hdulist[0].header['CTYPE3'] = 'KLMODES'
                # write them individually
                for i, klmode in enumerate(zaxis):
                    hdulist[0].header['KLMODE{0}'.format(i)] = (klmode, "KL Mode of slice {0}".format(i))
                hdulist[0].header['CUNIT3'] = "N/A"
                hdulist[0].header['CRVAL3'] = 1
                hdulist[0].header['CRPIX3'] = 1.
                hdulist[0].header['CD3_3'] = 1.

        if "Spectral" in filetype:
            uniquewvs = np.unique(self.wvs)
            # do spectral stuff instead
            # because wavelength solutoin is nonlinear, we're not going to store it here
            hdulist[0].header['CTYPE3'] = 'WAVE'
            hdulist[0].header['CUNIT3'] = "N/A"
            hdulist[0].header['CRPIX3'] = 1.
            hdulist[0].header['CRVAL3'] = 0
            hdulist[0].header['CD3_3'] = 1
            # write it out instead
            for i, wv in enumerate(uniquewvs):
                hdulist[0].header['WV{0}'.format(i)] = (wv, "Wavelength of slice {0}".format(i))

        center = self.centers[0]
        hdulist[0].header.update({'PSFCENTX': center[0], 'PSFCENTY': center[1]})
        hdulist[0].header.update({'CRPIX1': center[0], 'CRPIX2': center[1]})
        hdulist[0].header.add_history("Image recentered to {0}".format(str(center)))

        if more_keywords is not None:
            hdulist[0].header.update(more_keywords)

        try:
            hdulist.writeto(filepath, overwrite=True)
        except TypeError:
            hdulist.writeto(filepath, clobber=True)
        hdulist.close()

    def calibrate_output(self, img, spectral=False):
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

        Return:
            calib_img: calibrated image of the same shape
        """
        return img