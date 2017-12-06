__author__ = 'JB'

import os
from copy import copy
from glob import glob
from sys import stdout
import multiprocessing as mp
import itertools

import astropy.io.fits as pyfits
import numpy as np

from pyklip.kpp.utils.kppSuperClass import KPPSuperClass
from pyklip.instruments import GPI
import pyklip.kpp.utils.mathfunc as kppmath
import pyklip.spectra_management as spec


class Matchedfilter(KPPSuperClass):
    """
    Class calculating the matched filter of a 2D image or a 3D cube.
    """
    def __init__(self,read_func=None,filename=None,
                 folderName = None,
                 mute=None,
                 N_threads=None,
                 label = None,
                 overwrite=False,
                 SpT_file_csv=None,
                 kernel_type = None,
                 kernel_para = None,
                 PSF_read_func = None,
                 PSF_cube = None,
                 spectrum = None,
                 add2prefix = None,
                 keepPrefix = None,
                 compact_date_func = None,
                 filter_name_func = None,
                 PSF_size=None):
        """
        Define the general parameters of the matched filter.

        Args:
            read_func: lambda function treturning a instrument object where the only input should be a list of filenames
                    to read.
                    For e.g.:
                    read_func = lambda filenames:GPI.GPIData(filenames,recalc_centers=False,recalc_wvs=False,highpass=False)
            filename: Filename of the file to process.
                        It should be the complete path unless inputDir is used in initialize().
                        It can include wild characters. The files will be reduced as given by glob.glob().
            folderName: foldername used in the definition of self.outputDir (where files shoudl be saved) in initialize().
                        folderName could be the name of the spectrum used for the reduction for e.g.
                        Default folder name is "default_out".
                        Convention is self.outputDir = #outputDir#/kpop_#labe#/#folderName#/
            mute: If True prevent printed log outputs.
            N_threads: Number of threads to be used.
                        If None use mp.cpu_count().
                        If -1 do it sequentially.
                        Note that it is not used for this super class.
            label: label used in the definition of self.outputDir (where files shoudl be saved) in initialize().
                   Default is "default".
                   Convention is self.outputDir = #outputDir#/kpop_#labe#/#folderName#/
            overwrite: Boolean indicating whether or not files should be overwritten if they exist.
                       See check_existence().
            SpT_file_csv: Filename (.csv) of the table containing the target names and their spectral type.
                    Can be generated from quering Simbad.
                    If None (default), the function directly tries to query Simbad.
            kernel_type: String defining type of model to be used for the cross correlation:
                    - "PSF" (default): ?????? 
                            Use the PSF cube saved as fits with filename PSF_cube_filename.
                            There is no need to define PSF_cube_filename in the case of GPI campaign epochs folders.
                            Indeed it will try to find a PSF in inputDir (make sure there is only one PSF in there)
                            and if it can't find it it will try to generate one based on the raw spectral cubes in
                            inputDir. It uses the cubes listed in the extension header of pyklip images.
                            The PSF is flattened if the data is 2D.
                    - "hat": Define the kernel as a simple aperture photometry with radius kernel_para.
                            Default radius is 1.5 pixels.
                    - "Gaussian" (Default): define the kernel as a symmetric 2D gaussian with width (ie standard deviation) equal
                            to kernel_para. Default value of the width is 1.25.
                    - If kernel_type is a np.ndarray then kernel_type is the user defined template.
            kernel_para: Define the width of the Kernel depending on kernel_type. See kernel_type.
            PSF_read_func: lambda function used to read the PSF_cube.
                    It should return an object with at least two attributes input and wvs corresponding to the PSF cube and the associated wavelengths per slide.
                    The input shoudl be a list of filenames,
                    For e.g.: read_func = lambda filenames:GPI.GPIData(filenames,highpass=False)
            PSF_cube: Array or string defining the planet PSF if kernel_type == "PSF".
                      - np.ndarray: Should have the same number of dimensions as the image.
                      - string: Read file using read_func().
                            (make sure the file follows the conventions of the instrument).
                            - if absolute path: Simply picks up that file
                            - otherwise search for a matching file based on the directory of the image.
                      - None: use default value "*-original_radial_PSF_cube.fits"
            spectrum: spectrum name (string) or array used in the cube weighted mean when collapse is True.
                        - "host_star_spec": The spectrum from the star or the satellite spots is directly used.
                                            It is derived from the inverse of the calibrate_output() output.
                        - "constant": Use a constant spectrum np.ones(self.nl).
                        - other strings: name of the spectrum file in #pykliproot#/spectra/*/ with pykliproot the
                                        directory in which pyklip is installed. It that case it should be a spectrum
                                        from Mark Marley or one following the same convention.
                                        Spectrum will be corrected for transmission.
                        - ndarray: 1D array with a user defined spectrum. Spectrum will be corrected for transmission.
            add2prefix: Add user defined string to the prefix of the output filename.
            keepPrefix: (default = True) Keep the prefix of the input file instead of using the default:
                    self.prefix = self.star_name+"_"+self.compact_date+"_"+self.filter+self.add2prefix
            compact_date_func: lambda function returning the compact date (e.g. yyyymmdd) of the observation to be used
                                in the output filename if keepPrefix is False.
            filter_name_func: lambda function returning the name of the filter (e.g. "H", "J","K1"...) of the
                                observation to be used in the output filename if keepPrefix is False.
            PSF_size: Width of the PSF stamp to be used. Trim or pad with zeros the available PSF stamp.

        Return: instance of Matchedfilter.
        """
        # allocate super class
        super(Matchedfilter, self).__init__(read_func,filename,
                                     folderName = folderName,
                                     mute=mute,
                                     N_threads=N_threads,
                                     label=label,
                                     overwrite=overwrite,
                                     SpT_file_csv=SpT_file_csv)

        # Default value of kernel_type is "PSF"
        if kernel_type == None:
            self.kernel_type = "PSF"
        else:
            self.kernel_type = kernel_type
        # The default value is defined later
        self.kernel_para = kernel_para

        if self.kernel_type == "PSF":
            self.PSF_cube = PSF_cube
        self.PSF_read_func = PSF_read_func
        self.PSF_size = PSF_size

        if add2prefix is not None:
            self.add2prefix = "_"+add2prefix
        else:
            self.add2prefix = ""

        if keepPrefix is not None:
            self.keepPrefix = keepPrefix
        else:
            self.keepPrefix = True

        self.SpT_file_csv=SpT_file_csv
        self.spectrum = spectrum

        self.compact_date_func = compact_date_func
        self.filter_name_func = filter_name_func

        self.spectrum_name = ""
        self.prefix = ""
        self.filename_path = ""

    def spectrum_iter_available(self):
        """
        Should indicate whether or not the class is equipped for iterating over spectra.
        That depends on the number of dimensions of the input file. If it is 2D there is no spectrum template so it
        can't and shouldn't iterate over spectra however if the input file is a cube then a spectrum can be used.

        In order to iterate over spectra the function new_init_spectrum() can be called.
        spectrum_iter_available is a utility function for campaign data processing.

        :return: True if the file read is 3D.
                False if it is 2D.
        """

        return len(self.image.shape) == 3

    def init_new_spectrum(self,spectrum=None,SpT_file_csv=None):
        """
        Function allowing the reinitialization of the class with a new spectrum without reinitializing everything.

        See spectrum_iter_available()

        Args:
            spectrum: spectrum name (string) or array
                        - "host_star_spec": The spectrum from the star or the satellite spots is directly used.
                                            It is derived from the inverse of the calibrate_output() output.
                        - "constant": Use a constant spectrum np.ones(self.nl).
                        - other strings: name of the spectrum file in #pykliproot#/spectra/*/ with pykliproot the
                                        directory in which pyklip is installed. It that case it should be a spectrum
                                        from Mark Marley or one following the same convention.
                                        Spectrum will be corrected for transmission.
                        - ndarray: 1D array with a user defined spectrum. Spectrum will be corrected for transmission.
            SpT_file_csv: Filename (.csv) of the table containing the target names and their spectral type.
                    Can be generated from quering Simbad.
                    If None (default), the function directly tries to query Simbad, which requires internet access.

        Return: None
        """
        if not self.mute:
            print("~~ UPDATE Spectrum "+self.__class__.__name__+" ~~")

        if SpT_file_csv is not None:
            self.SpT_file_csv = SpT_file_csv
        if spectrum is not None:
            self.spectrum = spectrum

        # use super class
        super(Matchedfilter, self).init_new_spectrum(self.spectrum,SpT_file_csv=self.SpT_file_csv)

        for k in range(self.nl):
            self.PSF_cube_arr[k,:,:] *= self.spectrum_vec[k]/np.nanmax(self.PSF_cube_arr[k,:,:])
        # normalize spectrum with norm 2.
        self.spectrum_vec = self.spectrum_vec / np.sqrt(np.nansum(self.spectrum_vec**2))
        # # normalize PSF with norm 2.
        self.PSF_cube_arr = self.PSF_cube_arr / np.nansum(self.PSF_cube_arr)

        return None

    def initialize(self,inputDir = None,
                         outputDir = None,
                         spectrum = None,
                         folderName = None,
                         label = None):
        """
        Read the file using read_func (see the class  __init__ function) and define the matched filter kernel
        according to kernel_type.

        Can be called several time to process all the files matching the filename.

        Also define the output filename (if it were to be saved) such that check_existence() can be used.

        Args:
            inputDir: If defined it allows filename to not include the whole path and just the filename.
                            Files will be read from inputDir.
                            If inputDir is None then filename is assumed to have the absolute path.
            outputDir: Directory where to create the folder containing the outputs.
                    A kpop folder will be created to save the data. Convention is:
                    self.outputDir = outputDir+os.path.sep+"kpop_"+label+os.path.sep+folderName
            spectrum: spectrum name (string) or array
                        - "host_star_spec": The spectrum from the star or the satellite spots is directly used.
                                            It is derived from the inverse of the calibrate_output() output.
                        - "constant": Use a constant spectrum np.ones(self.nl).
                        - other strings: name of the spectrum file in #pykliproot#/spectra/*/ with pykliproot the
                                        directory in which pyklip is installed. It that case it should be a spectrum
                                        from Mark Marley or one following the same convention.
                                        Spectrum will be corrected for transmission.
                        - ndarray: 1D array with a user defined spectrum. Spectrum will be corrected for transmission.
            folderName: Name of the folder containing the outputs. It will be located in outputDir+os.path.sep+"kpop_"+label
                        Default folder name is "default_out".
                        A nice convention is to have one folder per spectral template.
                        If the file read has been created with KPOP, folderName is automatically defined from that
                        file.
            label: Define the suffix of the kpop output folder when it is not defined. cf outputDir. Default is "default".

        Return: True if all the files matching the filename (with wildcards) have been processed. False otherwise.
        """
        if not self.mute:
            print("~~ INITializing "+self.__class__.__name__+" ~~")

        # The super class already read the fits file
        init_out = super(Matchedfilter, self).initialize(inputDir = inputDir,
                                         outputDir = outputDir,
                                         folderName = folderName,
                                         label=label)

        if self.compact_date_func is None:
            self.compact_date = "noDate"
        else:
            self.compact_date = self.compact_date_func(self.filename_path)
        if self.filter_name_func is None:
            self.filter_name = "noFilter"
        else:
            self.filter_name = self.filter_name_func(self.filename_path)


        # If the Kernel is a PSF build it here
        if self.kernel_type=="PSF":

            if isinstance(self.PSF_cube, np.ndarray):
                self.PSF_cube_arr = self.PSF_cube
            else:
                self.PSF_cube_filename = self.PSF_cube

                if self.PSF_cube_filename is None:
                    self.PSF_cube_filename = "*-original_radial_PSF_cube.fits"
                    if not self.mute:
                        print("Using default filename for PSF cube: "+self.PSF_cube_filename)

                if os.path.isabs(self.PSF_cube_filename):
                    try:
                        self.PSF_cube_path = os.path.abspath(glob(self.PSF_cube_filename)[0])
                    except:
                        raise Exception("File "+self.PSF_cube_filename+"doesn't exist.")
                else:
                    base_path = os.path.dirname(self.filename_path)
                    try:
                        self.PSF_cube_path = os.path.abspath(glob(os.path.join(base_path,self.PSF_cube_filename))[0])
                    except:
                        raise Exception("File "+os.path.join(base_path,self.PSF_cube_filename)+"doesn't exist.")

                if not self.mute:
                    print("Loading PSF cube: "+self.PSF_cube_path)

                # Load the PSF cube if a file has been found or was just generated
                self.PSF_obj = self.PSF_read_func([self.PSF_cube_path])
                self.PSF_cube_arr = self.PSF_obj.input

            if (len(self.image.shape) == 2) and not (len(self.PSF_cube_arr.shape) == 3):
                # flatten the PSF cube if the data is 2D
                self.PSF_cube_arr =  np.nanmean(self.PSF_cube_arr,axis=0)
            # Remove the spectral shape from the psf cube because it is dealt with independently
            if (len(self.PSF_cube_arr.shape) == 3):
                for l_id in range(self.PSF_cube_arr.shape[0]):
                    self.PSF_cube_arr[l_id,:,:] = self.PSF_cube_arr[l_id,:,:]/np.nanmax(self.PSF_cube_arr[l_id,:,:])


        # Define the PSF as a gaussian
        if self.kernel_type == "gaussian":
            if self.kernel_para == None:
                self.kernel_para = 1.25
                if not self.mute:
                    print("Default width sigma = {0} used for the gaussian".format(self.kernel_para))

            if not self.mute:
                print("Generate gaussian PSF")
            # Build the grid for PSF stamp.
            ny_PSF = 20 # should be even
            nx_PSF = 20 # should be even
            x_PSF_grid, y_PSF_grid = np.meshgrid(np.arange(0,ny_PSF,1)-ny_PSF//2,np.arange(0,nx_PSF,1)-nx_PSF//2)

            PSF = kppmath.gauss2d(x_PSF_grid, y_PSF_grid,1.0,0.0,0.0,self.kernel_para,self.kernel_para)

            if (len(self.image.shape) == 3):
                # Duplicate the PSF to get a PSF cube.
                # Caution: No spectral widening implemented here
                self.PSF_cube_arr = np.tile(PSF,(self.nl,1,1))
            else:
                self.PSF_cube_arr = PSF

        # Define the PSF as an aperture or "hat" function
        if self.kernel_type == "hat":
            if self.kernel_para == None:
                self.kernel_para = 2.
                if not self.mute:
                    print("Default radius = {0} used for the hat function".format(self.kernel_para))

            # Build the grid for PSF stamp.
            ny_PSF = 20 # should be even
            nx_PSF = 20 # should be even
            x_PSF_grid, y_PSF_grid = np.meshgrid(np.arange(0,ny_PSF,1)-ny_PSF//2,np.arange(0,nx_PSF,1)-nx_PSF//2)
            # Use aperture for the cross correlation.
            # Calculate the corresponding hat function
            PSF = kppmath.hat(x_PSF_grid, y_PSF_grid, self.kernel_para)

            if (len(self.image.shape) == 3):
                # Duplicate the PSF to get a PSF cube.
                # Caution: No spectral widening implemented here
                self.PSF_cube_arr = np.tile(PSF,(self.nl,1,1))
            else:
                self.PSF_cube_arr = PSF


        # Change size of the PSF stamp.
        if self.PSF_size is not None:
            if (len(self.image.shape) == 3):
                old_shape = self.PSF_cube_arr.shape
                if old_shape[1] != old_shape[2]:
                    raise Exception("PSF cube must be square")
                w1 = self.PSF_size
                w0 = old_shape[1]
                PSF_cube_arr_new = np.zeros((old_shape[0],self.PSF_size,self.PSF_size))
                dw = w1-w0
                if dw >= 0:
                    if (dw % 2) == 0:
                        PSF_cube_arr_new[:,(dw//2):(dw//2+w0),dw//2:(dw//2+w0)] = self.PSF_cube_arr
                    else:
                        PSF_cube_arr_new[:,(dw//2 + (w0 % 2)):(dw//2 + (w0 % 2)+w0),(dw//2 + (w0 % 2)):(dw//2 + (w0 % 2)+w0)] = self.PSF_cube_arr
                else:
                    dw = -dw
                    if (dw % 2) == 0:
                        PSF_cube_arr_new = self.PSF_cube_arr[:,(dw//2):(dw//2+w1),dw//2:(dw//2+w1)]
                    else:
                        PSF_cube_arr_new = self.PSF_cube_arr[:,(dw//2 + (w1 % 2)):(dw//2 + (w1 % 2)+w1),(dw//2 + (w1 % 2)):(dw//2 + (w1 % 2)+w1)]
                self.PSF_cube_arr = PSF_cube_arr_new
            else:
                old_shape = self.PSF_cube_arr.shape
                if old_shape[0] != old_shape[1]:
                    raise Exception("PSF cube must be square")
                w1 = self.PSF_size
                w0 = old_shape[1]
                PSF_cube_arr_new = np.zeros((self.PSF_size,self.PSF_size))
                dw = w1-w0
                if dw >= 0:
                    if (dw % 2) == 0:
                        PSF_cube_arr_new[(dw//2):(dw//2+w0),dw//2:(dw//2+w0)] = self.PSF_cube_arr
                    else:
                        PSF_cube_arr_new[(dw//2 + (w0 % 2)):(dw//2 + (w0 % 2)+w0),(dw//2 + (w0 % 2)):(dw//2 + (w0 % 2)+w0)] = self.PSF_cube_arr
                else:
                    dw = -dw
                    if (dw % 2) == 0:
                        PSF_cube_arr_new = self.PSF_cube_arr[(dw//2):(dw//2+w1),dw//2:(dw//2+w1)]
                    else:
                        PSF_cube_arr_new = self.PSF_cube_arr[(dw//2 + (w1 % 2)):(dw//2 + (w1 % 2)+w1),(dw//2 + (w1 % 2)):(dw//2 + (w1 % 2)+w1)]
                self.PSF_cube_arr = PSF_cube_arr_new


        if (len(self.image.shape) == 3):
            self.nl_PSF, self.ny_PSF, self.nx_PSF = self.PSF_cube_arr.shape
        else:
            self.ny_PSF, self.nx_PSF = self.PSF_cube_arr.shape


        if (len(self.image.shape) == 3):
            if spectrum is not None:
                self.spectrum = spectrum
            if self.spectrum is not None:
                self.init_new_spectrum(self.spectrum,self.SpT_file_csv)

        # self.PSF_cube = self.PSF_cube / np.sqrt(np.sum(self.PSF_cube**2))
        self.PSF_cube_arr = self.PSF_cube_arr / np.nansum(self.PSF_cube_arr)

        # Define the suffix used when saving files
        if (len(self.image.shape) == 3):
            dim_suffix = "3D"
        else:
            dim_suffix = "2D"
        self.suffix = dim_suffix+self.kernel_type

        if self.keepPrefix:
            file_ext_ind = os.path.basename(self.filename_path)[::-1].find(".")
            self.prefix = os.path.basename(self.filename_path)[:-(file_ext_ind+1)]
        else:
            self.prefix = self.star_name+"_"+self.compact_date+"_"+self.filter+self.add2prefix

        return init_out

    def check_existence(self):
        """
        Return whether or not a filename of the processed data can be found.

        If overwrite is True, the output is always false.

        Return: boolean
        """

        suffix = "MF"+self.suffix
        file_exist = (len(glob(os.path.join(self.outputDir,self.folderName,self.prefix+'-'+suffix+'.fits'))) >= 1)

        if file_exist and not self.mute:
            print("Output already exist: "+os.path.join(self.outputDir,self.folderName,self.prefix+'-'+suffix+'.fits'))

        if self.overwrite and not self.mute:
            print("Overwriting is turned ON!")

        return file_exist and not self.overwrite


    def calculate(self,image=None, PSF=None,spectrum = None):
        """
        Perform a matched filter on the current loaded file.

        Args:
            image: image to get the cross correlation from.
            PSF: Template for the cross correlation
            spectrum: Spectrum to collapse the datacube if collapse has been set to true.

        Return: Processed images (matched filter, cross corr).
        """
        if image is not None:
            self.image = image
            print(self.image.shape)
            if np.size(self.image.shape) == 2:
                self.ny,self.nx = self.image.shape
            if np.size(self.image.shape) == 3:
                self.nl,self.ny,self.nx = self.image.shape
        if PSF is not None:
            self.PSF_cube_arr = PSF
            if np.size(self.PSF_cube_arr.shape) == 2:
                self.ny_PSF,self.nx_PSF = self.PSF_cube_arr.shape
            if np.size(self.PSF_cube_arr.shape) == 3:
                self.nl_PSF,self.ny_PSF,self.nx_PSF = self.PSF_cube_arr.shape
        if spectrum is not None:
            self.spectrum_vec = spectrum

        if not self.mute:
            print("~~ Calculating "+self.__class__.__name__)

        if (len(self.image.shape) == 3):
            flat_cube = np.nanmean(self.image,axis=0)
        else:
            flat_cube = self.image

        # Get the nans pixels of the flat_cube. We won't bother trying to calculate metrics for those.
        flat_cube_nans = np.where(np.isnan(flat_cube))

        # Remove the very edges of the image. We can't calculate a proper projection of an image stamp onto the PSF if we
        # are too close from the edges of the array.
        flat_cube_mask = np.ones((self.ny,self.nx))
        flat_cube_mask[flat_cube_nans] = np.nan
        flat_cube_noEdges_mask = copy(flat_cube_mask)
        # remove the edges if not already nans
        flat_cube_noEdges_mask[0:self.ny_PSF//2,:] = np.nan
        flat_cube_noEdges_mask[:,0:self.nx_PSF//2] = np.nan
        flat_cube_noEdges_mask[(self.ny-self.ny_PSF//2):self.ny,:] = np.nan
        flat_cube_noEdges_mask[:,(self.nx-self.nx_PSF//2):self.nx] = np.nan
        # Get the pixel coordinates corresponding to non nan pixels and not too close from the edges of the array.
        flat_cube_noNans_noEdges = np.where(np.isnan(flat_cube_noEdges_mask) == 0)

        mf_map = np.ones((self.ny,self.nx)) + np.nan
        cc_map = np.ones((self.ny,self.nx)) + np.nan
        flux_map = np.ones((self.ny,self.nx)) + np.nan

        # Calculate the criterion map.
        # For each pixel calculate the dot product of a stamp around it with the PSF.
        # We use the PSF cube to consider also the spectrum of the planet we are looking for.
        if not self.mute:
            print("Calculate the matched filter maps. It is done pixel per pixel so it might take a while...")
        stamp_PSF_x_grid, stamp_PSF_y_grid = np.meshgrid(np.arange(0,self.nx_PSF,1)-self.nx_PSF//2,
                                                         np.arange(0,self.ny_PSF,1)-self.ny_PSF//2)
        aper_radius = np.min([self.ny_PSF,self.nx_PSF])*7./20.
        r_PSF_stamp = (stamp_PSF_x_grid)**2 +(stamp_PSF_y_grid)**2
        where_sky_mask = np.where(r_PSF_stamp < (aper_radius**2))
        stamp_PSF_sky_mask = np.ones((self.ny_PSF,self.nx_PSF))
        stamp_PSF_sky_mask[where_sky_mask] = np.nan
        where_aper_mask = np.where(r_PSF_stamp > (aper_radius**2))
        stamp_PSF_aper_mask = np.ones((self.ny_PSF,self.nx_PSF))
        stamp_PSF_aper_mask[where_aper_mask] = np.nan
        if (len(self.PSF_cube_arr.shape) == 3):
            # Duplicate the mask to get a mask cube.
            # Caution: No spectral widening implemented here
            stamp_PSF_aper_mask = np.tile(stamp_PSF_aper_mask,(self.nl,1,1))

        N_pix = flat_cube_noNans_noEdges[0].size
        chunk_size = N_pix//self.N_threads

        if self.N_threads > 0 and chunk_size != 0:
            pool = mp.Pool(processes=self.N_threads)

            ## cut images in N_threads part
            N_chunks = N_pix//chunk_size

            # Get the chunks
            chunks_row_indices = []
            chunks_col_indices = []
            for k in range(N_chunks-1):
                chunks_row_indices.append(flat_cube_noNans_noEdges[0][(k*chunk_size):((k+1)*chunk_size)])
                chunks_col_indices.append(flat_cube_noNans_noEdges[1][(k*chunk_size):((k+1)*chunk_size)])
            chunks_row_indices.append(flat_cube_noNans_noEdges[0][((N_chunks-1)*chunk_size):N_pix])
            chunks_col_indices.append(flat_cube_noNans_noEdges[1][((N_chunks-1)*chunk_size):N_pix])

            outputs_list = pool.map(calculate_matchedfilter_star, itertools.izip(chunks_row_indices,
                                                       chunks_col_indices,
                                                       itertools.repeat(self.image),
                                                       itertools.repeat(self.PSF_cube_arr),
                                                       itertools.repeat(stamp_PSF_sky_mask),
                                                       itertools.repeat(stamp_PSF_aper_mask)))

            for row_indices,col_indices,out in zip(chunks_row_indices,chunks_col_indices,outputs_list):
                mf_map[(row_indices,col_indices)] = out[0]
                cc_map[(row_indices,col_indices)] = out[1]
                flux_map[(row_indices,col_indices)] = out[2]
            pool.close()
        else:
            out = calculate_matchedfilter(flat_cube_noNans_noEdges[0],
                                                       flat_cube_noNans_noEdges[1],
                                                       self.image,
                                                       self.PSF_cube_arr,
                                                       stamp_PSF_sky_mask,
                                                       stamp_PSF_aper_mask)

            mf_map[flat_cube_noNans_noEdges] = out[0]
            cc_map[flat_cube_noNans_noEdges] = out[1]
            flux_map[flat_cube_noNans_noEdges] = out[2]


        self.metricMap = (mf_map,cc_map,flux_map)
        return self.metricMap


    def save(self,dataset=None,outputDir = None,folderName = None,prefix=None):
        """
        Save the processed files as:
        #user_outputDir#+os.path.sep+"kpop_"+self.label+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.fits'

        Args:
            dataset: Instrument object. Needs the savedata() method.
            outputDir: Output directory where to save the processed files.
            folderName: subfolder of outputDir where to save the processed files. Set to "" to disable.
            prefix: prefix of the filename to be saved

        :return: None
        """

        if outputDir is not None:
            self.outputDir = outputDir
        if folderName is not None:
            self.folderName = folderName
        if prefix is not None:
            self.prefix = prefix
        if prefix == "":
            self.prefix = "unknown"
        if not hasattr(self,"suffix"):
            self.suffix = ""
        if dataset is not None:
            self.image_obj = dataset

        if not os.path.exists(self.outputDir+os.path.sep+self.folderName):
            os.makedirs(self.outputDir+os.path.sep+self.folderName)


        # Save the parameters as fits keywords
        extra_keywords = {"KPPFILEN":os.path.basename(self.filename_path),
                         "KPPFOLDN":self.folderName,
                         "KPPLABEL":self.label,
                         "KPPKERTY":str(self.kernel_type),
                         "KPPKERPA":str(self.kernel_para)}

        if (len(self.image.shape) == 3):
            extra_keywords["KPPSPNAM"] = str(self.spectrum_name)

        suffix = "MF"+self.suffix
        extra_keywords["KPPSUFFI"] = suffix
        if not self.mute:
            print("Saving: "+os.path.join(self.outputDir,self.folderName,self.prefix+'-'+suffix+'.fits'))
        self.image_obj.savedata(os.path.join(self.outputDir,self.folderName,self.prefix+'-'+suffix+'.fits'),
                         self.metricMap[0],
                         filetype=suffix,
                         more_keywords = extra_keywords,pyklip_output=False)
        suffix = "CC"+self.suffix
        extra_keywords["KPPSUFFI"] = suffix
        if not self.mute:
            print("Saving: "+os.path.join(self.outputDir,self.folderName,self.prefix+'-'+suffix+'.fits'))
        self.image_obj.savedata(os.path.join(self.outputDir,self.folderName,self.prefix+'-'+suffix+'.fits'),
                         self.metricMap[1],
                         filetype=suffix,
                         more_keywords = extra_keywords,pyklip_output=False)
        suffix = "Flux"+self.suffix
        extra_keywords["KPPSUFFI"] = suffix
        if not self.mute:
            print("Saving: "+os.path.join(self.outputDir,self.folderName,self.prefix+'-'+suffix+'.fits'))
        self.image_obj.savedata(os.path.join(self.outputDir,self.folderName,self.prefix+'-'+suffix+'.fits'),
                         self.metricMap[2],
                         filetype=suffix,
                         more_keywords = extra_keywords,pyklip_output=False)

        return None


    def load(self):
        """
        Load the metric map. One should check that it exist first using self.check_existence().

        Define the attribute self.metricMap.

        :return: self.metricMap
        """


        return self.metricMap


def calculate_matchedfilter_star(params):
    """
    Convert `f([1,2])` to `f(1,2)` call.
    It allows one to call calculate_shape3D_metric() with a tuple of parameters.
    """
    return calculate_matchedfilter(*params)

def calculate_matchedfilter(row_indices,col_indices,image,PSF,stamp_PSF_sky_mask,stamp_PSF_aper_mask, mute = True):
    '''
    Calculate the matched filter, cross correlation and flux map on a given image or datacube for the pixels targeted by
    row_indices and col_indices.
    These lists of indices can basically be given from the numpy.where function following the example:
        import numpy as np
        row_indices,col_indices = np.where(np.finite(np.mean(cube,axis=0)))
    By truncating the given lists in small pieces it is then easy to parallelized.

    Args:
        row_indices: Row indices list of the pixels where to calculate the metric in cube.
                            Indices should be given from a 2d image.
        col_indices: Column indices list of the pixels where to calculate the metric in cube.
                            Indices should be given from a 2d image.
        image: 2D or 3D image from which one wants the metric map. PSF_cube should be norm-2 normalized.
                    PSF_cube /= np.sqrt(np.sum(PSF_cube**2))
        PSF: 2D or 3D PSF template used for calculated the metric. If nl,ny_PSF,nx_PSF = PSF_cube.shape, nl is the
                         number of wavelength samples, ny_PSF and nx_PSF are the spatial dimensions of the PSF_cube.
        stamp_PSF_sky_mask: 2d mask of size (ny_PSF,nx_PSF) used to mask the central part of a stamp slice. It is used as
                            a type of a high pass filter. Before calculating the metric value of a stamp cube around a given
                            pixel the average value of the surroundings of each slice of that stamp cube will be removed.
                            The pixel used for calculating the average are the one equal to one in the mask.
        stamp_PSF_aper_mask: 3d mask for the aperture.
        mute: If True prevent printed log outputs.

    Return: Vector of length row_indices.size with the value of the metric for the corresponding pixels.
    '''

    image = np.array(image)
    if len(image.shape) == 2:
        cube = np.array([image])
        PSF_cube = np.array([PSF])
    else:
        cube = image
        PSF_cube = np.array(PSF)

    # Shape of the PSF cube
    nl,ny_PSF,nx_PSF = PSF_cube.shape

    # Number of rows and columns to add around a given pixel in order to extract a stamp.
    row_m = int(np.floor(ny_PSF/2.0))    # row_minus
    row_p = int(np.ceil(ny_PSF/2.0))     # row_plus
    col_m = int(np.floor(nx_PSF/2.0))    # col_minus
    col_p = int(np.ceil(nx_PSF/2.0))     # col_plus

    # Number of pixels on which the metric has to be computed
    N_it = row_indices.size
    # Define an shape vector full of nans
    mf_map = np.zeros((N_it,)) + np.nan
    cc_map = np.zeros((N_it,)) + np.nan
    flux_map = np.zeros((N_it,)) + np.nan
    # Loop over all pixels (row_indices[id],col_indices[id])
    for id,k,l in zip(range(N_it),row_indices,col_indices):
        if not mute:
            # Print the progress of the function
            stdout.write("\r{0}/{1}".format(id,N_it))
            stdout.flush()

        # Extract stamp cube around the current pixel from the whoel cube
        stamp_cube = copy(cube[:,(k-row_m):(k+row_p), (l-col_m):(l+col_p)])
        # wavelength dependent variance in the image
        var_per_wv = np.zeros(nl)
        # Remove average value of the surrounding pixels in each slice of the stamp cube
        for slice_id in range(nl):
            stamp_cube[slice_id,:,:] -= np.nanmean(stamp_cube[slice_id,:,:]*stamp_PSF_sky_mask)
            var_per_wv[slice_id] = np.nanvar(stamp_cube[slice_id,:,:]*stamp_PSF_sky_mask)
        try:
            mf_map[id] = np.nansum((stamp_PSF_aper_mask*PSF_cube*stamp_cube)/var_per_wv[:,None,None]) \
                         /np.sqrt(np.nansum((stamp_PSF_aper_mask*PSF_cube)**2/var_per_wv[:,None,None]))
            cc_map[id] = np.nansum(stamp_PSF_aper_mask*PSF_cube*stamp_cube)/np.sqrt(np.nansum((stamp_PSF_aper_mask*PSF_cube)**2))
            flux_map[id] = np.nansum((stamp_PSF_aper_mask*PSF_cube*stamp_cube)/var_per_wv[:,None,None]) \
                         /np.nansum((stamp_PSF_aper_mask*PSF_cube)**2/var_per_wv[:,None,None])
        except:
            # In case ones divide by zero...
            mf_map[id] =  np.nan
            cc_map[id] =  np.nan
            flux_map[id] =  np.nan

    return (mf_map,cc_map,flux_map)

def run_matchedfilter(image, PSF,N_threads=None):
        """
        Perform a matched filter on the current loaded file.

        Args:
            image: image for which to get the matched filter.
            PSF: Template for the matched filter. It should include any kind of spectrum you which to use of the data is 3d.

        Return: Processed images (matched filter,cross correlation,estimated flux).
        """
        # Number of threads to be used in case of parallelization.
        if N_threads is None:
            N_threads = mp.cpu_count()
        else:
            N_threads = N_threads

        if image is not None:
            image = image
            print(image.shape)
            if np.size(image.shape) == 2:
                ny,nx = image.shape
            if np.size(image.shape) == 3:
                nl,ny,nx = image.shape
        if PSF is not None:
            PSF_cube_arr = PSF
            if np.size(PSF_cube_arr.shape) == 2:
                ny_PSF,nx_PSF = PSF_cube_arr.shape
            if np.size(PSF_cube_arr.shape) == 3:
                nl_PSF,ny_PSF,nx_PSF = PSF_cube_arr.shape

        if (len(image.shape) == 3):
            flat_cube = np.nanmean(image,axis=0)
        else:
            flat_cube = image

        # Get the nans pixels of the flat_cube. We won't bother trying to calculate metrics for those.
        flat_cube_nans = np.where(np.isnan(flat_cube))

        # Remove the very edges of the image. We can't calculate a proper projection of an image stamp onto the PSF if we
        # are too close from the edges of the array.
        flat_cube_mask = np.ones((ny,nx))
        flat_cube_mask[flat_cube_nans] = np.nan
        flat_cube_noEdges_mask = copy(flat_cube_mask)
        # remove the edges if not already nans
        flat_cube_noEdges_mask[0:ny_PSF//2,:] = np.nan
        flat_cube_noEdges_mask[:,0:nx_PSF//2] = np.nan
        flat_cube_noEdges_mask[(ny-ny_PSF//2):ny,:] = np.nan
        flat_cube_noEdges_mask[:,(nx-nx_PSF//2):nx] = np.nan
        # Get the pixel coordinates corresponding to non nan pixels and not too close from the edges of the array.
        flat_cube_noNans_noEdges = np.where(np.isnan(flat_cube_noEdges_mask) == 0)

        mf_map = np.ones((ny,nx)) + np.nan
        cc_map = np.ones((ny,nx)) + np.nan
        flux_map = np.ones((ny,nx)) + np.nan

        # Calculate the criterion map.
        # For each pixel calculate the dot product of a stamp around it with the PSF.
        # We use the PSF cube to consider also the spectrum of the planet we are looking for.
        stamp_PSF_x_grid, stamp_PSF_y_grid = np.meshgrid(np.arange(0,nx_PSF,1)-nx_PSF//2,
                                                         np.arange(0,ny_PSF,1)-ny_PSF//2)
        aper_radius = np.min([ny_PSF,nx_PSF])*7./20.
        r_PSF_stamp = (stamp_PSF_x_grid)**2 +(stamp_PSF_y_grid)**2
        where_sky_mask = np.where(r_PSF_stamp < (aper_radius**2))
        stamp_PSF_sky_mask = np.ones((ny_PSF,nx_PSF))
        stamp_PSF_sky_mask[where_sky_mask] = np.nan
        where_aper_mask = np.where(r_PSF_stamp > (aper_radius**2))
        stamp_PSF_aper_mask = np.ones((ny_PSF,nx_PSF))
        stamp_PSF_aper_mask[where_aper_mask] = np.nan
        if (len(PSF_cube_arr.shape) == 3):
            # Duplicate the mask to get a mask cube.
            # Caution: No spectral widening implemented here
            stamp_PSF_aper_mask = np.tile(stamp_PSF_aper_mask,(nl,1,1))

        N_pix = flat_cube_noNans_noEdges[0].size
        chunk_size = N_pix//N_threads

        if N_threads > 0 and chunk_size != 0:
            pool = mp.Pool(processes=N_threads)

            ## cut images in N_threads part
            N_chunks = N_pix//chunk_size

            # Get the chunks
            chunks_row_indices = []
            chunks_col_indices = []
            for k in range(N_chunks-1):
                chunks_row_indices.append(flat_cube_noNans_noEdges[0][(k*chunk_size):((k+1)*chunk_size)])
                chunks_col_indices.append(flat_cube_noNans_noEdges[1][(k*chunk_size):((k+1)*chunk_size)])
            chunks_row_indices.append(flat_cube_noNans_noEdges[0][((N_chunks-1)*chunk_size):N_pix])
            chunks_col_indices.append(flat_cube_noNans_noEdges[1][((N_chunks-1)*chunk_size):N_pix])

            outputs_list = pool.map(calculate_matchedfilter_star, itertools.izip(chunks_row_indices,
                                                       chunks_col_indices,
                                                       itertools.repeat(image),
                                                       itertools.repeat(PSF_cube_arr),
                                                       itertools.repeat(stamp_PSF_sky_mask),
                                                       itertools.repeat(stamp_PSF_aper_mask)))

            for row_indices,col_indices,out in zip(chunks_row_indices,chunks_col_indices,outputs_list):
                mf_map[(row_indices,col_indices)] = out[0]
                cc_map[(row_indices,col_indices)] = out[1]
                flux_map[(row_indices,col_indices)] = out[2]
            pool.close()
        else:
            out = calculate_matchedfilter(flat_cube_noNans_noEdges[0],
                                                       flat_cube_noNans_noEdges[1],
                                                       image,
                                                       PSF_cube_arr,
                                                       stamp_PSF_sky_mask,
                                                       stamp_PSF_aper_mask)

            mf_map[flat_cube_noNans_noEdges] = out[0]
            cc_map[flat_cube_noNans_noEdges] = out[1]
            flux_map[flat_cube_noNans_noEdges] = out[2]


        metricMap = (mf_map,cc_map,flux_map)
        return metricMap