__author__ = 'JB'
import os
import astropy.io.fits as pyfits
from scipy.signal import correlate2d
import numpy as np
from glob import glob
from copy import copy

from pyklip.kpp.utils.kppSuperClass import KPPSuperClass
import pyklip.kpp.utils.mathfunc as kppmath
import pyklip.spectra_management as spec

class CrossCorr(KPPSuperClass):
    """
    Cross correlate data.
    """
    def __init__(self,read_func=None,filename=None,
                 folderName = None,
                 mute=None,
                 N_threads=None,
                 label = None,
                 overwrite = False,
                 SpT_file_csv = None,
                 kernel_type = None,
                 kernel_para = None,
                 collapse = None,
                 spectrum = None,
                 nans2zero = None,
                 PSF_size = None):
        """
        Define the general parameters of the cross correlation:
            - cross correlation template
            - weighted mean of the input data (if collapsing a cube)

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
            N_threads: Number of threads to be used for the metrics and the probability calculations.
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
                    - "hat": Define the kernel as a simple aperture photometry with radius kernel_para.
                            Default radius is 1.5 pixels.
                    - "Gaussian" (default): define the kernel as a symmetric 2D gaussian with width (ie standard deviation) equal
                            to kernel_para. Default value of the width is 1.25.
                    - If kernel_type is a np.ndarray then kernel_type is the user defined template.
            kernel_para: Define the width of the Kernel depending on kernel_type. See kernel_type.
            collapse: If true and input is 3D then it will collapse the final map. Requires to define spectrum.
                        Each slice is weighted by spectrum before collapsing.
            spectrum: spectrum name (string) or array used in the cube weighted mean when collapse is True.
                        - "host_star_spec": The spectrum from the star or the satellite spots is directly used.
                                            It is derived from the inverse of the calibrate_output() output.
                        - "constant": Use a constant spectrum np.ones(self.nl).
                        - other strings: name of the spectrum file in #pykliproot#/spectra/*/ with pykliproot the
                                        directory in which pyklip is installed. It that case it should be a spectrum
                                        from Mark Marley or one following the same convention.
                                        Spectrum will be corrected for transmission.
                        - ndarray: 1D array with a user defined spectrum. Spectrum will be corrected for transmission.
            nans2zero: If True, replace all nans values with zeros.
            PSF_size: Width of the PSF stamp to be used. Trim or pad with zeros the available PSF stamp.


        Return: instance of CrossCorr.
        """
        # allocate super class
        super(CrossCorr, self).__init__(read_func,filename,
                                     folderName = folderName,
                                     mute=mute,
                                     N_threads=N_threads,
                                     label=label,
                                     overwrite = overwrite,
                                     SpT_file_csv=SpT_file_csv)



        # Default value of kernel_type is "PSF"
        if kernel_type == None:
            self.kernel_type = "Gaussian"
        else:
            self.kernel_type = kernel_type
        # The default value is defined later
        self.kernel_para = kernel_para

        if collapse is None:
            self.collapse = False
        else:
            self.collapse = collapse
        self.spectrum = spectrum
        if nans2zero is None:
            self.nans2zero = True
        else:
            self.nans2zero = nans2zero

        self.spectrum_name = ""
        self.prefix = ""
        self.filename_path = ""
        self.PSF_size = PSF_size

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
        super(CrossCorr, self).init_new_spectrum(self.spectrum,SpT_file_csv=self.SpT_file_csv)

        return None

    def initialize(self,inputDir = None,
                         outputDir = None,
                         spectrum = None,
                         folderName = None,
                         label = None):
        """
        Read the file using read_func (see the class  __init__ function) and define the cross correlation kernel
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
        init_out = super(CrossCorr, self).initialize(inputDir = inputDir,
                                         outputDir = outputDir,
                                         folderName = folderName,
                                         label=label)

        try:
            self.folderName = self.prihdr["KPPFOLDN"]+os.path.sep
        except:
            try:
                self.folderName = self.exthdr["METFOLDN"]+os.path.sep
                print("/!\ CAUTION: Reading deprecated data.")
            except:
                try:
                    self.folderName = self.exthdr["STAFOLDN"]+os.path.sep
                    print("/!\ CAUTION: Reading deprecated data.")
                except:
                    pass

        file_ext_ind = os.path.basename(self.filename_path)[::-1].find(".")
        self.prefix = os.path.basename(self.filename_path)[:-(file_ext_ind+1)]
        self.suffix = "crossCorr"+self.kernel_type

        if self.kernel_type is not None:
            self.ny_PSF = 20 # should be even
            self.nx_PSF = 20 # should be even
            # Define the PSF as a gaussian
            if self.kernel_type == "gaussian":
                if self.kernel_para == None:
                    self.kernel_para = 1.25
                    if not self.mute:
                        print("Default width sigma = {0} used for the gaussian".format(self.kernel_para))

                if not self.mute:
                    print("Generate gaussian PSF")
                # Build the grid for PSF stamp.
                x_PSF_grid, y_PSF_grid = np.meshgrid(np.arange(0,self.ny_PSF,1)-self.ny_PSF//2,
                                                     np.arange(0,self.nx_PSF,1)-self.nx_PSF//2)

                self.PSF = kppmath.gauss2d(x_PSF_grid, y_PSF_grid,1.0,0.0,0.0,self.kernel_para,self.kernel_para)

            # Define the PSF as an aperture or "hat" function
            if self.kernel_type == "hat":
                if self.kernel_para == None:
                    self.kernel_para = 1.5
                    if not self.mute:
                        print("Default radius = {0} used for the hat function".format(self.kernel_para))

                # Build the grid for PSF stamp.
                x_PSF_grid, y_PSF_grid = np.meshgrid(np.arange(0,self.ny_PSF,1)-self.ny_PSF//2,
                                                     np.arange(0,self.nx_PSF,1)-self.nx_PSF//2)
                # Use aperture for the cross correlation.
                # Calculate the corresponding hat function
                self.PSF = kppmath.hat(x_PSF_grid, y_PSF_grid, self.kernel_para)

            if isinstance(self.kernel_type, np.ndarray):
                self.PSF = self.kernel_type

            self.PSF = self.PSF / np.sqrt(np.nansum(self.PSF**2))

        # Change size of the PSF stamp. If the size is even, it will be changed to an even size in the next few lines
        if self.PSF_size is not None:
            old_shape = self.PSF.shape
            if old_shape[0] != old_shape[1]:
                raise Exception("PSF cube must be square")
            w1 = self.PSF_size
            w0 = old_shape[1]
            PSF_cube_arr_new = np.zeros((self.PSF_size,self.PSF_size))
            dw = w1-w0
            if dw >= 0:
                if (dw % 2) == 0:
                    PSF_cube_arr_new[(dw//2):(dw//2+w0),dw//2:(dw//2+w0)] = self.PSF
                else:
                    PSF_cube_arr_new[(dw//2 + (w0 % 2)):(dw//2 + (w0 % 2)+w0),(dw//2 + (w0 % 2)):(dw//2 + (w0 % 2)+w0)] = self.PSF
            else:
                dw = -dw
                if (dw % 2) == 0:
                    PSF_cube_arr_new = self.PSF[(dw//2):(dw//2+w1),dw//2:(dw//2+w1)]
                else:
                    PSF_cube_arr_new = self.PSF[(dw//2 + (w1 % 2)):(dw//2 + (w1 % 2)+w1),(dw//2 + (w1 % 2)):(dw//2 + (w1 % 2)+w1)]
            self.PSF = PSF_cube_arr_new
            self.ny_PSF,self.nx_PSF = self.PSF.shape

        if spectrum is not None:
            self.spectrum = spectrum
        if self.spectrum is not None:
            self.init_new_spectrum(self.spectrum,self.SpT_file_csv)

        return init_out

    def check_existence(self):
        """
        Return whether or not a filename of the processed data can be found.

        If overwrite is True, the output is always false.

        Return: boolean
        """

        file_exist = (len(glob(self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.fits')) >= 1)

        if file_exist and not self.mute:
            print("Output already exist: "+self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.fits')

        if self.overwrite and not self.mute:
            print("Overwriting is turned ON!")

        return file_exist and not self.overwrite


    def calculate(self,image=None, PSF=None,spectrum = None):
        """
        Perform a cross correlation on the current loaded file.

        Args:
            image: image to get the cross correlation from.
            PSF: Template for the cross correlation
            spectrum: Spectrum to collapse the datacube if collapse has been set to true.

        Return: Processed image.
        """
        if image is not None:
            self.image = image
            print(self.image.shape)
            if np.size(self.image.shape) == 2:
                self.ny,self.nx = self.image.shape
            if np.size(self.image.shape) == 3:
                self.nl,self.ny,self.nx = self.image.shape
        if PSF is not None:
            self.PSF = PSF
            if np.size(self.PSF.shape) == 2:
                self.ny_PSF,self.nx_PSF = self.PSF.shape
            if np.size(self.PSF.shape) == 3:
                self.nl_PSF,self.ny_PSF,self.nx_PSF = self.PSF.shape
        if spectrum is not None:
            self.spectrum_vec = spectrum

        if not self.mute:
            print("~~ Calculating "+self.__class__.__name__)

        self.image_cpy = copy(self.image)

        if self.collapse:
            image_collapsed = np.zeros((self.ny,self.nx))
            for k in range(self.nl):
                image_collapsed = image_collapsed + self.spectrum_vec[k]*self.image_cpy[k,:,:]
            self.image_cpy = image_collapsed/np.sum(self.spectrum_vec)

        if self.nans2zero:
            where_nans = np.where(np.isnan(self.image_cpy))
            self.image_cpy = np.nan_to_num(self.image_cpy)

        # We have to make sure the PSF dimensions are odd because correlate2d shifts the image otherwise...
        if (self.nx_PSF % 2 ==0):
            PSF_tmp = np.zeros((self.ny_PSF,self.nx_PSF+1))
            PSF_tmp[0:self.ny_PSF,0:self.nx_PSF] = self.PSF
            self.PSF = PSF_tmp
            self.nx_PSF = self.nx_PSF +1
        if (self.ny_PSF % 2 ==0):
            PSF_tmp = np.zeros((self.ny_PSF+1,self.nx_PSF))
            PSF_tmp[0:self.ny_PSF,0:self.nx_PSF] = self.PSF
            self.PSF = PSF_tmp
            self.ny_PSF = self.ny_PSF +1


        if self.kernel_type is not None:
            # Check if the input file is 2D or 3D
            if np.size(self.image_cpy.shape) == 3: # If the file is a 3D cube
                self.image_convo = np.zeros(self.image_cpy.shape)
                for l_id in np.arange(self.nl):
                    self.image_convo[l_id,:,:] = correlate2d(self.image_cpy[l_id,:,:],self.PSF,mode="same")
            else: # image is 2D
                self.image_convo = correlate2d(self.image_cpy,self.PSF,mode="same")

        if self.nans2zero:
            self.image_convo[where_nans] = np.nan

        return self.image_convo


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
            self.suffix = "crossCorr"
        if dataset is not None:
            self.image_obj = dataset

        if not os.path.exists(self.outputDir+os.path.sep+self.folderName):
            os.makedirs(self.outputDir+os.path.sep+self.folderName)

        if not self.mute:
            print("Saving: "+os.path.join(self.outputDir,self.folderName,self.prefix+'-'+self.suffix+'.fits'))

        # Save the parameters as fits keywords
        extra_keywords = {"KPPFILEN":os.path.basename(self.filename_path),
                          "KPPFOLDN":self.folderName,
                          "KPPLABEL":self.label,
                          "KPPKERTY":str(self.kernel_type),
                          "KPPKERPA":str(self.kernel_para),
                          "KPPCOLLA":self.collapse,
                          "KPPNAN2Z":str(self.nans2zero)}

        if self.collapse:
            extra_keywords["KPPSPNAM"] = str(self.spectrum_name)

        print(os.path.join(self.outputDir,self.folderName,self.prefix+'-'+self.suffix+'.fits'))
        self.image_obj.savedata(os.path.join(self.outputDir,self.folderName,self.prefix+'-'+self.suffix+'.fits'),
                         self.image_convo,
                         filetype=self.suffix,
                         more_keywords = extra_keywords,pyklip_output=False)

        return None

    def load(self):
        """

        :return: None
        """

        return None