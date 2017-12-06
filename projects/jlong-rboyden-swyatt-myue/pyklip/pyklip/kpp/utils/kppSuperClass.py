__author__ = 'JB'

import os
import astropy.io.fits as pyfits
from glob import glob
import multiprocessing as mp
import numpy as np
from copy import copy


import pyklip.spectra_management as spec

class KPPSuperClass(object):
    """
    Super class for all kpop classes (ie FMMF, matched filter, SNR calculation...).
    The initialize function is implemented to read a file but can be disable.

    Using KPPSuperClass would simply read a file and return them as they are.

    The idea of this class is that it can be used several times on different files when using wildcards in the filename.
    - Parameters defined in the builder function should not change from one file to the other.
    - Parameters that depend on the file (for e.g the spectral type of the star might change) should be defined in
    initiliaze after the file has been read.

    After the object has been created, initialize(), run() and save() can be called reapeatedly to reduce all the files
    matching the filanme if wildcards were used.
    (Use kppPerDir for an automated version of this)
    """
    def __init__(self,read_func,filename,
                 folderName = None,
                 mute=None,
                 N_threads=None,
                 label = None,
                 overwrite = False,
                 SpT_file_csv = None):
        """
        Define the general parameters of the algorithm..
        For e.g, which matched filter template to use, like gaussian or hat, and its width.

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
            .      Default is "default".
                   Convention is self.outputDir = #outputDir#/kpop_#labe#/#folderName#/
            overwrite: Boolean indicating whether or not files should be overwritten if they exist.
                       See check_existence().
            SpT_file_csv: Filename (.csv) of the table containing the target names and their spectral type.
                    Can be generated from quering Simbad.
                    If None (default), the function directly tries to query Simbad.

        Return: instance of kppSuperClass.
        """
        self.read_func = read_func

        self.overwrite = overwrite

        self.SpT_file_csv = SpT_file_csv

        # Define a default folderName is the one given is None.
        if folderName is None:
            self.folderName = "default_out"
        else:
            self.folderName = folderName

        self.filename = filename

        if label is None:
            self.label = "default"
        else:
            self.label = label

        # Number of threads to be used in case of parallelization.
        if N_threads is None:
            self.N_threads = mp.cpu_count()
        else:
            self.N_threads = N_threads

        self.mute = mute
        self.inputDir = None

    def spectrum_iter_available(self):
        """
        Should indicate wether or not the class is equipped for iterating over spectra.
        If the metric requires a spectrum, one might want to iterate over several spectra without rereading the input
        files. In order to iterate over spectra the function init_new_spectrum() should be defined.

        Args:
        :return: False
        """

        return False

    def init_new_spectrum(self,spectrum,
                                SpT_file_csv = None):
        """
        Function allowing the reinitialization of the class with a new spectrum without reinitializing everything.
        Generate a transmission corrected spectrum.

        This function can be redefined in the inherited class if not suitable.

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
        self.dn_per_contrast = 1./self.image_obj.calibrate_output(np.ones((self.nl,1,1)),spectral=True).squeeze()
        self.host_star_spec = self.dn_per_contrast/np.mean(self.dn_per_contrast)

        if SpT_file_csv is not None:
            self.SpT_file_csv = SpT_file_csv
        self.star_type = spec.get_specType(self.star_name,self.SpT_file_csv)
        # Interpolate a spectrum of the star based on its spectral type/temperature
        wv,self.star_sp = spec.get_star_spectrum(self.image_obj.wvs,self.star_type)

        # Define the output Foldername
        if isinstance(spectrum, str):

            # Do the best it can with the spectral information given in inputs.
            if spectrum == "host_star_spec":
                # If spectrum_filename is an empty string the function takes the sat spot spectrum by default.
                if not self.mute:
                    print("Default host star specrum will be used.")
                self.spectrum_vec = copy(self.host_star_spec)
                self.spectrum_name = "host_star_spec"
            elif spectrum == "constant":
                if not self.mute:
                    print("Spectrum is not or badly defined so taking flat spectrum")
                self.spectrum_vec = np.ones(self.nl)
                self.spectrum_name = "constant"
            else :
                pykliproot = os.path.dirname(os.path.realpath(spec.__file__))
                self.spectrum_filename = os.path.abspath(glob(os.path.join(pykliproot,"spectra","*",spectrum+".flx"))[0])
                spectrum_name = self.spectrum_filename.split(os.path.sep)
                self.spectrum_name = spectrum_name[len(spectrum_name)-1].split(".")[0]

                # spectrum_filename is not empty it is assumed to be a valid path.
                if not self.mute:
                    print("Spectrum model: "+self.spectrum_filename)
                # Interpolate the spectrum of the planet based on the given filename
                wv,self.planet_sp = spec.get_planet_spectrum(self.spectrum_filename,self.image_obj.wvs)

                # Correct the ideal spectrum given in spectrum_filename for atmospheric and instrumental absorption.
                self.spectrum_vec = (self.host_star_spec/self.star_sp)*self.planet_sp

        elif isinstance(spectrum, np.ndarray):
            self.planet_sp = spectrum
            self.spectrum_name = "custom"

            self.star_type = spec.get_specType(self.star_name,self.SpT_file_csv)

            # Correct the ideal spectrum given in spectrum_filename for atmospheric and instrumental absorption.
            self.spectrum_vec = (self.host_star_spec/self.star_sp)*self.planet_sp
        else:
            raise ValueError("Invalid spectrum: {0}".format(spectrum))

        self.spectrum_vec = self.spectrum_vec/np.mean(self.spectrum_vec)

        if self.folderName == "default_out":
            self.folderName = self.spectrum_name

        return None


    def initialize(self,inputDir = None,
                         outputDir = None,
                         folderName = None,
                         label = None,
                         read = True):
        """
        First read the file using read_func (see the class  __init__ function) and setup the file dependent parameters.

        The idea of this class is that it can be used several times on different files when using wildcards in the filename.
        - Parameters defined in the builder function should not change from one file to the other.
        - Parameters that depend on the file (for e.g the spectral type of the star might change) should be defined in
        initiliaze (here) after the file has been read.

        After the object has been created, initialize(), run() and save() can be called reapeatedly to reduce all the files
        matching the filanme if wildcards were used.
        (Use kppPerDir for an automated version of this)

        The file is assumed here to be a fits containing a 2D image or a GPI 3D cube (assumes 37 spectral slice).

        Define the following attribute:
            - self.image: the image/cube to be processed
            - (self.nl,)self.ny,self.nx the dimensions of the image. self.nl is only defined if 3D.
            - self.center: The centers of the images
            - self.prihdr and self.exthdr if prihdrs and exthdrs are attributes of the instrument class used for read_func.
            - self.outputDir based on outputDir, folderName and label. Convention is:
                    self.outputDir = outputDir+os.path.sep+"kpop_"+label+os.path.sep+folderName

        Args:
            inputDir: If defined it allows filename to not include the whole path and just the filename.
                            Files will be read from inputDir.
                            If inputDir is None then filename is assumed to have the absolute path.
            outputDir: Directory where to create the folder containing the outputs.
                    A kpop folder will be created to save the data. Convention is:
                    self.outputDir = outputDir+os.path.sep+"kpop_"+label+os.path.sep+folderName
            folderName: Name of the folder containing the outputs. It will be located in outputDir+os.path.sep+"kpop_"+label
                            Default folder name is "default_out".
                            A nice convention is to have one folder per spectral template.
            label: Define the suffix of the kpop output folder when it is not defined. cf outputDir. Default is "default".
            read: If true (default) read the fits file according to inputDir and filename otherwise only define self.outputDir.

        Return: True if all the files matching the filename (with wildcards) have been processed. False otherwise.
        """

        if not hasattr(self,"id_matching_file"):
            self.id_matching_file = 0
        if not hasattr(self,"process_all_files"):
            self.process_all_files = True

        if label is not None:
            self.label = label

        # Define a default folderName is the one given is None.
        if folderName is not None:
            self.folderName = folderName

        # Define the actual filename path
        if inputDir is None:
            self.inputDir = None
        else:
            self.inputDir = os.path.abspath(inputDir)

        if not hasattr(self,"outputDir"):
            self.outputDir = None

        if read:
            # Check file existence and define filename_path
            if self.inputDir is None or os.path.isabs(self.filename):
                try:
                    self.filename_path = os.path.abspath(glob(self.filename)[self.id_matching_file])
                    self.N_matching_files = len(glob(self.filename))
                except:
                    raise Exception("File "+self.filename+"doesn't exist.")
            else:
                try:
                    self.filename_path = os.path.abspath(glob(self.inputDir+os.path.sep+self.filename)[self.id_matching_file])
                    self.N_matching_files = len(glob(self.inputDir+os.path.sep+self.filename))
                except:
                    raise Exception("File "+self.inputDir+os.path.sep+self.filename+" doesn't exist.")

            self.id_matching_file = self.id_matching_file+1

            # Read the image using the user defined reading function
            self.image_obj = self.read_func([self.filename_path])
            self.image = self.image_obj.input
            try:
                self.star_name = self.image_obj.object_name.replace(" ","_")
            except:
                self.star_name = None
            # print(self.image_obj.centers)
            # print(self.image_obj.wvs)

            # Get input cube dimensions
            self.image = np.squeeze(self.image)
            if len(self.image.shape) == 3:
                self.nl,self.ny,self.nx = self.image.shape
                # # Checking that the cube has the 37 spectral slices of a normal GPI cube.
                # if self.nl != 37:
                #     raise Exception("Returning None. Spectral dimension of "+self.filename_path+" is not correct...")
            elif len(self.image.shape) == 2:
                self.ny,self.nx = self.image.shape
            else:
                raise Exception("Returning None. fits file "+self.filename_path+" was not a 2D image or a 3D cube...")

            # Get center of the image (star position)
            try:
                # Retrieve the center of the image from the fits headers.
                self.center = self.image_obj.centers
            except:
                # If the keywords could not be found the center is defined as the middle of the image
                if not self.mute:
                    print("Couldn't find PSFCENTX and PSFCENTY keywords.")
                if len(self.image.shape) == 3:
                    self.center = [[(self.nx-1)//2,(self.ny-1)//2],]*self.nl
                else:
                    self.center = [[(self.nx-1)//2,(self.ny-1)//2],]

            try:
                hdulist = pyfits.open(self.filename_path)
                self.prihdr = hdulist[0].header
                hdulist.close()
            except:
                self.prihdr = None
            try:
                hdulist = pyfits.open(self.filename_path)
                self.exthdr = hdulist[1].header
                hdulist.close()
            except:
                self.exthdr = None

            # Figure out which header
            self.fakeinfohdr = None
            if self.prihdr is not None:
                if np.sum(["FKPA" in key for key in self.prihdr.keys()]):
                    self.fakeinfohdr = self.prihdr
            if self.exthdr is not None:
                if np.sum(["FKPA" in key for key in self.exthdr.keys()]):
                    self.fakeinfohdr = self.exthdr

            # If outputDir is None define it as the project directory.
            if outputDir is not None:
                if self.label is not None:
                    self.outputDir = os.path.abspath(outputDir+os.path.sep+"kpop_"+self.label)
                else:
                    self.outputDir = os.path.abspath(outputDir)
            else: # if self.outputDir is None:
                split_path = np.array(os.path.dirname(self.filename_path).split(os.path.sep))
                if np.sum(word.startswith("planet_detec_") for word in split_path):
                    planet_detec_label = split_path[np.where(["planet_detec" in mystr for mystr in split_path])][-1]
                    self.outputDir = os.path.abspath(self.filename_path.split(planet_detec_label)[0]+planet_detec_label)
                elif np.sum(word.startswith("kpop_") for word in split_path):
                    split_path = np.array(os.path.dirname(self.filename_path).split(os.path.sep))
                    planet_detec_label = split_path[np.where(["kpop" in mystr for mystr in split_path])][-1]
                    self.outputDir = os.path.abspath(self.filename_path.split(planet_detec_label)[0]+planet_detec_label)
                else:
                    if self.label is not None:
                        self.outputDir = os.path.join(os.path.dirname(self.filename_path),"kpop_"+self.label)
                    else:
                        self.outputDir = os.path.dirname(self.filename_path)

            if self.process_all_files:
                if self.id_matching_file < self.N_matching_files:
                    return True
                else:
                    self.id_matching_file = 0
                    return False
            else:
                return False
        else:
            # If outputDir is None define it as the project directory.
            if outputDir is not None:
                self.outputDir = os.path.abspath(outputDir+os.path.sep+"kpop_"+self.label)

            if self.outputDir is None:
                if self.inputDir is None:
                    self.outputDir = os.path.abspath("."+os.path.sep+"kpop_"+self.label)
                else:
                    self.outputDir = os.path.abspath(self.inputDir+os.path.sep+"kpop_"+self.label)

            return False



    def check_existence(self):
        """
        Check if the file corresponding to the processed data already exist.
        In this case one could decide to skip the reduction of this particular file.

        Args:

        Return: False
        """

        return False


    def calculate(self):
        """
        Process the data.
        Make sure initialize has been called first.

        Args:

        Return: self.image (the input fits file.)
        """

        return self.image


    def save(self):
        """
        Save the processed files.

        KPOP convention is that it should be saved in:
        self.outputDir = #user_outputDir#+os.path.sep+"kpop_"+self.label+os.path.sep+self.folderName


        Args:

        Return: None
        """

        return None


    def load(self):
        """
        Load the processed files.

        The idea of that this function knows the output directory and filename as a function of the reduction parameters.

        Return: None
        """

        return None