__author__ = 'JB'
import os
import astropy.io.fits as pyfits
from glob import glob
import multiprocessing as mp
import numpy as np
from scipy.signal import correlate2d

from pyklip.kpp.utils.kppSuperClass import KPPSuperClass
from pyklip.kpp.stat.statPerPix_utils import *
from pyklip.kpp.utils.oi import *
import pyklip.kpp.utils.mathfunc as kppmath

class Stat(KPPSuperClass):
    """
    Class for SNR calculation on a per pixel basis.
    """
    def __init__(self,read_func=None,filename=None,
                 folderName = None,
                 mute=None,
                 N_threads=None,
                 label = None,
                 overwrite = False,
                 mask_radius = None,
                 IOWA = None,
                 N = None,
                 Dr = None,
                 Dth = None,
                 type = None,
                 rm_edge = None,
                 OI_list_folder = None,
                 filename_noPlanets = None,
                 resolution = None,
                 image_wide = False,
                 r_step = None,
                 pix2as=None):
        """
        Define the general parameters of the SNR calculation.

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
                        Only available if "pixel based: is defined,
            label: label used in the definition of self.outputDir (where files shoudl be saved) in initialize().
            .      Default is "default".
                   Convention is self.outputDir = #outputDir#/kpop_#labe#/#folderName#/
            overwrite: Boolean indicating whether or not files should be overwritten if they exist.
                       See check_existence().
            mask_radius: Radius of the mask used for masking point sources or the surroundings of the current pixel out
                        of the data. Default value is 7 pixels.
            IOWA: (IWA,OWA) inner working angle, outer working angle. It defines boundary to the zones in which the
                        statistic is calculated.
                        If None, kpp.utils.GPIimage.get_IOWA() is used.
            N: Defines the width of the ring by the number of pixels it has to include.
                    The width of the annuli will therefore vary with sepration.
            Dr: If not None defines the width of the ring as Dr. N is then ignored if Dth is defined.
                Default value is 2 pixels.
            Dth: Define the angular size of a sector in degree (will apply for either Dr or N).
                 Only available if "pixel based" is defined,
            type: Indicate the type of statistic to be calculated from:
                    {"pixel based SNR","pixel based stddev","pixel based proba","SNR","stddev","proba"}
                    The "pixel based" mention indicates that the calculation will be done for each pixel by masking its
                    surroundings to avoid having a planet signal biasing its own SNR estimate.
                    Otherwise:
                        If "SNR" (default value) simple stddev calculation and returns SNR.
                        If "stddev" returns the pure standard deviation map.
                        If "proba" triggers proba calculation with pdf fitting.
            rm_edge: Remove the edge of the image based on the outermask of kpp.utils.GPIimage.get_occ().
                    (Not the edge of the array but the edge of the finite values in the image when there is some nan
                    padding)
            OI_list_folder: List of Object of Interest (OI) that should be masked from any standard deviation
                            calculation. See the online documentation for instructions on how to define it.
            filename_noPlanets: Filename pointing to the planet free version of filename.
                                The planet free images are used to estimate the standard deviation.
                                If filename_noPlanets has only one matching file from the function glob.glob(),
                                then it will be used for all matching filename.
                                If it has as many matching files as filename, then they will be used with a
                                one to one correspondance. Any othercase is ill-defined.
            resolution: Diameter of the resolution elements (in pix) used to do do the small sample statistic.
                    For e.g., FWHM of the PSF.
                    Only available if "pixel based: is defined,
                    /!\ I am not sure the implementation is correct. We should probably do better.
            image_wide: Don't divide the image in annuli or sectors when computing the statistic.
                        Use the entire image directly. Not available if "pixel based: is defined,
            r_step: Distance between two consecutive annuli mean separation. Not available if "pixel based" is defined,
            pix2as: Platescale (arcsec per pixel).

        """
        # allocate super class
        super(Stat, self).__init__(read_func,filename,
                                     folderName = folderName,
                                     mute=mute,
                                     N_threads=N_threads,
                                     label=label,
                                     overwrite = overwrite)

        if mask_radius is None:
            self.mask_radius = 7
        else:
            self.mask_radius = mask_radius

        if Dr is None:
            self.Dr = 2
        else:
            self.Dr = Dr

        if type is None:
            self.type = "pixel based SNR"
        else:
            self.type = type

        self.image_wide = image_wide

        self.IOWA = IOWA
        self.N = N
        self.Dth = Dth
        self.rm_edge = rm_edge
        self.resolution=resolution
        self.OI_list_folder = OI_list_folder
        self.filename_noPlanets = filename_noPlanets
        self.image_wide = image_wide

        if r_step is None:
            self.r_step = 2
        else:
            self.r_step = r_step

        self.pix2as = pix2as

        self.prefix = ""
        self.filename_path = ""

    def initialize(self,inputDir = None,
                         outputDir = None,
                         folderName = None,
                         label = None):
        """
        Read the files using read_func (see the class  __init__ function).

        Can be called several time to process all the files matching the filename.

        Also define the output filename (if it were to be saved) such that check_existence() can be used.

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
                        If the file read has been created with KPOP, folderName is automatically defined from that
                        file.
            label: Define the suffix of the kpop output folder when it is not defined. cf outputDir. Default is "default".

        Return: True if all the files matching the filename (with wildcards) have been processed. False otherwise.
        """
        if not self.mute:
            print("~~ INITializing "+self.__class__.__name__+" ~~")
        # The super class already read the fits file
        init_out = super(Stat, self).initialize(inputDir = inputDir,
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
        #self.prefix = "".join(os.path.basename(self.filename_path).split(".")[0:-1])
        self.suffix = self.type.replace("pixel based ","")
        if "pixel based" in self.type:
            self.suffix = self.suffix+"PerPix"
        tmp_suffix = ""
        if self.image_wide is False:
            if self.Dr is not None:
                tmp_suffix = tmp_suffix+"Dr"+str(self.Dr)
            elif self.N is not None:
                tmp_suffix = tmp_suffix+"N"+str(self.N)
            if self.Dth is not None:
                tmp_suffix = tmp_suffix+"Dth"+str(self.Dth)
            if "pixel based" not in self.type:
                if self.r_step is not None:
                    tmp_suffix = tmp_suffix+"rs"+str(self.r_step)
        else:
            tmp_suffix = "IW"
        self.suffix = self.suffix+tmp_suffix

        if self.filename_noPlanets is not None:# Check file existence and define filename_path
            if self.inputDir is None or os.path.isabs(self.filename_noPlanets):
                try:
                    if len(glob(self.filename_noPlanets)) == self.N_matching_files:
                        self.filename_noPlanets_path = os.path.abspath(glob(self.filename_noPlanets)[self.id_matching_file-1])
                    else:
                        self.filename_noPlanets_path = os.path.abspath(glob(self.filename_noPlanets)[0])
                except:
                    raise Exception("File "+self.filename_noPlanets+"doesn't exist.")
            else:
                try:
                    if len(glob(self.inputDir+os.path.sep+self.filename_noPlanets)) == self.N_matching_files:
                        self.filename_noPlanets_path = os.path.abspath(glob(self.inputDir+os.path.sep+self.filename_noPlanets)[self.id_matching_file-1])
                    else:
                        self.filename_noPlanets_path = os.path.abspath(glob(self.inputDir+os.path.sep+self.filename_noPlanets)[0])
                except:
                    raise Exception("File "+self.inputDir+os.path.sep+self.filename_noPlanets+" doesn't exist.")

            # Open the fits file on which the metric will be applied
            hdulist = pyfits.open(self.filename_noPlanets_path)
            if not self.mute:
                print("Opened: "+self.filename_noPlanets_path)

            # Read the image using the user defined reading function
            self.image_noPlanets_obj = self.read_func([self.filename_noPlanets_path])
            self.image_noPlanets = self.image_noPlanets_obj.input

            try:
                hdulist = pyfits.open(self.filename_noPlanets_path)
                self.prihdr_noPlanets = hdulist[0].header
                hdulist.close()
            except:
                self.prihdr_noPlanets = None
            try:
                hdulist = pyfits.open(self.filename_noPlanets_path)
                self.exthdr_noPlanets = hdulist[1].header
                hdulist.close()
            except:
                self.exthdr_noPlanets = None

            # Figure out which header
            self.fakeinfohdr_noPlanets = None
            if self.prihdr_noPlanets is not None:
                if np.sum(["FKPA" in key for key in self.prihdr_noPlanets.keys()]):
                    self.fakeinfohdr_noPlanets = self.prihdr_noPlanets
            if self.exthdr_noPlanets is not None:
                if np.sum(["FKPA" in key for key in self.exthdr_noPlanets.keys()]):
                    self.fakeinfohdr_noPlanets = self.exthdr_noPlanets

        return init_out

    def check_existence(self):
        """
        Return whether or not a filename of the processed data can be found.

        If overwrite is True, the output is always false.

        Return: boolean
        """

        print(self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.fits')
        file_exist = (len(glob(self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.fits')) >= 1)

        if file_exist and not self.mute:
            print("Output already exist: "+self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.fits')

        if self.overwrite and not self.mute:
            print("Overwriting is turned ON!")

        return file_exist and not self.overwrite


    def calculate(self,image=None,image_without_planet=None,center=None):
        """
        Calculate SNR map of the current image/cube.

        Args:
            image: Image from which to get the SNR map
            image_without_planet: Image in which the real signal has been masked
            center: center of the image (y_cen, x_cen)

        :return: Processed image.
        """
        if not self.mute:
            print("~~ Calculating "+self.__class__.__name__)

        if image is None and image_without_planet is None:
            if self.rm_edge is not None:
                # Mask out a band of 10 pixels around the edges of the finite pixels of the image.
                IWA,OWA,inner_mask,outer_mask = get_occ(self.image, centroid = self.center[0])
                conv_kernel = np.ones((self.rm_edge,self.rm_edge))
                wider_mask = correlate2d(outer_mask,conv_kernel,mode="same")
                self.image[np.where(np.isnan(wider_mask))] = np.nan

            if self.OI_list_folder is not None:
                try:
                    MJDOBS = self.prihdr['MJD-OBS']
                except:
                    raise ValueError("Could not find MJDOBS. Probably because non GPI data. Code needs to be improved")
            else:
                MJDOBS = None

            if self.filename_noPlanets is not None:
                self.image_without_planet = mask_known_objects(self.image_noPlanets,self.fakeinfohdr_noPlanets,self.star_name,
                                                               self.pix2as,self.center[0],MJDOBS=MJDOBS,
                                                               OI_list_folder=self.OI_list_folder, mask_radius = self.mask_radius)
            else:
                self.image_without_planet =  mask_known_objects(self.image,self.fakeinfohdr,self.star_name,self.pix2as,
                                                                self.center[0],MJDOBS=MJDOBS,OI_list_folder=self.OI_list_folder,
                                                                mask_radius = self.mask_radius)
        else:
            self.image = image
            if image_without_planet is not None:
                self.image_without_planet = image_without_planet
            else:
                self.image_without_planet = self.image
            self.center = [center]


        if np.size(self.image.shape) == 3:
            # Not tested
            self.stat_cube_map = np.zeros(self.image.shape)
            for k in range(self.nl):
                if "pixel based" in self.type:
                    self.stat_cube_map[k,:,:] = get_image_stat_map_perPixMasking(self.image[k,:,:],
                                                                            self.image_without_planet[k,:,:],
                                                                            mask_radius = self.mask_radius,
                                                                            IOWA = self.IOWA,
                                                                            N = self.N,
                                                                            centroid = self.center[0],
                                                                            mute = self.mute,
                                                                            N_threads = self.N_threads,
                                                                            Dr= self.Dr,
                                                                            Dth = self.Dth,
                                                                            type = self.type.replace("pixel based ",""),
                                                                            resolution = self.resolution)
                else:
                    self.stat_cube_map[k,:,:] = get_image_stat_map(self.image[k,:,:],
                                                                   self.image_without_planet[k,:,:],
                                                                   IOWA = self.IOWA,
                                                                   N = self.N,
                                                                   centroid = self.center[0],
                                                                   r_step = self.r_step,
                                                                   mute = self.mute,
                                                                   Dr= self.Dr,
                                                                   type = self.type,
                                                                   image_wide=self.image_wide)

        elif np.size(self.image.shape) == 2:
            if "pixel based" in self.type:
                self.stat_cube_map = get_image_stat_map_perPixMasking(self.image,
                                                                 self.image_without_planet,
                                                                 mask_radius = self.mask_radius,
                                                                 IOWA = self.IOWA,
                                                                 N = self.N,
                                                                 centroid = self.center[0],
                                                                 mute = self.mute,
                                                                 N_threads = self.N_threads,
                                                                 Dr= self.Dr,
                                                                 Dth = self.Dth,
                                                                 type = self.type.replace("pixel based ",""),
                                                                 resolution = self.resolution)
            else:
                self.stat_cube_map = get_image_stat_map(self.image,
                                                        self.image_without_planet,
                                                        IOWA = self.IOWA,
                                                        N = self.N,
                                                        centroid = self.center[0],
                                                        r_step = self.r_step,
                                                        mute = self.mute,
                                                        Dr= self.Dr,
                                                        type = self.type,
                                                        image_wide=self.image_wide)


        return self.stat_cube_map


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
            self.suffix = "SNR"
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
                          "KPPMASKR":self.mask_radius,
                          "KPP_IOWA":str(self.IOWA),
                          "KPP_N":self.N,
                          "KPP_DR":self.Dr,
                          "KPP_DTH":self.Dth,
                          "KPP_TYPE":self.type,
                          "KPPRMEDG":self.rm_edge,
                          "KPPGOILF":self.OI_list_folder}

        if hasattr(self,"filename_noSignal_path"):
            extra_keywords["KPPFILNS"] = self.filename_noSignal_path

        self.image_obj.savedata(os.path.join(self.outputDir,self.folderName,self.prefix+'-'+self.suffix+'.fits'),
                         self.stat_cube_map,
                         filetype=self.suffix,
                         more_keywords = extra_keywords,pyklip_output=False)

        return None

    def load(self):
        """

        :return: None
        """

        return None