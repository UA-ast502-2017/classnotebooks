__author__ = 'JB'
import os
import astropy.io.fits as pyfits
from glob import glob
import multiprocessing as mp
import numpy as np
from scipy.signal import convolve2d

from pyklip.kpp.utils.kppSuperClass import KPPSuperClass
from pyklip.kpp.stat.stat_utils import *
from pyklip.kpp.utils.oi import *
import pyklip.kpp.utils.mathfunc as kppmath
import pyklip.kpp.utils.GPIimage as gpiim

class Detection(KPPSuperClass):
    """
    Class for detecting blobs in a image.
    """
    def __init__(self,read_func=None,filename=None,
                 folderName = None,
                 mute=None,
                 overwrite = False,
                 mask_radius = None,
                 threshold = None,
                 maskout_edge = None,
                 IWA = None,
                 OWA = None,
                 pix2as=None):
        """
        Define the general parameters for the blob detection.

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
            overwrite: Boolean indicating whether or not files should be overwritten if they exist.
                       See check_existence().
            mask_radius: Radius of the mask used for masking point sources or the surroundings of the current pixel out
                        of the data. Default value is 4 pixels.
            threshold: Threshold under which blob should be ignore.
            maskout_edge: mask a 10 pixels border around each NaN pixel.
            IWA: inner working angle in pixels.
            OWA: outer working angle in pixels.
            pix2as: Platescale (arcsec per pixel).
        """
        # allocate super class
        super(Detection, self).__init__(read_func,filename,
                                     folderName = folderName,
                                     mute=mute,
                                     N_threads=None,
                                     label=None,
                                     overwrite = overwrite)

        if mask_radius is None:
            self.mask_radius = 4
        else:
            self.mask_radius = mask_radius

        if threshold is None:
            self.threshold = 3
        else:
            self.threshold = threshold

        # If true mask out a band of 10pix at the edge of the image following the nan boundary.
        if maskout_edge is None:
            self.maskout_edge = False
        else:
            self.maskout_edge = maskout_edge

        self.IWA = IWA
        self.OWA = OWA

        self.pix2as = pix2as

        self.prefix = ""
        self.filename_path = ""

    def  initialize(self,inputDir = None,
                         outputDir = None,
                         folderName = None):
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

        Return: True if all the files matching the filename (with wildcards) have been processed. False otherwise.
        """
        if not self.mute:
            print("~~ INITializing "+self.__class__.__name__+" ~~")

        # The super class already read the fits file
        init_out = super(Detection, self).initialize(inputDir = inputDir,
                                         outputDir = outputDir,
                                         folderName = folderName,
                                         label=None)

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
                    self.folderName = None

        file_ext_ind = os.path.basename(self.filename_path)[::-1].find(".")
        self.prefix = os.path.basename(self.filename_path)[:-(file_ext_ind+1)]
        self.suffix = "DetecTh{0}Mr{1}".format(self.threshold,self.mask_radius)

        return init_out

    def check_existence(self):
        """
        Return whether or not a filename of the processed data can be found.

        If overwrite is True, the output is always false.

        Return: boolean
        """


        if self.folderName is not None:
            myname = self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.csv'
        else:
            myname = self.outputDir+os.path.sep+self.prefix+'-'+self.suffix+'.csv'

        file_exist = (len(glob(myname)) >= 1)

        if file_exist and not self.mute:
            print("Output already exist: "+myname)

        if self.overwrite and not self.mute:
            print("Overwriting is turned ON!")

        return file_exist and not self.overwrite


    def calculate(self,image=None, center = None):
        """
        Find the brightest blobs in the image/cube.

        Args:
            image: Image from which to get the SNR map
            center: center of the image (y_cen, x_cen)

        :return: Detection table..
        """
        if not self.mute:
            print("~~ Calculating "+self.__class__.__name__)

        if image is not None:
            self.image = image
            if np.size(self.image.shape) == 2:
                self.ny,self.nx = self.image.shape
        if center is not None:
            self.center = [center]

        # Make a copy of the criterion map because it will be modified in the following.
        # Local maxima are indeed masked out when checked
        image_cpy = copy(self.image)

        # Build as grids of x,y coordinates.
        # The center is in the middle of the array and the unit is the pixel.
        # If the size of the array is even 2n x 2n the center coordinate in the array is [n,n].
        x_grid, y_grid = np.meshgrid(np.arange(0,self.nx,1)-self.center[0][0],np.arange(0,self.ny,1)-self.center[0][1])


        # Definition of the different masks used in the following.
        stamp_size = self.mask_radius * 2 + 2
        # Mask to remove the spots already checked in criterion_map.
        stamp_x_grid, stamp_y_grid = np.meshgrid(np.arange(0,stamp_size,1)-6,np.arange(0,stamp_size,1)-6)
        stamp_mask = np.ones((stamp_size,stamp_size))
        r_stamp = abs((stamp_x_grid) +(stamp_y_grid)*1j)
        stamp_mask[np.where(r_stamp < self.mask_radius)] = np.nan

        # Mask out a band of 10 pixels around the edges of the finite pixels of the image.
        if self.maskout_edge:
            IWA,OWA,inner_mask,outer_mask = get_occ(self.image, centroid = self.center[0])
            conv_kernel = np.ones((10,10))
            flat_cube_wider_mask = convolve2d(outer_mask,conv_kernel,mode="same")
            image_cpy[np.where(np.isnan(flat_cube_wider_mask))] = np.nan


        # Number of rows and columns to add around a given pixel in order to extract a stamp.
        row_m = int(np.floor(stamp_size/2.0))    # row_minus
        row_p = int(np.ceil(stamp_size/2.0))     # row_plus
        col_m = int(np.floor(stamp_size/2.0))    # col_minus
        col_p = int(np.ceil(stamp_size/2.0))     # col_plus

        # Table containing the list of the local maxima with their info
        # Description by column:
        # 1/ index of the candidate
        # 2/ Value of the maximum
        # 3/ Position angle in degree from North in [0,360]
        # 4/ Separation in pixel
        # 5/ Separation in arcsec
        # 6/ x position in pixel
        # 7/ y position in pixel
        # 8/ row index
        # 9/ column index
        self.candidate_table = []
        self.table_labels = ["index","value","PA","Sep (pix)","Sep (as)","x","y","row","col"]
        ## START WHILE LOOP.
        # Each iteration looks at one local maximum in the criterion map.
        k = 0
        max_val_criter = np.nanmax(image_cpy)
        while max_val_criter >= self.threshold:# and k <= max_attempts:
            k += 1
            # Find the maximum value in the current criterion map. At each iteration the previous maximum is masked out.
            max_val_criter = np.nanmax(image_cpy)
            # Locate the maximum by retrieving its coordinates
            max_ind = np.where( image_cpy == max_val_criter )
            row_id,col_id = max_ind[0][0],max_ind[1][0]
            x_max_pos, y_max_pos = x_grid[row_id,col_id],y_grid[row_id,col_id]
            sep_pix = np.sqrt(x_max_pos**2+y_max_pos**2)
            sep_arcsec = self.pix2as *sep_pix
            pa = np.mod(np.rad2deg(np.arctan2(-x_max_pos,y_max_pos)),360)


            # Mask the spot around the maximum we just found.
            image_cpy[(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)] *= stamp_mask

            if self.IWA is not None:
                if sep_pix < self.IWA:
                    continue
            if self.OWA is not None:
                if sep_pix > self.OWA:
                    continue

            # Store the current local maximum information in the table
            self.candidate_table.append([k,max_val_criter,pa,sep_pix,sep_arcsec,x_max_pos,y_max_pos,row_id,col_id])
        ## END WHILE LOOP.

        return self.candidate_table


    def save(self,outputDir = None,folderName = None,prefix=None):
        """
        Save the processed files as:
        #user_outputDir#+os.path.sep+self.prefix+'-'+self.suffix+'.fits'
        or if self.label and self.folderName are not None:
        #user_outputDir#+os.path.sep+"kpop_"+self.label+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.fits'

        Args:
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
            self.suffix = "Detec"

        if self.folderName is not None:
            if not os.path.exists(self.outputDir+os.path.sep+self.folderName):
                os.makedirs(self.outputDir+os.path.sep+self.folderName)
            myname = self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.csv'
        else:
            if not os.path.exists(self.outputDir):
                os.makedirs(self.outputDir)
            myname = self.outputDir+os.path.sep+self.prefix+'-'+self.suffix+'.csv'


        if not self.mute:
            print("Saving: "+myname)
        with open(myname, 'w+') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=';')
            csvwriter.writerows([self.table_labels])
            csvwriter.writerows(self.candidate_table)
        return None

    def load(self):
        """

        :return: None
        """

        return None
