__author__ = 'jruffio'

from pyklip.kpp.utils.kppSuperClass import KPPSuperClass
from pyklip.kpp.utils.oi import *

class ROC(KPPSuperClass):
    """
    Class for calculating the ROC curve for a dataset.
    """
    def __init__(self,read_func,filename,filename_detec,
                 mute=None,
                 overwrite = False,
                 detec_distance = None,
                 ignore_distance = None,
                 OI_list_folder = None,
                 threshold_sampling = None,
                 IWA = None,
                 OWA = None,
                 pix2as=None):
        """
        Define the general parameters of the ROC calculation.

        Args:
            read_func: lambda function treturning a instrument object where the only input should be a list of filenames
                    to read.
                    For e.g.:
                    read_func = lambda filenames:GPI.GPIData(filenames,recalc_centers=False,recalc_wvs=False,highpass=False)
            filename: Filename of the file containing the simulated or real planets.
                        It should be the complete path unless inputDir is used in initialize().
                        It can include wild characters. The files will be reduced as given by glob.glob().
            filename_detec: Filename of the .csv file with the list of blobs in the image as produced by the class
                            pyklip.kpp.detection.detection.Detection.
            mute: If True prevent printed log outputs.
            overwrite: Boolean indicating whether or not files should be overwritten if they exist.
                       See check_existence().
            detec_distance: Distance in pixel between a candidate a true planet to claim the detection. (default 2 pixels)
            ignore_distance: Distance in pixel from a true planet up to which detections are ignored. (default 10 pixels)
            OI_list_folder: List of Object of Interest (OI) that should be masked from any standard deviation
                            calculation. See the online documentation for instructions on how to define it.
            threshold_sampling: Sampling used for the detection threshold (default np.linspace(0.0,20,200))
            IWA: inner working angle in pixels.
            OWA: outer working angle in pixels.
            pix2as: Platescale (arcsec per pixel).
        """
        # allocate super class
        super(ROC, self).__init__(read_func,filename,
                                     folderName = None,
                                     mute=mute,
                                     N_threads=None,
                                     label=None,
                                     overwrite = overwrite)

        if detec_distance is None:
            self.detec_distance = 2
        else:
            self.detec_distance = detec_distance

        if ignore_distance is None:
            self.ignore_distance = 10
        else:
            self.ignore_distance = ignore_distance

        if threshold_sampling is None:
            self.threshold_sampling = np.linspace(0.0,20,200)
        else:
            self.threshold_sampling = threshold_sampling

        self.filename_detec = filename_detec
        self.OI_list_folder = OI_list_folder

        self.IWA = IWA
        self.OWA = OWA

        self.pix2as = pix2as



    def initialize(self,inputDir = None,
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
        init_out = super(ROC, self).initialize(inputDir = inputDir,
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

        # Check file existence and define filename_path
        if self.inputDir is None or os.path.isabs(self.filename_detec):
            try:
                self.filename_detec_path = os.path.abspath(glob(self.filename_detec)[self.id_matching_file])
                self.N_matching_files = len(glob(self.filename_detec))
            except:
                raise Exception("File "+self.filename_detec+"doesn't exist.")
        else:
            try:
                self.filename_detec_path = os.path.abspath(glob(self.inputDir+os.path.sep+self.filename_detec)[self.id_matching_file])
                self.N_matching_files = len(glob(self.inputDir+os.path.sep+self.filename_detec))
            except:
                raise Exception("File "+self.inputDir+os.path.sep+self.filename_detec+" doesn't exist.")

        with open(self.filename_detec_path, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            csv_as_list = list(reader)
            self.detec_table_labels = csv_as_list[0]
            self.detec_table = np.array(csv_as_list[1::], dtype='string').astype(np.float)
        if not self.mute:
            print("Opened: "+self.filename_detec_path)


        self.N_detec = self.detec_table.shape[0]
        self.val_id = self.detec_table_labels.index("value")
        self.x_id = self.detec_table_labels.index("x")
        self.y_id = self.detec_table_labels.index("y")

        file_ext_ind = os.path.basename(self.filename_detec_path)[::-1].find(".")
        self.prefix = os.path.basename(self.filename_detec_path)[:-(file_ext_ind+1)]
        self.suffix = "ROC"
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


    def calculate(self):
        """
        Calculate the number of false positives and the number true positives.

        :return: FPR,TPR table
        """
        if not self.mute:
            print("~~ Calculating "+self.__class__.__name__+" with parameters " + self.suffix+" ~~")

        if self.OI_list_folder is not None:
            try:
                MJDOBS = self.prihdr.header['MJD-OBS']
            except:
                raise ValueError("Could not find MJDOBS. Probably because non GPI data. Code needs to be improved")
            x_real_object_list,y_real_object_list = \
                get_pos_known_objects(self.fakeinfohdr,self.star_name,self.pix2as,center=self.center[0],
                                      MJDOBS=MJDOBS,OI_list_folder=self.OI_list_folder,
                                      xy = True,ignore_fakes=True,IWA=self.IWA,OWA=self.OWA)

        row_object_list,col_object_list = get_pos_known_objects(self.fakeinfohdr,self.star_name,self.pix2as,center=self.center[0],
                                                                IWA=self.IWA,OWA=self.OWA)

        self.false_detec_proba_vec = []
        # Loop over all the local maxima stored in the detec csv file
        for k in range(self.N_detec):
            val_criter = self.detec_table[k,self.val_id]
            x_pos = self.detec_table[k,self.x_id]
            y_pos = self.detec_table[k,self.y_id]

            #remove the detection if it is a real object
            if self.OI_list_folder is not None:
                reject = False
                for x_real_object,y_real_object  in zip(x_real_object_list,y_real_object_list):
                    if (x_pos-x_real_object)**2+(y_pos-y_real_object)**2 < self.ignore_distance**2:
                        reject = True
                        break
                if reject:
                    continue

            if self.IWA is not None:
                if np.sqrt( (x_pos)**2+(y_pos)**2) < self.IWA:
                    continue
            if self.OWA is not None:
                if np.sqrt( (x_pos)**2+(y_pos)**2) > self.OWA:
                    continue

            self.false_detec_proba_vec.append(val_criter)

        self.true_detec_proba_vec = [self.image[np.round(row_real_object),np.round(col_real_object)] \
                                     for row_real_object,col_real_object in zip(row_object_list,col_object_list)]
        self.true_detec_proba_vec = np.array(self.true_detec_proba_vec)[np.where(~np.isnan(self.true_detec_proba_vec))]

        self.N_false_pos = np.zeros(self.threshold_sampling.shape)
        self.N_true_detec = np.zeros(self.threshold_sampling.shape)
        for id,threshold_it in enumerate(self.threshold_sampling):
            self.N_false_pos[id] = np.sum(self.false_detec_proba_vec >= threshold_it)
            self.N_true_detec[id] = np.sum(self.true_detec_proba_vec >= threshold_it)


        return zip(self.threshold_sampling,self.N_false_pos,self.N_true_detec)


    def save(self):
        """
        Save the processed files as:
        #user_outputDir#+os.path.sep+self.prefix+'-'+self.suffix+'.fits'
        or if self.label and self.folderName are not None:
        #user_outputDir#+os.path.sep+"kpop_"+self.label+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.fits'

        :return: None
        """

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
            csvwriter.writerows([["value","N false pos","N true pos"]])
            csvwriter.writerows(zip(self.threshold_sampling,self.N_false_pos,self.N_true_detec))
        return None

    def load(self):
        """

        :return: None
        """

        return None
