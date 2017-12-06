__author__ = 'JB'

import os
from glob import glob

import astropy.io.fits as pyfits
import numpy as np
import scipy.interpolate as interp

from pyklip.kpp.utils.kppSuperClass import KPPSuperClass
from pyklip.instruments import GPI
import pyklip.spectra_management as spec
import pyklip.fm as fm
import pyklip.fmlib.matchedFilter as mf
import pyklip.klip as klip
import pyklip.kpp.utils.oi as oi
import pyklip.kpp.utils.GPIimage as gpiim


class FMMF(KPPSuperClass):
    """
    Class calculating the Forward model matched filter of a dataset.
    """
    def __init__(self,read_func = None, filename = None,
                 folderName = None,
                 mute=None,
                 N_threads=None,
                 label = None,
                 overwrite=False,
                 SpT_file_csv=None,
                 PSF_read_func = None,
                 PSF_cube = None,
                 PSF_cube_wvs = None,
                 spectrum = None,
                 numbasis = None,
                 maxnumbasis = None,
                 flux_overlap = None,
                 mvt = None,
                 OWA = None,
                 N_pix_sector = None,
                 subsections = None,
                 annuli = None,
                 predefined_sectors = None,
                 quickTest = False,
                 mute_progression = False,
                 fakes_only = None,
                 disable_FM = None,
                 mvt_noTemplate = None,
                 true_fakes_pos = None,
                 keepPrefix = None,
                 compact_date_func = None,
                 filter_name_func = None,
                 pix2as=None,
                 highpass = None,
                 padding = None,
                 PSF_size = None,
                 rm_edge = None):
        """
        Define the general parameters of the matched filter.

        Args:
            read_func: lambda function returning a instrument object where the only input should be a list of filenames
                    to read.
                    For e.g.:
                    read_func = lambda filenames:GPI.GPIData(filenames,recalc_centers=False,recalc_wvs=False,highpass=True)
            filename: Filename of the file to process. Default "S*distorcorr.fits".
                        It should be the complete path unless inputDir is used in initialize().
                        It can include wild characters. The files will be reduced as given by glob.glob().
            folderName: foldername used in the definition of self.outputDir (where files shoudl be saved) in initialize().
                        folderName could be the name of the spectrum used for the reduction for e.g.
                        Default folder name is "default_out".
                        Convention is self.outputDir = #outputDir#/kpop_#labe#/#folderName#/
            mute: If True prevent printed log outputs.
            N_threads: Number of threads to be used.
                        If None use mp.cpu_count().
                        A sequential option is hard coded for debugging purposes only.
            label: label used in the definition of self.outputDir (where files shoudl be saved) in initialize().
                   Default is "default".
                   Convention is self.outputDir = #outputDir#/kpop_#labe#/#folderName#/
            overwrite: Boolean indicating whether or not files should be overwritten if they exist.
                       See check_existence().
            SpT_file_csv: Filename (.csv) of the table containing the target names and their spectral type.
                    Can be generated from quering Simbad.
                    If None (default), the function directly tries to query Simbad.
            PSF_read_func: lambda function used to read the PSF_cube.
                    It should return an object with at least two attributes input and wvs corresponding to the PSF cube and the associated wavelengths per slide.
                    The input shoudl be a list of filenames,
                    For e.g.: read_func = lambda filenames:GPI.GPIData(filenames,highpass=False)
            PSF_cube: Array or string defining the planet PSF.
                      - np.ndarray: Should have the same number of dimensions as the image.
                      - string: Read file using PSF_read_func()
                            (make sure the file follows the conventions of the instrument).
                            - if absolute path: Simply picks up that file
                            - otherwise search for a matching file based on the directory of the images.
                      - None: use default value "*-original_PSF_cube.fits"
            PSF_cube_wvs: List of wavelengths corresponding to PSF_cube slices.
            spectrum: spectrum name (string) or array used in the cube weighted mean when collapse is True.
                        - "host_star_spec": The spectrum from the star or the satellite spots is directly used.
                                            It is derived from the inverse of the calibrate_output() output.
                        - "constant": Use a constant spectrum np.ones(self.nl).
                        - other strings: name of the spectrum file in #pykliproot#/spectra/*/ with pykliproot the
                                        directory in which pyklip is installed. It that case it should be a spectrum
                                        from Mark Marley or one following the same convention.
                                        Spectrum will be corrected for transmission.
                        - ndarray: 1D array with a user defined spectrum. Spectrum will be corrected for transmission.
            numbasis: number of KL basis vectors to use (can be a scalar or list like). Length of b
                    If numbasis is [None] the number of KL modes to be used is automatically picked based on the eigenvalues.
            maxnumbasis: Number of KL modes to be calculated from which numbasis modes will be taken.
            flux_overlap: Maximum fraction of flux overlap between a slice and any reference frames included in the
                        covariance matrix. Flux_overlap should be used instead of "movement" when a template spectrum is used.
                        However if movement is not None then the old code is used and flux_overlap is ignored.
                        The overlap is estimated for 1D gaussians with FWHM defined by PSF_FWHM. So note that the overlap is
                        not exactly the overlap of two real 2D PSF for a given instrument but it will behave similarly.
            mvt: minimum amount of movement (in pixels) of an astrophysical source
                      to consider using that image for a refernece PSF
            OWA: (disabled) if defined, the outer working angle for pyklip. Otherwise, it will pick it as the cloest distance to a
                nan in the first frame
            N_pix_sector: Rough number of pixels in a sector. Overwriting subsections and making it sepration dependent.
                      The number of subsections is defined such that the number of pixel is just higher than N_pix_sector.
                      I.e. subsections = floor(pi*(r_max^2-r_min^2)/N_pix_sector)
                      Warning: There is a bug if N_pix_sector is too big for the first annulus. The annulus is defined from
                                0 to 2pi which create a bug later on. It is probably in the way pa_start and pa_end are
                                defined in fm_from_eigen().
            subsections: Sections to break each annuli into. Can be a number [integer], or a list of 2-element tuples (a, b)
                         specifying the positon angle boundaries (a <= PA < b) for each section [radians]
            anuuli: Annuli to use for KLIP. Can be a number, or a list of 2-element tuples (a, b) specifying
                    the pixel bondaries (a <= r < b) for each annulus
            predefined_sectors: Use predefined KLIP sectors instead of manually defining it with
                                    anuuli,subsections,N_pix_sector,OWA.
                                - "0.6 as": cover a disk of 0.6 as.
                                - "1 as": cover a disk of 1.0 as.
                                - "c_Eri": sector around expected position for 51 Eri b.
            quickTest: Read only two files (the first and the last) instead of the all sequence
            mute_progression: Mute the printing of the progression percentage in fm.klip_parallelized.
                            Indeed sometimes the overwriting feature doesn't work and one ends up with thousands of
                            printed lines. Therefore muting it can be a good idea.
            fakes_only: Only calculate the FMMF at the position of injected fake planets.
                    The following keywords need to be defined in the extension header of the input data:
                    FKPA##, FKSEP##,FKCONT##,FKPOSX##,FKPOSY## with ## and index for the simulated planet.
                    (Still only working for GPI... To be generalized)
            disable_FM: Disable the calculation of the forward model in the code.
                        The unchanged original PSF will be used instead. (Default False)
            mvt_noTemplate: If True, the reference library exclusion criterion is a simple displacement criterion.
                            Otherwise, adapt the exclusion depending on the expected planet spectrum.
                            (Default False)
            true_fakes_pos: If True and fakes_only is True, calculate the forward model at the exact position of the
                    fakes and not at the center of the pixels. (Default False)
            keepPrefix: (default = True) Keep the prefix of the input file instead of using the default:
                    self.prefix = self.star_name+"_"+self.compact_date+"_"+self.filter+self.add2prefix
            compact_date_func: lambda function returning the compact date (e.g. yyyymmdd) of the observation to be used
                                in the output filename if keepPrefix is False.
            filter_name_func: lambda function returning the name of the filter (e.g. "H", "J","K1"...) of the
                                observation to be used in the output filename if keepPrefix is False.
            pix2as: Platescale (arcsec per pixel). (Used if fakes_only is True)
            highpass: if True, run a Gaussian high pass filter (default size is sigma=imgsize/10)
                      can also be a number specifying FWHM of box in pixel units.
            PSF_size: Width of the PSF stamp to be used. Trim or pad with zeros the available PSF stamp.
            rm_edge: When True (default), remove image edges to avoid edge effect. When there is more than 25% of NaNs
                    in the projection of the FM model on the data, the result of the projection is set to NaNs right away.

        Return: instance of FMMF.
        """
        # allocate super class
        super(FMMF, self).__init__(read_func,filename,
                                     folderName = folderName,
                                     mute=mute,
                                     N_threads=N_threads,
                                     label=label,
                                     overwrite=overwrite,
                                     SpT_file_csv=SpT_file_csv)

        # Prevent the class to iterate over all the files matching filename
        self.process_all_files = False
        self.quickTest = quickTest
        self.mute_progression = mute_progression
        self.SpT_file_csv = SpT_file_csv
        self.spectrum = spectrum

        self.highpass = highpass
        self.rm_edge = rm_edge

        if filename is None:
            self.filename = "S*distorcorr.fits"
        else:
            self.filename = filename

        self.fakes_only = fakes_only
        if disable_FM is None:
            self.disable_FM = False
        else:
            self.disable_FM = disable_FM
        if mvt_noTemplate is None:
            self.mvt_noTemplate = False
        else:
            self.mvt_noTemplate = mvt_noTemplate

        self.PSF_cube = PSF_cube
        self.PSF_read_func = PSF_read_func
        self.PSF_cube_wvs = PSF_cube_wvs

        self.save_per_sector = None
        if padding is None:
            self.padding = 10
        else:
            self.padding = padding

        self.save_klipped = True
        self.true_fakes_pos = true_fakes_pos

        if numbasis is None:
            self.numbasis = np.array([30])
        else:
            self.numbasis = numbasis

        if maxnumbasis is None:
            self.maxnumbasis = 150
        else:
            self.maxnumbasis = maxnumbasis

        if flux_overlap is None:
            if mvt is None:
                self.flux_overlap = 0.7
                self.mvt = None
            else:
                self.flux_overlap = None
                self.mvt = mvt
        else:
            self.flux_overlap = flux_overlap
            self.mvt = None


        # self.OWA = OWA
        self.N_pix_sector = N_pix_sector

        if subsections is None:
            self.subsections = 4
        else:
            self.subsections = subsections

        if annuli is None:
            self.annuli = 5
        else:
            self.annuli = annuli

        if predefined_sectors == "0.6 as":
            self.N_pix_sector = 100
            self.subsections = None
            # Define 3 thin annuli (each ~5pix wide) and a big one (~20pix) to cover up to 0.6''
            self.annuli = [(8.7, 14.3), (14.3, 20), (20, 25.6),(25.6, 40.5)]#
        if predefined_sectors == "1.0 as":
            self.N_pix_sector = 100
            self.subsections = None
            # Define 3 thin annuli (each ~5pix wide) and 3 big ones (~20pix) to cover up to 1.0''
            self.annuli = [(8.7, 14.3), (14.3, 20), (20, 25.6),(25.6, 40.5),(40.5,55.5),(55.5,70.8)]
        elif predefined_sectors == "c_Eri":
            self.subsections = [[150./180.*np.pi,190./180.*np.pi]]
            self.annuli = [[23,41]]

        if keepPrefix is not None:
            self.keepPrefix = keepPrefix
        else:
            self.keepPrefix = True
            
        self.compact_date_func = compact_date_func
        self.compact_date = ""
        self.filter_name_func = filter_name_func

        self.pix2as = pix2as

        self.spectrum_name = ""
        self.spectrum_filename = ""
        self.PSF_cube_path = ""
        self.prefix = None
        self.PSF_size = PSF_size

    def spectrum_iter_available(self):
        """
        Indicates wether or not the class is equipped for iterating over spectra.
        Forward modelling matched filter requires a spectrum so the answer is yes.

        In order to iterate over spectra the function new_init_spectrum() can be called.
        spectrum_iter_available is a utility function for campaign data processing.

        :return: True
        """

        return True

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

        # Reread the dataset.
        self.image_obj = self.read_func(self.filename_path_list)

        if SpT_file_csv is not None:
            self.SpT_file_csv = SpT_file_csv
        if spectrum is not None:
            self.spectrum = spectrum

        # use super class
        super(FMMF, self).init_new_spectrum(self.spectrum,SpT_file_csv=self.SpT_file_csv)

        if not self.keepPrefix:
            if self.flux_overlap is not None:
                self.prefix = self.star_name+"_"+self.compact_date+"_"+self.filter+"_"+self.spectrum_name +"_{0:.2f}".format(self.flux_overlap)
            else:
                self.prefix = self.star_name+"_"+self.compact_date+"_"+self.filter+"_"+self.spectrum_name +"_{0:.2f}".format(self.mvt)

        # Make sure the total flux of each PSF is unity for all wavelengths
        # So the peak value won't be unity.
        self.PSF_cube_arr = self.PSF_cube_arr/np.nansum(self.PSF_cube_arr,axis=(1,2))[:,None,None]
        # Get the conversion factor from peak spectrum to aperture based spectrum
        self.aper_over_peak_ratio = 1/np.nanmax(self.PSF_cube_arr,axis=(1,2))
        aper_over_peak_ratio_tiled = np.zeros(self.nl)#wavelengths
        for k,wv in enumerate(self.image_obj.wvs):
            aper_over_peak_ratio_tiled[k] = self.aper_over_peak_ratio[spec.find_nearest(self.PSF_cube_wvs,wv)[1]]
        # Summed DN flux of the star in the entire dataset calculated from dn_per_contrast
        self.star_flux = np.sum(aper_over_peak_ratio_tiled*self.dn_per_contrast)
        self.fake_contrast = 1. # ratio of flux of the planet/flux of the star (broad band flux)
        # normalize the spectra to unit contrast.
        self.spectrum_vec = self.spectrum_vec/np.sum(self.spectrum_vec)*self.star_flux*self.fake_contrast

        # Build the FM class to do matched filter
        self.fm_class = mf.MatchedFilter(self.image_obj.input.shape,self.numbasis, self.PSF_cube_arr, self.PSF_cube_wvs,
                                         spectrallib = [self.spectrum_vec],
                                         save_per_sector = self.save_per_sector,
                                         fakes_sepPa_list = self.fakes_sepPa_list,
                                         disable_FM=self.disable_FM,
                                         true_fakes_pos= self.true_fakes_pos,
                                         ref_center=[np.mean(self.image_obj.centers[:,0]), np.mean(self.image_obj.centers[:,1])],
                                         flipx=self.image_obj.flipx,
                                         rm_edge=self.rm_edge)
        return None

    def initialize(self,inputDir = None,
                         outputDir = None,
                         spectrum = None,
                         folderName = None,
                         label = None):
        """
        Read the file using read_func (see the class  __init__ function).

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
        init_out = super(FMMF, self).initialize(inputDir = inputDir,
                                             outputDir = outputDir,
                                             folderName = folderName,
                                             label=label,
                                             read=False)

        # Check file existence and define filename_path
        if self.inputDir is None or os.path.isabs(self.filename):
            try:
                self.filename_path_list = [os.path.abspath(filename) for filename in glob(self.filename)]
            except:
                raise Exception("File "+self.filename+"doesn't exist.")
        else:
            try:
                self.filename_path_list = [os.path.abspath(filename) for filename in glob(self.inputDir+os.path.sep+self.filename)]
            except:
                raise Exception("File "+self.inputDir+os.path.sep+self.filename+" doesn't exist.")

        self.filename_path_list.sort()
        if self.quickTest:
            self.filename_path_list = [self.filename_path_list[0],self.filename_path_list[-1]]

        # Read the image using the user defined reading function
        self.image_obj = self.read_func(self.filename_path_list)
        self.nl,self.ny,self.nx = self.image_obj.input.shape
        try:
            self.star_name = self.image_obj.object_name.replace(" ","_")
        except:
            self.star_name = None

        if spectrum is not None:
            self.spectrum = spectrum

        # use super class
        super(FMMF, self).init_new_spectrum(self.spectrum,SpT_file_csv=self.SpT_file_csv)

        try:
            hdulist = pyfits.open(self.filename_path_list[0])
            self.prihdr = hdulist[0].header
            hdulist.close()
        except:
            self.prihdr = None
        try:
            hdulist = pyfits.open(self.filename_path_list[0])
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
            self.outputDir = os.path.abspath(outputDir+os.path.sep+"kpop_"+self.label)
        else: # if self.outputDir is None:
            self.outputDir = os.path.join(os.path.dirname(self.filename_path_list[0]),"kpop_"+self.label)

        if self.compact_date_func is None:
            self.compact_date = "noDate"
        else:
            self.compact_date = self.compact_date_func(self.filename_path_list[0])
        if self.filter_name_func is None:
            self.filter_name = "noFilter"
        else:
            self.filter_name = self.filter_name_func(self.filename_path_list[0])

        if self.keepPrefix:
            if self.flux_overlap is not None:
                file_ext_ind = os.path.basename(self.filename_path_list[0])[::-1].find(".")
                self.prefix = os.path.basename(self.filename_path_list[0])[:-(file_ext_ind+1)]+"_"+self.folderName +"_{0:.2f}".format(self.flux_overlap)
            else:
                file_ext_ind = os.path.basename(self.filename_path_list[0])[::-1].find(".")
                self.prefix = os.path.basename(self.filename_path_list[0])[:-(file_ext_ind+1)]+"_"+self.folderName +"_{0:.2f}".format(self.mvt)
        else:
            if self.flux_overlap is not None:
                self.prefix = self.star_name+"_"+self.compact_date+"_"+self.filter+"_"+self.folderName +"_{0:.2f}".format(self.flux_overlap)
            else:
                self.prefix = self.star_name+"_"+self.compact_date+"_"+self.filter+"_"+self.folderName +"_{0:.2f}".format(self.mvt)

        # Build the PSF cube
        if isinstance(self.PSF_cube, np.ndarray):
            self.PSF_cube_arr = self.PSF_cube
            self.PSF_cube_path = "None"
        else:
            self.PSF_cube_filename = self.PSF_cube

            if self.PSF_cube_filename is None:
                self.PSF_cube_filename = "*-original_PSF_cube.fits"
                if not self.mute:
                    print("Using default filename for PSF cube: "+self.PSF_cube_filename)

            if os.path.isabs(self.PSF_cube_filename):
                self.PSF_cube_path = os.path.abspath(glob(self.PSF_cube_filename)[0])
            else:
                base_path = os.path.dirname(self.filename_path_list[0])
                self.PSF_cube_path = os.path.abspath(glob(os.path.join(base_path,self.PSF_cube_filename))[0])

            if not self.mute:
                print("Loading PSF cube: "+self.PSF_cube_path)
            # Load the PSF cube if a file has been found or was just generated
            self.PSF_obj = self.PSF_read_func([self.PSF_cube_path])
            self.PSF_cube_arr = self.PSF_obj.input
            self.PSF_cube_wvs = self.PSF_obj.wvs

        if self.PSF_size is not None:
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

        # read fakes from headers and give sepPa list to MatchedFilter
        if self.fakes_only:
            sep_list, pa_list = oi.get_pos_known_objects(self.fakeinfohdr,self.star_name,self.pix2as,OI_list_folder=None,
                                                         xy = False,pa_sep = True,fakes_only=True)
            self.fakes_sepPa_list = [(1./self.pix2as*sep,pa) for sep,pa in zip(sep_list, pa_list)]
        else:
            self.fakes_sepPa_list = None

        # Make sure the total flux of each PSF is unity for all wavelengths
        # So the peak value won't be unity.
        self.PSF_cube_arr = self.PSF_cube_arr/np.nansum(self.PSF_cube_arr,axis=(1,2))[:,None,None]
        # Get the conversion factor from peak spectrum to aperture based spectrum
        self.aper_over_peak_ratio = 1/np.nanmax(self.PSF_cube_arr,axis=(1,2))
        aper_over_peak_ratio_tiled = np.zeros(self.nl)#wavelengths
        for k,wv in enumerate(self.image_obj.wvs):
            aper_over_peak_ratio_tiled[k] = self.aper_over_peak_ratio[spec.find_nearest(self.PSF_cube_wvs,wv)[1]]
        # Summed DN flux of the star in the entire dataset calculated from dn_per_contrast
        self.star_flux = np.nansum(aper_over_peak_ratio_tiled*self.dn_per_contrast)
        self.fake_contrast = 1. # ratio of flux of the planet/flux of the star (broad band flux)
        # normalize the spectra to unit contrast.
        self.spectrum_vec = self.spectrum_vec/np.sum(self.spectrum_vec)*self.star_flux*self.fake_contrast

        # Build the FM class to do matched filter
        self.fm_class = mf.MatchedFilter(self.image_obj.input.shape,self.numbasis, self.PSF_cube_arr, self.PSF_cube_wvs,
                                         spectrallib = [self.spectrum_vec],
                                         save_per_sector = self.save_per_sector,
                                         fakes_sepPa_list = self.fakes_sepPa_list,
                                         disable_FM=self.disable_FM,
                                         true_fakes_pos= self.true_fakes_pos,
                                         # ref_center=[np.mean(self.image_obj.centers[:,0]), np.mean(self.image_obj.centers[:,1])],
                                         # flipx=self.image_obj.flipx,
                                         rm_edge=self.rm_edge)

        return init_out

    def check_existence(self):
        """
        Check if FMMF has already been calculated for this dataset.

        If self.overwrite is True then the output will always be False

        :return: Boolean indicating the existence of the reduced data.
        """

        if self.quickTest:
            susuffix = "QT"
        else:
            susuffix = ""

        if self.disable_FM:
            presuffix = "no"
        else:
            presuffix = ""

        file_exist = True
        for nmbasis in self.numbasis:
            suffix1 = presuffix+"FMMF-KL{0}".format(nmbasis)+susuffix
            file_exist= file_exist and (len(glob(self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+suffix1+'.fits')) >= 1)

        if file_exist and not self.mute:
            print("Output already exist.")

        return file_exist and not self.overwrite


    def calculate(self,dataset=None,spectrum=None,fm_class=None):
        """
        Perform a matched filter on the current loaded file.

        Args:
            dataset: Instance of an instrument class.
            spectrum: Transmission corrected spectrum to be used in reference library selection.
            fm_class: Instance of the fmlib.MatchedFilter class.
                    (Note: the parameters defined in the definition of the FMMF class meant for the MatchedFilter class
                    won't be taken into account in this case)



        Return: [self.FMMF_map,self.FMCC_map,self.contrast_map,self.final_cube_modes]
            FMMF_map: Forward model matched filter map
            FMCC_map: Forward model cross correlation map
            contrast_map: Forward model estimated contrast map
            final_cube_modes: Classic klipped cubes for each # of KL modes.
        """
        if dataset is not None:
            self.image_obj = dataset

        if spectrum is not None:
            self.spectrum_vec = spectrum

        if fm_class is not None:
            self.fm_class = fm_class

        # high pass filter?
        from pyklip.fm import high_pass_filter_imgs
        if isinstance(self.highpass, bool):
            if self.highpass:
                self.image_obj.input = high_pass_filter_imgs(self.image_obj.input, numthreads=self.N_threads)
        else:
            # should be a number
            if isinstance(self.highpass, (float, int)):
                highpass = float(self.highpass)
                fourier_sigma_size = (self.image_obj.input.shape[1]/(highpass)) / (2*np.sqrt(2*np.log(2)))
                self.image_obj.input = high_pass_filter_imgs(self.image_obj.input, numthreads=self.N_threads, filtersize=fourier_sigma_size)

        # import matplotlib.pyplot as plt
        # plt.imshow(self.PSF_cube_arr[10,:,:],interpolation="nearest")
        # plt.show()
        # plt.figure(1)
        # plt.plot(self.image_obj.wvs,self.spectrum_vec)
        # plt.figure(2)
        # plt.plot(self.image_obj.wvs,self.host_star_spec)
        # plt.figure(3)
        # plt.plot(self.image_obj.wvs,self.star_sp)
        # plt.figure(4)
        # print(self.image_obj.input.shape)
        # # plt.plot(self.image_obj.wvs,np.sum(self.image_obj.input[:,33:35,9:11],axis=(1,2)))
        # plt.plot(self.image_obj.wvs,np.nansum(self.image_obj.input[:,32:36,8:12],axis=(1,2)))
        # plt.show()
        #todo to remove
        # self.spectrum_vec = np.nansum(self.image_obj.input[:,32:36,8:12],axis=(1,2))
        # self.spectrum_vec=self.spectrum_vec/self.spectrum_vec

        # Run KLIP with the forward model matched filter
        sub_imgs, fmout,tmp, klipped_center = fm.klip_parallelized(self.image_obj.input, self.image_obj.centers, self.image_obj.PAs, self.image_obj.wvs, self.image_obj.IWA, self.fm_class,
                                   numbasis=self.numbasis,
                                   maxnumbasis=self.maxnumbasis,
                                   flux_overlap=self.flux_overlap,
                                   movement=self.mvt,
                                   spectrum=self.spectrum_vec,
                                   annuli=self.annuli,
                                   subsections=self.subsections,
                                   numthreads=self.N_threads,
                                   padding = self.padding,
                                   N_pix_sector=self.N_pix_sector,
                                   save_klipped = self.save_klipped,
                                   OWA = self.image_obj.OWA,
                                   mute_progression = self.mute_progression,
                                   flipx = self.image_obj.flipx)

        #fmout_shape = (3,self.N_spectra,self.N_numbasis,self.N_frames,self.ny,self.nx)
        fmout[np.where(fmout==0)] = np.nan

        # The mf.MatchedFilter class calculate the projection of the FM on the data for each pixel and images.
        # The final combination to form the cross  cross correlation, matched filter and contrast maps is done right
        # here.
        FMCC_map = np.nansum(fmout[0,:,:,:,:,:],axis=2) \
                        / np.sqrt(np.nansum(fmout[1,:,:,:,:,:],axis=2))
        FMCC_map[np.where(FMCC_map==0)]=np.nan
        self.FMCC_map = FMCC_map

        FMMF_map = np.nansum(fmout[0,:,:,:,:,:]/fmout[2,:,:,:,:,:],axis=2) \
                        / np.sqrt(np.nansum(fmout[1,:,:,:,:,:]/fmout[2,:,:,:,:,:],axis=2))
        FMMF_map[np.where(FMMF_map==0)]=np.nan
        self.FMMF_map = FMMF_map

        contrast_map = np.nansum(fmout[0,:,:,:,:,:]/fmout[2,:,:,:,:,:],axis=2) \
                        / np.nansum(fmout[1,:,:,:,:,:]/fmout[2,:,:,:,:,:],axis=2)
        contrast_map[np.where(contrast_map==0)]=np.nan
        self.contrast_map = contrast_map

        self.N_pix_mf = np.nansum(fmout[3,:,:,:,:,:],axis=2)

        # Update the wcs headers to indicate North up
        if self.image_obj.wcs[0] is not None:
            [klip._rotate_wcs_hdr(astr_hdr, angle, flipx=True) for angle, astr_hdr in zip(self.image_obj.PAs, self.image_obj.wcs)]


        self.image_obj.output_wcs = np.array([w.deepcopy() for w in self.image_obj.wcs])
        self.image_obj.output_centers = np.array([klipped_center for _ in range(sub_imgs.shape[1])])

        # Form regular klipped cubes
        self.sub_imgs = sub_imgs
        N_unique_wvs = len(np.unique(self.image_obj.wvs))
        self.N_cubes = len(self.image_obj.wvs)//N_unique_wvs
        cubes_list = []
        for k in range(self.N_cubes):
            cubes_list.append(sub_imgs[:,k*N_unique_wvs:(k+1)*N_unique_wvs,:,:])
        self.final_cube_modes = np.sum(cubes_list,axis = 0)
        self.final_cube_modes[np.where(self.final_cube_modes==0)] = np.nan

        self.metricMap = [self.FMMF_map,self.FMCC_map,self.contrast_map,self.final_cube_modes]

        return self.metricMap


    def save(self,outputDir = None,folderName = None,prefix=None):
        """
        Save the processed files as:
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

        if not os.path.exists(self.outputDir+os.path.sep+self.folderName):
            os.makedirs(self.outputDir+os.path.sep+self.folderName)

        # Save the parameters as fits keywords
        extra_keywords = {"KPPFOLDN":self.folderName,
                          "KPPLABEL":self.label,
                          "KPPNUMBA":str(self.numbasis),
                          "KPPMAXNB":self.maxnumbasis,
                          "KPPFLXOV":str(self.flux_overlap),
                          "KPP_MVT":str(self.mvt),
                          "KPPNPIXS":self.N_pix_sector,
                          "KPPSUBSE":str(self.subsections),
                          "KPPANNUL":str(self.annuli),
                          "KPPMAXNB":self.maxnumbasis,
                          "KPPINDIR":self.inputDir,
                          "KPPOUTDI":self.outputDir,
                          "KPPFOLDN":self.folderName,
                          "KPPCDATE":self.compact_date,
                          "KPPSPECN":self.spectrum_name,
                          "KPPSPECF":self.spectrum_filename,
                          "KPPPSFDI":self.PSF_cube_path,
                          "KPPPREFI":self.prefix,
                          "KPPQUICT":self.quickTest}

        if hasattr(self,"star_type"):
            extra_keywords["KPPSTTYP"] = self.star_type

        extra_keywords["FMMFVERS"] = 2.0

        if self.quickTest:
            susuffix = "QT"
        else:
            susuffix = ""

        if self.disable_FM:
            presuffix = "no"
        else:
            presuffix = ""

        for k in range(self.final_cube_modes.shape[0]):
            # Save the outputs (matched filter, shape map and klipped image) as fits files
            suffix = presuffix+"FMMF-KL{0}".format(self.numbasis[k])+susuffix
            extra_keywords["KPPSUFFI"]=suffix
            self.image_obj.savedata(self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+suffix+'.fits',
                             self.FMMF_map[0,k,:,:],
                             filetype=suffix,
                             more_keywords = extra_keywords,pyklip_output=True)
            suffix = presuffix+"FMCont-KL{0}".format(self.numbasis[k])+susuffix
            extra_keywords["KPPSUFFI"]=suffix
            self.image_obj.savedata(self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+suffix+'.fits',
                             self.contrast_map[0,k,:,:],
                             filetype=suffix,
                             more_keywords = extra_keywords,pyklip_output=True)
            suffix = presuffix+"FMCC-KL{0}".format(self.numbasis[k])+susuffix
            extra_keywords["KPPSUFFI"]=suffix
            self.image_obj.savedata(self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+suffix+'.fits',
                             self.FMCC_map[0,k,:,:],
                             filetype=suffix,
                             more_keywords = extra_keywords,pyklip_output=True)
            suffix = presuffix+"FMNpix-KL{0}".format(self.numbasis[k])+susuffix
            extra_keywords["KPPSUFFI"]=suffix
            self.image_obj.savedata(self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+suffix+'.fits',
                             self.N_pix_mf[0,k,:,:],
                             filetype=suffix,
                             more_keywords = extra_keywords,pyklip_output=True)
            suffix = "speccube-KL{0}".format(self.numbasis[k])+susuffix
            extra_keywords["KPPSUFFI"]=suffix
            self.image_obj.savedata(self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+suffix+'.fits',
                             self.final_cube_modes[k],
                             filetype="PSF subtracted spectral cube with fmpyklip",
                             more_keywords = extra_keywords,pyklip_output=True)

        return None


    def load(self):
        """
        Load the metric map. One should check that it exist first using self.check_existence().

        Define the attribute self.metricMap.

        :return: self.metricMap
        """
        # suffix = "FMMF"
        # hdulist = pyfits.open(self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+suffix+'.fits')
        # self.metric_MF = hdulist[1].data
        # hdulist.close()
        # suffix = "FMSH"
        # hdulist = pyfits.open(self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+suffix+'.fits')
        # self.FMMF_map = hdulist[1].data
        # hdulist.close()
        # suffix = "speccube-PSFs"
        # hdulist = pyfits.open(self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+suffix+'.fits')
        # self.sub_imgs = hdulist[1].data
        # hdulist.close()
        # self.metricMap = [self.metric_MF,self.FMMF_map,self.sub_imgs]


        return None
