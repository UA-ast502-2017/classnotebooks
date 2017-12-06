import multiprocessing as mp
import numpy as np
import os
import copy
import pickle
import glob
import scipy.ndimage as ndimage
import ctypes

from pyklip.fmlib.nofm import NoFM
import pyklip.fm as fm
from pyklip.klip import rotate
import itertools



class DiskFM(NoFM):
    def __init__(self, inputs_shape, numbasis, dataset, model_disk, basis_filename = 'klip-basis.p', load_from_basis = False, save_basis = False, annuli = None, subsections = None, OWA = None, numthreads = None, mode = 'ADI'):
        '''
        Takes an input model and runs KLIP-FM. Can be used in MCMCs by saving the basis 
        vectors. When disk is updated, FM can be run on the new disk without computing new basis
        vectors. 

        For first time, instantiate DiskFM with no save_basis and nominal model disk.
        Specify number of annuli and subsections used to save basis vectors

        Currently only supports mode = ADI
        '''
        super(DiskFM, self).__init__(inputs_shape, numbasis)

        # Attributes of input/output
        self.inputs_shape = inputs_shape
        self.numbasis = numbasis
        self.numims = inputs_shape[0]
        self.mode = mode

        # Input dataset attributes
        self.dataset = dataset
        self.IWA = dataset.IWA
        self.images = dataset.input
        self.pas = dataset.PAs
        self.centers = dataset.centers
        self.wvs = dataset.wvs
        
        # Outputs attributes
        output_imgs_shape = self.images.shape + self.numbasis.shape
        self.output_imgs_shape = output_imgs_shape
        self.outputs_shape = output_imgs_shape
        self.np_data_type = ctypes.c_float

        # Coords where align_and_scale places model center (default is inputs center).
        self.aligned_center = [int(self.inputs_shape[2]//2), int(self.inputs_shape[1]//2)]

        # Make disk reference PSFS
        self.update_disk(model_disk)

        self.save_basis = save_basis
        self.annuli = annuli
        self.subsections = subsections
        self.OWA = OWA

        self.basis_filename = basis_filename
        self.load_from_basis = load_from_basis

        x,y = np.meshgrid(np.arange(inputs_shape[2] * 1.0),np.arange(inputs_shape[1]*1.0))
        nanpix = np.where(np.isnan(dataset.input[0]))
        if OWA is None:
            if np.size(nanpix) == 0:
                OWA = np.sqrt(np.max((x - self.centers[0][0]) ** 2 + (y - self.centers[0][1]) ** 2))
            else:
                # grab the NaN from the 1st percentile (this way we drop outliers)    
                OWA = np.sqrt(np.percentile((x[nanpix] - self.centers[0][0]) ** 2 + (y[nanpix] - self.centers[0][1]) ** 2, 1))
        self.OWA = OWA



        if numthreads == None:
            self.numthreads = mp.cpu_count()
        else:
            self.numthreads = numthreads

        if self.save_basis == True:
            # Need to know r and phi indicies in fm from eigen
            assert annuli is not None, "need annuli keyword to save basis"
            assert subsections is not None, "need subsections keyword to save basis"
            x, y = np.meshgrid(np.arange(inputs_shape[2] * 1.0), np.arange(inputs_shape[1] * 1.0))
            self.dr = (OWA - dataset.IWA) / annuli
            self.dphi = 2 * np.pi / subsections
            
            # Set up dictionaries for saving basis
            manager = mp.Manager()
            global klmodes_dict, evecs_dict, evals_dict, ref_psfs_indicies_dict, section_ind_dict
            klmodes_dict = manager.dict()
            evecs_dict = manager.dict()
            evals_dict = manager.dict()
            ref_psfs_indicies_dict = manager.dict()
            section_ind_dict = manager.dict()

        if load_from_basis is True:
            self.load_basis_files(basis_filename)


    def alloc_fmout(self, output_img_shape):
        ''' 
       Allocates shared memory for output image 
        '''
        fmout_size = np.prod(output_img_shape)
        fmout_shape = output_img_shape
        fmout = mp.Array(self.data_type, fmout_size)
        return fmout, fmout_shape

    def fm_from_eigen(self, klmodes=None, evals=None, evecs=None, input_img_shape=None, input_img_num=None, ref_psfs_indicies=None, section_ind=None,section_ind_nopadding=None, aligned_imgs=None, pas=None,
                     wvs=None, radstart=None, radend=None, phistart=None, phiend=None, padding=None,IOWA = None, ref_center=None,
                     parang=None, ref_wv=None, numbasis=None, fmout=None, perturbmag=None, klipped=None, covar_files=None, **kwargs):
        '''
        FIXME
        '''
        sci = aligned_imgs[input_img_num, section_ind[0]]

        refs = aligned_imgs[ref_psfs_indicies, :]
        refs = refs[:, section_ind[0]]
        refs[np.where(np.isnan(refs))] = 0

        model_sci = self.model_disks[input_img_num, section_ind[0]]

        model_ref = self.model_disks[ref_psfs_indicies, :]
        model_ref = model_ref[:, section_ind[0]]
        model_ref[np.where(np.isnan(model_ref))] = 0

        delta_KL= fm.perturb_specIncluded(evals, evecs, klmodes, refs, model_ref, return_perturb_covar = False)
        postklip_psf, oversubtraction, selfsubtraction = fm.calculate_fm(delta_KL, klmodes, numbasis, sci, model_sci, inputflux = None)

        for thisnumbasisindex in range(np.size(numbasis)):
            self._save_rotated_section(input_img_shape, postklip_psf[thisnumbasisindex], section_ind,
                                       fmout[input_img_num, :, :,thisnumbasisindex], None, parang,
                                       radstart, radend, phistart, phiend,  padding,IOWA, ref_center, flipx=True) # FIXME


        if self.save_basis is True:
            curr_rad = str(int(np.round((radstart - self.dataset.IWA) / self.dr)))
            curr_sub = str(int(np.round(phistart / self.dphi)))
            curr_im = str(input_img_num)
            if len(curr_im) < 2:
                curr_im = '0' + curr_im

            # FIXME save per wavelength
            nam = 'r' + curr_rad + 's' + curr_sub + 'i' + curr_im 
            
            klmodes_dict[nam] = klmodes
            evals_dict[nam] = evals
            evecs_dict[nam] = evecs
            ref_psfs_indicies_dict[nam] = ref_psfs_indicies
            section_ind_dict[nam] = section_ind
            
    def fm_parallelized(self):
        '''
        Functions like klip_parallelized, but doesn't find new 
        evals and evecs. 
        '''



        fmout_data, fmout_shape = self.alloc_fmout(self.output_imgs_shape)
        fmout_np = fm._arraytonumpy(fmout_data, fmout_shape, dtype = self.np_data_type)
        
        # Parallelilze this
        for key in self.dict_keys:
            rad = int(key[1])
            phi = int(key[3])
            img_num = int(key[5:])
            
            radstart = self.dr * rad + self.IWA
            radend = self.dr * (rad + 1) + self.IWA
            phistart = self.dphi * phi
            phiend = self.dphi  * (phi + 1)
            # FIXME This next line makes subsection != 1 have overlapping sections, but breaks
            # subsection = 1. Need to add padding option to fix this
#            phi_bounds[-1][1] = 2. * np.pi - 0.0001  
 
            section_ind = self.section_ind_dict[key]
            sector_size = np.size(section_ind)
            original_KL = self.klmodes_dict[key]
            evals = self.evals_dict[key]
            evecs = self.evecs_dict[key]
            ref_psfs_indicies = self.ref_psfs_indicies_dict[key] 
            
            parallel = False 
        
            if not parallel:
                self.fm_from_eigen(klmodes=original_KL, evals=evals, evecs=evecs,
                                   input_img_shape=[original_shape[1], original_shape[2]], input_img_num=img_num,
                                   ref_psfs_indicies=ref_psfs_indicies, section_ind=section_ind, aligned_imgs=self.aligned_imgs_np,

                                   pas=self.pa_imgs_np[ref_psfs_indicies], wvs=self.wvs_imgs_np[ref_psfs_indicies], radstart=radstart,
                                   radend=radend, phistart=phistart, phiend=phiend, padding=0.,IOWA = (self.IWA, self.OWA), ref_center=self.aligned_center,
                                   parang=self.pa_imgs_np[img_num], ref_wv=None, numbasis=self.numbasis,
                                   fmout=fmout_np,perturbmag = None, klipped=None, covar_files=None)

            else:
                pass

        fmout_np = fm._arraytonumpy(fmout_data, fmout_shape, dtype = self.np_data_type)
        fmout_np = self.cleanup_fmout(fmout_np)

        return fmout_np
            


    def load_basis_files(self, basis_file_pattern):
        '''
        Loads in previously saved basis files and sets variables for fm_from_eigen
        '''

        # Load in file
        f = open(basis_file_pattern)
        self.klmodes_dict = pickle.load(f)
        self.evecs_dict = pickle.load(f)
        self.evals_dict = pickle.load(f)
        self.ref_psfs_indicies_dict = pickle.load(f)
        self.section_ind_dict = pickle.load(f)

        # Set extents for each section
        self.dict_keys = sorted(self.klmodes_dict.keys())
        rads = [int(key[1]) for key in self.dict_keys]
        phis = [int(key[3]) for key in self.dict_keys]
        self.annuli = np.max(rads) + 1
        self.subsections = np.max(phis) + 1

        # More geometry parameters
        x, y = np.meshgrid(np.arange(self.inputs_shape[2] * 1.0), np.arange(self.inputs_shape[1] * 1.0))
        nanpix = np.where(np.isnan(self.dataset.input[0]))
        self.dr = (self.OWA - self.dataset.IWA) / self.annuli
        self.dphi = 2 * np.pi / self.subsections
        
        # Make flattened images for running paralellized
        original_imgs = mp.Array(self.data_type, np.size(self.images))
        original_imgs_shape = self.images.shape
        original_imgs_np = fm._arraytonumpy(original_imgs, original_imgs_shape,dtype=self.np_data_type)
        original_imgs_np[:] = self.images


        # make array for recentered/rescaled image for each wavelength                               
        unique_wvs = np.unique(self.wvs)
        recentered_imgs = mp.Array(self.data_type, np.size(self.images)*np.size(unique_wvs))
        recentered_imgs_shape = (np.size(unique_wvs),) + self.images.shape

        # remake the PA, wv, and center arrays as shared arrays                  
        pa_imgs = mp.Array(self.data_type, np.size(self.pas))
        pa_imgs_np = fm._arraytonumpy(pa_imgs,dtype=self.np_data_type)
        pa_imgs_np[:] = self.pas
        wvs_imgs = mp.Array(self.data_type, np.size(self.wvs))
        wvs_imgs_np = fm._arraytonumpy(wvs_imgs,dtype=self.np_data_type)
        wvs_imgs_np[:] = self.wvs
        centers_imgs = mp.Array(self.data_type, np.size(self.centers))
        centers_imgs_np = fm._arraytonumpy(centers_imgs, self.centers.shape,dtype=self.np_data_type)
        centers_imgs_np[:] = self.centers
        output_imgs = None
        output_imgs_numstacked = None
        output_imgs_shape = self.images.shape + self.numbasis.shape
        self.output_imgs_shape = output_imgs_shape
        self.outputs_shape = output_imgs_shape

        perturbmag, perturbmag_shape = self.alloc_perturbmag(self.output_imgs_shape, self.numbasis)

        
        # Making MP arrays
        fmout_data = None
        fmout_shape = None
        tpool = mp.Pool(processes=self.numthreads, initializer=fm._tpool_init,initargs=(original_imgs, original_imgs_shape, recentered_imgs, recentered_imgs_shape, output_imgs, self.output_imgs_shape, output_imgs_numstacked, pa_imgs, wvs_imgs, centers_imgs, None, None, fmout_data, fmout_shape,perturbmag,perturbmag_shape), maxtasksperchild=50)
        
        # Okay if these are global variables right now, can make them local later
        self._tpool_init(original_imgs, original_imgs_shape, recentered_imgs, recentered_imgs_shape, output_imgs,self.output_imgs_shape, output_imgs_numstacked, pa_imgs, wvs_imgs, centers_imgs, None, None,fmout_data, fmout_shape,perturbmag,perturbmag_shape)

        fmout_data = None
        fmout_shape = None
    
        print("Begin align and scale images for each wavelength")
        aligned_outputs = []
        for threadnum in range(self.numthreads):
            aligned_outputs += [tpool.apply_async(fm._align_and_scale_subset, args=(threadnum, self.aligned_center,self.numthreads,self.np_data_type))]         
            #save it to shared memory                                           
        for aligned_output in aligned_outputs:
            aligned_output.wait()


        self.aligned_imgs_np = fm._arraytonumpy(aligned, shape = (original_imgs_shape[0], original_imgs_shape[1] * original_imgs_shape[2]))

        self.wvs_imgs_np = wvs_imgs_np
        self.pa_imgs_np = pa_imgs_np

        #Don't need to save the bases again! 
        self.save_basis = False

        # Delete global variables so it can pickle
        del pa_imgs
        del wvs_imgs
        del original_imgs
        del original_imgs_shape
        del original_imgs_np
        del recentered_imgs
        del recentered_imgs_shape
        del centers_imgs_np
        del fmout_data
        del fmout_shape
        del output_imgs
        del output_imgs_shape
        del output_imgs_numstacked
        del centers_imgs
        del wvs_imgs_np
        del pa_imgs_np


    def save_fmout(self, dataset, fmout, outputdir, fileprefix, numbasis, klipparams=None, calibrate_flux=False, spectrum=None):
        '''
        Uses self.dataset parameters to save fmout, the output of
        fm_paralellized or klip_dataset
        '''

        #Collapsed across all files (and wavelenths) and divide by number of images to keep units as ADU/coadd
        KLmode_cube = np.nanmean(fmout, axis = 1)/self.inputs_shape[0] 

        #Check if we have a disk model at multiple wavelengths
        model_disk_shape = np.shape(self.model_disk)        
        #If true then it's a spec mode diskand save indivudal specmode cubes for each KL mode
        if np.size(model_disk_shape) > 2: 

            nfiles = int(np.nanmax(self.dataset.filenums))+1 #Get the number of files  
            n_wv_per_file = self.inputs_shape[0]/nfiles #Number of wavelenths per file. 

            ##Collapse across all files, keeping the wavelengths intact. 
            KLmode_spectral_cubes = np.zeros([np.size(numbasis),n_wv_per_file,self.inputs_shape[1],self.inputs_shape[2]])
            for i in np.arange(n_wv_per_file):
                KLmode_spectral_cubes[:,i,:,:] = np.nansum(fmout[:,i::n_wv_per_file,:,:], axis =1)/nfiles
            
            for KLcutoff, spectral_cube in zip(numbasis, KLmode_spectral_cubes):
                # calibrate spectral cube if needed
                dataset.savedata(outputdir + '/' + fileprefix + "-fmpsf-KL{0}-speccube.fits".format(KLcutoff),
                                 spectral_cube, klipparams=klipparams.format(numbasis=KLcutoff),
                                 filetype="PSF Subtracted Spectral Cube")


        dataset.savedata(outputdir + '/' + fileprefix + "-fmpsf-KLmodes-all.fits", KLmode_cube,
                         klipparams=klipparams.format(numbasis=str(numbasis)), filetype="KL Mode Cube",
                         zaxis=numbasis)


    def cleanup_fmout(self, fmout):
        # will need to fix later
        """
        After running KLIP-FM, we need to reshape fmout so that the numKL dimension is the first one and not the last

        Args:
            fmout: numpy array of ouput of FM

        Returns:
            fmout: same but cleaned up if necessary
        """
        if self.save_basis == True:
            f = open(self.basis_filename, 'wb')
            pickle.dump(dict(klmodes_dict), f)
            pickle.dump(dict(evecs_dict), f)
            pickle.dump(dict(evals_dict), f)
            pickle.dump(dict(ref_psfs_indicies_dict), f)
            pickle.dump(dict(section_ind_dict), f)
        dims = fmout.shape
        fmout = np.rollaxis(fmout.reshape((dims[0], dims[1], dims[2], dims[3])), 3)
        return fmout

    def update_disk(self, model_disk):
        '''
        Takes model disk and rotates it to the PAs of the input images for use as reference PSFS
       
        Args: 
             model_disk: Disk to be forward modeled, can be either a 2D array ([N,N], where N is the width and height of your image)
             in which case, if the dataset is multiwavelength then the same model is used for all wavelenths. Otherwise, the model_disk is 
             input as a 3D arary, [nwvs, N,N], where nwvs is the number of wavelength channels).  
        Returns:
             None
        '''

        self.model_disk = model_disk
        self.model_disks = np.zeros(self.inputs_shape)

        model_disk_shape = np.shape(model_disk)        
        # print("Rotating Disk Model to PAs of data")
        
        # Check if we have a disk at multiple wavelengths
        if np.size(model_disk_shape) > 2: #Then it's a spec mode disk

            #If we do, then let's make sure that the number of wavelenth channels matches the data. 
            #Note this only works if all your data files have the same number of wavelenth channels. Which it likely does. 
            nfiles = int(np.nanmax(self.dataset.filenums))+1 #Get the number of files  
            n_wv_per_file = self.inputs_shape[0]/nfiles #Number of wavelenths per file. 
            n_disk_wvs = model_disk_shape[0]

            if n_disk_wvs == n_wv_per_file: #If your model wvs match the number of dataset wvs
                for k in np.arange(nfiles):
                    for j,wvs in enumerate(range(n_disk_wvs)):
                        model_copy = copy.deepcopy(model_disk[j,:,:])
                        model_copy = rotate(model_copy, self.pas[k*n_wv_per_file+j], self.aligned_center, flipx = True)
                        model_copy[np.where(np.isnan(model_copy))] = 0.
                        self.model_disks[k*n_wv_per_file+j,:,:] = model_copy 

                # for j,wvs in enumerate(range(n_disk_wvs)):
                #     for i, pa in enumerate(self.pas[j*n_wv_per_file:(j+1)*n_wv_per_file]):   
                #         model_copy = copy.deepcopy(model_disk[j,:,:])
                #         model_copy = rotate(model_copy, pa, self.aligned_center, flipx = True)
                #         model_copy[np.where(np.isnan(model_copy))] = 0.
                #         self.model_disks[j*n_wv_per_file+i,:,:] = model_copy #This line is incorrect!


            else: #Then we just use the first model in the stack. (Not the best solution, but whatever)
                print("The number of wavelenths in your data don't match the number of wavelenths in your disk model.")
                print("Using the first model in the model disk stack for all cases")

                #K, now do things how you would for just a 2D disk
                for i, pa in enumerate(self.pas):
                    model_copy = copy.deepcopy(model_disk[0,:,:])
                    model_copy = rotate(model_copy, pa, self.aligned_center, flipx = True)
                    model_copy[np.where(np.isnan(model_copy))] = 0.
                    self.model_disks[i] = model_copy
                


        else: #If we have a 2D disk model, then we'll just do it like this. 

            # FIXME add align and scale
            for i, pa in enumerate(self.pas):
                model_copy = copy.deepcopy(model_disk)
                model_copy = rotate(model_copy, pa, self.aligned_center, flipx = True)
                model_copy[np.where(np.isnan(model_copy))] = 0.
                self.model_disks[i] = model_copy
        
        self.model_disks = np.reshape(self.model_disks, (self.inputs_shape[0], self.inputs_shape[1] * self.inputs_shape[2])) 

    def _tpool_init(self, original_imgs, original_imgs_shape, aligned_imgs, aligned_imgs_shape, output_imgs, output_imgs_shape,
                    output_imgs_numstacked,
                    pa_imgs, wvs_imgs, centers_imgs, interm_imgs, interm_imgs_shape, fmout_imgs, fmout_imgs_shape,
                    perturbmag_imgs, perturbmag_imgs_shape):
        """
        Initializer function for the thread pool that initializes various shared variables. Main things to note that all
        except the shapes are shared arrays (mp.Array) - output_imgs does not need to be mp.Array and can be anything. Need another version of this for load_image because global variables made in fm.py won't work in here. 
        
        Args:
        original_imgs: original images from files to read and align&scale.
        original_imgs_shape: (N,y,x), N = number of frames = num files * num wavelengths
        aligned: aligned and scaled images for processing.
        aligned_imgs_shape: (wv, N, y, x), wv = number of wavelengths per datacube
        output_imgs: PSF subtraceted images
        output_imgs_shape: (N, y, x, b)
        output_imgs_numstacked: number of images stacked together for each pixel due to geometry overlap. Shape of
        (N, y x). Output without the b dimension
        pa_imgs, wvs_imgs: arrays of size N with the PA and wavelength
        centers_img: array of shape (N,2) with [x,y] image center for image frame
        interm_imgs: intermediate data product shape - what is saved on a sector to sector basis before combining to
        form the output of that sector. The first dimention should be N (i.e. same thing for each science
        image)
        interm_imgs_shape: shape of interm_imgs. The first dimention should be N.
        fmout_imgs: array for output of forward modelling. What's stored in here depends on the class
        fmout_imgs_shape: shape of fmout
        perturbmag_imgs: array for output of size of linear perturbation to assess validity
        perturbmag_imgs_shape: shape of perturbmag_imgs
        """
        global original, original_shape, aligned, aligned_shape, outputs, outputs_shape, outputs_numstacked, img_pa, img_wv, img_center, interm, interm_shape, fmout, fmout_shape, perturbmag, perturbmag_shape
        # original images from files to read and align&scale. Shape of (N,y,x)
        original = original_imgs
        original_shape = original_imgs_shape
        # aligned and scaled images for processing. Shape of (wv, N, y, x)
        aligned = aligned_imgs
        aligned_shape = aligned_imgs_shape
        # output images after KLIP processing
        outputs = output_imgs
        outputs_shape = output_imgs_shape
        outputs_numstacked = output_imgs_numstacked
        # parameters for each image (PA, wavelegnth, image center)
        img_pa = pa_imgs
        img_wv = wvs_imgs
        img_center = centers_imgs
        
        #intermediate and FM arrays
        interm = interm_imgs
        interm_shape = interm_imgs_shape
        fmout = fmout_imgs
        fmout_shape = fmout_imgs_shape
        perturbmag = perturbmag_imgs
        perturbmag_shape = perturbmag_imgs_shape

    #@profile
    # @jit
    def _save_rotated_section(self, input_shape, sector, sector_ind, output_img, output_img_numstacked, angle, radstart, radend, phistart, phiend, padding,IOWA, img_center, flipx=True,
                             new_center=None):
        """
        Rotate and save sector in output image at desired ranges

        Args:
            input_shape: shape of input_image
            sector: data in the sector to save to output_img
            sector_ind: index into input img (corresponding to input_shape) for the original sector
            output_img: the array to save the data to
            output_img_numstacked: array to increment region where we saved output to to bookkeep stacking. None for
                                   skipping bookkeeping
            angle: angle that the sector needs to rotate (I forget the convention right now)

            The next 6 parameters define the sector geometry in input image coordinates
            radstart: radius from img_center of start of sector
            radend: radius from img_center of end of sector
            phistart: azimuthal start of sector
            phiend: azimuthal end of sector
            padding: amount of padding around each sector
            IOWA: tuple (IWA,OWA) where IWA = Inner working angle and OWA = Outer working angle both in pixels.
                    It defines the separation interva in which klip will be run.
            img_center: center of image in input image coordinate
d
            flipx: if true, flip the x coordinate to switch coordinate handiness
            new_center: if not none, center of output_img. If none, center stays the same
        """
        # convert angle to radians
        angle_rad = np.radians(angle)

        #wrap phi
        phistart %= 2 * np.pi
        phiend %= 2 * np.pi

        #incorporate padding
        IWA,OWA = IOWA
        radstart_padded = np.max([radstart-padding,IWA])
        if OWA is not None:
            radend_padded = np.min([radend+padding,OWA])
        else:
            radend_padded = radend+padding
        phistart_padded = (phistart - padding/np.mean([radstart, radend])) % (2 * np.pi)
        phiend_padded = (phiend + padding/np.mean([radstart, radend])) % (2 * np.pi)

        # create the coordinate system of the image to manipulate for the transform
        dims = input_shape
        x, y = np.meshgrid(np.arange(dims[1], dtype=np.float32), np.arange(dims[0], dtype=np.float32))

        # if necessary, move coordinates to new center
        if new_center is not None:
            dx = new_center[0] - img_center[0]
            dy = new_center[1] - img_center[1]
            x -= dx
            y -= dy

        # flip x if needed to get East left of North
        if flipx is True:
            x = img_center[0] - (x - img_center[0])

        # do rotation. CW rotation formula to get a CCW of the image
        xp = (x-img_center[0])*np.cos(angle_rad) + (y-img_center[1])*np.sin(angle_rad) + img_center[0]
        yp = -(x-img_center[0])*np.sin(angle_rad) + (y-img_center[1])*np.cos(angle_rad) + img_center[1]

        if new_center is None:
            new_center = img_center

        rp = np.sqrt((xp - new_center[0])**2 + (yp - new_center[1])**2)
        phip = (np.arctan2(yp-new_center[1], xp-new_center[0]) + angle_rad) % (2 * np.pi)

        # grab sectors based on whether the phi coordinate wraps
        # padded sector
        # check to see if with padding, the phi coordinate wraps
        if phiend_padded > phistart_padded:
            # doesn't wrap
            in_padded_sector = ((rp >= radstart_padded) & (rp < radend_padded) &
                                (phip >= phistart_padded) & (phip < phiend_padded))
        else:
            # wraps
            in_padded_sector = ((rp >= radstart_padded) & (rp < radend_padded) &
                                ((phip >= phistart_padded) | (phip < phiend_padded)))
        rot_sector_pix = np.where(in_padded_sector)

        # do NaN detection by defining any pixel in the new coordiante system (xp, yp) as a nan
        # if any one of the neighboring pixels in the original image is a nan
        # e.g. (xp, yp) = (120.1, 200.1) is nan if either (120, 200), (121, 200), (120, 201), (121, 201)
        # is a nan
        dims = input_shape
        blank_input = np.zeros(dims[1] * dims[0])
        blank_input[sector_ind] = sector
        blank_input.shape = [dims[0], dims[1]]

        xp_floor = np.clip(np.floor(xp).astype(int), 0, xp.shape[1]-1)[rot_sector_pix]
        xp_ceil = np.clip(np.ceil(xp).astype(int), 0, xp.shape[1]-1)[rot_sector_pix]
        yp_floor = np.clip(np.floor(yp).astype(int), 0, yp.shape[0]-1)[rot_sector_pix]
        yp_ceil = np.clip(np.ceil(yp).astype(int), 0, yp.shape[0]-1)[rot_sector_pix]
        rotnans = np.where(np.isnan(blank_input[yp_floor.ravel(), xp_floor.ravel()]) | 
                           np.isnan(blank_input[yp_floor.ravel(), xp_ceil.ravel()]) |
                           np.isnan(blank_input[yp_ceil.ravel(), xp_floor.ravel()]) |
                           np.isnan(blank_input[yp_ceil.ravel(), xp_ceil.ravel()]))

        # resample image based on new coordinates, set nan values as median
        nanpix = np.where(np.isnan(blank_input))
        medval = np.median(blank_input[np.where(~np.isnan(blank_input))])
        input_copy = np.copy(blank_input)
        input_copy[nanpix] = medval
        rot_sector = ndimage.map_coordinates(input_copy, [yp[rot_sector_pix], xp[rot_sector_pix]], cval=np.nan)

        # mask nans
        rot_sector[rotnans] = np.nan
        sector_validpix = np.where(~np.isnan(rot_sector))

        # need to define only where the non nan pixels are, so we can store those in the output image
        blank_output = np.zeros([dims[0], dims[1]]) * np.nan
        blank_output[rot_sector_pix] = rot_sector
        blank_output.shape = (dims[0], dims[1])
        rot_sector_validpix_2d = np.where(~np.isnan(blank_output))

        # save output sector. We need to reshape the array into 2d arrays to save it
        output_img.shape = [self.outputs_shape[1], self.outputs_shape[2]]
        output_img[rot_sector_validpix_2d] = np.nansum([output_img[rot_sector_pix][sector_validpix], rot_sector[sector_validpix]], axis=0)
        output_img.shape = [self.outputs_shape[1] * self.outputs_shape[2]]

        # Increment the numstack counter if it is not None
        if output_img_numstacked is not None:
            output_img_numstacked.shape = [self.outputs_shape[1], self.outputs_shape[2]]
            output_img_numstacked[rot_sector_validpix_2d] += 1
            output_img_numstacked.shape = [self.outputs_shape[1] *  self.outputs_shape[2]]
