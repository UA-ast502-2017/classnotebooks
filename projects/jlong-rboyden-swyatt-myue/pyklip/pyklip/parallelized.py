import pyklip.klip as klip
import pyklip.spectra_management as spec
import pyklip.fakes as fakes
import multiprocessing as mp
import ctypes
import numpy as np
import cProfile
import os
import itertools
import copy
import astropy.io.fits as fits
import scipy.interpolate as interp
from scipy.stats import norm

#Logic to test mkl exists
try:
    import mkl
    mkl_exists = True
except ImportError:
    mkl_exists = False

def _tpool_init(original_imgs, original_imgs_shape, aligned_imgs, aligned_imgs_shape, output_imgs, output_imgs_shape,
                pa_imgs, wvs_imgs, centers_imgs, psf_library, psf_library_shape):
    """
    Initializer function for the thread pool that initializes various shared variables. Main things to note that all
    except the shapes are shared arrays (mp.Array).

    Args:
        original_imgs: original images from files to read and align&scale.
        original_imgs_shape: (N,y,x), N = number of frames = num files * num wavelengths
        aligned: aligned and scaled images for processing.
        aligned_imgs_shape: (wv, N, y, x), wv = number of wavelengths per datacube
        output_imgs: output images after KLIP processing
        output_imgs_shape: (b, N, y, x), b = number of different KL basis cutoffs for KLIP routine
        pa_imgs, wvs_imgs: arrays of size N with the PA and wavelength
        centers_img: array of shape (N,2) with [x,y] image center for image frame
        psf_library: array of shape (N_lib, y, x) with N_lib PSF library images
    """
    global original, original_shape, aligned, aligned_shape, output, output_shape, img_pa, img_wv, img_center, psf_lib, \
        psf_lib_shape
    # original images from files to read and align&scale. Shape of (N,y,x)
    original = original_imgs
    original_shape = original_imgs_shape
    # aligned and scaled images for processing. Shape of (wv, N, y, x)
    aligned = aligned_imgs
    aligned_shape = aligned_imgs_shape
    # output images after KLIP processing
    output = output_imgs
    output_shape = output_imgs_shape
    # parameters for each image (PA, wavelegnth, image center)
    img_pa = pa_imgs
    img_wv = wvs_imgs
    img_center = centers_imgs
    psf_lib = psf_library
    psf_lib_shape = psf_library_shape


def _arraytonumpy(shared_array, shape=None, dtype=None):
    """
    Covert a shared array to a numpy array
    Args:
        shared_array: a multiprocessing.Array array
        shape: a shape for the numpy array. otherwise, will assume a 1d array
        dtype: data type of the arrays. Should be either ctypes.c_float(default) or ctypes.c_double

    Returns:
        numpy_array: numpy array for vectorized operation. still points to the same memory!
                     returns None is shared_array is None
    """
    if dtype is None:
        dtype = ctypes.c_float

    # if you passed in nothing you get nothing
    if shared_array is None:
        return None

    numpy_array = np.frombuffer(shared_array.get_obj(), dtype=dtype)
    if shape is not None:
        numpy_array.shape = shape

    return numpy_array


def _align_and_scale_per_image(img_index, aligned_center, ref_wv, dtype=None):
    """
    Aligns and scales the an individual image (used for pyklip lite)

    Args:
        img_index: index of image for the shared arrays
        algined_center: center to align things to
        ref_wv: wavelength to scale images to
        dtype: data type of the arrays. Should be either ctypes.c_float(default) or ctypes.c_double

    Returns:
        None
    """
    if dtype is None:
        dtype = ctypes.c_float

    original_imgs = _arraytonumpy(original, original_shape,dtype=dtype)
    wvs_imgs = _arraytonumpy(img_wv,dtype=dtype)
    centers_imgs = _arraytonumpy(img_center, (np.size(wvs_imgs),2),dtype=dtype)
    aligned_imgs = _arraytonumpy(aligned, aligned_shape,dtype=dtype)

    aligned_imgs[img_index,:,:] = klip.align_and_scale(original_imgs[img_index], aligned_center,
                                                       centers_imgs[img_index], ref_wv/wvs_imgs[img_index])
    return
        

def _align_and_scale(iterable_arg):
    """
    Aligns and scales the set of original images about a reference center and scaled to a reference wavelength.
    Note: is a helper function to only be used after initializing the threadpool!

    Args:
        iterable_arg: a tuple of three elements:
            ref_wv_iter: a tuple of two elements. First is the index of the reference wavelength (between 0 and 36).
                         second is the value of the reference wavelength. This is to determine scaling
            ref_center: a two-element array with the [x,y] center position to align all the images to.
            dtype: Should be equal to float or np.float32. Define the data type of the arrays.
                    float is actually the default double.

    Returns:
        just returns ref_wv_iter again
    """

    # extract out arguments from the iteration argument
    ref_wv_iter = iterable_arg[0]
    ref_center = iterable_arg[1]
    dtype = iterable_arg[2]
    ref_wv_index = ref_wv_iter[0]
    ref_wv = ref_wv_iter[1]

    original_imgs = _arraytonumpy(original, original_shape,dtype=dtype)
    wvs_imgs = _arraytonumpy(img_wv,dtype=dtype)
    centers_imgs = _arraytonumpy(img_center, (np.size(wvs_imgs),2),dtype=dtype)

    aligned_imgs = _arraytonumpy(aligned, aligned_shape,dtype=dtype)
    aligned_imgs[ref_wv_index, :, :, :] =  np.array([klip.align_and_scale(frame, ref_center, old_center, ref_wv/old_wv,dtype=dtype)
                                                     for frame, old_center, old_wv in zip(original_imgs, centers_imgs, wvs_imgs)])

    return ref_wv_index, ref_wv


def _klip_section(img_num, parang, wavelength, wv_index, numbasis, radstart, radend, phistart, phiend, minmove,
                  ref_center, dtype=None):
    """
    DEPRECIATED. Still being preserved in case we want to change size of atomization. But will need some fixing

    Runs klip on a section of an image as given by the geometric parameters. Helper fucntion of klip routines and
    requires thread pool to be initialized! Currently is designed only for ADI+SDI. Not yet that flexible.

    Args:
        img_num: file index for the science image to process
        parang: PA of science iamge
        wavelength: wavelength of science image
        wv_index: array index of the wavelength of the science image
        numbasis: number of KL basis vectors to use (can be a scalar or list like). Length of b
        avg_rad: average radius of this annulus
        radstart: inner radius of the annulus (in pixels)
        radend: outer radius of the annulus (in pixels)
        phistart: lower bound in CCW angle from x axis for the start of the section
        phiend: upper boundin CCW angle from y axis for the end of the section
        minmove: minimum movement between science image and PSF reference image to use PSF reference image (in pixels)
        ref_center: 2 element list for the center of the science frames. Science frames should all be aligned.
        dtype: data type of the arrays. Should be either ctypes.c_float(default) or ctypes.c_double

    Returns:
        Returns True on success and False on failure.
        Output images are stored in output array as defined by _tpool_init()
    """
    global output, aligned

    if dtype is None:
        dtype = ctypes.c_float

    #create a coordinate system
    x, y = np.meshgrid(np.arange(original_shape[2] * 1.0), np.arange(original_shape[1] * 1.0))
    x.shape = (x.shape[0] * x.shape[1])
    y.shape = (y.shape[0] * y.shape[1])
    r = np.sqrt((x - ref_center[0])**2 + (y - ref_center[1])**2)
    phi = np.arctan2(y - ref_center[1], x - ref_center[0])

    #grab the pixel location of the section we are going to anaylze
    section_ind = np.where((r >= radstart) & (r < radend) & (phi >= phistart) & (phi < phiend))
    if np.size(section_ind) == 0:
        print("section is empty, skipping...")
        return False

    #grab the files suitable for reference PSF
    #load shared arrays for wavelengths and PAs
    wvs_imgs = _arraytonumpy(img_wv,dtype=dtype)
    pa_imgs = _arraytonumpy(img_pa,dtype=dtype)
    #calculate average movement in this section
    avg_rad = (radstart + radend) / 2.0
    moves = klip.estimate_movement(avg_rad, parang, pa_imgs, wavelength, wvs_imgs)
    file_ind = np.where(moves >= minmove)
    if np.size(file_ind[0]) < 2:
        print("less than 2 reference PSFs available for minmove={0}, skipping...".format(minmove))
        return False

    #load aligned images and make reference PSFs
    aligned_imgs = _arraytonumpy(aligned, (aligned_shape[0], aligned_shape[1], aligned_shape[2]*aligned_shape[3]),dtype=dtype)[wv_index]
    ref_psfs = aligned_imgs[file_ind[0], :]
    ref_psfs = ref_psfs[:,  section_ind[0]]
    #ref_psfs = ref_psfs[:, section_ind]
    #print(img_num, avg_rad, ref_psfs.shape)
    #print(sub_imgs.shape)
    #print(sub_imgs[img_num, section_ind, :].shape)

    #write to output
    output_imgs = _arraytonumpy(output, (output_shape[0], output_shape[1]*output_shape[2], output_shape[3]),dtype=dtype)
    klipped = klip.klip_math(aligned_imgs[img_num, section_ind], ref_psfs, numbasis)
    output_imgs[img_num, section_ind, :] = klipped
    return True


def _klip_section_profiler(img_num, parang, wavelength, wv_index, numbasis, radstart, radend, phistart, phiend, minmove,
                           ref_center=None):
    """
    DEPRECIATED. Still being preserved in case we want to change size of atomization. But will need some fixing

    Profiler wrapper for _klip_section. Outputs a file openable by pstats.Stats for each annulus wavelength.
    However there is no guarentee which wavelength and which subsection of the annulus is saved to disk.

    Args: Same arguments as _klip_section
    """
    cProfile.runctx("_klip_section(img_num, parang, wavelength, wv_index, numbasis, radstart, radend, phistart, phiend,"
                    " minmove, ref_center)", globals(), locals(), 'profile-{0}.out'.format(int(radstart+radend)/2))
    return True


def _klip_section_multifile_profiler(scidata_indicies, wavelength, wv_index, numbasis, radstart, radend, phistart,
                                     phiend, minmove, ref_center=None, minrot=0):
    """
    Profiler wrapper for _klip_section_multifile. Outputs a file openable by pstats.Stats for each annulus wavelength.
    However there is no guarentee which wavelength and which subsection of the annulus is saved to disk. There
    is the ability to output a profiler file for each subsection and wavelength but it's too many files and who
    actually looks at all of them.

    Args: Same arguments as _klip_section_multifile()
    """
    cProfile.runctx("_klip_section_multifile(scidata_indicies, wavelength, wv_index, numbasis, radstart, radend, "
                    "phistart, phiend, minmove, ref_center, minrot)", globals(), locals(),
                    'profile-{0}.out'.format(int(radstart + radend)/2))
    return True


def _klip_section_multifile(scidata_indicies, wavelength, wv_index, numbasis, maxnumbasis, radstart, radend, phistart,
                            phiend, minmove, ref_center, minrot, maxrot, spectrum, mode, psflib_good=None,
                            psflib_corr=None, lite=False, dtype=None, algo='klip'):
    """
    Runs klip on a section of the image for all the images of a given wavelength.
    Bigger size of atomization of work than _klip_section but saves computation time and memory. Currently no need to
    break it down even smaller when running on machines on the order of 32 cores.

    Args:
        scidata_indicies: array of file indicies that are the science images for this wavelength
        wavelength: value of the wavelength we are processing
        wv_index: index of the wavelenght we are processing
        numbasis: number of KL basis vectors to use (can be a scalar or list like). Length of b
        maxnumbasis: if not None, maximum number of KL basis/correlated PSFs to use for KLIP. Otherwise, use max(numbasis)    
        radstart: inner radius of the annulus (in pixels)
        radend: outer radius of the annulus (in pixels)
        phistart: lower bound in CCW angle from x axis for the start of the section
        phiend: upper boundin CCW angle from y axis for the end of the section
        minmove: minimum movement between science image and PSF reference image to use PSF reference image (in pixels)
        ref_center: 2 element list for the center of the science frames. Science frames should all be aligned.
        minrot: minimum PA rotation (in degrees) to be considered for use as a reference PSF (good for disks)
        maxrot: maximum PA rotation (in degrees) to be considered for use as a reference PSF (temporal variability)
        spectrum: if not None, a array of length N with the flux of the template spectrum at each wavelength. Uses
                    minmove to determine the separation from the center of the segment to determine contamination
                    (e.g. minmove=3, checks how much containmination is within 3 pixels of the hypothetical source)
                    if smaller than 10%, (hard coded quantity), then use it for reference PSF
        mode: one of ['ADI', 'SDI', 'ADI+SDI'] for ADI, SDI, or ADI+SDI
        lite: if True, use low memory footprint mode
        dtype: data type of the arrays. Should be either ctypes.c_float(default) or ctypes.c_double
        algo (str): algorithm to use ('klip', 'nmf')

    Returns:
        returns True on success, False on failure. Does not return whether KLIP on each individual image was sucessful.
        Saves data to output array as defined in _tpool_init()
    """
    if dtype is None:
        dtype = ctypes.c_float

    #create a coordinate system. Can use same one for all the images because they have been aligned and scaled
    x, y = np.meshgrid(np.arange(original_shape[2] * 1.0), np.arange(original_shape[1] * 1.0))
    x.shape = (x.shape[0] * x.shape[1]) #Flatten
    y.shape = (y.shape[0] * y.shape[1])
    r = np.sqrt((x - ref_center[0])**2 + (y - ref_center[1])**2)
    phi = np.arctan2(y - ref_center[1], x - ref_center[0])
    # make sure phi is in range [-pi, pi)
    phi = (phi % (2*np.pi)) - np.pi

    #grab the pixel location of the section we are going to anaylze
    section_ind = np.where((r >= radstart) & (r < radend) & (phi >= phistart) & (phi < phiend))
    if np.size(section_ind) <= 1:
        print("section is too small ({0} pixels), skipping...".format(np.size(section_ind)))
        return False

    #export some of klip.klip_math functions to here to minimize computation repeats

    #load aligned images for this wavelength
    #if lite memory, the aligend array has a different size
    if lite:
        aligned_imgs = _arraytonumpy(aligned, (aligned_shape[0], aligned_shape[1] * aligned_shape[2]),dtype=dtype)
    else:
        aligned_imgs = _arraytonumpy(aligned, (aligned_shape[0], aligned_shape[1], aligned_shape[2] * aligned_shape[3]),dtype=dtype)[wv_index]


    ref_psfs = aligned_imgs[:,  section_ind[0]]

    #do the same for the reference PSFs
    #playing some tricks to vectorize the subtraction of the mean for each row
    ref_psfs_mean_sub = ref_psfs - np.nanmean(ref_psfs, axis=1)[:, None]
    ref_psfs_mean_sub[np.where(np.isnan(ref_psfs_mean_sub))] = 0

    #calculate the covariance matrix for the reference PSFs
    #note that numpy.cov normalizes by p-1 to get the NxN covariance matrix
    #we have to correct for that in the klip.klip_math routine when consturcting the KL
    #vectors since that's not part of the equation in the KLIP paper
    covar_psfs = np.cov(ref_psfs_mean_sub)
    #also calculate correlation matrix since we'll use that to select reference PSFs
    covar_diag = np.diagflat(1./np.sqrt(np.diag(covar_psfs)))
    corr_psfs = np.dot( np.dot(covar_diag, covar_psfs ), covar_diag)

    #grab the parangs
    parangs = _arraytonumpy(img_pa,dtype=dtype)

    for file_index,parang in zip(scidata_indicies, parangs[scidata_indicies]):
        try:
            _klip_section_multifile_perfile(file_index, section_ind, ref_psfs_mean_sub, covar_psfs, corr_psfs,
                                            parang, wavelength, wv_index, (radstart + radend) / 2.0, numbasis,
                                            maxnumbasis, minmove, minrot, maxrot, mode, psflib_good=psflib_good,
                                            psflib_corr=psflib_corr, spectrum=spectrum, lite=lite, dtype=dtype,
                                            algo=algo)
        except (ValueError, RuntimeError, TypeError) as err:
            print("({0}): {1}".format(err.errno, err.strerror))
            return False


    #   [_klip_section_multifile_perfile(file_index, section_ind, ref_psfs_mean_sub, covar_psfs,
    #                                   parang, wavelength, wv_index, (radstart + radend) / 2.0, numbasis, minmove)
    #      for file_index,parang in zip(scidata_indicies, parangs[scidata_indicies])]

    return True


def _klip_section_multifile_perfile(img_num, section_ind, ref_psfs, covar,  corr, parang, wavelength, wv_index, avg_rad,
                                    numbasis, maxnumbasis, minmove, minrot, maxrot, mode,
                                    psflib_good=None, psflib_corr=None,
                                    spectrum=None, lite=False, dtype=None, algo='klip'):
    """
    Imitates the rest of _klip_section for the multifile code. Does the rest of the PSF reference selection

    Args:
        img_num: file index for the science image to process
        section_ind: np.where(pixels are in this section of the image). Note: coordinate system is collapsed into 1D
        ref_psfs: reference psf images of this section
        covar: the covariance matrix of the reference PSFs. Shape of (N,N)
        corr: the correlation matrix of the refernece PSFs. Shape of (N,N)
        parang: PA of science iamge
        wavelength: wavelength of science image
        wv_index: array index of the wavelength of the science image
        avg_rad: average radius of this annulus
        numbasis: number of KL basis vectors to use (can be a scalar or list like). Length of b
        maxnumbasis: if not None, maximum number of KL basis/correlated PSFs to use for KLIP. Otherwise, use max(numbasis)           
        minmove: minimum movement between science image and PSF reference image to use PSF reference image (in pixels)
        mode: one of ['ADI', 'SDI', 'ADI+SDI'] for ADI, SDI, or ADI+SDI
        psflib_good: array of size N_lib indicating which N_good are good are selected in the passed in corr matrix
        psflib_corr: matrix of size N_sci x N_good with correlation between the target franes and the good RDI PSFs
        spectrum: if not None, a array of length N with the flux of the template spectrum at each wavelength. Uses
                    minmove to determine the separation from the center of the segment to determine contamination and
                    the size of the PSF (TODO: make PSF size another quanitity)
                    (e.g. minmove=3, checks how much containmination is within 3 pixels of the hypothetical source)
                    if smaller than 10%, (hard coded quantity), then use it for reference PSF
        lite: if True, in memory-lite mode
        dtype: data type of the arrays. Should be either ctypes.c_float(default) or ctypes.c_double

    Returns:
        return True on success, False on failure.
        Saves image to output array defined in _tpool_init()
    """
    if dtype is None:
        dtype = ctypes.c_float
        
    # grab the files suitable for reference PSF
    # load shared arrays for wavelengths and PAs
    wvs_imgs = _arraytonumpy(img_wv,dtype=dtype)
    pa_imgs = _arraytonumpy(img_pa,dtype=dtype)
    # calculate average movement in this section for each PSF reference image w.r.t the science image
    moves = klip.estimate_movement(avg_rad, parang, pa_imgs, wavelength, wvs_imgs, mode)
    # check all the PSF selection criterion
    # enough movement of the astrophyiscal source
    if spectrum is None:
        goodmv = (moves >= minmove)
    else:
        # optimize the selection based on the spectral template rather than just an exclusion principle
        goodmv = (spectrum * norm.sf(moves-minmove/2.355, scale=minmove/2.355) <= 0.1 * spectrum[wv_index])

    # enough field rotation
    if minrot > 0:
        goodmv = (goodmv) & (np.abs(pa_imgs - parang) >= minrot)

    # if no SDI, don't use other wavelengths
    if "SDI" not in mode.upper():
        goodmv = (goodmv) & (wvs_imgs == wavelength)
    # if no ADI, don't use other parallactic angles
    if "ADI" not in mode.upper():
        goodmv = (goodmv) & (pa_imgs == parang)
    include_rdi = "RDI" in mode.upper()

    good_file_ind = np.where(goodmv)
    if (np.size(good_file_ind[0]) < 2) and (not include_rdi):
        print("less than 2 reference PSFs available for minmove={0}, skipping...".format(minmove))
        return False
    # pick out a subarray. Have to play around with indicies to get the right shape to index the matrix
    covar_files = covar[good_file_ind[0].reshape(np.size(good_file_ind), 1), good_file_ind[0]]

    # pick only the most correlated reference PSFs if there's more than enough PSFs
    if maxnumbasis is None:
        maxnumbasis = np.max(numbasis)
    maxbasis_possible = np.size(good_file_ind)
    # if RDI, also include the size of the RDI PSF library
    # and load in PSF library
    if include_rdi:
        num_good_rdi = np.size(psflib_good)
        maxbasis_possible += num_good_rdi
        psf_library = _arraytonumpy(psf_lib, (psf_lib_shape[0], psf_lib_shape[1]*psf_lib_shape[2]), dtype=dtype)

    # load input/output data
    if lite:
        aligned_imgs = _arraytonumpy(aligned, (aligned_shape[0], aligned_shape[1] * aligned_shape[2]),dtype=dtype)
    else:
        aligned_imgs = _arraytonumpy(aligned, (aligned_shape[0], aligned_shape[1], aligned_shape[2] * aligned_shape[3]),dtype=dtype)[wv_index]
    numpix = np.size(section_ind[0])

    # do we want to downselect out of all the possible references
    if maxbasis_possible > maxnumbasis:
        # grab the x-correlation with the sci img for valid PSFs
        xcorr = corr[img_num, good_file_ind[0]]
        if include_rdi:
            # calculate real xcorr between image and RDI PSFs for this sector for only the maxnumbasis
            # best reference PSFs.
            # grab the maxnumbasis most correlated PSFs from the library
            num_rdi_psfs_first_downselect = np.min([maxnumbasis, num_good_rdi])
            rdi_best_corr_max_possbile_indices = np.argsort(psflib_corr[img_num, psflib_good])[-num_rdi_psfs_first_downselect:]
            # grab these PSFs
            rdi_best_corr_max_possible = psf_library[psflib_good[rdi_best_corr_max_possbile_indices]]
            rdi_best_corr_max_possible = rdi_best_corr_max_possible[:, section_ind[0]]
            # recalculate their correlations in this sector
            sci_img = aligned_imgs[img_num, section_ind[0]].reshape(1, numpix)
            # to calculate correlation, first subtract off mean for each image
            sci_img_mean_sub = sci_img - np.mean(sci_img, axis=1)[:,None]
            rdi_best_corr_max_possible_mean_sub = rdi_best_corr_max_possible - np.mean(rdi_best_corr_max_possible, axis=1)[:,None]
            # we will then calcualte the covariance between the science image with the PSF Library
            sci_x_rdi_best_covar = np.dot(sci_img_mean_sub, rdi_best_corr_max_possible_mean_sub.T ) / (numpix - 1)
            # to convert from covariance to correlation matrix, we just need to divide by normalization of the datasets
            sci_norm = np.sum(sci_img_mean_sub*sci_img_mean_sub, axis=1)
            rdi_best_norm = np.sum(rdi_best_corr_max_possible_mean_sub*rdi_best_corr_max_possible_mean_sub, axis=1)
            # convert form covaraince matrix to correlation matrix
            sci_x_rdi_best_corr = sci_x_rdi_best_covar / np.sqrt(rdi_best_norm) / np.sqrt(sci_norm)
            # now we've recalculated the correlation for this sector for the best RDI PSFs

            # indicate which are RDI frames (remember we are using onyl the top correlated ones that we just comptued
            # correlations for
            is_rdi_psf = np.append(np.repeat(False, np.size(xcorr)), np.repeat(True, num_rdi_psfs_first_downselect))
            # indices for both the dataset and PSF library arrays squished together
            psfindices = np.append(np.arange(np.size(xcorr)), psflib_good[rdi_best_corr_max_possbile_indices])
            # cross correlation now includes both
            xcorr = np.append(xcorr, sci_x_rdi_best_corr)

        sort_ind = np.argsort(xcorr)
        closest_matched = sort_ind[-maxnumbasis:]  # sorted smallest first so need to grab from the end
        if include_rdi:
            # separate out the RDI ones
            rdi_selected = np.where(is_rdi_psf[closest_matched])
            rdi_closest_matched = psfindices[closest_matched[rdi_selected]]
            # remove the RDI ones from closest_matched to imitate non-RDI behavior
            closest_matched = psfindices[closest_matched[np.where(~is_rdi_psf[closest_matched])]]

        # grab smaller set of reference PSFs
        ref_psfs_selected = ref_psfs[good_file_ind[0][closest_matched], :]
        # grab the new and smaller covariance matrix
        covar_files = covar_files[closest_matched.reshape(np.size(closest_matched), 1), closest_matched]

        if include_rdi:
            rdi_psfs_selected = psf_library[rdi_closest_matched]
            rdi_psfs_selected = rdi_psfs_selected[:, section_ind[0]]
    else:
        # else just grab the reference PSFs for all the valid files
        ref_psfs_selected = ref_psfs[good_file_ind[0], :]

        if include_rdi:
            rdi_psfs_selected = psf_library[:, section_ind[0]]

    # add PSF library to reference psf list and covariance matrix if needed
    if include_rdi:
        # compute covariances. I could just grab these from ~20 lines above, but too lazy
        rdi_covar = np.cov(rdi_psfs_selected) # N_rdi_sel x N_rdi_sel
        # compute cross term
        # cross term has shape N_dataset_ref x N_rdi_selected
        covar_ref_x_rdi = np.dot((ref_psfs_selected - np.nanmean(ref_psfs_selected, axis=1)[:,None]),
                                 (rdi_psfs_selected - np.mean(rdi_psfs_selected, axis=1)[:,None]).T) / (numpix - 1)
        # piece together covariance matrix. It should looke like
        # [ cov_ref, cov_ref_x_rdi ]
        # [ cov_rdi_x_ref, cov_rdi ]
        # first append the horizontal component to get shape of N_all_refs x N_dataset_ref
        covar_files = np.append(covar_files, covar_ref_x_rdi, axis=1)
        # now append the bottom half
        covar_files_bottom = np.append(covar_ref_x_rdi.T, rdi_covar, axis=1)
        covar_files = np.append(covar_files, covar_files_bottom, axis=0)

        # append the rdi psfs to the reference PSFs
        ref_psfs_selected = np.append(ref_psfs_selected, rdi_psfs_selected, axis=0)



    output_imgs = _arraytonumpy(output, (output_shape[0], output_shape[1]*output_shape[2], output_shape[3]),dtype=dtype)

    # run KLIP
    try:
        if algo.lower() == 'klip':
            klipped = klip.klip_math(aligned_imgs[img_num, section_ind[0]], ref_psfs_selected, numbasis, covar_psfs=covar_files)
        elif algo.lower() == 'nmf':
            import pyklip.nmf_imaging as nmf_imaging
            klipped = nmf_imaging.nmf_math(aligned_imgs[img_num, section_ind].ravel(), ref_psfs, componentNum=numbasis[0])
            klipped = klipped.reshape(klipped.shape[0], 1)
    except (ValueError, RuntimeError, TypeError) as err:
        print("({0}): {1}".format(err.errno, err.strerror))
        return False

    # write to output
    output_imgs[img_num, section_ind[0], :] = klipped

    return True


def rotate_imgs(imgs, angles, centers, new_center=None, numthreads=None, flipx=True, hdrs=None,
                disable_wcs_rotation = False):
    """
    derotate a sequences of images by their respective angles

    Args:
        imgs: array of shape (N,y,x) containing N images
        angles: array of length N with the angle to rotate each frame. Each angle should be CW in degrees.
                (TODO: fix this angle convention)
        centers: array of shape N,2 with the [x,y] center of each frame
        new_centers: a 2-element array with the new center to register each frame. Default is middle of image
        numthreads: number of threads to be used
        flipx: flip the x axis to get a left handed coordinate system (oh astronomers...)
        hdrs: array of N wcs astrometry headers

    Returns:
        derotated: array of shape (N,y,x) containing the derotated images
    """

    tpool = mp.Pool(processes=numthreads)

    # klip.rotate(img, -angle, oldcenter, [152,152]) for img, angle, oldcenter
    # multithreading the rotation for each image
    if hdrs is None:
        tasks = [tpool.apply_async(klip.rotate, args=(img, angle, center, new_center, flipx, None))
                 for img, angle, center in zip(imgs, angles, centers)]
    else:
        tasks = [tpool.apply_async(klip.rotate, args=(img, angle, center, new_center, flipx, None))
                 for img, angle, center in zip(imgs, angles, centers)]
        # lazy hack around the fact that wcs objects don't preserve wcs.cd fields when sent to other processes
        # so let's just do it manually outside of the rotation
        if not disable_wcs_rotation:
            for angle, astr_hdr in zip(angles, hdrs):
                if astr_hdr is None:
                    continue
                klip._rotate_wcs_hdr(astr_hdr, angle, flipx=flipx)

    # reform back into a giant array
    derotated = np.array([task.get() for task in tasks])

    tpool.close()

    return derotated

def high_pass_filter_imgs(imgs, numthreads=None, filtersize=10):
    """
    filters a sequences of images using a FFT

    Inputs:
        imgs: array of shape (N,y,x) containing N images
        numthreads: number of threads to be used
        filtersize: size in Fourier space of the size of the space. In image space, size=img_size/filtersize

    Output:
        filtered: array of shape (N,y,x) containing the filtered images
    """

    tpool = mp.Pool(processes=numthreads)


    tasks = [tpool.apply_async(klip.high_pass_filter, args=(img, filtersize)) for img in imgs]

    #reform back into a giant array
    filtered = np.array([task.get() for task in tasks])

    tpool.close()

    return filtered


def klip_parallelized_lite(imgs, centers, parangs, wvs, IWA, OWA=None, mode='ADI+SDI', annuli=5, subsections=4,
                           movement=3, numbasis=None, aligned_center = None, numthreads=None, minrot=0, maxrot=360,
                           annuli_spacing="constant", maxnumbasis=None,
                           spectrum=None, dtype=None, algo='klip', **kwargs):
    """
    multithreaded KLIP PSF Subtraction, has a smaller memory foot print than the original

    Args:
        imgs: array of 2D images for ADI. Shape of array (N,y,x)
        centers: N by 2 array of (x,y) coordinates of image centers
        parangs: N length array detailing parallactic angle of each image
        wvs: N length array of the wavelengths
        IWA: inner working angle (in pixels)
        OWA: outer working angle (in pixels)
        mode: one of ['ADI', 'SDI', 'ADI+SDI'] for ADI, SDI, or ADI+SDI
        anuuli: number of annuli to use for KLIP
        subsections: number of sections to break each annuli into
        movement: minimum amount of movement (in pixels) of an astrophysical source
                  to consider using that image for a refernece PSF
        numbasis: number of KL basis vectors to use (can be a scalar or list like). Length of b
        annuli_spacing: how to distribute the annuli radially. Currently three options. Constant (equally spaced), 
                log (logarithmical expansion with r), and linear (linearly expansion with r)
        maxnumbasis: if not None, maximum number of KL basis/correlated PSFs to use for KLIP. Otherwise, use max(numbasis)           
        aligned_center: array of 2 elements [x,y] that all the KLIP subtracted images will be centered on for image
                        registration
        numthreads: number of threads to use. If none, defaults to using all the cores of the cpu
        minrot: minimum PA rotation (in degrees) to be considered for use as a reference PSF (good for disks)
        maxrot: maximum PA rotation (in degrees) to be considered for use as a reference PSF (temporal variability)
        spectrum: if not None, a array of length N with the flux of the template spectrum at each wavelength. Uses
                    minmove to determine the separation from the center of the segment to determine contamination and
                    the size of the PSF (TODO: make PSF size another quanitity)
                    (e.g. minmove=3, checks how much containmination is within 3 pixels of the hypothetical source)
                    if smaller than 10%, (hard coded quantity), then use it for reference PSF
        kwargs: in case you pass it stuff that we don't use in the lite version
        dtype: data type of the arrays. Should be either ctypes.c_float (default) or ctypes.c_double
        algo (str): algorithm to use ('klip', 'nmf')

    Returns:
        sub_imgs: array of [array of 2D images (PSF subtracted)] using different number of KL basis vectors as
                    specified by numbasis. Shape of (b,N,y,x).
    """

    ################## Interpret input arguments ####################

    # default numbasis if none
    if numbasis is None:
        totalimgs = imgs.shape[0]
        maxbasis = np.min([totalimgs, 100]) # only going up to 100 KL modes by default
        numbasis = np.arange(1, maxbasis + 10, 10)
        print("KL basis not specified. Using default.", numbasis)
    else:
        if hasattr(numbasis, "__len__"):
            numbasis = np.array(numbasis)
        else:
            numbasis = np.array([numbasis])

    # default aligned_center if none:
    if aligned_center is None:
        aligned_center = [np.mean(centers[:,0]), np.mean(centers[:,1])]

    # save all bad pixels
    allnans = np.where(np.isnan(imgs))

    # use first image to figure out how to divide the annuli
    dims = imgs.shape
    x, y = np.meshgrid(np.arange(dims[2] * 1.0), np.arange(dims[1] * 1.0))
    nanpix = np.where(np.isnan(imgs[0]))

    # if user didn't supply how to define IWA
    if OWA is None:
        full_image = True # reduce the full image
        # define OWA as either the closest NaN pixel or edge of image if no NaNs exist
        if np.size(nanpix) == 0:
            OWA = np.sqrt(np.max((x - centers[0][0]) ** 2 + (y - centers[0][1]) ** 2))
        else:
            # grab the NaN from the 1st percentile (this way we drop outliers)
            OWA = np.sqrt(np.percentile((x[nanpix] - centers[0][0]) ** 2 + (y[nanpix] - centers[0][1]) ** 2, 1))
    else:
        full_image = False # don't reduce the full image, only up the the IWA

    #error checking for too small of annuli go here


    #calculate the annuli ranges
    rad_bounds = klip.define_annuli_bounds(annuli, IWA, OWA, annuli_spacing)

    # if OWA wasn't passed in, we're going to assume we reduce the full image, so last sector emcompasses everything
    if full_image:
        # last annulus should mostly emcompass everything
        rad_bounds[annuli - 1] = (rad_bounds[annuli - 1][0], imgs[0].shape[0])

    #divide annuli into subsections
    dphi = 2 * np.pi / subsections
    phi_bounds = [[dphi * phi_i - np.pi, dphi * (phi_i + 1) - np.pi] for phi_i in range(subsections)]
    phi_bounds[-1][1] = np.pi


    #calculate how many iterations we need to do
    global tot_iter
    tot_iter = np.size(np.unique(wvs)) * len(phi_bounds) * len(rad_bounds)

    #before we start, create the output array in flattened form
    #sub_imgs = np.zeros([dims[0], dims[1] * dims[2], numbasis.shape[0]])

    ########################### Create Shared Memory ###################################

    if dtype is None:
        dtype = ctypes.c_float
    mp_data_type = dtype

    #implement the thread pool
    #make a bunch of shared memory arrays to transfer data between threads
    #make the array for the original images and initalize it
    original_imgs = mp.Array(mp_data_type, np.size(imgs))
    original_imgs_shape = imgs.shape
    original_imgs_np = _arraytonumpy(original_imgs, original_imgs_shape,dtype=dtype)
    original_imgs_np[:] = imgs
    #make array for recentered/rescaled image (only big enough for one wavelength at a time)
    unique_wvs = np.unique(wvs)
    recentered_imgs = mp.Array(mp_data_type, np.size(imgs))
    recentered_imgs_shape = imgs.shape
    #make output array which also has an extra dimension for the number of KL modes to use
    output_imgs = mp.Array(mp_data_type, np.size(imgs)*np.size(numbasis))
    output_imgs_np = _arraytonumpy(output_imgs,dtype=dtype)
    output_imgs_np[:] = np.nan
    output_imgs_shape = imgs.shape + numbasis.shape
    #remake the PA, wv, and center arrays as shared arrays
    pa_imgs = mp.Array(mp_data_type, np.size(parangs))
    pa_imgs_np = _arraytonumpy(pa_imgs,dtype=dtype)
    pa_imgs_np[:] = parangs
    wvs_imgs = mp.Array(mp_data_type, np.size(wvs))
    wvs_imgs_np = _arraytonumpy(wvs_imgs,dtype=dtype)
    wvs_imgs_np[:] = wvs
    centers_imgs = mp.Array(mp_data_type, np.size(centers))
    centers_imgs_np = _arraytonumpy(centers_imgs, centers.shape,dtype=dtype)
    centers_imgs_np[:] = centers

    tpool = mp.Pool(processes=numthreads, initializer=_tpool_init,
                    initargs=(original_imgs, original_imgs_shape, recentered_imgs, recentered_imgs_shape, output_imgs,
                              output_imgs_shape, pa_imgs, wvs_imgs, centers_imgs, None, None), maxtasksperchild=50)

    print("Total number of tasks for KLIP processing is {0}".format(tot_iter))
    jobs_complete = 0
    #align and scale the images for each image. Use map to do this asynchronously
    for wv_index, this_wv in enumerate(np.unique(wvs)):
        print("Begin processing of wv {0:.4} with index {1}".format(this_wv, wv_index))
        print("Aligning and scaling imgs")
        recentered_imgs_np = _arraytonumpy(recentered_imgs, recentered_imgs_shape,dtype=dtype)
        
        #multitask this
        tasks = [tpool.apply_async(_align_and_scale_per_image, args=(img_index, aligned_center, this_wv,dtype))
                 for img_index in range(recentered_imgs_shape[0])]

        #save it to shared memory
        for img_index, aligned_img_task in enumerate(tasks):
            aligned_img_task.wait()

        #list to store each threadpool task
        outputs = []
    
        print("Wavelength {1:.4} with index {0} has finished align and scale. Queuing for KLIP".format(wv_index, this_wv))

        #pick out the science images that need PSF subtraction for this wavelength
        scidata_indicies = np.where(wvs == this_wv)[0]

        # commented out code to do _klip_section instead of _klip_section_multifile
        # outputs += [tpool.apply_async(_klip_section_profiler, args=(file_index, parang, wv_value, wv_index, numbasis,
        #                                                   radstart, radend, phistart, phiend, movement))
        #                     for phistart,phiend in phi_bounds
        #                 for radstart, radend in rad_bounds
        #             for file_index,parang in zip(scidata_indicies, parangs[scidata_indicies])]

        #perform KLIP asynchronously for each group of files of a specific wavelength and section of the image
        lite = True
        outputs += [tpool.apply_async(_klip_section_multifile, args=(scidata_indicies, this_wv, wv_index, numbasis,
                                                                     maxnumbasis,
                                                                     radstart, radend, phistart, phiend, movement,
                                                                     aligned_center, minrot, maxrot, spectrum,
                                                                     mode, None, None, lite, dtype, algo))
                    for phistart,phiend in phi_bounds
                    for radstart, radend in rad_bounds]

        #harness the data!
        #check make sure we are completely unblocked before outputting the data
        for out in outputs:
            out.wait()
            if (jobs_complete + 1) % 10 == 0:
                print("{0:.4}% done ({1}/{2} completed)".format((jobs_complete+1)*100.0/tot_iter, jobs_complete, tot_iter))
            jobs_complete += 1



    #close to pool now and make sure there's no processes still running (there shouldn't be or else that would be bad)
    print("Closing threadpool")
    tpool.close()
    tpool.join()

    #finished. Let's reshape the output images
    #move number of KLIP modes as leading axis (i.e. move from shape (N,y,x,b) to (b,N,y,x)
    sub_imgs = _arraytonumpy(output_imgs, output_imgs_shape,dtype=dtype)
    sub_imgs = np.rollaxis(sub_imgs.reshape((dims[0], dims[1], dims[2], numbasis.shape[0])), 3)

    #restore bad pixels
    sub_imgs[:, allnans[0], allnans[1], allnans[2]] = np.nan

    return sub_imgs, aligned_center



def klip_parallelized(imgs, centers, parangs, wvs, IWA, OWA=None, mode='ADI+SDI', annuli=5, subsections=4, movement=3,
                      numbasis=None, aligned_center=None, numthreads=None, minrot=0, maxrot=360, 
                      annuli_spacing="constant", maxnumbasis=None,
                      spectrum=None,
                      psf_library=None, psf_library_good=None, psf_library_corr=None,
                      save_aligned = False, restored_aligned = None, dtype=None, algo='klip'):
    """
    multithreaded KLIP PSF Subtraction

    Args:
        imgs: array of 2D images for ADI. Shape of array (N,y,x)
        centers: N by 2 array of (x,y) coordinates of image centers
        parangs: N length array detailing parallactic angle of each image
        wvs: N length array of the wavelengths
        IWA: inner working angle (in pixels)
        mode: one of ['ADI', 'SDI', 'ADI+SDI'] for ADI, SDI, or ADI+SDI
        anuuli: number of annuli to use for KLIP
        subsections: number of sections to break each annuli into
        movement: minimum amount of movement (in pixels) of an astrophysical source
                  to consider using that image for a refernece PSF
        numbasis: number of KL basis vectors to use (can be a scalar or list like). Length of b
        aligned_center: array of 2 elements [x,y] that all the KLIP subtracted images will be centered on for image
                        registration
        numthreads: number of threads to use. If none, defaults to using all the cores of the cpu
        minrot: minimum PA rotation (in degrees) to be considered for use as a reference PSF (good for disks)
        maxrot: maximum PA rotation (in degrees) to be considered for use as a reference PSF (temporal variability)
        annuli_spacing: how to distribute the annuli radially. Currently three options. Constant (equally spaced), 
                        log (logarithmical expansion with r), and linear (linearly expansion with r)
        maxnumbasis: if not None, maximum number of KL basis/correlated PSFs to use for KLIP. Otherwise, use max(numbasis)
        spectrum: if not None, a array of length N with the flux of the template spectrum at each wavelength. Uses
                    minmove to determine the separation from the center of the segment to determine contamination and
                    the size of the PSF (TODO: make PSF size another quanitity)
                    (e.g. minmove=3, checks how much containmination is within 3 pixels of the hypothetical source)
                    if smaller than 10%, (hard coded quantity), then use it for reference PSF
        psf_library: array of (N_lib, y, x) with N_lib PSF library PSFs
        psf_library_good: array of size N_lib indicating which N_good are good are selected in the passed in corr matrix
        psf_library_corr: matrix of size N_sci x N_good with correlation between the target franes and the good RDI PSFs
        save_aligned:	Save the aligned and scaled images (as well as various wcs information), True/False
        restore_aligned: The aligned and scaled images from a previous run of klip_dataset
        				(usually restored_aligned = dataset.aligned_and_scaled)
        dtype: data type of the arrays. Should be either ctypes.c_float(default) or ctypes.c_double
        algo (str): algorithm to use ('klip', 'nmf')

    Returns:
        sub_imgs: array of [array of 2D images (PSF subtracted)] using different number of KL basis vectors as
                    specified by numbasis. Shape of (b,N,y,x).
        aligned_center: (x,y) specifying the common center the output images are aligned to
    """

    ################## Interpret input arguments ####################

    #defaullt numbasis if none
    if numbasis is None:
        totalimgs = imgs.shape[0]
        maxbasis = np.min([totalimgs, 100]) #only going up to 100 KL modes by default
        numbasis = np.arange(1, maxbasis + 5, 5)
        print("KL basis not specified. Using default.", numbasis)
    else:
        if hasattr(numbasis, "__len__"):
            numbasis = np.array(numbasis)
        else:
            numbasis = np.array([numbasis])

    #default aligned_center if none:
    if aligned_center is None:
        #aligned_center = [int(imgs.shape[2]//2), int(imgs.shape[1]//2)]
        aligned_center = [np.mean(centers[:,0]), np.mean(centers[:,1])]

    # validate RDI has an RDI Library with supporting cast
    if "RDI" in mode.upper():
        if psf_library is None:
            raise ValueError("Need to pass in PSF library fields if you want to do RDI.")
    if psf_library is not None:
        if psf_library_corr is None or psf_library_good is None:
            raise ValueError("Need to pass in correlatoin matrix and good selection array for PSF library")


    #save all bad pixels
    allnans = np.where(np.isnan(imgs))

    #use first image to figure out how to divide the annuli
    #TODO: what to do with OWA
    #need to make the next 10 lines or so much smarter
    dims = imgs.shape
    x, y = np.meshgrid(np.arange(dims[2] * 1.0), np.arange(dims[1] * 1.0))
    nanpix = np.where(np.isnan(imgs[0]))
    # if user didn't supply how to define NaNs
    if OWA is None:
        full_image = True # reduce the full image
        # define OWA as either the closest NaN pixel or edge of image if no NaNs exist
        if np.size(nanpix) == 0:
            OWA = np.sqrt(np.max((x - centers[0][0]) ** 2 + (y - centers[0][1]) ** 2))
        else:
            # grab the NaN from the 1st percentile (this way we drop outliers)
            OWA = np.sqrt(np.percentile((x[nanpix] - centers[0][0]) ** 2 + (y[nanpix] - centers[0][1]) ** 2, 1))
    else:
        full_image = False # don't reduce the full image, only up the the IWA

    #error checking for too small of annuli go here


    #calculate the annuli ranges
    rad_bounds = klip.define_annuli_bounds(annuli, IWA, OWA, annuli_spacing)

    # if OWA wasn't passed in, we're going to assume we reduce the full image, so last sector emcompasses everything
    if full_image:
        # last annulus should mostly emcompass everything
        rad_bounds[annuli - 1] = (rad_bounds[annuli - 1][0], imgs[0].shape[0])

    #divide annuli into subsections
    dphi = 2 * np.pi / subsections
    phi_bounds = [[dphi * phi_i - np.pi, dphi * (phi_i + 1) - np.pi] for phi_i in range(subsections)]
    phi_bounds[-1][1] = np.pi

    # print(rad_bounds)

    #calculate how many iterations we need to do
    global tot_iter
    tot_iter = np.size(np.unique(wvs)) * len(phi_bounds) * len(rad_bounds)

    #before we start, create the output array in flattened form
    #sub_imgs = np.zeros([dims[0], dims[1] * dims[2], numbasis.shape[0]])


    ########################### Create Shared Memory ###################################

    # default to using float precision
    if dtype is None:
        dtype = ctypes.c_float
    # we should use the same datatype for both
    mp_data_type = dtype

    #implement the thread pool
    #make a bunch of shared memory arrays to transfer data between threads
    #make the array for the original images and initalize it
    original_imgs = mp.Array(mp_data_type, np.size(imgs))
    original_imgs_shape = imgs.shape
    original_imgs_np = _arraytonumpy(original_imgs, original_imgs_shape,dtype=dtype)
    original_imgs_np[:] = imgs
    #make array for recentered/rescaled image for each wavelength
    unique_wvs = np.unique(wvs)
    recentered_imgs = mp.Array(mp_data_type, np.size(imgs)*np.size(unique_wvs))
    recentered_imgs_shape = (np.size(unique_wvs),) + imgs.shape
    #make output array which also has an extra dimension for the number of KL modes to use
    output_imgs = mp.Array(mp_data_type, np.size(imgs)*np.size(numbasis))
    output_imgs_np = _arraytonumpy(output_imgs,dtype=dtype)
    output_imgs_np[:] = np.nan
    output_imgs_shape = imgs.shape + numbasis.shape
    #remake the PA, wv, and center arrays as shared arrays
    pa_imgs = mp.Array(mp_data_type, np.size(parangs))
    pa_imgs_np = _arraytonumpy(pa_imgs,dtype=dtype)
    pa_imgs_np[:] = parangs
    wvs_imgs = mp.Array(mp_data_type, np.size(wvs))
    wvs_imgs_np = _arraytonumpy(wvs_imgs,dtype=dtype)
    wvs_imgs_np[:] = wvs
    centers_imgs = mp.Array(mp_data_type, np.size(centers))
    centers_imgs_np = _arraytonumpy(centers_imgs, centers.shape,dtype=dtype)
    centers_imgs_np[:] = centers
    if psf_library is not None:
        psf_lib = mp.Array(mp_data_type, np.size(psf_library))
        psf_lib_np = _arraytonumpy(psf_lib, psf_library.shape, dtype=dtype)
        psf_lib_np[:] = psf_library
        psf_lib_shape = psf_library.shape
    else:
        psf_lib = None
        psf_lib_shape = None

    if restored_aligned is not None:
        recentered_imgs_np = _arraytonumpy(recentered_imgs, recentered_imgs_shape,dtype=dtype)
        recentered_imgs_np[:] = restored_aligned

    tpool = mp.Pool(processes=numthreads, initializer=_tpool_init,
                    initargs=(original_imgs, original_imgs_shape, recentered_imgs, recentered_imgs_shape, output_imgs,
                              output_imgs_shape, pa_imgs, wvs_imgs, centers_imgs, psf_lib, psf_lib_shape),
                    maxtasksperchild=50)

    if restored_aligned is None:
        #align and scale the images for each image. Use map to do this asynchronously
        print("Begin align and scale images for each wavelength")
        realigned_index = tpool.imap_unordered(_align_and_scale, zip(enumerate(unique_wvs), itertools.repeat(aligned_center),itertools.repeat(dtype)))
    else:
        #align and scale the images for each image. Use map to do this asynchronously
        realigned_index = enumerate(unique_wvs)

    #list to store each threadpool task
    outputs = []
    #as each is finishing, queue up the aligned data to be processed with KLIP
    for wv_index, wv_value in realigned_index:
        print("Wavelength {1:.4} with index {0} has finished align and scale. Queuing for KLIP".format(wv_index, wv_value))

        #pick out the science images that need PSF subtraction for this wavelength
        scidata_indicies = np.where(wvs == wv_value)[0]

        # commented out code to do _klip_section instead of _klip_section_multifile
        # outputs += [tpool.apply_async(_klip_section_profiler, args=(file_index, parang, wv_value, wv_index, numbasis,
        #                                                   radstart, radend, phistart, phiend, movement))
        #                     for phistart,phiend in phi_bounds
        #                 for radstart, radend in rad_bounds
        #             for file_index,parang in zip(scidata_indicies, parangs[scidata_indicies])]

        #perform KLIP asynchronously for each group of files of a specific wavelength and section of the image
        lite = False
        outputs += [tpool.apply_async(_klip_section_multifile, args=(scidata_indicies, wv_value, wv_index, numbasis,
                                                                     maxnumbasis,
                                                                     radstart, radend, phistart, phiend, movement,
                                                                     aligned_center, minrot, maxrot, spectrum,
                                                                     mode, psf_library_good, psf_library_corr, False,
                                                                     dtype, algo))
                    for phistart,phiend in phi_bounds
                    for radstart, radend in rad_bounds]

    #harness the data!
    #check make sure we are completely unblocked before outputting the data
    print("Total number of tasks for KLIP processing is {0}".format(tot_iter))
    for index, out in enumerate(outputs):
        out.wait()
        if (index + 1) % 10 == 0:
            print("{0:.4}% done ({1}/{2} completed)".format((index+1)*100.0/tot_iter, index, tot_iter))



    #close to pool now and make sure there's no processes still running (there shouldn't be or else that would be bad)
    print("Closing threadpool")
    tpool.close()
    tpool.join()

    #finished. Let's reshape the output images
    #move number of KLIP modes as leading axis (i.e. move from shape (N,y,x,b) to (b,N,y,x)
    sub_imgs = _arraytonumpy(output_imgs, output_imgs_shape,dtype=dtype)
    sub_imgs = np.rollaxis(sub_imgs.reshape((dims[0], dims[1], dims[2], numbasis.shape[0])), 3)

    #restore bad pixels
    sub_imgs[:, allnans[0], allnans[1], allnans[2]] = np.nan

    if save_aligned:
        aligned_and_scaled = _arraytonumpy(recentered_imgs, recentered_imgs_shape, dtype=dtype)
        return sub_imgs, aligned_center, aligned_and_scaled
    else:
        return sub_imgs, aligned_center


def klip_dataset(dataset, mode='ADI+SDI', outputdir=".", fileprefix="", annuli=5, subsections=4, movement=3,
                 numbasis=None, numthreads=None, minrot=0, calibrate_flux=False, aligned_center=None,
                 annuli_spacing="constant", maxnumbasis=None, spectrum=None, psf_library=None, highpass=False,
                 lite=False, save_aligned = False, restored_aligned = None, dtype=None, algo='klip'):
    """
    run klip on a dataset class outputted by an implementation of Instrument.Data

    Args:
        dataset:        an instance of Instrument.Data (see instruments/ subfolder)
        mode:           some combination of ADI, SDI, and RDI (e.g. "ADI+SDI", "RDI")
        outputdir:      directory to save output files
        fileprefix:     filename prefix for saved files
        anuuli:         number of annuli to use for KLIP
        subsections:    number of sections to break each annuli into
        movement:       minimum amount of movement (in pixels) of an astrophysical source
                        to consider using that image for a refernece PSF
        numbasis:       number of KL basis vectors to use (can be a scalar or list like). Length of b
        numthreads:     number of threads to use. If none, defaults to using all the cores of the cpu
        minrot:         minimum PA rotation (in degrees) to be considered for use as a reference PSF (good for disks)
        calibrate_flux: if True calibrate flux of the dataset, otherwise leave it be
        aligned_center: array of 2 elements [x,y] that all the KLIP subtracted images will be centered on for image
                        registration
        annuli_spacing: how to distribute the annuli radially. Currently three options. Constant (equally spaced), 
                        log (logarithmical expansion with r), and linear (linearly expansion with r)
        maxnumbasis: if not None, maximum number of KL basis/correlated PSFs to use for KLIP. Otherwise, use max(numbasis)

        spectrum:       (only applicable for SDI) if not None, optimizes the choice of the reference PSFs based on the
                        spectrum shape. Currently only supports "methane" between 1 and 10 microns.
        psf_library:    if not None, a rdi.PSFLibrary object with a PSF Library for RDI
        highpass:       if True, run a Gaussian high pass filter (default size is sigma=imgsize/10)
                            can also be a number specifying FWHM of box in pixel units

        lite:           if True, run a low memory version of the alogirhtm

        save_aligned	Save the aligned and scaled images (as well as various wcs information), True/False
        restore_aligned The aligned and scaled images from a previous run of klip_dataset
        				(usually restored_aligned = dataset.aligned_and_scaled)
        dtype: data type of the arrays. Should be either ctypes.c_float(default) or ctypes.c_double
        algo (str): algorithm to use ('klip', 'nmf')

    Returns
        Saved files in the output directory
        Returns: nothing, but saves to dataset.output: (b, N, wv, y, x) 5D cube of KL cutoff modes (b), number of images
                            (N), wavelengths (wv), and spatial dimensions. Images are derotated.
                            For ADI only, the wv is omitted so only 4D cube
    """
    # defaullt numbasis if none
    if numbasis is None:
        totalimgs = dataset.input.shape[0]
        maxbasis = np.min([totalimgs, 100]) # only going up to 100 KL modes by default
        numbasis = np.arange(1, maxbasis + 5, 5)
        print("KL basis not specified. Using default.", numbasis)
    else:
        if hasattr(numbasis, "__len__"):
            numbasis = np.array(numbasis)
        else:
            numbasis = np.array([numbasis])

    # check which algo we will use and whether the inputs are correct
    if algo.lower() == 'klip':
        pass
    elif algo.lower() == 'nmf':
        # check to see the correct nmf packages are installed 
        import pyklip.nmf_imaging as nmf_imaging
        if np.size(numbasis) > 1:
            raise ValueError("NMF can only be run with one basis")
    else:
        raise ValueError("Algo {0} is not supported".format(algo))

    # RDI Sanity Checks to make sure PSF Library is properly configured
    if "RDI" in mode:
        if psf_library is None:
            raise ValueError("You need to pass in a psf_library if you want to run RDI")
        if psf_library.dataset is dataset:
            raise ValueError("The PSF Library is not prepared for this dataset. Run psf_library.prepare_library()")
        if aligned_center is not None:
            if np.array_equal(aligned_center, psf_library.aligned_center): 
                raise ValueError("The images need to be aligned to the same center as the RDI Library")

        else:
            aligned_center = psf_library.aligned_center
        # good rdi_library
        master_library = psf_library.master_library
        rdi_corr_matrix = psf_library.correlation
        rdi_good_psfs = psf_library.isgoodpsf
    else:
        master_library = None
        rdi_corr_matrix = None
        rdi_good_psfs = None

    # Save the WCS and centers info, incase we need it again!
    if save_aligned is True:
        dataset.old_wcs = copy.deepcopy(dataset.wcs)
        dataset.old_centers = copy.deepcopy(dataset.centers)
        dataset.old_PAs = copy.deepcopy(dataset.PAs)

    # select memory-lite version if requested
    # If lite, we are not running iwth save_aligned and restore_aligned keywords. Throw error if this happens
    if lite:
        klip_function = klip_parallelized_lite
        if (save_aligned is True) or (restored_aligned is True):
            print('save_aligned and restored_aligned are not compatible with lite mode')
            raise ValueError('save_aligned and restored_aligned are not compatible with lite mode')
        # save_aligned = False
        # restored_aligned = None
    else:
        klip_function = klip_parallelized

    # If re-running KLIP with same data, restore centers to old value
    # We don't need to highpass the data if the aligned and scaled images are being restored
    if restored_aligned is not None:
        dataset.centers = copy.deepcopy(dataset.old_centers)
        dataset.wcs = copy.deepcopy(dataset.old_wcs)
        dataset.PAs = copy.deepcopy(dataset.old_PAs)
    else:
        if isinstance(highpass, bool):
            if highpass:
                dataset.input = high_pass_filter_imgs(dataset.input, numthreads=numthreads)
        else:
            # should be a number
            if isinstance(highpass, (float, int)):
                highpass = float(highpass)
                fourier_sigma_size = (dataset.input.shape[1]/(highpass)) / (2*np.sqrt(2*np.log(2)))
                dataset.input = high_pass_filter_imgs(dataset.input, numthreads=numthreads, filtersize=fourier_sigma_size)

    # if no outputdir specified, then current working directory (don't want to write to '/'!)
    if outputdir == "":
        outputdir = "."

    if spectrum is not None:
        if spectrum.lower() == "methane":
            pykliproot = os.path.dirname(os.path.realpath(__file__))
            spectrum_dat = np.loadtxt(os.path.join(pykliproot,"spectra","t800g100nc.flx"))[:160] #skip wavelegnths longer of 10 microns
            spectrum_wvs = spectrum_dat[:,1]
            spectrum_fluxes = spectrum_dat[:,3]
            spectrum_interpolation = interp.interp1d(spectrum_wvs, spectrum_fluxes, kind='cubic')

            spectra_template = spectrum_interpolation(dataset.wvs)
            #unique_wvs = np.unique(wvs)
            #spectra_template = spectrum_interpolation(unique_wvs)
        else:
            raise ValueError("{0} is not a valid spectral template. Only currently supporting 'methane'"
                             .format(spectrum))
    else:
        spectra_template = None

    # save klip parameters as a string
    klipparams = "mode={mode},annuli={annuli},subsect={subsections},minmove={movement}," \
                 "numbasis={numbasis}/{maxbasis},minrot={minrot},calibflux={calibrate_flux},spectrum={spectrum}," \
                 "highpass={highpass}".format(mode=mode, annuli=annuli, subsections=subsections, movement=movement,
                                              numbasis="{numbasis}", maxbasis=np.max(numbasis), minrot=minrot,
                                              calibrate_flux=calibrate_flux, spectrum=spectrum, highpass=highpass)
    dataset.klipparams = klipparams

    #Set MLK parameters
    if mkl_exists:
        old_mkl = mkl.get_max_threads()
        mkl.set_num_threads(1)

    # some bookkeeping
    unique_wvs = np.unique(dataset.wvs)
    num_wvs = int(np.size(unique_wvs))
    number_of_klmodes = np.size(numbasis)
    num_cubes = np.size(dataset.wvs) // num_wvs

    # run KLIP
    # For SDI(+ADI)(+RDI) reductions
    if "SDI" in mode:
        print("Beginning {0} KLIP".format(mode))

        # Actually run the PSF Subtraction with all the arguments
        klip_outputs = klip_function(dataset.input, dataset.centers, dataset.PAs, dataset.wvs, dataset.IWA,
                                     OWA=dataset.OWA, mode=mode,
                                     annuli=annuli, subsections=subsections, movement=movement, numbasis=numbasis,
                                     numthreads=numthreads, minrot=minrot, aligned_center=aligned_center,
                                     annuli_spacing=annuli_spacing, maxnumbasis=maxnumbasis,
                                     spectrum=spectra_template, psf_library=master_library,
                                     psf_library_corr=rdi_corr_matrix, psf_library_good=rdi_good_psfs,
                                     save_aligned = save_aligned, restored_aligned = restored_aligned, dtype=dtype,
                                     algo=algo)

        # parse the output of klip. Normally, it is just the klipped_imgs,
        # but some optional arguments return more things
        if not save_aligned:
            klipped_imgs, klipped_center = klip_outputs
        else:
            klipped_imgs, klipped_center, dataset.aligned_and_scaled = klip_outputs

        # save output and image center for each output
        dataset.output = klipped_imgs
        dataset.output_centers = np.array([klipped_center for _ in range(klipped_imgs.shape[1])])
        # construct the output wcs info, but it's currently just a copy of the input one until we rotate it
        dataset.output_wcs = np.array([w.deepcopy() if w is not None else None for w in dataset.wcs])

    # For ADI only datasets, can run KLIP on each wavelenght separately
    else:
        # set up output, output centers, and output wcs variables but they are the same as the input for now
        dataset.output_centers = np.copy(dataset.centers)
        dataset.output_wcs = np.array([w.deepcopy() if w is not None else None for w in dataset.wcs])

        # append output to a list at first since we are running it a bunch of times
        dataset.output = []

        # save the output of aligned_and_scaled optionally in a list
        if save_aligned:
            dataset.aligned_and_scaled = []

        for wvindex,unique_wv in enumerate(unique_wvs):
            if num_wvs > 1:
                print("Running KLIP ADI on slice {0}/{1}: {2:.3f} um".format(wvindex+1, num_wvs, unique_wv))
            thiswv = np.where(dataset.wvs == unique_wv)

            if restored_aligned is not None:
                restored_aligned_thiswv = restored_aligned[np.where(unique_wv == unique_wvs)]
            else:
                restored_aligned_thiswv = None

            klip_output = klip_function(dataset.input[thiswv], dataset.centers[thiswv], dataset.PAs[thiswv], dataset.wvs[thiswv],
                                        dataset.IWA, OWA=dataset.OWA, mode=mode, annuli=annuli, subsections=subsections,
                                        movement=movement, numbasis=numbasis, numthreads=numthreads, minrot=minrot,
                                        maxnumbasis=maxnumbasis, annuli_spacing=annuli_spacing,
                                        aligned_center=aligned_center, psf_library=master_library,
                                        psf_library_corr=rdi_corr_matrix, psf_library_good=rdi_good_psfs,
                                        save_aligned = save_aligned, restored_aligned=restored_aligned_thiswv,
                                        dtype=dtype)
            
            klipped_imgs = klip_output[0]
            klipped_center = klip_output[1]
            
            # save data for this wavelength
            dataset.output.append(klipped_imgs)
            dataset.output_centers[thiswv[0], 0] = klipped_center[0]
            dataset.output_centers[thiswv[0], 1] = klipped_center[1]

            # if we need to save the aligned_and_scaled images, put them in a list
            if save_aligned:
                dataset.aligned_and_scaled.append(klip_output[2][0])

        # convert lists to numpy arrays
        dataset.output = np.array(dataset.output)

        if save_aligned:
            dataset.aligned_and_scaled = np.array(dataset.aligned_and_scaled)

        # reformat the output to be consistent with the other modes.
        # Currently shape is (wv,b,N,y,x). Switch to (b,N,wv,y,x), then flatten in wavelength dimension
        dataset.output = np.swapaxes(dataset.output, 0, 1) # shape of (b, wv, N, y, x)
        dataset.output = np.swapaxes(dataset.output, 1, 2) # shape of (b, N, wv, y, x)
        # then collapse N/wv together
        dataset.output = np.reshape(dataset.output, (dataset.output.shape[0], dataset.output.shape[1]*dataset.output.shape[2], dataset.output.shape[3], dataset.output.shape[4]) )


    # TODO: handling of only a single numbasis
    # derotate all the images
    # flatten so it's just a 3D array (collapse KL and Nframes dimensions)
    oldshape = dataset.output.shape
    dataset.output = dataset.output.reshape(oldshape[0]*oldshape[1], oldshape[2], oldshape[3])

    # we need to duplicate PAs and centers for the different KL mode cutoffs we supplied
    flattend_parangs = np.tile(dataset.PAs, oldshape[0])
    flattened_centers = np.tile(dataset.output_centers.reshape(oldshape[1]*2), oldshape[0]).reshape(oldshape[1]*oldshape[0],2)

    # align center to center of image if not specified
    # note that klip_parallelized aligns everything to the mean of the input centers, whereas now we will re align it
    # to the middle of the array for cosmetic purposes. 
    if aligned_center is None:
        aligned_center = [int(dataset.input.shape[2]//2), int(dataset.input.shape[1]//2)]

    # parallelized rotate images
    print("Derotating Images...")
    rot_imgs = rotate_imgs(dataset.output, flattend_parangs, flattened_centers, numthreads=numthreads, flipx=dataset.flipx,
                           hdrs=dataset.output_wcs, new_center=aligned_center)

    # give rot_imgs dimensions of (num KLmode cutoffs, num cubes, num wvs, y, x)
    rot_imgs = rot_imgs.reshape(oldshape[0], oldshape[1]//num_wvs, num_wvs, oldshape[2], oldshape[3])

    # save modified data and centers
    dataset.output = rot_imgs
    dataset.output_centers[:,0] = aligned_center[0]
    dataset.output_centers[:,1] = aligned_center[1]

    # valid output path and write iamges
    outputdirpath = os.path.realpath(outputdir)
    print("Writing Images to directory {0}".format(outputdirpath))

    # collapse in time and wavelength to examine KL modes
    if spectrum is None or num_wvs == 1:
        KLmode_cube = np.nanmean(dataset.output, axis=(1,2))
    else:
        #do the mean combine by weighting by the spectrum
        spectra_template = spectra_template.reshape(dataset.output.shape[1:3]) #make same shape as dataset.output
        KLmode_cube = np.nanmean(dataset.output * spectra_template[None,:,:,None,None], axis=(1,2))\
                        / np.mean(spectra_template)
    # broadband flux calibration for KL mode cube
    if calibrate_flux:
        KLmode_cube = dataset.calibrate_output(KLmode_cube, spectral=False)
    numbasis_str = '[' + " ".join(str(basis) for basis in numbasis) + ']'
    dataset.savedata(outputdirpath + '/' + fileprefix + "-KLmodes-all.fits", KLmode_cube,
                     klipparams=klipparams.format(numbasis=numbasis_str), filetype="KL Mode Cube",
                     zaxis=numbasis)

    # if there is more than one wavelenght channel, save spectral data cubes too
    if num_wvs > 1:
        # for each KL mode, collapse in time to examine spectra
        KLmode_spectral_cubes = np.nanmean(dataset.output, axis=1)
        for KLcutoff, spectral_cube in zip(numbasis, KLmode_spectral_cubes):
            # calibrate spectral cube if needed
            if calibrate_flux:
                spectral_cube = dataset.calibrate_output(spectral_cube, spectral=True)
            dataset.savedata(outputdirpath + '/' + fileprefix + "-KL{0}-speccube.fits".format(KLcutoff),
                             spectral_cube, klipparams=klipparams.format(numbasis=KLcutoff),
                             filetype="PSF Subtracted Spectral Cube")



    #Restore old setting
    if mkl_exists:
        mkl.set_num_threads(old_mkl)

