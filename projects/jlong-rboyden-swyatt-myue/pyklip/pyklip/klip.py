import numpy as np
import numpy.fft as fft
import scipy.linalg as la
import scipy.ndimage as ndimage
from scipy.stats import t


def klip_math(sci, ref_psfs, numbasis, covar_psfs=None, return_basis=False, return_basis_and_eig=False):
    """
    Helper function for KLIP that does the linear algebra
    
    Args:
        sci: array of length p containing the science data
        ref_psfs: N x p array of the N reference PSFs that 
                  characterizes the PSF of the p pixels
        numbasis: number of KLIP basis vectors to use (can be an int or an array of ints of length b)
        covar_psfs: covariance matrix of reference psfs passed in so you don't have to calculate it here
        return_basis: If true, return KL basis vectors (used when onesegment==True)
        return_basis_and_eig: If true, return KL basis vectors as well as the eigenvalues and eigenvectors of the
                                covariance matrix. Used for KLIP Forward Modelling of Laurent Pueyo.

    Returns:
        sub_img_rows_selected: array of shape (p,b) that is the PSF subtracted data for each of the b KLIP basis
                               cutoffs. If numbasis was an int, then sub_img_row_selected is just an array of length p
        KL_basis: array of shape (max(numbasis),p). Only if return_basis or return_basis_and_eig is True.
        evals: Eigenvalues of the covariance matrix. The covariance matrix is assumed NOT to be normalized by (p-1).
                Only if return_basis_and_eig is True.
        evecs: Eigenvectors of the covariance matrix. The covariance matrix is assumed NOT to be normalized by (p-1).
                Only if return_basis_and_eig is True.
    """
    # for the science image, subtract the mean and mask bad pixels
    sci_mean_sub = sci - np.nanmean(sci)
    # sci_nanpix = np.where(np.isnan(sci_mean_sub))
    # sci_mean_sub[sci_nanpix] = 0

    # do the same for the reference PSFs
    # playing some tricks to vectorize the subtraction
    ref_psfs_mean_sub = ref_psfs - np.nanmean(ref_psfs, axis=1)[:, None]
    ref_psfs_mean_sub[np.where(np.isnan(ref_psfs_mean_sub))] = 0

    # calculate the covariance matrix for the reference PSFs
    # note that numpy.cov normalizes by p-1 to get the NxN covariance matrix
    # we have to correct for that a few lines down when consturcting the KL
    # vectors since that's not part of the equation in the KLIP paper
    if covar_psfs is None:
        covar_psfs = np.cov(ref_psfs_mean_sub)

    # maximum number of KL modes
    tot_basis = covar_psfs.shape[0]

    # only pick numbasis requested that are valid. We can't compute more KL basis than there are reference PSFs
    # do numbasis - 1 for ease of indexing since index 0 is using 1 KL basis vector
    numbasis = np.clip(numbasis - 1, 0, tot_basis-1)  # clip values, for output consistency we'll keep duplicates
    max_basis = np.max(numbasis) + 1  # maximum number of eigenvectors/KL basis we actually need to use/calculate

    # calculate eigenvalues and eigenvectors of covariance matrix, but only the ones we need (up to max basis)
    evals, evecs = la.eigh(covar_psfs, eigvals=(tot_basis-max_basis, tot_basis-1))

    # check if there are negative eignevalues as they will cause NaNs later that we have to remove
    # the eigenvalues are ordered smallest to largest
    #check_nans = evals[-1] < 0 # currently this checks that *all* the evals are neg, but we want just one.
    # also, include 0 because that is a bad value too
    check_nans = np.any(evals <= 0) # alternatively, check_nans = evals[0] <= 0

    # scipy.linalg.eigh spits out the eigenvalues/vectors smallest first so we need to reverse
    # we're going to recopy them to hopefully improve caching when doing matrix multiplication
    evals = np.copy(evals[::-1])
    evecs = np.copy(evecs[:,::-1], order='F') #fortran order to improve memory caching in matrix multiplication

    # keep an index of the negative eignevalues for future reference if there are any
    if check_nans:
        neg_evals = (np.where(evals <= 0))[0]

    # calculate the KL basis vectors
    kl_basis = np.dot(ref_psfs_mean_sub.T, evecs)
    # JB question: Why is there this [None, :]? (It adds an empty first dimension)
    kl_basis = kl_basis * (1. / np.sqrt(evals * (np.size(sci) - 1)))[None, :]  #multiply a value for each row

    # sort to KL basis in descending order (largest first)
    # kl_basis = kl_basis[:,eig_args_all]

    # duplicate science image by the max_basis to do simultaneous calculation for different k_KLIP
    sci_mean_sub_rows = np.tile(sci_mean_sub, (max_basis, 1))
    sci_rows_selected = np.tile(sci_mean_sub, (np.size(numbasis), 1)) # this is the output image which has less rows

    # bad pixel mask
    # do it first for the image we're just doing computations on but don't care about the output
    sci_nanpix = np.where(np.isnan(sci_mean_sub_rows))
    sci_mean_sub_rows[sci_nanpix] = 0
    # now do it for the output image
    sci_nanpix = np.where(np.isnan(sci_rows_selected))
    sci_rows_selected[sci_nanpix] = 0

    # do the KLIP equation, but now all the different k_KLIP simultaneously
    # calculate the inner product of science image with each of the different kl_basis vectors
    # TODO: can we optimize this so it doesn't have to multiply all the rows because in the next lines we only select some of them
    inner_products = np.dot(sci_mean_sub_rows, np.require(kl_basis, requirements=['F']))
    # select the KLIP modes we want for each level of KLIP by multiplying by lower diagonal matrix
    lower_tri = np.tril(np.ones([max_basis, max_basis]))
    inner_products = inner_products * lower_tri
    # if there are NaNs due to negative eigenvalues, make sure they don't mess up the matrix multiplicatoin
    # by setting the appropriate values to zero
    if check_nans:
        needs_to_be_zeroed = np.where(lower_tri == 0)
        inner_products[needs_to_be_zeroed] = 0
        # make a KLIP PSF for each amount of klip basis, but only for the amounts of klip basis we actually output
        kl_basis[:, neg_evals] = 0
        klip_psf = np.dot(inner_products[numbasis,:], kl_basis.T)
        # for KLIP PSFs that use so many KL modes that they become nans, we have to put nan's back in those
        badbasis = np.where(numbasis >= np.min(neg_evals)) #use basis with negative eignevalues
        klip_psf[badbasis[0], :] = np.nan
    else:
        # make a KLIP PSF for each amount of klip basis, but only for the amounts of klip basis we actually output
        klip_psf = np.dot(inner_products[numbasis,:], kl_basis.T)

    # make subtracted image for each number of klip basis
    sub_img_rows_selected = sci_rows_selected - klip_psf

    # restore NaNs
    sub_img_rows_selected[sci_nanpix] = np.nan


    if return_basis is True:
        return sub_img_rows_selected.transpose(), kl_basis.transpose()
    elif return_basis_and_eig is True:
        return sub_img_rows_selected.transpose(), kl_basis.transpose(),evals*(np.size(sci)-1), evecs
    else:
        return sub_img_rows_selected.transpose()


    # old code that only did one number of KL basis for truncation
    # #truncation either based on user input or maximum number of PSFs
    # trunc_basis = np.min([numbasis, tot_basis])
    # #eigenvalues are ordered largest first now
    # eig_args = eig_args_all[0: trunc_basis]
    # kl_basis = kl_basis[:, eig_args]
    #
    # #project KL vectors onto science image to construct model PSF
    # inner_products = np.dot(sci_mean_sub, kl_basis)
    # klip_psf = np.dot(inner_products, kl_basis.T)
    #
    # #subtract from original image to get final image
    # sub_img = sci_mean_sub - klip_psf
    #
    # #restore NANs
    # sub_img[sci_nanpix] = np.nan
    #
    # #pdb.set_trace()
    #
    # return sub_img


def estimate_movement(radius, parang0=None, parangs=None, wavelength0=None, wavelengths=None, mode=None):
    """
    Estimates the movement of a hypothetical astrophysical source in ADI and/or SDI at the given radius and
    given reference parallactic angle (parang0) and reference wavelegnth (wavelength0)

    Args:
        radius: the radius from the star of the hypothetical astrophysical source
        parang0: the parallactic angle of the reference image (in degrees)
        parangs: array of length N of the parallactic angle of all N images (in degrees)
        wavelength0: the wavelength of the reference image
        wavelengths: array of length N of the wavelengths of all N images
        NOTE: we expect parang0 and parangs to be either both defined or both None.
                Same with wavelength0 and wavelengths
        mode: one of ['ADI', 'SDI', 'ADI+SDI'] for ADI, SDI, or ADI+SDI

    Returns:
        moves: array of length N of the distance an astrophysical source would have moved from the
               reference image
    """
    #default no movement parameters
    dtheta = 0 # how much the images moved in theta (polar coordinate)
    scale_fac = 1 # how much the image moved radially (r/radius)

    if (parang0 is not None):
        dtheta = np.radians(parang0 - parangs)
    if (wavelength0 is not None):
        scale_fac = (wavelength0/wavelengths)

    #define cartesean coordinate system where astrophysical source is at (x,y) = (r,0)
    x0 = radius
    y0 = 0.

    #find x,y location of astrophysical source for the rest of the images
    r = radius * scale_fac
    x = r * np.cos(dtheta)
    y = r * np.sin(dtheta)

    moves = np.sqrt((x-x0)**2 + (y-y0)**2)
    return moves

def calc_scaling(sats, refwv=18):
    """
    Helper function that calculates the wavelength scaling factor from the satellite spot locations.
    Uses the movement of spots diagonally across from each other, to calculate the scaling in a 
    (hopefully? tbd.) centering-independent way. 
    This method is definitely temporary and will be replaced by better scaling strategies as we come
    up with them.
    Scaling is calculated as the average of (1/2 * sqrt((x_1-x_2)**2+(y_1-y_2))), over the two pairs
    of spots.

    Args:
        sats: [4 x Nlambda x 2] array of x and y positions for the 4 satellite spots
        refwv: reference wavelength for scaling (optional, default = 20)
    Returns:
        scaling_factors: Nlambda array of scaling factors
    """
    pairs = [(0,3), (1,2)] # diagonally-located spots (spot_num - 1 for indexing)
    separations = np.mean([0.5*np.sqrt(np.diff(sats[p,:,0], axis=0)[0]**2 + np.diff(sats[p,:,1], axis=0)[0]**2) 
                           for p in pairs], 
                          axis=0) # average over each pair, the first axis

    scaling_factors = separations/separations[refwv]
    return scaling_factors

def align_and_scale(img, new_center, old_center=None, scale_factor=1,dtype=float):
    """
    Helper function that realigns and/or scales the image

    Args:
        img: 2D image to perform manipulation on
        new_center: 2 element tuple (xpos, ypos) of new image center
        old_center: 2 element tuple (xpos, ypos) of old image center
        scale_factor: how much the stretch/contract the image. Will we
                      scaled w.r.t the new_center (done after relaignment).
                      We will adopt the convention
                        >1: stretch image (shorter to longer wavelengths)
                        <1: contract the image (longer to shorter wvs)
                        This means scale factor should be lambda_0/lambda
                        where lambda_0 is the wavelength you want to scale to
    Returns:
        resampled_img: shifted and/or scaled 2D image
    """
    #import scipy.interpolate as interp
    #import pdb

    #create the coordinate system of the image to manipulate for the transform
    dims = img.shape
    x, y = np.meshgrid(np.arange(dims[1], dtype=dtype), np.arange(dims[0], dtype=dtype))
    mod_flag = 0 #check how many modifications we are making

    #if old_center is specified, realign the images
    if ((old_center is not None) & ~(np.array_equal(new_center, old_center))):
        dx = new_center[0] - old_center[0]
        dy = new_center[1] - old_center[1]
        x -= dx
        y -= dy
        mod_flag += 1
        #6/10/2016: Next line is a Bug fix
        new_center = old_center

    #if scale_factor is specified, scale the images
    if scale_factor != 1:
        #conver to polar for scaling
        r = np.sqrt((x - new_center[0]) ** 2 + (y - new_center[1]) ** 2)
        theta = np.arctan2(y - new_center[1], x - new_center[0])  #theta range is [-pi,pi]

        #Because x and y are the coordinates where we want to interpolate in the original image. See the following lines
        r /= scale_factor

        #convert back to cartesian
        x = r * np.cos(theta) + new_center[0]
        y = r * np.sin(theta) + new_center[1]
        mod_flag += 1

    #if nothing is to be changed, return a copy of the image
    if mod_flag == 0:
        return np.copy(img)


    #resample image based on new coordinates
    #scipy uses y,x convention when meshgrid uses x,y
    #stupid scipy functions can't work with masked arrays (NANs)
    #and trying to use interp2d with sparse arrays is way to slow
    #hack my way out of this by picking a really small value for NANs and try to detect them after the interpolation
    #then redo the transformation setting NaN to zero to reduce interpolation effects, but using the mask we derived
    minval = np.min([np.nanmin(img), 0.0])
    nanpix = np.where(np.isnan(img))
    medval = np.median(img[np.where(~np.isnan(img))])
    img_copy = np.copy(img)
    img_copy[nanpix] = minval * 5.0
    resampled_img_mask = ndimage.map_coordinates(img_copy, [y, x], cval=minval * 5.0)
    img_copy[nanpix] = medval
    resampled_img = ndimage.map_coordinates(img_copy, [y,x], cval=np.nan)
    resampled_img[np.where(resampled_img_mask < minval)] = np.nan

    # # JB debug
    # nanpix = np.where(np.isnan(img))
    # medval = np.median(img[np.where(~np.isnan(img))])
    # img_copy = np.copy(img)
    # img_copy[nanpix] = medval

    resampled_img = ndimage.map_coordinates(img_copy, [y, x], cval = np.nan)
    resampled_img[nanpix] = np.nan

    #broken attempt at using sparse arrays with interp2d. Warning: takes forever to run
    #good_dat = np.where(~(np.isnan(img)))
    ##recreate old coordinate system
    #x0,y0 = np.meshgrid(np.arange(dims[0], dtype=np.float32), np.arange(dims[1], dtype=np.float32))
    #interpolated = interp.interp2d(x0[good_dat], y0[good_dat], img[good_dat], kind='cubic')
    #resampled_img = np.ones(img.shape) + np.nan
    #resampled_img[good] = interpolated(y[good],x[good])

    return resampled_img

def rotate(img, angle, center, new_center=None, flipx=True, astr_hdr=None):
    """
    Rotate an image by the given angle about the given center.
    Optional: can shift the image to a new image center after rotation. Also can reverse x axis for those left
              handed astronomy coordinate systems

    Args:
        img: a 2D image
        angle: angle CCW to rotate by (degrees)
        center: 2 element list [x,y] that defines the center to rotate the image to respect to
        new_center: 2 element list [x,y] that defines the new image center after rotation
        flipx: default is True, which reverses x axis.
        astr_hdr: wcs astrometry header for the image
    Returns:
        resampled_img: new 2D image
    """
    #convert angle to radians
    angle_rad = np.radians(angle)

    #create the coordinate system of the image to manipulate for the transform
    dims = img.shape
    x, y = np.meshgrid(np.arange(dims[1], dtype=np.float32), np.arange(dims[0], dtype=np.float32))

    #if necessary, move coordinates to new center
    if new_center is not None:
        dx = new_center[0] - center[0]
        dy = new_center[1] - center[1]
        x -= dx
        y -= dy

    #flip x if needed to get East left of North
    if flipx is True:
        x = center[0] - (x - center[0])

    #do rotation. CW rotation formula to get a CCW of the image
    xp = (x-center[0])*np.cos(angle_rad) + (y-center[1])*np.sin(angle_rad) + center[0]
    yp = -(x-center[0])*np.sin(angle_rad) + (y-center[1])*np.cos(angle_rad) + center[1]


    #resample image based on new coordinates
    #scipy uses y,x convention when meshgrid uses x,y
    #stupid scipy functions can't work with masked arrays (NANs)
    #and trying to use interp2d with sparse arrays is way to slow
    #hack my way out of this by picking a really small value for NANs and try to detect them after the interpolation
    #then redo the transformation setting NaN to zero to reduce interpolation effects, but using the mask we derived
    minval = np.min([np.nanmin(img), 0.0])
    nanpix = np.where(np.isnan(img))
    medval = np.median(img[np.where(~np.isnan(img))])
    img_copy = np.copy(img)
    img_copy[nanpix] = minval * 5.0
    resampled_img_mask = ndimage.map_coordinates(img_copy, [yp, xp], cval=minval * 5.0)
    img_copy[nanpix] = medval
    resampled_img = ndimage.map_coordinates(img_copy, [yp, xp], cval=np.nan)
    resampled_img[np.where(resampled_img_mask < minval)] = np.nan

    #edit the astrometry header if given to compensate for orientation
    if astr_hdr is not None:
        _rotate_wcs_hdr(astr_hdr, angle, flipx=flipx)

    return resampled_img


def _rotate_wcs_hdr(wcs_header, rot_angle, flipx=False, flipy=False):
    """
    Modifies the wcs header when rotating/flipping an image.

    Args:
        wcs_header: wcs astrometry header
        rot_angle: in degrees CCW, the specified rotation desired
        flipx: after the rotation, reverse x axis? Yes if True
        flipy: after the rotation, reverse y axis? Yes if True
    """
    # rotate WCS header by a rotation matrix
    rot_angle_rad = np.radians(rot_angle)
    cos_rot = np.cos(rot_angle_rad)
    sin_rot = np.sin(rot_angle_rad)
    rot_matrix = np.array([[cos_rot, sin_rot], [-sin_rot, cos_rot]])
    wcs_header.wcs.cd = np.dot(wcs_header.wcs.cd, rot_matrix)

    # flip RA if true to be North up East left
    if flipx is True:
        wcs_header.wcs.cd[:,0] *= -1
    if flipy is True:
        wcs_header.wcs.cd[:,1] *= -1


def meas_contrast(dat, iwa, owa, resolution, center=None, low_pass_filter=True):
    """
    Measures the contrast in the image. Image must already be in contrast units and should be corrected for algorithm
    thoughput.

    Args:
        dat: 2D image - already flux calibrated
        iwa: inner working angle
        owa: outer working angle
        resolution: size of resolution element in pixels (FWHM or lambda/D)
        center: location of star (x,y). If None, defaults the image size // 2.
        low_pass_filter: if True, run a low pass filter.
                         Can also be a float which specifices the width of the Gaussian filter (sigma).
                         If False, no Gaussian filter is run

    Returns:
        (seps, contrast): tuple of separations in pixels and corresponding 5 sigma FPF

    """

    if center is None:
        starx = dat.shape[1]//2
        stary = dat.shape[0]//2
    else:
        starx, stary = center

    # figure out how finely to sample the radial profile
    dr = resolution/2.0
    numseps = int((owa-iwa)/dr)
    # don't want to start right at the edge of the occulting mask
    # but also want to well sample the contrast curve so go at twice the resolution
    seps = np.arange(numseps) * dr + iwa + resolution/2.0
    dsep = resolution
    # find equivalent Gaussian PSF for this resolution


    # run a low pass filter on the data, check if input is boolean or a number
    if not isinstance(low_pass_filter, bool):
        # manually passed in low pass filter size
        sigma = low_pass_filter
        filtered = nan_gaussian_filter(dat, sigma)
    elif low_pass_filter:
        # set low pass filter size to be same as resolution element
        sigma = dsep / 2.355  # assume resolution element size corresponds to FWHM
        filtered = nan_gaussian_filter(dat, sigma)
    else:
        # no filtering
        filtered = dat

    contrast = []
    # create a coordinate grid
    x,y = np.meshgrid(np.arange(float(dat.shape[1])), np.arange(float(dat.shape[0])))
    r = np.sqrt((x-starx)**2 + (y-stary)**2)
    theta = np.arctan2(y-stary, x-starx) % 2*np.pi
    for sep in seps:
        # calculate noise in an annulus with width of the resolution element
        annulus = np.where((r < sep + resolution/2) & (r > sep - resolution/2))
        noise_mean = np.nanmean(filtered[annulus])
        noise_std = np.nanstd(filtered[annulus], ddof=1)
        # account for small sample statistics
        num_samples = int(np.floor(2*np.pi*sep/resolution))

        # find 5 sigma flux using student-t statistics
        # Correction based on Mawet et al. 2014
        fpf_flux = t.ppf(0.99999971334, num_samples-1, scale=noise_std) * np.sqrt(1 + 1./num_samples) + noise_mean
        contrast.append(fpf_flux)

    return seps, np.array(contrast)


def nan_gaussian_filter(img, sigma):
    """
    Gaussian low-pass filter that handles nans

    Args:
        img: 2-D image
        sigma: float specifiying width of Gaussian

    Returns:
        filtered: 2-D image that has been smoothed with a Gaussian

    """
    # make a copy to mask with nans
    masked = np.copy(img)
    nan_locs = np.where(np.isnan(img))
    masked[nan_locs] = 0

    # filter the image
    filtered = ndimage.gaussian_filter(masked, sigma=sigma, truncate=4)

    # because of NaNs, we need to renormalize the gaussian filter, since NaNs shouldn't contribute
    norm_dat = np.ones(filtered.shape)
    norm_dat[nan_locs] = 0
    filter_norm = ndimage.gaussian_filter(norm_dat, sigma=sigma, truncate=4)
    filtered /= filter_norm
    filtered[nan_locs] = np.nan

    # for some reason, the fitlered image peak pixel fluxes get decreased by 2
    filtered *= 2

    return filtered


def high_pass_filter(img, filtersize=10):
    """
    A FFT implmentation of high pass filter.

    Args:
        img: a 2D image
        filtersize: size in Fourier space of the size of the space. In image space, size=img_size/filtersize

    Returns:
        filtered: the filtered image
    """
    # mask NaNs
    nan_index = np.where(np.isnan(img))
    img[nan_index] = 0

    transform = fft.fft2(img)

    # coordinate system in FFT image
    u,v = np.meshgrid(fft.fftfreq(transform.shape[1]), fft.fftfreq(transform.shape[0]))
    # scale u,v so it has units of pixels in FFT space
    rho = np.sqrt((u*transform.shape[1])**2 + (v*transform.shape[0])**2)
    # scale rho up so that it has units of pixels in FFT space
    # rho *= transform.shape[0]
    # create the filter
    filt = 1. - np.exp(-(rho**2/filtersize**2))

    filtered = np.real(fft.ifft2(transform*filt))

    # restore NaNs
    filtered[nan_index] = np.nan
    img[nan_index] = np.nan

    return filtered


def define_annuli_bounds(annuli, IWA, OWA, annuli_spacing='constant'):
    """
    Defines the annuli boundaries radially

    Args:
        annuli: number of annuli
        IWA: inner working angle (pixels)
        OWA: outer working anglue (pixels)
        annuli_spacing: how to distribute the annuli radially. Currently three options. Constant (equally spaced),
                        log (logarithmical expansion with r), and linear (linearly expansion with r)

    Returns:
        rad_bounds: array of 2-element tuples that specify the beginning and end radius of that annulus

    """
    #calculate the annuli ranges
    if annuli_spacing.lower() == "constant":
        dr = float(OWA - IWA) / annuli
        rad_bounds = [(dr * rad + IWA, dr * (rad + 1) + IWA) for rad in range(annuli)]
    elif annuli_spacing.lower() == "log":
        # calculate normalization of log scaling
        unnormalized_log_scaling = np.log(np.arange(annuli) + 1) + 1
        log_coeff = float(OWA - IWA)/np.sum(unnormalized_log_scaling)
        # construct the radial spacing
        rad_bounds = []
        for i in range(annuli):
            # lower bound is either mask or end of previous annulus
            if i == 0:
                lower_bound = IWA
            else:
                lower_bound = rad_bounds[-1][1]
            upper_bound = lower_bound + log_coeff * unnormalized_log_scaling[i]
            rad_bounds.append((lower_bound, upper_bound))
    elif annuli_spacing.lower() == "linear":
        # scale linaer scaling to OWA-IWA
        linear_coeff = float(OWA - IWA)/np.sum(np.arange(annuli) + 1)
        rad_bounds = [(IWA + linear_coeff * rad, IWA + linear_coeff * (rad + 1)) for rad in range(annuli)]
    else:
        raise ValueError("annuli_spacing currently only supports 'constant', 'log', or 'linear'")

    # check to make sure the annuli are all greater than 1 pixel
    min_width = np.min(np.diff(rad_bounds, axis=1))
    if min_width < 1:
        raise ValueError("Too many annuli, some annuli are less than 1 pixel")

    return rad_bounds



