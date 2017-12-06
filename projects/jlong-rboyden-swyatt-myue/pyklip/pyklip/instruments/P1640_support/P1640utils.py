#!/usr/bin/env

import numpy as np
import os
import glob

# import pyklip.instruments.P1640_support.P1640cores
# import pyklip.instruments.P1640_support.P1640contrast

"""
Various useful functions specific to the P1640 data
"""

def set_zeros_to_nan(data):
    """
    PyKLIP expects values outside the detector to be set to nan.
    P1640 sets these (and also saturated pixels) to identically 0.
    Find all the zeros and convert them to nans
    Input:
        data: N x Npix x Npix datacube or appended set of datacubes
    Returns:
        nandata: data with nans instead of zeros
    """
    zeros = np.where(data==0)
    data[zeros] = np.nan
    return data



##############################################################
# Spots
##############################################################
def get_spot_files(fitsfile, spot_file_dir):
    """
    Search in spot_file_dir for the spot files associated with fitsfile
    Return spot files if they are found, otherwise return an empty list
    """
    spot_file_re = os.path.splitext(os.path.basename(fitsfile))[0]+"-spot[0-3].csv"
    spot_files = glob.glob(os.path.join(spot_file_dir, spot_file_re))
    return spot_files


##############################################################
# Bad Pixel Filtering
##############################################################
def find_bad_pix(img, median_img, std_img, thresh=3):
    """
    Find the bad pixels
    """
    return np.where(np.abs(img-median_img)/std_img > thresh)

def clean_bad_pixels(img, boxrad=2, thresh=3):
    """
    Clean the image of outlier pixels using a median filter.
    Input:
        img: 2-d array
        boxrad: 1/2 the fitler size (2*boxrad+1)
        thresh: threshold (in stdev) for deciding a hot pixel
    Returns:
        cleaned_img: 2_d array where hot pixels have been replaced by median values
    """
    imgrid = np.indices(img.shape)
    side=2*boxrad+1
  
    median_img = generic_filter(img, np.nanmedian, (side, side))
    
    # now get the standard deviation of the boxes
    std_img = make_std_map(img, boxrad)

    #std_img = set_zeros_to_nan(std_img)
    
    #bad_pix = np.where(np.abs(img-median_img)/std_img > thresh)
    bad_pix = find_bad_pix(img, median_img, std_img, thresh)
    
    # set hot pixels to median value
    img[bad_pix] = median_img[bad_pix]
    
    return img

def clean_bad_pixels_cube(cube, boxrad=2, thresh=10):
    """
    Clean the image of outlier pixels using a median filter.
    Input:
        cube: 3-D data cube
        boxrad: 1/2 the fitler size (2*boxrad+1)
        thresh: threshold (in stdev) for deciding a hot pixel
    """    
    
    cleaned_cube = np.array([clean_bad_pixels(i, boxrad, thresh) for i in cube])
    return cleaned_cube

##############################################################

##############################################################
# PSF centering
##############################################################
def centroid_image(orig_img):
    """
    Centroid an image - weighted sum of pixels
    Input:
        orig_img: 2D array
    Return:
        [y,x] floating-point coordinates of the centroid
    """
    img = orig_img.copy()
    ypix, xpix = np.indices(img.shape)
    tot = np.nansum(img)
    ycenter = np.dot(np.ravel(ypix), np.ravel(img))/tot
    xcenter = np.dot(np.ravel(xpix), np.ravel(img))/tot
    return ycenter, xcenter

    
def get_cube_xsection(orig_cube, center, width):
    """
    Select the cross-section of a cube centered in center with 1/2-width width
    Input:
        orig_cube: Nlambda x Npix x Npix datacube
        center: [row, col] index
        width: 1/2-width of cross-section
    Returns:
       cube_cutout: Nlambda x (2*width+1) x (2*width+1) cube cross-section
    """
    cube = orig_cube.copy()
    shape = cube.shape
    try:
        assert(np.all(np.where(center < 0)))
        assert(center[0] < shape[1])
        assert(center[1] < shape[2])
    except AssertionError:
        print("bad value for center")
        return None
    # [(xlow, xhigh),(ylow,yhigh)]
    center_lims = np.array([(np.max([0,center[0]-width]), np.min([shape[-1],center[0]+width])),
                            (np.max([0,center[1]-width]), np.min([shape[-1],center[1]+width]))]) 
    cube_cutout = cube[:, center_lims[0][0]:center_lims[0][1], center_lims[1][0]:center_lims[1][1]].copy()
    return cube_cutout

def get_PSF_center(cube, refchan=26, fine=False):
    """
    Return the PSF center at the pixel level (default) or subpixel level (fine=True)
    Input:
        cube: Nlambda x Npix x Npix datacube
        refchan(=26): Reference channel for the initial center estimate
        fine_centering(=False): After getting a rough estimate of the center, centroid the image
    Returns:
        Nlamdba x 2 array of pixel indices for the PSF center
    """
    core_cube = cube.copy()
    shape = core_cube.shape
    nchan = shape[0]
    refchan = 26
    init_center = np.array(np.unravel_index(np.nanargmax(core_cube[26]), shape[-2:]))
    width = 25
    center_cutout = get_cube_xsection(cube, init_center, width)
    try:
        centers = np.array([np.unravel_index(np.nanargmax(center_cutout[chan]), center_cutout[chan].shape)
                            for chan in range(nchan)] + init_center - width)
    except ValueError:
        print("All-NaN slice encountered:", chan)
        return None
    # If you want centroiding, do a second pass
    if fine == True:
        init_center = np.floor(np.nanmean(centers,axis=0)).astype(np.int)
        center_cutout = get_cube_xsection(cube, init_center, width)
        centers = np.array([centroid_image(img) for img in center_cutout]) + init_center - width
    centers = np.fmax(0, centers)
    return centers

#######################################################
# Encircled energy radius
#######################################################
def get_encircled_energy_image(im, center, frac=0.5):
    """
    Given an image, find the fraction of encircled energy around the center.
    Input:
        im: unocculted core cube Npix x Npix
        frac: encircled energy cutoff
    Returns:
        Pandas Series with the following indices:
        [starx, stary, radius, flux, bgnd_mean, bgnd_std, bgnd_npix]
    
    """
    shape = im.shape
    ind = ['starx','stary','radius','flux','bgnd_mean','bgnd_std', 'bgnd_npix']
    ee_data = pd.Series(np.zeros(len(ind)), index=ind)
    ee_data[['stary', 'starx']] = center
    
    # get star-centered coordinates
    yx_centered = np.array((np.indices(im.shape).T - center).T)
    # convert to radii
    rad_coord = np.linalg.norm(yx_centered, axis=0)
    
    #radpix = zip(np.ravel(rad_coord, core_cube))
    # find the encircled energy radius
    radpix = pd.DataFrame(zip(np.ravel(rad_coord), np.ravel(im)), 
                          columns=['rad', 'pixval'])
    # sort by pixel distance from PSF center
    radpix.sort(columns='rad', inplace=True)
    radpix.reset_index(inplace=True)
        
    # sort into radial rings (may not be azimuthally contiguous)
    unique_rad = np.unique(radpix['rad'])
    # group by common radius, god bless pandas for the time saving
    grouped = radpix.groupby('rad') 
    unique_flux = grouped.pixval.sum().as_matrix()
    pix_per_rad = grouped.size().as_matrix()
    
    unique_cumsum = np.cumsum(unique_flux)
    psf_edge_pix = np.nanargmax(unique_cumsum) + 1 # +1 is added for array indexing convenience
    psf_edge_rad = unique_rad[psf_edge_pix-1]
    
    # subtract background
    bgnd_mean = np.nanmean(unique_flux[psf_edge_pix:])
    bgnd_std = np.nanstd(radpix[radpix['rad'] > psf_edge_rad]['pixval'])
    unique_flux -= bgnd_mean*pix_per_rad

    # redo cumsum now with bgnd subtracted
    unique_cumsum = np.cumsum(unique_flux)
    total_flux = unique_cumsum[psf_edge_pix-1]
    encircled_energy = unique_cumsum[:psf_edge_pix]/total_flux
    # fill in the dataframe
    ee_data['bgnd_mean'] = bgnd_mean
    ee_data['bgnd_std'] = bgnd_std
    ee_data['bgnd_npix'] = grouped.size().ix[psf_edge_rad:].sum()
    ee_data['flux'] = total_flux
    # get the radius, taking care of the case of a crappy channel
    try:
        lolim = np.where(encircled_energy <= frac)[0][-1]
        hilim = np.where(encircled_energy > frac)[0][0]
    except IndexError:
        ee_data['radius'] = 1
        return ee_data
    encirc_interp = interpolate.interp1d(x=encircled_energy[[lolim, hilim]], 
                                         y=unique_rad[[lolim, hilim]])
    ee_data['radius'] = encirc_interp(frac)

    return ee_data

def get_encircled_energy_cube(cube, frac=0.5, refchan=26):
    """
    Get the fractional encircled energy of a PSF in each channel of a datacube. 
    Basically a wrapper for get_encircled_energy_image
    Input:
        core_cube: unocculted core cube Nlambda x Npix x Npix
        frac: encircled energy cutoff
    Returns:
        Pandas Dataframe with Nlambda columns:
        [starx, stary, radius, flux]
    """
    core_cube = cube.copy()
    nchan = core_cube.shape[0]
    centers = get_PSF_center(core_cube, fine=True)
    ee_data = pd.DataFrame([get_encircled_energy_image(im, center, frac)
                            for im, center in zip(core_cube, centers)])
    return ee_data


#################################################################
# Wrapper for easily converting astropy tables to FITS format
#################################################################
def table_to_TableHDU(table, kwargs={}):
    """
    Accept a table with a .colnames element and return it as  an astropy 
    fits.TableHDU object. Only works with floating-point data atm.
    Input:
        table: astropy.table.Table object
        kwargs: dict of keywords and arguments to pass to the HDU
    Returns:
        TableHDU: fits TableHDU with an empty header
    """
    coldef = fits.ColDefs([fits.Column(name=name,
                                       format='E',
                                       array=table[name])
                           for name in table.colnames])
    hdu = fits.TableHDU.from_columns(coldef, **kwargs)
    return hdu
