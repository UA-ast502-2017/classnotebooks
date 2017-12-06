#!/usr/bin/env python

"""
This library has method for operating on P1640 core images
"""

import sys
import os
import glob
import re

import numpy as np
import pandas as pd
from scipy import interpolate

from astropy.io import fits
from photutils import aperture_photometry, CircularAperture

sys.path.append("~/pyklip")
from pyklip.klip import align_and_scale

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
    xlims = (np.floor(np.nanmax([0,center[0]-width])).astype(np.int),
             np.ceil(np.nanmin([shape[-1],center[0]+width])).astype(np.int))
    ylims = (np.floor(np.nanmax([0,center[1]-width])).astype(np.int),
             np.ceil(np.nanmin([shape[-1],center[1]+width])).astype(np.int))
    center_lims = np.array([xlims, ylims])
    cube_cutout = cube[:, 
                       center_lims[0][0]:center_lims[0][1], 
                       center_lims[1][0]:center_lims[1][1]].copy()
    return cube_cutout


def centroid_image(orig_img):
    """
    Centroid an image - weighted sum of pixels
    Input:
        orig_img: 2D array
    Return:
        [y,x] floating-point coordinates of the centroid
    """
    img = orig_img.copy()
    # don't include negative pixels; they are not part of the PSF
    img[img<0] = 0.
    ypix, xpix = np.indices(img.shape)
    tot = np.nansum(img)
    ycenter = np.dot(np.ravel(ypix), np.ravel(img))/tot
    xcenter = np.dot(np.ravel(xpix), np.ravel(img))/tot
    return ycenter, xcenter

def get_PSF_center(orig_cube, refchan=26, fine=False):
    """
    Return the PSF center at the pixel level (default) or subpixel level (fine=True)
    Input:
        orig_cube: [Nlambda x] Npix x Npix datacube (or image)
        refchan(=26): Reference channel for the initial center estimate
        fine_centering(=False): After getting a rough estimate of the center, centroid the image
    Returns:
        Nlamdba x 2 array of pixel indices for the PSF center
    """
    cube = orig_cube.copy()
    orig_shape = cube.shape
    IMG_FLAG = True if len(cube.shape) == 2 else False
    if IMG_FLAG:
        cube = cube[np.newaxis,:,:]
    shape = cube.shape
    nchan = shape[0]
    if nchan < refchan:
        refchan = nchan-1
    init_center = np.array(np.unravel_index(np.nanargmax(cube[refchan]), shape[-2:]))
    width = 15
    center_cutout = get_cube_xsection(cube, init_center, width)
    try:
        centers = np.array([np.unravel_index(np.nanargmax(center_cutout[chan]), center_cutout[chan].shape)
                            for chan in range(nchan)] + init_center - width)
    except ValueError:
        print("All-NaN slice encountered:", chan)
        return None
    # If you want centroiding, do a second pass on a cross-section of the cube centered on the initial guess
    if fine == True:
        init_center = np.floor(np.nanmean(centers,axis=0)).astype(np.int)
        center_cutout = get_cube_xsection(cube, init_center, width)
        centers = np.array([centroid_image(img) for img in center_cutout]) + init_center - width
    # negative values are illegal, set to bottom corner
    centers = np.fmax(0, centers)
    
    if IMG_FLAG: # take care of dimensionality
        return centers[0]
    return centers


def _get_encircled_energy_image(im, center, frac=0.5):
    """
    Get the encircled energy for a 2-D image.
    Calculate the radial pixel coordinates centered on 'center'. 
    """
    yx_centered = np.array((np.indices(im.shape).T - center).T)
    # convert to radii
    rad_coord = np.linalg.norm(yx_centered, axis=0)

    radpix = pd.DataFrame(zip(np.ravel(rad_coord), np.ravel(im)), 
                          columns=['rad', 'pixval'])
    radpix.sort(columns='rad', inplace=True)
    radpix.reset_index(inplace=True)
    radpix['cumsum'] = radpix['pixval'].cumsum()
    unique_rad = np.unique(radpix['rad'])
    # group by common radius
    grouped = radpix.groupby('rad') 
    unique_flux = grouped.pixval.sum().as_matrix()
    pix_per_rad = grouped.size().as_matrix()

    unique_cumsum = np.cumsum(unique_flux)
    psf_edge_pix = np.nanargmax(unique_cumsum) + 1 # +1 is added for array indexing reasons
    psf_edge_rad = unique_rad[psf_edge_pix-1]
    bgnd_mean = np.nanmean(unique_flux[psf_edge_pix:])
    bgnd_std = np.nanstd(radpix[radpix['rad'] > psf_edge_rad]['pixval'])
    # subtract background
    unique_flux -= bgnd_mean*pix_per_rad
    # redo cumsum now with bgnd subtracted
    unique_cumsum = np.cumsum(unique_flux)
    total_flux = unique_cumsum[psf_edge_pix-1]
    encircled_energy = unique_cumsum[:psf_edge_pix]/total_flux

    ind = ['starx','stary','radius','flux','bgnd_mean','bgnd_std', 'bgnd_npix']
    ee_data = pd.Series(np.zeros(len(ind)), index=ind)
    ee_data[['stary','starx']] = center
    ee_data['flux'] = total_flux
    ee_data['bgnd_mean'] = bgnd_mean
    ee_data['bgnd_std'] = bgnd_std
    ee_data['bgnd_npix'] = grouped.size().ix[psf_edge_rad:].sum()

    # get the radius, taking care of the case of a crappy channel
    try:
        lolim = np.where(encircled_energy < frac)[0][-1]
        hilim = np.where(encircled_energy >= frac)[0][0]
    except IndexError:
        ee_data['radius'] = 1
        return ee_data
    encirc_interp = interpolate.interp1d(x=encircled_energy[[lolim, hilim]], 
                                         y=unique_rad[[lolim, hilim]])
    ee_data['radius'] = encirc_interp(frac)

    return ee_data


def get_encircled_energy_cube(orig_cube, frac=0.5):
    """
    Get the fractional encircled energy of a PSF in each channel of a datacube. 
    Basically a wrapper for _get_encircled_energy_image. Accepts 2-D and 3-D input.
    Input:
        orig_cube: unocculted core cube [Nlambda x ]Npix x Npix
        frac: encircled energy cutoff
    Returns:
        Pandas Dataframe with Nlambda columns:
        [starx, stary, radius, flux]
    """
    cube = orig_cube.copy()
    if len(cube.shape) == 2:
        cube = cube[np.newaxis,:,:]
    nchan = cube.shape[0]
    centers = get_PSF_center(cube, fine=True)
    ee_data = pd.DataFrame([_get_encircled_energy_image(im, center, frac)
                            for im, center in zip(cube, centers)])
    return ee_data


def _aperture_convolve_img(orig_img, aperture_radius, apkwargs={'method':'subpixel','subpixels':4}):
    """
    Perform aperture photometry on every pixel in a 2-D image
    Input:
        orig_img: Npix x Npix image
        aperture_radius: radius of the circular aperture
    Returns:
        phot_img: image of the aperture photometry on every non-nan pixel
    """
    phot_img = np.zeros(orig_img.shape)
    nanpix = np.where(np.isnan(orig_img))
    notnanpix = np.where(~np.isnan(orig_img))

    orig_shape = orig_img.shape
    img = orig_img.copy() # let's not modify the original cube
    img[nanpix] = 0.0
    
    # put an aperture on every non-nan pixel
    img_apertures = CircularAperture(positions = zip(*notnanpix[::-1]),
                                     r = aperture_radius)
    img_phot = aperture_photometry(img, img_apertures, **apkwargs)
    # set up to calculate the flattened indices
    ncols = np.ones_like(img_phot['ycenter'])*img.shape[-1]
    flattened_ind = lambda n, row, col: n*row+col
    ind = np.array(map(flattened_ind, ncols,
                       img_phot['ycenter'],
                       img_phot['xcenter']),
                   dtype=np.int)
    phot_img = np.ravel(phot_img)
    # map the photometry to the cube
    phot_img[ind] = img_phot['aperture_sum']
    phot_img = phot_img.reshape(orig_shape)
    # re-assign the nan pixels
    phot_img[nanpix] = np.nan
    
    return phot_img


def aperture_convolve_cube(orig_cube, aperture_radii, apkwargs={'method':'subpixel','subpixels':4}):
    """
    Perform apeture photometry on every pixel in a datacube or image
    Wrapper for _aperture_convolve_img to handle 2-D and 3-D data
    Input:
        orig_cube: [Nlambda x] Npix x Npix datacube
        aperture_radii: Nlambda array of aperture radii
        apkwargs: dictionary of arguments to pass to aperture_photometry
            Default: {'method':'subpixel','subpixels':4}
    Returns:
        phot_cube: cube with shape of orig_cube of the aperture photometry
    """
    cube = orig_cube.copy()
    orig_shape = cube.shape
    aperture_radii = np.array(aperture_radii) # because pandas indexing can cause problems
    # image or cube?
    IMG_FLAG = True if len(orig_shape) == 2 else False
    if len(orig_shape) == 2:
        cube = cube[np.newaxis,:,:]
    phot_cube = np.zeros(cube.shape)
    for i, im in enumerate(cube):
        phot_cube[i] = _aperture_convolve_img(im, aperture_radii[i], apkwargs)

    if IMG_FLAG:
        return phot_cube[0]
    return phot_cube


def combine_multiple_cores(multiple_core_info):
    """
    Combine the stellar flux and radii information from multiple cores in the proper way.
    """
    rad_mean = np.nanmean([c['radius'] for c in multiple_core_info], axis=0)
    flux_mean = np.nanmean([c['flux'] for c in multiple_core_info], axis=0)
    rad_err = np.nanstd([c['radius'] for c in multiple_core_info], axis=0)
    flux_err = np.nanstd([c['flux'] for c in multiple_core_info], axis=0)

    core_info = pd.DataFrame(zip(rad_mean, rad_err, flux_mean, flux_err),
                             columns=['radius','drad','flux','dflux'])
    return core_info

def zero_pad_core_box(core_cutout, centers, radii):
    """
    Get a cube of core cutouts with a center and a radius for each channel
    Inside/outside the radius is determined by the center of the pixel.
    Negative pixels are set to 0.
    """
    pass
    

def get_injection_core(core_cubes):
    """
    Remember, the injected PSF needs to be the SAME as the reference PSF, except for a scaling factor!
    Combine multiple cubes into a single core file for injection.
    Make sure that the total injected flux is the sum of the pixels!
    Outline:
    1. Get the encircled energy fraction for all the cores (frac=1)
    2. For each core, prepare a zero-cube with width of the largest radius + 1
    3. Add the core from each channel to the zero-cube
    
    """
    core_info_all = [get_encircled_energy_cube(c, frac=1) for c in core_cubes]
    core_cutouts = []
    for cube, core_df in zip(core_cubes, core_info_all):
        core_cutouts.append(get_core_xsection(cube,
                                        core_df[['stary','starx']].mean(), 
                                        core['radius'].max()))
        # set it so the cutout sum gives the total flux in each channel
        core_cutouts[-1][core_cutout<0] = 0
        core_cutouts[-1][np.where(np.isnan(core_cutout))] = 0

    core_cutouts = np.array(core_cutouts)
    final_psf = np.median(core_cutouts, axis=0)
    return final_psf

def make_median_core(core_cubes):
    """
    Take a set of core cubes and assemble a median cube out of them. Set all non-PSF pixels to 0
    Input:
        core_cubes: Ncubes x Nlambda x Npix x Npix set of cores
    Returns:
        median_core: Nlambda x Npix x Npix 
    """
    """
    Pseudocode:
    For each core:
        1. Get full radius for each channel in each core.
        2. Make an aligned cutout
        3. Mask all non-PSF pixels
        4. Combine the cutouts into an array of length core_cubes
    Median-combine the unmasked pixels
    """
    core_dfs = [get_encircled_energy_cube(cube, frac=1.) for cube in core_cubes]
    tmp_core_cutouts = [get_cube_xsection(core_cubes[i],
                                          df[['stary','starx']].mean(),
                                          df['radius'].max()+i)
                        for i,df in enumerate(core_dfs)]
    # make sure all the core cutouts have the same dimensions
    # if the radii are different, the difference in dims is 2 pixels
    max_shape = np.max([c.shape for c in tmp_core_cutouts], axis=0)
    core_cutouts = np.zeros((len(tmp_core_cutouts), max_shape[-3],
                             max_shape[-2], max_shape[-1]))*np.nan # make a nan-cube
    for i, cutout in enumerate(tmp_core_cutouts):
        shift = np.abs((cutout.shape[-2:] - max_shape[-2:])/2).astype(np.int)
        core_cutouts[i,:,shift[-2]:max_shape[-2]-shift[-2],shift[-1]:max_shape[-1]-shift[-1]] = cutout[:,:,:].copy()
    # align the cutouts so you can median-combine them
    # only need to align corresponding channels
    aligned_cutouts = core_cutouts.copy()
    dims = aligned_cutouts.shape
    aligned_cutouts = aligned_cutouts.reshape(reduce(lambda x,y:x*y,dims[:-2]), dims[-2], dims[-1])
    new_center = map(lambda x: np.floor(x/2.).astype(np.int), core_cutouts.shape[-2:])[::-1]
    old_centers = get_PSF_center(aligned_cutouts, fine=True)[:,::-1]
    for i in range(len(aligned_cutouts)):
        aligned_cutouts[i] = align_and_scale(aligned_cutouts[i], new_center, old_centers[i])
    aligned_cutouts = aligned_cutouts.reshape(dims)

    # before subtracting off the background, make a mask to ID background pixels
    bgnd_mask = np.ma.mask_or((aligned_cutouts < 0), np.isnan(aligned_cutouts))
    # before median-combining, subtract off the background, and:
    background = np.array([df['bgnd_mean'] for df in core_dfs])
    cutout_psfs = np.ma.masked_array([(psf.T - bg).T for psf,bg in zip(aligned_cutouts, background)],
                                     mask=bgnd_mask,
                                     fill_value=0.0)
    median_psf = np.ma.median(cutout_psfs, axis=0)
    # return an unmasked array
    return median_psf.data
    
