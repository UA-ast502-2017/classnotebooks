#!/usr/bin/env python

from __future__ import division

import sys
import os
import warnings
import glob

import numpy as np
from numpy import ma
from scipy import interpolate

from astropy.io import fits
from astropy import units
from astropy.modeling import models, fitting

try:
    from photutils import aperture_photometry, CircularAperture
except:
    print("P1640: photutils not available; spot photometry will fail.")

#for handling different python versions
if sys.version_info < (3,0):
    import ConfigParser
else:
    import configparser as ConfigParser

class P1640params:
    num_spots = 4
    refchan = 26
    nchan = 32 # can be overridden by cube
    channels = np.arange(32, dtype=np.int) # can be overridden by cube
    wlsol = np.linspace(969, 1797, 32)*1e-3 # used for estimating scaling
    aperture_refchan = 3.5 # aperture size in the reference channel
P1640params.reflambda = P1640params.wlsol[P1640params.refchan]
P1640params.scale_factors = P1640params.wlsol/P1640params.reflambda

##################################################
# Grid operations
##################################################
def get_centered_grid(img_shape, center):
    """
    Return a coordinate grid shifted to the center
    """
    grid = np.indices(img_shape)
    centered_grid = np.array([grid[0]-center[0], grid[1]-center[1]])
    return centered_grid

def get_rotated_grid(img_shape, center, angle):
    """
    Rotate a coordinate grid by some angle around a center
    """
    centered_grid = get_centered_grid(img_shape, center)
    rad_grid = np.linalg.norm(centered_grid, axis=0)
    

##################################################
# Masks
##################################################
def make_mask_circle(img_shape, center, R):
    """
    Make a circular mask, where everything inside a radius R around the center 
    is False and outside is True
    Input:
        img_shape: the shape of the image in (row, col)
        center: the center of the mask, in (row, col)
        R: the radius of the circle
    Returns:
        mask: a masked array of shape img_shape
    """
    mask = np.zeros(shape=img_shape, dtype=np.int)
    centered_grid = get_centered_grid(img_shape, center)
    rad_grid = np.linalg.norm(centered_grid, axis=0)
    mask[rad_grid <= R] = 1
    return ~ma.make_mask(mask)

def make_mask_donut(img_shape, center, R0, R1):
    """
    Make a donut mask centered on 'center' where the inside of the donut is 
    False and the outside of the donut is True
    """
    inner_mask = make_mask_circle(img_shape, center, R0)
    outer_mask = make_mask_circle(img_shape, center, R1)
    donut_mask = (outer_mask != inner_mask)
    return donut_mask

def make_mask_half_img(img_shape, center, angle):
    """
    Mask half the image, cutting it through the center at an arbitrary angle.
    Angle is measured *counterclockwise* from vertical, and should be an 
    astropy units object, otherwise assume degrees. 
    Input:
        img_shape: shape of image in (row, col)
        center: the center of the mask in (row, col)
        angle: angle measured counterclockwise from vertical, default in deg
    Output:
        mask: masked_array with a plane running through point (center) at angle
            (angle)
    """
    if type(angle) != units.quantity.Quantity:
        angle = angle*units.degree
    mask = np.zeros(shape=img_shape, dtype=np.int)
    centered_grid = get_centered_grid(img_shape, center)
    rad_grid = np.linalg.norm(centered_grid, axis=0)
    ang_grid = np.arctan2(centered_grid[0], centered_grid[1]) * units.radian
    rot_grid = ang_grid - angle
    #dx = centered_grid[1] + rad_grid * np.cos(rot_grid)
    dx = rad_grid * np.cos(rot_grid)
    mask[dx>=0] = 1
    return ~ma.make_mask(mask)


def make_mask_bar(img_shape, center, angle, width):
    """
    Make a bar mask where all the pixels inside a bar through the center of
    the image within some width are 1 and everything outside is 0
    Inputs:
        img_shape: the shape of the image in (row, col)
        center: the center of the mask, in (row, col)
        angle: angle measured counterclockwise from vertical, default in deg
        width: with of bar in pixels
    Returns:
        mask: a masked array where the values inside the bar are False and 
            outside the bar are True
    """
    if type(angle) != units.quantity.Quantity:
        angle = angle*units.degree
    dcenter = np.array([(width/2) * np.sin(angle),
                        (width/2) * np.cos(angle)])
    mask1 = make_mask_half_img(img_shape, center+dcenter, angle)
    mask2 = make_mask_half_img(img_shape, center-dcenter, angle)
   
    # now mask the half-plane you don't want
    pmask = make_mask_half_img(img_shape, center, angle-90*units.degree)
    
    mask = ((mask1 != mask2) & pmask) #np.abs(mask2 - mask1)*pmask
    
    return mask

def make_mask_grid_spots(img_shape, centers, rotated_spots=False, nchan=P1640params.nchan):
    """
    Make a mask that shows only the grid spots
    Input:
        img_shape: the shape of the image to mask in (row, col)
        centers: (Nchan x 2) array of centers of the mask in (row, col)
        rotated_spots: [False] make mask for normal (False) or rotated (True) grid spots
        nchan: number of spectral channels in the cube
    Returns:
        masks: Nspot x Nchan cube of masks
    """
    if rotated_spots is True:
        angles = units.Quantity([21, 108, 194, 290], unit=units.degree)
        center_region = np.linspace(45, 85, 32) # centers mask regions

    else: # default values
        angles = units.Quantity([-25, 63, 155, 243], unit=units.degree)
        center_region = np.linspace(65, 110, 32) # centers mask regions
    ang_width = 40
    region_width = 20*P1640params.scale_factors
    inner_radii = center_region - region_width
    outer_radii = center_region + region_width
    
    donuts = [make_mask_donut(img_shape, i[0], i[1], i[2])
              for i in zip(centers, inner_radii, outer_radii)]
    #donut = make_mask_donut(img_shape, center, inner_radius, outer_radius)

    bars = [[make_mask_bar(img_shape, center, ang, ang_width) for center in centers]
            for ang in angles]

    masks = [[~(donuts[j] & bar[j]) for j in range(nchan)] for i,bar in enumerate(bars)]
    
    return masks


def make_mask_refined_grid_spots(img_shape, centers, spots, nchan=32):
    """
    Make a new set of masks that are centered on the interpolated 
    grid spot locations
    Input:
        img_shape: x- and y-dimensions of image
        centers: nchan x 2 array of star positions
        spots: num_spots  x nchan x 2 array of spot positions
    """
    drad = 10 # pixel half-width in radius
    dtheta = 20 # pixel full-width in theta
    masks = [[] for i in spots]
    for i,spot in enumerate(spots):
        for chan in range(nchan):
        # get spot radius
            rad = np.linalg.norm((spot[chan] - centers[chan]))
            theta = np.arctan2(spot[chan][0]-centers[chan][0],
                               spot[chan][1]-centers[chan][1])
            donut_mask = make_mask_donut(img_shape, centers[chan], 
                                         rad-drad, rad+drad)
            bar_mask = make_mask_bar(img_shape, centers[chan],
                                     theta*180/np.pi, dtheta)
            mask = ~(donut_mask & bar_mask)
            masks[i].append(mask)
    return masks
            

##################################################
# Fitting
##################################################
def guess_grid_spot_loc(img):
    """
    get max pixel as initial guess of location
    """
    spot_pos = np.unravel_index(np.argmax(img), dims=img.shape)
    return spot_pos

def fit_grid_spot(img, center, loc=None):
    """
    Fit spot with a 2-D Gaussian
    Inputs:
        img: 2-D masked_array to fit
        center: center of the image in (row, col) order
        loc (optional): initial position guess in (row, col) order
    """
    cleaned_img = (~img.mask)*img.data # image that is 0 outside mask
    if loc is None:
        loc = guess_grid_spot_loc(cleaned_img)

    x_stddev = 3#img[:,loc[1]].std()
    y_stddev = 3#img[loc[0],:].std()
    init_theta = np.arctan2(loc[0]-center[0], loc[1]-center[1])
    g_init = models.Gaussian2D(amplitude=img.max(), 
                               x_mean=loc[1], y_mean=loc[0],
                               x_stddev=x_stddev, y_stddev=y_stddev,
                               theta = init_theta)
    g_init.x_stddev.fixed = True
    g_init.y_stddev.fixed = True


    fit = fitting.LevMarLSQFitter()
    y,x = np.indices(img.shape)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        g = fit(g_init, x, y, img)

    return g, np.array([g.y_mean.value, g.x_mean.value])


def fit_poly(ind, dep, order):
    """
    Takes in an array of positions in row, col format and returns a polynomial
    that fits them to col = b + C*row, where C is a vector of coefficients
    Fitting is done by least squares
    Input:
        ind: dependent variable (prob channel number)
        dep: independent variable (prob x or y spot position)
    Returns:
        array of polynomial coefficients
    """
    ind = np.array(ind)
    dep = np.array(dep)
    x = np.array([ind**i for i in range(order+1)]).T
    y = dep
    a = np.dot(x.T, x)
    b = np.dot(x.T, y)
    coeffs = np.dot(np.linalg.inv(a), b)
    return coeffs

def get_points_from_poly(ind, coeffs):
    """
    Get a polynomials coefficients and return the y-values given the independent values
    """
    ind = np.array(ind)
    x = np.array([ind**i for i in range(coeffs.size)]).T
    y = np.dot(x, coeffs)
    return y


def check_bad_spots(spot, centers):
    """
    Input:
        spot: y,x positions for a spot
        centers: y,x positions for the center
    Output: 
        fixed_spot: a fixed spot position y,x
    """
    fixed_spot = spot.copy()
    centered_spot = spot-centers
    rad_spot = np.linalg.norm(centered_spot, axis=1)
    bad_spots = np.where(rad_spot[1:]<rad_spot[:-1])[0]+1
    for bad in bad_spots:
        good_channels = np.concatenate((P1640params.channels[:bad-1], 
                                        P1640params.channels[bad+1:]))
        # fit them with a chosen polynomial (cubic for now)
        good_rad = np.linalg.norm(centered_spot[good_channels], axis=1)
        good_ang = np.arctan2(centered_spot[good_channels][:,0],
                              centered_spot[good_channels][:,1]).mean()
        rad_line = fit_poly(good_channels, good_rad, 3)
        new_rad = get_points_from_poly(P1640params.channels, rad_line)
        newyx = (new_rad*np.sin(good_ang), new_rad*np.cos(good_ang)) + centers.T
        residuals = abs(rad_spot - new_rad)
        worse = bad if residuals[bad] > residuals[bad-1] else bad-1
        fixed_spot[worse,0] = newyx[0][worse]
        fixed_spot[worse,1] = newyx[1][worse]   
    return fixed_spot

def check_bad_channels(rad_spot):
    """
    Check that the spot positions increase monotonically in radius.
    If they don't, return the positions that do not follow monotonically.
    Input:
        rad_spot: Nchan array of radial sep of a spot from star 
    Output: 
        bad_chans: a list of channel pairs that fail the check and need fixing
    """
    try:
        assert rad_spot.ndim==1
    except AssertionError:
        print("check_bad_channels received array with wrong dimensions, exiting")
        sys.exit()
    # get bad channels, in ambiguous pairs
    bad_chans0 = np.where(rad_spot[1:] < rad_spot[:-1])[0]
    bad_chans1 = bad_chans0+1
    # convert to list and sort
    bad_chans = list(zip(bad_chans0, bad_chans1))
    temp3 = list(bad_chans)
    return bad_chans

def fix_bad_channels(spot, centers, bad_chans):
    """
    Two cases: 
        1) a spot jumps inwards
        2) a spot jumps outwards
    In either case, remove both the failing spot and the one before it
    and fit a cubic to the remaining points. Then, fix the spot that is
    further from the fit.
    Input: 
       spot: Nchan x 2 array of positions for one spot
       centers: y,x positions for the star in each channel
       bad_chan: index of a spot that does not monotonically increase in radius
    Output:
       fixed_spot: Nchan x 2 array of fixed positions for the spot
    """
    fixed_spot = np.copy(spot)
    centered_spot = spot - centers
    rad_spot = np.linalg.norm(centered_spot, axis=1)
    channels = range(spot.shape[0])

    # remove all bad channels from fitting and interpolation channels
    good_channels = list(channels[:])
    try:
        for i in np.unique(np.ravel(bad_chans)): good_channels.remove(i)
    except ValueError:
        pass
    '''
    # separate procedures for middle channel vs end channels
    # check if 1 is a bad channel (the lowest possible bad channel)
    if 1 in np.ravel(bad_chans): 
        # use the four lowest good channels to fit a quadratic
        fit_channels = good_channels[:5]
        #good_channels = channels[bad_chan+1:bad_chan+5]
        fit_rad = rad_spot[fit_channels] 
        rad_line = fit_poly(fit_channels, fit_rad, 2)
        new_rad = get_points_from_poly(channels, rad_line)
    # check if last channel is a bad channel
    elif channels[-1] in np.ravel(bad_chans): 
        # use the four highest good channels to fit a quadratic
        fit_channels = good_channels[-4:]
        fit_rad = rad_spot[fit_channels]
        rad_line = fit_poly(fit_channels, fit_rad, 2)
        new_rad = get_points_from_poly(channels, rad_line)
    # if bad channel is in the middle, we can interpolate 
    else:  # interpolation NOPE fit a cubic to all the good points
        # remove two channels from the good set
        good_rad = rad_spot[good_channels]
        rad_line = fit_poly(good_channels, good_rad, 3)
        new_rad = get_points_from_poly(channels, rad_line)
        #cubeinterp = interpolate.interp1d(good_channels, good_rad, kind='cubic')
        # splice in the interpolated values
        #new_rad = good_rad.tolist()
        #for i in np.ravel(bad_chans):
        #    new_rad.insert(i, cubeinterp(i))
        #new_rad = np.array(new_rad)
    '''
    good_rad = rad_spot[good_channels]
    rad_line = fit_poly(good_channels, good_rad, 3)
    new_rad = get_points_from_poly(P1640params.channels, rad_line)
    # to get x and y, just use the average angle
    good_ang = np.arctan2(centered_spot[good_channels][:,0],
                          centered_spot[good_channels][:,1]).mean()
    newyx = (new_rad*np.sin(good_ang), new_rad*np.cos(good_ang)) + centers.T

    # test bad channel pairs against fit
    residuals = abs(rad_spot - new_rad)
    for bad_chan in bad_chans:
        worse = bad_chan[0] if residuals[bad_chan[0]] > residuals[bad_chan[1]] else bad_chan[1]
        fixed_spot[worse,0] = newyx[0][worse]
        fixed_spot[worse,1] = newyx[1][worse]   

    return fixed_spot

##################################################
# Write spots to memory or disk
##################################################

def write_spots_to_header(spots, fitsfile):
    """
    Write the spot positions to a fits header
    Input:
        spots: 4 x Nchan x 2 array
        fitsfile: full path to a fits file whose header you want to modify
    """
    hdulist = fits.open(fitsfile)
    hdu = hdulist[0]
    header = hdu.header
    header['Spot0x'] = spots[0,:,1]
    header['Spot0y'] = spots[0,:,0]
    header['Spot1x'] = spots[1,:,1]
    header['Spot1y'] = spots[1,:,0]
    header['Spot2x'] = spots[2,:,1]
    header['Spot2y'] = spots[2,:,0]
    header['Spot3x'] = spots[3,:,1]
    header['Spot3y'] = spots[3,:,0]
    return hdu

def write_spots_to_file(data_filepath, spot_positions, output_dir=None,
                        spotid=None, ext=None, overwrite=True):
    """
    Write one file for each spot to the directory defined at the top of
    this file. Output file name is data_filename -fits +spoti.csv.
    Format is (row, col). Will overwrite existing files.
    Input:
        data_filename: the base name of the file with the spots
        spot_positions: Nspot x Nchan x 2 array of spot positions
        output_dir: directory to write the output files
        overwrite: (True) overwrite existing spot files
        spotid: (-spoti) identifier for the 4 different spot files
        ext: (csv) file extension
    Returns:
        None
        writes a file to the output dir whose name corresponds to the cube 
        used to generate the spots + spotidN.ext (N is 0-3)
    """
    data_filename = os.path.basename(data_filepath)
    exists = glob.glob(os.path.join(output_dir,data_filename)+"*")

    # If you shouldn't overwrite existing files, quit here
    if (exists) and (not overwrite):
        print("Spot files exist and overwrite is False, skipping...")
        return

    # if output_dir, spotid, and ext are NOT specified, used P1640.ini as defaults
    if np.any([i is None for i in [output_dir, spotid, ext]]):
        # default config file
        config = ConfigParser.ConfigParser()
        config.read("/data/home/jaguilar/pyklip/pyklip/instruments/P1640.ini")
        if output_dir is None:
            output_dir = config.get("spots", "spot_file_path")
            print("Using value in P1640.ini for spot output directory: " + output_dir)
        if spotid is None:
            spotid = config.get("spots", "spot_file_postfix")
            print("Using value in P1640.ini for spot file ID: " + spotid)
        if ext is None:
            ext = config.get("spots", "spot_file_ext")
            print("Using value in P1640.ini for spot file ext: " + ext)
    
    try:
        for i, spot in enumerate(spot_positions):
            data_filename = os.path.basename(data_filepath)
            spotnum = spotid+'{0}'.format(i)
            output_filename = os.path.splitext(data_filename)[0]+spotnum+'.'+ext
            output_filepath = os.path.join(output_dir, output_filename)
            np.savetxt(output_filepath, spot, delimiter=",",
                           header='row,column')
            print(os.path.basename(output_filepath) + " written")
    except:
        # implement error handling later?
        pass
    return
        

##################################################
# Complete spot extraction
##################################################

def get_initial_spot_guesses(cube, rotated_spots=False):
    """
    """
    nchan = cube.shape[0]
    channels = np.arange(nchan, dtype=np.int)
    img_shape = cube.shape[1:]
    init_centers = img_shape*np.ones((nchan, 2))/2

    spot_masks = make_mask_grid_spots(img_shape, init_centers, rotated_spots=rotated_spots)

    spot_locs = np.zeros((P1640params.num_spots, nchan, 2))
    masked_cubes =  ma.masked_array([ma.masked_array(cube, mask=i) 
                                     for i in spot_masks])
   # get reference channel grid spots:
    for i in range(P1640params.num_spots):
        img = masked_cubes[i, P1640params.refchan]
        g, spot_locs[i,P1640params.refchan] = fit_grid_spot(img, init_centers[P1640params.refchan])

    # get initial guesses for the rest
    centered_spots = spot_locs[:,P1640params.refchan] - init_centers[P1640params.refchan]
    init_rad = np.linalg.norm(centered_spots, axis=1)
    init_theta = np.arctan2(centered_spots[:,0], centered_spots[:,1])
    init_spots = np.rollaxis(np.rollaxis(
        np.array([np.outer(init_rad, P1640params.scale_factors).T*np.sin(init_theta),
                  np.outer(init_rad, P1640params.scale_factors).T*np.cos(init_theta)]),
        2, 0), 2, 1) + init_centers

    return init_spots

def fit_grid_spots(masked_cubes, centers, spots_guesses):
    """
    Wrapper for fit_grid_spot to loop over all four spots
    Input:
        masked_cubes: Nspot x Nchan x Npix x Npix array of masks
        centers: Nchan x 2 array of (row, col) guesses for spot centers
        spot_guesses: Nspot x Nchan x 2 (row, col) guesses for spot locations
    Output:
        spot_fits: List of astropy.model fits
        spot_locs: Nspot x Nchan x 2 array of spot locations from fitting
    """
    nchan = masked_cubes.shape[1]
    spot_locs = np.zeros((P1640params.num_spots, nchan, 2))
    spot_fits = [[] for i in range(P1640params.num_spots)]
    for i in range(P1640params.num_spots):
        images = masked_cubes[i] 
        for chan in range(nchan):
            g, spot_locs[i,chan] = fit_grid_spot(images[chan], centers[chan], 
                                                     loc=spots_guesses[i,chan])
            spot_fits[i].append(g)
    return spot_fits, spot_locs



#################################
# Single-cube methods
#################################

def get_single_cube_spot_positions(cube, rotated_spots=False):
    """
    Return the spot positions for a single cube
    Input:
        cube: a data cube from P1640
        rotated_spots: (False) if True, use the rotated masks
    Output:
        spot_array: Nspots x Nchan x 2 array of spot positions. 
    """
    #################################
    # some unavoidable initializations
    nchan = cube.shape[0]
    channels = np.arange(nchan, dtype=np.int)
    img_shape = cube.shape[1:]
    init_centers = img_shape*np.ones((nchan, 2))/2
    spot_masks = make_mask_grid_spots(img_shape, init_centers, rotated_spots=rotated_spots)
    masked_cubes =  ma.masked_array([ma.masked_array(cube, mask=i) 
                                     for i in spot_masks])
    #################################

    #################################
    # Initial pass
    init_spots = get_initial_spot_guesses(cube, rotated_spots)

    # now, fit rest of spots using initial guesses
    spot_fits, spot_locs = fit_grid_spots(masked_cubes, init_centers, init_spots)
    #################################
    # Calculate centers
    # At each channel, fit lines through opposing spots
    centers = get_single_cube_star_positions(spot_locs)
    # Fix 'bad' spots:
    # Fit the radial separation and get x and y from that
    fixed_spot_locs = np.copy(spot_locs)
    centered_spots = fixed_spot_locs - centers
    for i in range(P1640params.num_spots):
        rad_spots = np.linalg.norm(fixed_spot_locs[i] - centers,
                                   axis=-1)
        bad_channels = check_bad_channels(rad_spots)
        while len(list(bad_channels)) != 0:
            fixed_spot_locs[i] = fix_bad_channels(fixed_spot_locs[i],
                                                  centers, bad_channels)
            # update radial spot distances to check they've all been corrected
            rad_spots = np.linalg.norm(fixed_spot_locs[i] - centers, 
                                       axis=-1)
            bad_channels = check_bad_channels(rad_spots)

    # update centers using new positions
    centers = get_single_cube_star_positions(fixed_spot_locs)

    #################################
    # Second pass, with new masks
    refined_masks = make_mask_refined_grid_spots(img_shape, centers, 
                                                 fixed_spot_locs)
    refined_masked_cubes =  ma.masked_array([ma.masked_array(cube, mask=i) 
                                             for i in refined_masks])
    # now, fit spots and again check for bad spots
    refined_spot_locs = np.zeros(spot_locs.shape)
    for i in range(P1640params.num_spots):
        images = refined_masked_cubes[i]
        for chan in range(nchan):
            g, refined_spot_locs[i, chan] = fit_grid_spot(images[chan], 
                                                          centers[chan], 
                                                          loc=fixed_spot_locs[i,chan])
            spot_fits[i][chan] = g
    # check bad spots again
    for i in range(P1640params.num_spots):
        rad_spots = np.linalg.norm(refined_spot_locs[i] - centers, 
                                   axis=-1)
        bad_channels = check_bad_channels(rad_spots)
        while len(list(bad_channels)) != 0:
            refined_spot_locs[i] = fix_bad_channels(refined_spot_locs[i],
                                                    centers, bad_channels)
            # update radial spot distances to check they've all been corrected
            rad_spots = np.linalg.norm(refined_spot_locs[i] - centers,
                                       axis=-1)
            bad_channels = check_bad_channels(rad_spots)
    '''
    for i in range(P1640params.num_spots):
        while np.any(np.where(rad_spots[i,1:]<rad_spots[i,:-1])[0]+1):
            refined_spot_locs[i] = check_bad_spots(spot_locs[i], centers)
            # update radial spot distances to check they've all been corrected
            centered_spots[i] = refined_spot_locs[i] - centers
            rad_spots[i] = np.linalg.norm(centered_spots[i], axis=-1)
    '''

    return refined_spot_locs

def get_single_file_spot_positions(fitsfile, rotated_spots=False):
    """
    Wrapper for get_single_cube_spot_positions
    """
    hdulist = fits.open(fitsfile)
    cube = hdulist[0].data
    spot_positions = get_single_cube_spot_positions(cube, rotated_spots)
    hdulist.close()
    return spot_positions


#################################
# Operations on spot arrays
#################################

def get_single_cube_star_positions(spot_array):
    """
    Using the spot positions for a single cube, find the star position at each wavelength.
    Input:
        spot_array: Nspots x Nchan x 2 array of (row, column) spot positions
    Output:
        star_array: Nchan x 2 array of (row, column) star positions
    """
    nspots  = P1640params.num_spots # always 4, because physics 
    nchan = spot_array.shape[-2]

    star_position = np.zeros((nchan, 2))

    for chan in range(nchan):
        line20 = fit_poly(np.array((spot_array[0,chan,1], 
                                    spot_array[2,chan,1])),
                          np.array((spot_array[0,chan,0], 
                                    spot_array[2,chan,0])),
                          order=1)
        line31 = fit_poly(np.array((spot_array[1,chan,1], 
                                    spot_array[3,chan,1])),
                          np.array((spot_array[1,chan,0], 
                                    spot_array[3,chan,0])),
                          order=1)
        x_star = (line20[0]-line31[0])/(line31[1]-line20[1])
        y_star = (line20[1]*line31[0]-line31[1]*line20[0])/(line20[1]-line31[1])
        star_position[chan] = (y_star, x_star)
    
    return star_position
   
def get_single_cube_scaling_factors(spot_array, star_array=None):
    """
    Get the scaling factors for a single cube
    Input:
        spot_array: Nspots x Nchan x 2 array of (row, column) spot positions
        star_array: Nchan x 2 array of (row, column) star positions.
            If not supplied, spot_array will be calculated from spot_array
    Output:
        scaling: Nspots x Nchan array of scaling factors, normalized to
            P1640params.refchan
    """
    if star_array is None:
        star_array = get_single_cube_star_positions(spot_array)
    centered_spots = spot_array - star_array
    rad_spots = np.linalg.norm(centered_spots, axis=-1)
    scaling = rad_spots / rad_spots[:, P1640params.refchan][:,None] 
    #scaling = rad_spots[:, P1640params.refchan][:,None] / rad_spots
    return scaling

def get_single_file_scaling_and_centering(fitsfile):
    """
    Take a single fits file, and return the star positions and 
    scaling factors
    See also: get_scalign_and_centering_from_spots
    Input:
        fitsfile: a single fits file with a P1640 cube
    Output:
        scaling_factors: scaling factors for each slice of the cube
        star_positions: star positions in each slice of the cube
    """
    hdulist = fits.open(fitsfile)
    cube = hdulist[0].data
    spot_positions = get_single_cube_spot_positions(cube)
    star_positions = get_single_cube_star_positions(spot_positions)
    scaling_factors = get_single_cube_scaling_factors(spot_positions, star_positions)
    hdulist.close()
    return scaling_factors, star_positions



##################################
# Spot photometry
##################################

def get_single_cube_spot_photometry(cube, spot_positions):
    """
    Do aperture photometry on the spots. Will need to be careful about
    aperture size for future comparison
    Input:
        cube: Nchan x Npix x Npix data cube to do photometry
        spot_positions: Nspot x Nchan x 2 spot positions for apertures
        scaling_factors: Nchan array for scaling apertures with wavelength
    Output:
        spot_phot: Nspot x Nchan array of spot photometry and spot errors
    """
    nchan = cube.shape[0]
    apsize = P1640params.aperture_refchan
    photometry = []
    # need scaling factors to scale aperture
    scaling_factors = get_single_cube_scaling_factors(spot_positions).mean(axis=0)
    for chan in range(nchan):
        positions = [i[::-1] for i in spot_positions[:,chan,:]] # need to reverse row, col
        apertures = CircularAperture(positions = positions,
                                     r = apsize*scaling_factors[chan])
        photometry.append(aperture_photometry(cube[chan], apertures))
    spot_phot = np.array([i['aperture_sum'] for i in photometry]).T
    return spot_phot

def get_single_cube_spot_positions_and_photometry(cube):
    """
    Wrapper that combines get_single_cube_spot_positions and
    get_single_cube_spot_photometry
    Input:
        cube: a datacube in P1640 format
    Output:
        spot_positions: Nspots x Nchan x 2 array of spot positions. 
        spot_photometry: Nspots x Nchan x 1 array of spot fluxes
    """
    spot_positions = get_single_cube_spot_positions(cube)
    spot_photometry = get_single_cube_spot_photometry(cube, spot_positions)
    return spot_positions, spot_photometry


#################################
# Multi-cube wrappers
#################################

def get_spot_positions(fitsfiles):
    """
    Return the spot positions for a set of data cubes. Really just a wrapper
    for get_single_cube_spot_positions
    Input:
        fitsfiles: a list of P1640 fits files
    Output:
        spot_array: Nfile x 4 x Nchan x 2 array of spot positions
    """
    # support providing a single file
    if isinstance(fitsfiles, str):
        fitsfiles = [fitsfiles]

    spot_array  = np.array([get_single_cube_spot_positions(fits.getdata(f))
                           for f in fitsfiles])
    return spot_array

def get_star_positions(spot_array):
    """
    Get the center of a set of 4 spots for a single cube
    Input:
        spot_locations: Nspot x Nlambda x 2 array of [row, col] spot positions
    Returns:
        star_array: Nlambda x 2 array of [row, col] star positions
    """
    num_spots=4
    
    # find the pairs of opposite spots
    # sorry for the confusing python shorthand
    spot_list = list(range(num_spots))
    pairs=[]
    pair1 = 0
    pairs.append([pair1, np.linalg.norm(spot_array[pair1] - spot_array, axis=-1).mean(axis=-1).argmax()])
    for i in sorted(pairs[0])[::-1]:
        spot_list.pop(i)
    pair2 = spot_list.pop(0)
    pairs.append([pair2, np.linalg.norm(spot_array[pair2] - spot_array, axis=-1).mean(axis=-1).argmax()])

    # Re-order the array so that channels are on the first axis
    if spot_array.shape[-2] != num_spots:
        spotaxid = np.where(np.array(spot_array.shape) == num_spots)[0][0]
        spot_array = np.rollaxis(spot_array, spotaxid,-1)
    orig_shape = spot_array.shape
    orig_ndim = spot_array.ndim
    if orig_ndim == 4: # more than one cube - 
        # collapse channels since we'll be processing the slices independently
        spot_array = spot_array.reshape((orig_shape[-4]*orig_shape[-3], 
                                         orig_shape[-2], orig_shape[-1]))
    
    # each element of spot_array is a set of four co-wavelength spots to be centered
    nslices = len(spot_array)
    star_positions = np.zeros((len(spot_array), 2))
    for chan in range(nslices):
        b20,m20 = fit_poly(spot_array[chan,pairs[0],1], spot_array[chan,pairs[0],0], order=1)
        b31,m31 = fit_poly(spot_array[chan,pairs[1],1], spot_array[chan,pairs[1],0], order=1)
        x_star = (b31-b20)/(m20-m31)  
        y_star = (m31*b20-m20*b31)/(m31-m20)
        star_positions[chan] = (y_star, x_star)
    return star_positions
    # for the final shape, separate out the cubes again
    new_shape = list(orig_shape)
    new_shape.pop(-2) # remove the Nspots index
    star_positions = star_positions.reshape(new_shape)
    return star_positions


def get_spot_positions_and_photometry(fitsfiles):
    """
    Wrapper that combines get_single_cube_spot_positions and 
    get_single_cube_spot_photometry
    Accept a list of fits files and returns the spot positions and
    spot photometry
    Input:
        fitsfiles: a list of P1640 fits files
    Output: 
        spot_array: Nfiles x Nspots x Nchan x 2 array of (row, col) positions
        spot_phot: Nfils x Nspots x Nchan array of spot photometry
    """
    if isinstance(fitsfiles, str):
            filepaths = [fitsfiles]

    spot_positions = []
    photometry = []
    for f in fitsfiles:
        hdulist = fits.open(f)
        cube = hdustlst[0].data
        spot_positions.append(get_single_cube_spot_positions(cube))
        photometry.append(get_single_cube_spot_photometry(cube, spot_positions))
        hdulist.close()
    return np.array(spot_positions, spot_photometry)

def get_scaling(spot_array, star_array=None, return_mean=True):
    """
    Wrapper for get_single_cube_scaling_factors, to handle multiple cubes
    Input:
        spot_array: (Nfiles) x Nspots x Nchan x 2 array in (row, col) order
        star_array: (Nfiles) x Nchan x 2 array of (row, column) star positions.
            If not supplied, spot_array will be calculated from spot_array
        return_mean: (default False) If true, return mean scaling factor for
            each channel (useful for multiple cubes)
    Output:
        scaling_array: (Nfiles) x Nchan array of scaling fators
    """
    if spot_array.ndim == 3:
        spot_array = np.expand_dims(spot_array, 0)
    if star_array is None:
        star_array = np.array([None for i in spot_array])
    scaling = []
    for spots, star in zip(spot_array, star_array):
        scaling.append(get_single_cube_scaling_factors(spots, star))
    scaling = np.array(scaling)

    if return_mean == True:
        scaling = np.mean(scaling, axis=-2)

    return scaling

def get_scaling_and_centering_from_spots(spot_positions, mean_scaling=True):
    """
    Accepts an array of spots, and returns the scaling factors and centers.
    See also: get_scaling_and_centering_from_files
    Input:
        spot_positions: Ncube x Nspot x Nchan x 2 array of (row, col) spot positions
        mean_scaling: [True] return the average scaling of the 4 spots
    Output:
        scaling_factors: Ncube x  Nchan array of scaling factors
        star_positions: Ncube x Nchan x 2 array of (row, col) star positions
    """
    star_positions = get_star_positions(spot_positions)
    scaling_factors = get_scaling(spot_positions, star_positions, mean_scaling)
    return scaling_factors, star_positions
    
def get_scaling_and_centering_from_files(files, mean_scaling=True):
    """
    Take some csv spot files, and return the star positions and 
    scaling factors for each datacube
    Wrapper for get_scaling_and_centering_from_spots
    Input:
        files: a list of fits files with data cubes
               if the files end in fits or csv, call appropriate routines
        mean_scaling: [True] return the mean scaling of the 4 spots

    Output:
        scaling_factors: scaling factors for each slice of each cube
        star_positions: star positions in each slice of each cube
    """
    spot_positions = [np.genfromtxt(f, delimiter=',') for f in files]
    spot_positions = np.array(spot_positions)
    scaling_factors, star_positions = get_scaling_and_centering_from_spots(spot_positions, mean_scaling)    
    return scaling_factors, star_positions









##################################################
# Diagnostic
##################################################

if __name__ == "__main__":
    fitsfile = sys.argv[1]
    foldername = fitsfile[-8:-5]
    cube = fits.getdata(fitsfile)
    nchan = cube.shape[0]
    img_shape = cube.shape[1:]
    centers = 125*np.ones((nchan, 2))
    channels = np.arange(nchan)

    #################################
    # Initial pass
    spot_masks = make_mask_grid_spots(img_shape, centers)
    spot_locs = np.zeros((num_spots, nchan, 2))
    masked_cubes =  ma.masked_array([ma.masked_array(cube, mask=i) 
                                     for i in spot_masks])
    # get reference channel grid spots:
    for i in range(num_spots):
        img = masked_cubes[i,P1640params.refchan] 
        g, spot_locs[i,P1640params.refchan] = fit_grid_spot(img, centers[P1640params.refchan]) 
    # get starting guesses for the rest
    centered_spots = spot_locs[:,P1640params.refchan] - centers[P1640params.refchan]
    init_rad = np.linalg.norm(centered_spots, axis=1)
    init_theta = np.arctan2(centered_spots[:,0], centered_spots[:,1])
    init_spots = np.rollaxis(np.rollaxis(
        np.array([np.outer(init_rad,scale_factors).T*np.sin(init_theta),
                  np.outer(init_rad,scale_factors).T*np.cos(init_theta)]), 
        2, 0), 2, 1) + centers
    
    # now, fit the rest of the spots
    spot_fits = [[] for i in range(num_spots)]
    for i in range(num_spots):
        images = masked_cubes[i] #cube*spot_masks[i]
        for chan in range(nchan):
            g, spot_locs[i,chan] = fit_grid_spot(images[chan], centers[chan], 
                                                 loc=init_spots[i,chan])
            spot_fits[i].append(g)

    #################################
    # Calculate centers
    # At each channel, fit lines through opposing spots
    for chan in range(nchan):
        b20,m20 = fit_poly(np.array((spot_locs[0,chan,1], 
                                    spot_locs[2,chan,1])),
                          np.array((spot_locs[0,chan,0], 
                                    spot_locs[2,chan,0])),
                      order=1)
        b31,m31 = fit_poly(np.array((spot_locs[1,chan,1], 
                                    spot_locs[3,chan,1])),
                          np.array((spot_locs[0,chan,0], 
                                    spot_locs[2,chan,0])),
                      order=1)
        x_center = (b31-b20)/(m20-m31)  #(line20[0]-line31[0])/(line31[1]-line20[1])
        y_center = (m31*b20-m20*b31)/(m31-m20)  #(line20[1]*line31[0]-line31[1]*line20[0])/(line20[1]-line31[1])
        centers[chan] = (y_center, x_center)

    #re-calculate positions and radii given the new centers
    centered_spots = spot_locs - centers
    rad_spots = np.linalg.norm(spot_locs-centers, axis=2)


    # remake the masks with the new centers
    #spot_masks = make_mask_grid_spots(img_shape, centers)


