#!/usr/bin/env python

"""
Utilities specific to generating contrast curves
"""

import sys
import os
import glob
import re

#for handling different python versions
if sys.version_info < (3,0):
    import ConfigParser
else:
    import configparser as ConfigParser

import numpy as np
import pandas as pd
from scipy import interpolate

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm


from astropy.io import fits
from photutils import aperture_photometry, CircularAperture

import pyklip.instruments.P1640_support.P1640cores

def calc_contrast_single_file(filename, core_info=None, chans='all'):
    """
    Calculate the radiall-averaged variance for a pyklip-reduced file for a single channel
    Input: 
        filename: full path to a pyklip-processed datacube
        core_info: pandas DataFrame with 'radius' and 'flux' columns
        chans: iterative type list of channels
    """
    # pick an arbitrary reference channel
    cube = fits.getdata(filename)
    numfiles = fits.getval(filename, 'DRPNFILE')
    starx = fits.getval(filename, 'PSFCENTX')
    stary = fits.getval(filename, 'PSFCENTY')
    # normalize flux for co-added cubes
    cube /= numfiles
    if chans == 'all':
        chans = range(len(cube))
    if np.size(chans) == 1:
        chans = [chans]

    # get radial coordinates
    yx = (np.indices(cube.shape[-2:]).T - [stary,starx]).T
    rad_coord = np.ravel(np.linalg.norm(yx, axis=0))
    radlims = (np.floor(np.nanmin(rad_coord)), np.ceil(np.nanmax(rad_coord)))
    radbins = np.linspace(radlims[0], radlims[1], 50)
    radii = radbins[:-1] + np.diff(radbins)
    
    # convolve cube with aperture
    convolved_ims = aperture_convolve_cube(cube[chans], core_info['radius'][chans])
    # divide by flux
    contrast_ims = (convolved_ims.T/np.array(core_info['flux'][chans])).T

    contrast_df = {}
    contrast_df['rad'] = radii
    for i, chan in enumerate(chans):
        radpix = pd.DataFrame(zip(rad_coord, np.ravel(contrast_ims[i])), 
                              columns=['rad', 'pixval'])
        radpix.sort(columns='rad', inplace=True)
        radpix.reset_index(inplace=True)
        # set up windowed variance
        std = []
        for j in range(len(radbins)-1):
            std.append(radpix[(radpix['rad'] >= radbins[j]) & (radpix['rad'] < radbins[j+1])]['pixval'].std())
        std = np.array(std)
        contrast_df['chan{0:02d}'.format(chan)] = std
        
    return pd.DataFrame(contrast_df)

def calc_contrast_multifile(core_files, datacube):
    """
    Assemble a median core PSF out of the list of core_files, and return
    a datacube scaled by the core flux
    """
    pass

def make_contrast_plot(contrast_map, title=None, 
                       contrast_range=None, 
                       plate_scale=19.2, pckwargs=None):
    """
    Plot contrast against separation and channel number.
    Input:
        contrast_map: Pandas DataFrame. See required columns below.
        name: plot title (preferably the file name corresponding to the contrast map)
        plate_scale (19.2 mas/pix): convert between pixels and mas
        contrast_range (None): tuple of upper and lower bounds for contrast plot. If none, min-max.
        pckwargs: dictionary of arguments that can be passed to plt.pcolor
    Returns:
       fig: plt.figure() object
    contrast_map: one column is titled 'rad', this is the separation in pixels
    The rest of the columns are titled 'chan##', where ## is the channel number.
    """
    # x- and y-axes
    channels = [col for col in contrast_map.columns if re.match('chan[0-9][0-9]', col) is not None]
    sep = contrast_map['rad']*plate_scale
    masked_data = np.ma.masked_array(contrast_map[channels].as_matrix(),
                                     mask=np.isnan(contrast_map[channels].as_matrix()))
    fig = plt.figure()
    if contrast_range is None: 
        contrast_range=(np.nanmin(masked_data), np.nanmax(masked_data))
    ax = plt.pcolor(sep, np.arange(len(channels)),
                    masked_data.T, 
                    cmap = 'cubehelix',
                    norm=LogNorm(vmin=contrast_range[0], 
                                 vmax=contrast_range[1]))
    plt.colorbar()
    plt.xlim(0, 3000)
    plt.ylim(0, len(channels)-1)
    plt.grid(True, which='both')
    plt.xlabel("Separation [mas]", size='x-large')
    plt.ylabel("Channel", size='x-large')
    if title is not None:
        plt.title(title, size='xx-large')
    fig.tight_layout()
    return fig    

def make_contrast_summary_plot(contrast_map_dict, chan='chan23', title=None, plate_scale=19.2, kwargs=None):
    """
    Plot the mean, min, and max contrast in the given channel for all the reductions
    """
    # select the diagnostic channel
    chandata = pd.concat([pd.DataFrame(cm[['rad',chan]]) for cm in contrast_map_dict.values()])
    # put the values at different radii together
    grouped = chandata.groupby('rad')
    rad = np.array(sorted(grouped.groups.keys()))*plate_scale
    fig = plt.figure()
    plt.fill_between(rad, grouped.min()[chan], grouped.max()[chan], alpha=0.5, label='min-max')
    plt.plot(rad, grouped.median()[chan], 'k-', lw=2, label='median')
    plt.yscale('log')
    plt.xlabel("Separation [mas]")
    plt.ylabel("Contrast")
    plt.grid(True, which='major')
    if title is None:
        title = chan + ' summary'
    plt.title(title, size='x-large')
    plt.legend(numpoints=1)
    fig.tight_layout()
    return fig



################################
# Run as script
################################
if __name__ == "__main__":
    """Run with -s for saving figures"""
    # Get star info from config file
    config = sys.argv[1]
    SaveFlag = False
    try:
        if sys.argv[2] == '-s':
            SaveFlag = True
    except IndexError:
        pass
    
    cfp = ConfigParser.SafeConfigParser()
    cfp.read(config)

    StarName = cfp.get("Contrast","StarName")
    ObsDate =  cfp.get("Contrast","ObsDate")
    print("Star: {star}\nEpoch: {epoch}".format(star=StarName,epoch=ObsDate))

    # validate input paths
    reduced_files = eval(cfp.get("Contrast","reduced_file_search_command"))
    # if you messed up the reduced file path, you get to try again
    while len(reduced_files) == 0:
        print("No reduced files found with: {0}".format(cfp.get("Contrast","reduced_file_search_command")))
        reduced_files = eval(raw_input("Fix command: "))
        if len(reduced_files) >= 0: print("{0} reduced files found, good job. Moving on...\n".format(len(reduced_files)))
    
    # STAR FLUX
    # prepare core files
    core_files = cfp.get("Contrast","core_files").split()
    core_hdus = [fits.open(f) for f in core_files]
    core_cubes = [hdulist[0].data for hdulist in core_hdus]
    for core in core_cubes: 
        core[core == 0] = np.nan
    print("Determining core fluxes and radii")
    core_info_all = [P1640cores.get_encircled_energy_cube(c) for c in core_cubes]
    core_info = P1640cores.combine_multiple_cores(core_info_all)
    print("    ...finished.\n")
    
    # POST-PROCESSING FLUX
    contrast_maps = {}
    channels = 'all'
    print("Generating contrast maps")
    for i,ff in enumerate(reduced_files):
        fname = os.path.basename(ff)
        print("\t{n:2d}/{tot}\t{ff}".format(n=i,tot=len(reduced_files),ff=fname))
        contrast_maps[fname] = calc_contrast_single_file(ff, core_info, chans=channels)
        print("Processing complete, contrast maps stored in variable 'contrast_maps'")

    plate_scale = np.float(cfp.get("Contrast","plate_scale"))
    if SaveFlag is True:
        figure_path = cfp.get("Contrast","figure_path")
        sumchan = 'chan23'
        fig = make_contrast_summary_plot(contrast_maps,
                                         title=StarName+' '+ObsDate+'\n'+sumchan+' '+'summary',
                                         plate_scale=plate_scale)
        fig.savefig(os.path.join(figure_path, StarName+'_'+ObsDate+'-summary.png'))
        plt.close(fig.number)
        for fname, cm in contrast_maps.iteritems():
            name = os.path.splitext(os.path.basename(fname))[0]
            name1 = StarName+'_'+ObsDate
            name2 = name[len(name1)+1:]
            title = name1+'\n'+name2
            fig = make_contrast_plot(cm, title, plate_scale=plate_scale, 
                                     contrast_range=(1e-7,5e-2))
            fig.savefig(os.path.join(figure_path, name+'-contrast_map.png'))
            plt.close(fig.number)
    

