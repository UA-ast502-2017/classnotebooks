#!/usr/bin/env python
"""
Given a datacube, find the four corresponding spot files. 
Plot the calculated positions on top of the original cube.

Run from an ipython terminal with:
%run spot_checker.py full/path/to/cubes.fits
"""

from __future__ import division

import sys
import os
import glob
import warnings

import argparse
#for handling different python versions
if sys.version_info < (3,0):
    import ConfigParser
else:
    import configparser as ConfigParser

from pyklip.instruments.P1640_support.P1640spots import get_single_cube_star_positions
from multiprocessing import Pool, Process, Queue

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.collections import PatchCollection
from matplotlib.patches import CirclePolygon

from astropy.io import fits

dnah_spot_directory = '/data/p1640/data/users/spot_positions/jonathan/'


# open a fits file and draw the cube
def draw_cube(cube, cube_name, spots):
    """
    Make a figure and draw cube slices on it
    spots are a list of [row, col] positions for each spot
    """
    # mask center for better image scaling
    cube[:,100:150,100:150] = np.nan #cube[:,100:150,100:150]*1e-3
    chan=0
    nchan = cube.shape[0]
    # get star positions
    star_positions = get_single_cube_star_positions(np.array(spots))#P1640spots.get_single_cube_star_positions(np.array(spots))
    
    #try:
    fig = plt.figure()
    fig.suptitle(cube_name, fontsize='x-large')
    gridsize = (8,8)

    """ this was too fancy and just messed things up
    # scaling
    ax_radial = plt.subplot2grid(gridsize, (1,6), rowspan=2, colspan=3)
    ax_radial.plot(np.linalg.norm(np.array(spots) - star_positions, axis=-1).T)
    ax_radial.set_title("Radial position")
    ax_radial.grid(True, axis='both', which='major')
    ax_radial.set_xlabel("channel")
    ax_radial.set_ylabel("Separation")
    
    # centering
    ax_center = plt.subplot2grid(gridsize, (5,6), rowspan=2, colspan=3)
    ax_center.set_xlim(np.min(star_positions[:,1])-1,
                       np.max(star_positions[:,1])+1)

     main image axis object
    ax = plt.subplot2grid(gridsize, (1,0),
                          rowspan=6,
                          colspan=5)
    """
    
    ax = fig.add_subplot(111)
    while True:
        ax.clear()
    
        chan = chan % nchan

        patches1 = [CirclePolygon(xy=spot[chan][::-1], radius=5,
                                  fill=False, alpha=1, ec='k', lw=2)
                    for spot in spots] # large circles centered on spot
        patches2 = [CirclePolygon(xy=spot[chan][::-1], radius=1,
                                  fill=True, alpha=0.3, ec='k', lw=2)
                    for spot in spots] # dots in location of spot
        starpatches = [CirclePolygon(xy=star_positions[chan][::-1], radius=3,
                                     fill=True, alpha=0.3, ec='k', lw=2)
                       for spot in spots] # star position
        patchcoll = PatchCollection(patches1+patches2, match_original=True)
        
        imax = ax.imshow(cube[chan], norm=LogNorm())
        imax.axes.add_collection(patchcoll)
        ax.set_title("Channel {1:02d}".format(cube_name, chan))

        """
        ax_center.set_title("Star position")
        ax_center.plot(star_positions[:,1], star_positions[:,0], 'b-')
        ax_center.plot(star_positions[chan,1], star_positions[chan,0],
                       'bx',
                       ms=10, mew=2)
        ax_center.grid(True, axis='both', which='major')
        ax_center.set_xlabel("x")
        ax_center.set_ylabel("y")
        """
        plt.pause(0.2)
        chan += 1
    #except KeyboardInterrupt:
    #    pass

    
# Command-line option handling
# if switch --config is used, treat the following argument like the path to a config file
# if --config is not present, treat the following arguments like individual datacubes
class ConfigAction(argparse.Action):
    """
    Create a custom action to parse the 
    """
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        #if nargs is not None:
        #    raise ValueError("nargs not allowed")
        super(ConfigAction, self).__init__(option_strings, dest, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        configparser = ConfigParser.ConfigParser()
        configparser.read(values)
        # get the list of files
        filelist = configparser.get("Input","occulted_files").split()
        setattr(namespace, self.dest, filelist)
        setattr(namespace, 'spot_path', configparser.get("Spots","spot_file_path"))
        
        # get the spot file path
        # get the spot file name info

parser = argparse.ArgumentParser(prog="P1640_spot_checker.py",
                                 description='A utility to verify the grid spot fitting')
group = parser.add_mutually_exclusive_group()
group.add_argument('--files', dest='files', nargs='*', 
                    help='list of datacube files')
group.add_argument('--config', dest='files', nargs=1, action=ConfigAction,
                    help='config file containing fits files')
parser.add_argument('--spot_path', dest='spot_path', default=dnah_spot_directory,
                    help='directory where spot position files are stored')
        
    

def run_checker(files=None, config=None, spot_path=None):
    """
    Supply ONE OF:
    files: list of files
    config: config file with a list of files
    """
    if spot_path is None:
        # set this default here so it can be overridden by a config
        configparser = ConfigParser.ConfigParser()
        configparser.read("../../P1640.ini")
        spot_directory = configparser.get("spots","spot_file_path")

    # files vs config: two args enter! one arg leaves!
    if not (files is None) != (config is None):
        print("Please supply either a list if files or a config file")
        return None
    elif files is not None:
        fitsfiles = files
    elif config is not None:
        configparser = ConfigParser.ConfigParser()
        configparser.read(config)
        # get the list of files
        fitsfiles = configparser.get("Input","occulted_files").split()
        spot_directory = configparser.get("Spots","spot_file_path")   
    else:
        print("Please supply either list of files or a config file")
        return None

    # spot path - if None, read default
    
    good_cubes = dict(zip(fitsfiles, [None for f in fitsfiles]))
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #fig = plt.figure()
        for i, ff in enumerate(fitsfiles):
            # check file
            if not os.path.isfile(ff):
                print("File not found: {0}".format(ff))
                sys.exit(0)
                
            # get spots
            cubefile_name = os.path.splitext(os.path.basename(ff))[0]
            spot_files = glob.glob(os.path.join(spot_path, cubefile_name)+"*")
            if len(spot_files) == 0:
                print("No spot files found for {0}".format(os.path.basename(ff)))
                sys.exit(0)
            spots = [np.genfromtxt(f, delimiter=',') for f in spot_files]

            hdulist = fits.open(ff)
            cube = hdulist[0].data
            cube_name = os.path.splitext(os.path.basename(ff))[0]

            # start drawing subprocess
            p = Process(target=draw_cube, args=(cube, cube_name, spots))
            p.start()

            # print cube information
            print("\n{0}/{1} files".format(i+1, len(fitsfiles)))
            print("\nCube: {0}".format(cube_name))
            print("\tExposure time: {0}".format(fits.getval(ff, "EXP_TIME")))
            print("\tSeeing: {0}".format(fits.getval(ff, "SEEING")))
            print("\tAirmass: {0}".format(np.mean([fits.getval(ff, "INIT_AM"),
                                                   fits.getval(ff, "FINL_AM")])))
            # ask if cube is good or not
            keep_cube = None
            while keep_cube not in ['y', 'n']:
                keep_cube = raw_input('\t\tKeep? y/n: ').lower()[0]
            good_cubes[ff] = keep_cube

            # close drawing subprocess
            p.terminate()
            p.join()

        plt.close('all')
    for key, val in good_cubes.iteritems():
        if val == 'y': 
            good_cubes[key] = True
        elif val == 'n': 
            good_cubes[key] = False
        else:
            good_cubes[key] = None
    
    #print(good_cubes)
    #print("Good cubes: ")
    #for i in sorted([key for key, val inre good_cubes.iteritems() if val == True]):
    #    print(i)
    if np.all(good_cubes.values()):
        print("\nSpot fitting succeeded for all cubes.\n")
    else:
        print("\nSpot fitting failed for the following cubes:")
        print("\n".join([i for i in sorted(good_cubes.keys()) if good_cubes[i] == False]))
        print("\n")

    final_good_cubes = sorted([os.path.abspath(i) for i in good_cubes.keys() if good_cubes[i] == True])
    return final_good_cubes


if __name__ == "__main__":

    parseobj = parser.parse_args(sys.argv[1:])
    fitsfiles = parseobj.files
    spot_directory = parseobj.spot_path

    good_cubes = run_checker(fitsfiles, spot_path=spot_directory)
    
