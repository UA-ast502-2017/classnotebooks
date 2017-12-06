#!/usr/bin/env python
"""
Given a datacube, find the four corresponding spot files. 
Plot the calculated positions on top of the original cube.

Run from an ipython terminal with:
%run spot_checker.py full/path/to/cube.fits
"""

from __future__ import division

import sys
import os
import warnings
import glob

from multiprocessing import Pool, Process, Queue

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm
from matplotlib.collections import PatchCollection
from matplotlib.patches import CirclePolygon

from astropy.io import fits
from astropy import units

import argparse
#for handling different python versions
if sys.version_info < (3,0):
    import ConfigParser
else:
    import configparser as ConfigParser

# needed to import from outside this folder
base_dir = os.path.dirname(__file__) or '.'
sys.path.append(base_dir)
import pyklip.instruments.P1640_support.P1640spots

dnah_spot_directory = '/data/p1640/data/users/spot_positions/jonathan/'


# plt.ion()


# use multiple threads - one for drawing the figure, and another for handling user input

"""
Pseudocode:
1. Load list of files
2. Create the "good files" dictionary
3. For each file:
3a. Split offt a thread for drawing the cube
3b. Ask for user input
4. When the user provides 'y' or 'n', update the dictionary and kill the drawing thread
5. Move on to the next file
"""

def get_total_exposure_time(fitsfiles, unit=units.minute):
    """
    Accept a list of fits files and return the total exposure time
    Input:
      fitsfiles: single fits file or list of files with keyword 'EXPTIME' in the header
      units: [minute] astropy.units unit for the output
    Output:
      tot_exp_time: the sum of the exposure times for each cube, in minutes
    """
    exptimes = np.array([fits.getval(f, 'EXP_TIME') for f in fitsfiles]) * units.second
    return np.sum(exptimes).to(unit)


# open a fits file and draw the cube
def draw_cube(cube, cube_name, header, seeing, airmass, cube_ix):
    """
    Make a figure and draw cube slices on it
    """
    chan = 14
    nchan = cube.shape[0]
    max_val = np.max(cube)
    fig = plt.figure()
    fig.suptitle(cube_name, fontsize='x-large')
    gridsize=(8,8)
    
    ax_seeing = plt.subplot2grid(gridsize, (4,5), rowspan=2, colspan=3)
    ax_seeing.plot(seeing, 'b')
    ax_seeing.set_xlim(xmin=-0.5, xmax=len(seeing)+0.5)
    ax_seeing.set_yticks(np.sort(seeing)[[0,-1]])
    ax_seeing.axvline(cube_ix, c='k', ls='--')
    ax_seeing.axhline(seeing[cube_ix], c='k', ls='--')
    ax_seeing.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax_seeing.set_title("seeing")
    ax_airmass = plt.subplot2grid(gridsize, (6,5), rowspan=2, colspan=3)
    ax_airmass.plot(airmass, 'r')
    ax_airmass.set_xlim(xmin=-0.5, xmax=len(airmass)+0.5)
    ax_airmass.set_yticks(np.sort(airmass)[[0,-1]])
    ax_airmass.axvline(cube_ix, c='k', ls='--')
    ax_airmass.axhline(airmass[cube_ix], c='k', ls='--')
    ax_airmass.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax_airmass.set_title("airmass")
    ax_airmass.set_xlabel("cube #", size='large')

    #ax = fig.add_subplot(111)
    ax = plt.subplot2grid(gridsize, (1,0),
                          rowspan=6,
                          colspan=5)
#    ax.yaxis.tick_right()
#    fig.subplots_adjust(left=0.3)
    datainfo = """Exp time: {exptime:.3f}
Seeing: {seeing:>9.3f}
Airmass: {airmass:>8.3f}
Max val: {maxval:>8.1f}
Min val: {minval:>8.1f}""".format(exptime=header["EXP_TIME"],
                              seeing=header["SEEING"],
                              airmass=header["INIT_AM"],
                              maxval=np.nanmax(cube),
                              minval=np.nanmin(cube))
    
    fig.text(0.67, 0.85, datainfo, size='large', family='monospace',
             linespacing=2, 
             verticalalignment='top')

    plt.draw()
    plt.tight_layout(pad=1)
    while True:
        #plt.sca(ax)
        #plt.cla()
        ax.clear()
        chan = chan % nchan
        # you're gonna think you want a common scale for all the slices but you're wrong, leave LogNorm alone
        imax = ax.imshow(cube[chan], norm=LogNorm())#vmax=max_val))
        #ax.set_title("{name}\nChannel {ch:02d}".format(name=cube_name, ch=chan))
        ax.set_title("Channel {ch:02d}".format(name=cube_name, ch=chan))
        plt.pause(0.25)
        chan += 1

def plot_airmass_and_seeing(fitsfiles):
    # plot airmass and seeing
    seeing = [fits.getval(ff,'SEEING') for ff in fitsfiles]
    airmass = [fits.getval(ff,'INIT_AM') for ff in fitsfiles]
    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].plot(seeing,'b-')
    axes[0].axhline(np.mean(seeing), c='b',label='mean seeing')
    axes[0].set_title("Seeing")
    axes[1].plot(airmass, 'r-')
    axes[1].axhline(np.mean(airmass), c='r', label='mean airmass')
    axes[1].set_title("Airmass")
    plt.draw()

    

def run_checker(fitsfiles):
    """
    Run the checker
    """
    if fitsfiles == 'help':
        usage()
        return
    seeing = [fits.getval(ff,'SEEING') for ff in fitsfiles]
    airmass = [fits.getval(ff,'INIT_AM') for ff in fitsfiles]
    
    good_cubes = dict(zip(fitsfiles, [None for f in fitsfiles]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #fig = plt.figure()
        repeat = True
        while repeat:
#            proc_cube_stats = Process(target=plot_airmass_and_seeing,
#                                         args=[fitsfiles])
#            proc_cube_stats.start()
            for i, ff in enumerate(fitsfiles):
                # check file
                if not os.path.isfile(ff):
                    print("File not found: {0}".format(ff))
                    sys.exit(0)

                hdulist = fits.open(ff)
                cube = hdulist[0].data
                cube_name = os.path.splitext(os.path.basename(ff))[0]
                header = hdulist[0].header
                # start drawing subprocess
                p = Process(target=draw_cube, args=(cube, cube_name, header, seeing, airmass, i))
                p.start()

                # printcube information

                print("\n{0}/{1} files".format(i+1, len(fitsfiles)))
                print("Cube: {0}".format(cube_name))

                # ask if cube is good or not
                keep_cube = None

                while keep_cube not in ['y', 'n', 'r']:
                    try:
                        print("\ty: keep")
                        print("\tn: discard")
                        print("\tr: flag for re-extraction")
                        keep_cube = input("\tChoose y/n/r: ").lower()[0]
                    except IndexError:
                        continue
                good_cubes[ff] = keep_cube
                hdulist.close()
                # close drawing subprocess
                p.terminate()
                p.join()
#            proc_cube_stats.terminate()
#            proc_cube_stats.join()
            plt.close('all')
            repeat = input("Finished viewing cubes. Return list and quit? (n to loop again) Y/n: ").lower()[0]
            if repeat == 'y':
                repeat = False
            else:
                continue
    # remove the 're-extraction' cubes
    reextract_list = []
    # convert good_cubes dict to Boolean
    """
    for key, val in good_cubes.iteritems():
        if val == 'y': 
            good_cubes[key] = True
        elif val == 'n': 
            good_cubes[key] = False
        elif val == 'r':
            reextract_list.append(key)
        else:
            good_cubes[key] = None
    """
    reextract_cubes = sorted([os.path.abspath(key) for key, val in good_cubes.iteritems() if val == 'r'])
    final_good_cubes = sorted([os.path.abspath(key) for key, val in good_cubes.iteritems() if val == 'y'])
    tot_exp_time = get_total_exposure_time(final_good_cubes, units.minute)
    if reextract_cubes:
        print("\n")
        print("Cubes flagged for re-extraction: ")
        for i in sorted(reextract_cubes): print(i)
    if final_good_cubes:
        print("\n")
        print("Combined exposure time for good cubes: {0:0.2f}".format(tot_exp_time))
    else:
        print("No good cubes")
    return final_good_cubes


#########################
# Spot checker ##########
#########################
def draw_spot_cube(cube, cube_name, spots):
    """
    Make a figure and draw cube slices on it
    spots are a list of [row, col] positions for each spot
    """
    # mask center for better image scaling
    cube[:,100:150,100:150] = np.nan #cube[:,100:150,100:150]*1e-3
    chan=0
    nchan = cube.shape[0]
    # get star positions
    #star_positions = P1640spots.get_single_cube_star_positions(np.array(spots))
    
    #try:
    fig = plt.figure()
    fig.suptitle(cube_name, fontsize='x-large')
    gridsize = (8,8)

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
        #starpatches = [CirclePolygon(xy=star_positions[chan][::-1], radius=3,
        #                             fill=True, alpha=0.3, ec='k', lw=2)
        #               for spot in spots] # star position
        patchcoll = PatchCollection(patches1+patches2, match_original=True)
        
        imax = ax.imshow(cube[chan], norm=LogNorm())
        imax.axes.add_collection(patchcoll)
        ax.set_title("Channel {1:02d}".format(cube_name, chan))
        
        plt.pause(0.2)
        chan += 1
    #except KeyboardInterrupt:
    #    pass


# ALTERNATIVE TO REDEFINING SPOT CHECKER BUT THEN IT REQUIRES THE P1640_SPOT_CHECKER TO EXIST
#run_spot_checker = P1640_spot_checker.run_checker

def run_spot_checker(files=None, config=None, spot_path=None):
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
            p = Process(target=draw_spot_cube, args=(cube, cube_name, spots))
            p.start()

            # printcube information
            print("\n{0}/{1} files".format(i+1, len(fitsfiles)))
            print("\nCube: {0}".format(cube_name))
            print("\tExposure time: {0}".format(fits.getval(ff, "EXP_TIME")))
            print("\tSeeing: {0}".format(fits.getval(ff, "SEEING")))
            print("\tAirmass: {0}".format(np.mean([fits.getval(ff, "INIT_AM"),
                                                   fits.getval(ff, "FINL_AM")])))
            # ask if cube is good or not
            keep_cube = None
            while keep_cube not in ['y', 'n']:
                try:
                    keep_cube = input('\t\tKeep? y/n: ').lower()[0]
                except IndexError:
                    continue
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
    tot_exp_time = get_total_exposure_time(final_good_cubes, units.minute)
    print("Combined exposure time for good cubes: {0:0.2f} min".format(tot_exp_time))
    return final_good_cubes



def usage():
    print("""Required packages:
sys, os
warnings, multiprocessing
numpy, matplotlib
astropy
Usage:
    python cube_checker.py /path/to/data/cube.fits
    OR
    python cube_checker.py space.fits separated.fits sets.fits of.fits paths.fits to.fits cubes.fits
    OR
    python cube_checker.py `ls path/to/cubes/*fits`
    If running from IPython:
    files = glob.glob("all/the/fits/*fits")
    %run cube_checker.py {' '.join(files)}

Prints a list of the chosen file paths
""")

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
        try:
            setattr(namespace, 'spot_path', configparser.get("Spots","spot_file_path"))
        except:
            setattr(namespace, 'spot_path', None)


        
parser = argparse.ArgumentParser(prog="P1640_cube_checker.py",
                                 description='A utility to visually inspect P1640 cubes and spots')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--files', dest='files', nargs='*',
                    help='list of datacube files')
group.add_argument('--config', dest='files', nargs=1, action=ConfigAction,
                    help='config file containing fits files')
spotargs = parser.add_argument_group("spots", description="Use this if you want to check spot locations")
spotargs.add_argument("--spots", dest='spots',action='store_true', default=False,
                      help="use this flag if you want to overplot spot positions")
spotargs.add_argument("--spot_path", dest='spot_path', action='store',
                       default=dnah_spot_directory,
                       help='directory where spot position files are stored')
        
if __name__ == "__main__":

    if sys.argv[1] == 'help':
        usage()
        sys.exit()

    parseobj = parser.parse_args(sys.argv[1:])
    fitsfiles = parseobj.files

    if parseobj.spots:
        spot_directory = parseobj.spot_path
        good_cubes = run_spot_checker(fitsfiles, spot_path=spot_directory)
        #good_cubes = P1640_spot_checker.run_checker(fitsfiles, spot_path=spot_directory)
    else:
        good_cubes = run_checker(fitsfiles)
    print("Good cubes:")
    for i in good_cubes: print(i)

