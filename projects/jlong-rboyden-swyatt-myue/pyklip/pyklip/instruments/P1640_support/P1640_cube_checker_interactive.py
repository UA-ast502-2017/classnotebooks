#!/usr/bin/env python

"""
Observational SNR and exposure time calculator
Jonathan Aguilar
Nov. 15, 2013
"""
from __future__ import division

import sys
import os
import warnings
import glob

import argparse
#for handling different python versions
if sys.version_info < (3,0):
    import ConfigParser
else:
    import configparser as ConfigParser

import matplotlib as mpl
try:
    mpl.use('TkAgg')
except UserWarning:
    pass

import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm, Normalize
import matplotlib.ticker as mticker
from matplotlib.collections import PatchCollection
from matplotlib.patches import CirclePolygon

if sys.version_info[0] < 3:
    import Tkinter as Tk
    import tkFileDialog
else:
    import tkinter as Tk
    import filedialog

import tkFont

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg

from astropy.io import fits
from astropy import units

dnah_spot_directory = '/data/p1640/data/users/spot_positions/jonathan/'


# some text formatting
mpl.rcParams["font.size"]=16
mpl.rcParams["text.color"]='black'
mpl.rcParams["axes.labelcolor"]='black'
mpl.rcParams["xtick.color"]='black'
mpl.rcParams["ytick.color"]='black'



class CubeChecker:

    def __init__(self, master, fitsfiles,
                 spot_mode=False, spot_path=dnah_spot_directory):
        master.wm_title("P1640 Cube Inspector")
        self.customFont = tkFont.Font(family="Helvetica", size=12)

        # file list window
        frame = Tk.Frame(master)
        frame.grid()
        # cube display window
        frame_cube = Tk.Toplevel()
        frame_cube.title("Current cube")
        frame_cube.grid()
        # good cubes window
        self.frame_good_cubes = Tk.Toplevel()
        self.frame_good_cubes.title("Good cubes")
        self.frame_good_cubes.grid()

        self.spot_mode = spot_mode
        self.spot_path = spot_path
        self.fitsfiles = sorted([os.path.abspath(i) for i in fitsfiles])
        try:
            self.seeing = [fits.getval(ff,'SEEING') for ff in self.fitsfiles]
            self.airmass = [fits.getval(ff,'INIT_AM') for ff in self.fitsfiles]
        except KeyError:
            self.seeing = np.ones(len(self.fitsfiles))*10
            self.airmass = np.ones(len(self.fitsfiles))*10
        self.quit_button = Tk.Button(frame, text="Quit", font=self.customFont, fg='black', underline=0,
                                     command=frame.quit)

        # initialize the cube
        self.current_file_index = 0
        self.current_cube_path = Tk.StringVar()
        self.current_cube_path.set(self.fitsfiles[self.current_file_index])
        self.current_cube = fits.getdata(self.current_cube_path.get())
        self.current_header = fits.getheader(self.current_cube_path.get())


        # radio buttons to select linear or log scaling
        self.scaling_group = Tk.LabelFrame(frame_cube, text="Image Scaling", font=self.customFont)
        self.image_scaling = Tk.StringVar()
        self.image_scaling.set("linear")
        self.scale_select_radio = []
        self.scale_select_radio.append(Tk.Radiobutton(self.scaling_group, text="Linear",
                                                      font=self.customFont,
                                                      variable = self.image_scaling,
                                                      value="linear",
                                                      command=self.update_cubeax_display))
        self.scale_select_radio.append(Tk.Radiobutton(self.scaling_group, text="Log",
                                                      font=self.customFont,
                                                      variable=self.image_scaling,
                                                      value="log",
                                                      command=self.update_cubeax_display))

        ### Cube slider and display
        self.f_cube = mpl.figure.Figure()
        self.canvas_cube = FigureCanvasTkAgg(self.f_cube, master=frame_cube)
#        self.canvas_cube.show()
        self.ax_cube = self.f_cube.add_subplot(111)
        #self.ax_cube.xaxis.set_ticks_position("bottom")
        #self.ax_cube.set_xticklabels(self.ax_cube.get_xticklabels(), va='bottom')
        self.s_cube = Tk.Scale(frame_cube,
                               from_=0,to=len(self.current_cube)-1,
                               label='Channel',
                               orient='horizontal',
                               troughcolor='blue',
                               font=self.customFont,
                               length=400,
                               resolution=1,
                               command=self.update_cubeax_display,
                               repeatdelay=300,
                               repeatinterval=250,
                               tickinterval=5)
        self.s_cube.set(25)
        self.play_cube_var = Tk.IntVar()
        self.play_cube_var.set(0)
        self.play_cube_button = Tk.Checkbutton(frame_cube,
                                               text="Auto scroll",
                                               font=self.customFont,
                                               variable=self.play_cube_var,
                                               onvalue=1,
                                               offvalue=0,
                                               command=self.toggle_autoscroll)
                                          
        # plot seeing and airmass
        self.f_seeing = mpl.figure.Figure(figsize=(4,3))
        self.f_airmass = mpl.figure.Figure(figsize=(4,3))
        self.canvas_seeing = FigureCanvasTkAgg(self.f_seeing, master=frame)
        self.canvas_seeing.show()
        self.ax_seeing = self.f_seeing.add_subplot(111)
        self.canvas_airmass = FigureCanvasTkAgg(self.f_airmass, master=frame)
        self.canvas_airmass.show()
        self.ax_airmass = self.f_airmass.add_subplot(111)
   
        # Make a frame to organize the file selection tools
        self.file_lists = Tk.LabelFrame(frame, text="Cube list", font=self.customFont, labelanchor=Tk.N)
        # show the data directory
        self.data_dir = Tk.Label(frame,#self.file_lists,
                                 text='Path:\t'+os.path.dirname(os.path.commonprefix(self.fitsfiles)),
                                 justify=Tk.LEFT,
                                 #wraplength=300,
                                 anchor=Tk.NW,
                                 font=self.customFont)
        # radio buttons to select a fits file, and check boxes to select good files
        self.current_file_group = Tk.LabelFrame(self.file_lists, text="Select", font=self.customFont, padx=5, pady=5)
        self.good_file_group = Tk.LabelFrame(self.file_lists, text="Keep?", font=self.customFont, padx=0, pady=5)
        self.file_select_radio = []
        self.file_select_check = []
        self.good_files = []
        for i,ff in enumerate(self.fitsfiles):
            self.file_select_radio.append(Tk.Radiobutton(self.current_file_group,
                                                         anchor=Tk.W,
                                                         text=os.path.basename(ff),
                                                         font=self.customFont,
                                                         variable=self.current_cube_path,
                                                         value=ff,
                                                         indicatoron=0,
                                                         justify=Tk.LEFT,
                                                         offrelief=Tk.RIDGE,
                                                         command=self.load_selected_cube))
            gf_var = Tk.StringVar()
            self.file_select_check.append(Tk.Checkbutton(self.good_file_group,
                                                         text='',
                                                         font=self.customFont,
                                                         variable=gf_var,
                                                         onvalue=self.fitsfiles[i],
                                                         offvalue=""))
            self.good_files.append(gf_var)
        self.keep_button = Tk.Button(frame_cube, text="Keep?", font=self.customFont, fg="black", underline=0,
                                     command=self.toggle_check)
        self.next_button = Tk.Button(frame_cube, text="Next", font=self.customFont, fg="black", underline=0,
                                     command=self.next_cube)
        self.prev_button = Tk.Button(frame_cube, text="Prev", font=self.customFont, fg="black", underline=0,
                                     command=self.prev_cube)

        # bindings
        self.bind_buttons_to_frame(master)
        self.bind_buttons_to_frame(frame_cube)
        self.bind_buttons_to_frame(self.frame_good_cubes)
        
        # cube data fields
        self.cube_info_group = Tk.LabelFrame(frame_cube, text="Exposure Info", font=self.customFont)
        self.curr_exptime = Tk.StringVar()
        self.curr_seeing = Tk.StringVar()
        self.curr_airmass = Tk.StringVar()
        self.curr_minval = Tk.StringVar()
        self.curr_maxval = Tk.StringVar()
        # assign labels to these variables
        self.exptime_label = Tk.Label(self.cube_info_group,
                                      textvariable=self.curr_exptime, font=self.customFont)
        self.seeing_label = Tk.Label(self.cube_info_group,
                                     textvariable=self.curr_seeing, font=self.customFont)
        self.airmass_label = Tk.Label(self.cube_info_group,
                                      textvariable=self.curr_airmass, font=self.customFont)
        self.max_label = Tk.Label(self.cube_info_group,
                                  textvariable=self.curr_maxval, font=self.customFont)
        self.min_label = Tk.Label(self.cube_info_group,
                                  textvariable=self.curr_minval, font=self.customFont)

       
        # print good files button
        self.print_good_button = Tk.Button(frame, text="Print good files", font=self.customFont, underline=0,
                                           fg='black', command=self.print_good_cubes)
        self.wiki_format_flag = Tk.IntVar()
        self.wiki_format_flag.set(1)
        self.print_wiki_format = Tk.Checkbutton(frame, text="Output as wiki-pasteable list",
                                                font=self.customFont, variable=self.wiki_format_flag,
                                                fg='black')
        
        

        # File list window
        self.data_dir.grid(row=0,column=0, columnspan=4, padx=20)
        self.quit_button.grid(row=0, column=4, columnspan=2,  padx=20,pady=5, sticky=Tk.E)
        self.print_good_button.grid(row=1, column=0, columnspan=2, padx=20,pady=5, sticky=Tk.W)
        self.print_wiki_format.grid(row=2,column=0, padx=20, pady=2, sticky=Tk.NW)
        self.file_lists.grid(row=3,column=0,columnspan=2, rowspan=8, padx=20, sticky=Tk.NW)


        self.canvas_seeing.get_tk_widget().grid(row=1, column=3, rowspan=3, padx=20, pady=20)
        self.canvas_airmass.get_tk_widget().grid(row=4, column=3, rowspan=3, padx=20, pady=20)

        # file lists
        self.current_file_group.grid(row=0, column=0, rowspan=8, sticky=Tk.W)
        self.good_file_group.grid(row=0, column=1, rowspan=8)
        # file selection
        for i, fsr in enumerate(self.file_select_radio):
            fsr.grid(row=i,column=0)
        for i, fsc in enumerate(self.file_select_check):
            fsc.grid(row=i,column=1)

        # Current cube window
        self.canvas_cube.get_tk_widget().grid(row=1,column=1,rowspan=5,columnspan=5, padx=20)
        self.s_cube.grid(row=6,column=1,columnspan=5)

        self.scaling_group.grid(row=5,column=7)

        # self.play_cube_button.grid(row=7,column=0) # disabled for now because it doesn't work
        self.prev_button.grid(row=8,column=2)        
        self.keep_button.grid(row=8,column=3)
        self.next_button.grid(row=8,column=4)

        self.cube_info_group.grid(row=3,column=7, columnspan=2, sticky=Tk.W, padx=20)

        # current cube stats
        self.exptime_label.grid(row=1,column=0, sticky=Tk.W)
        self.seeing_label.grid(row=2,column=0, sticky=Tk.W)
        self.airmass_label.grid(row=3,column=0, sticky=Tk.W)
        self.max_label.grid(row=4,column=0, sticky=Tk.W)
        self.min_label.grid(row=5,column=0, sticky=Tk.W)
        
        # image scaling
        self.scale_select_radio[0].grid(row=1,column=1, sticky=Tk.W)
        self.scale_select_radio[1].grid(row=2,column=1,sticky=Tk.W)

        # initialize the whole widget based on the selected cube
        self.load_selected_cube()

    # key bindings
    def next_button_pushed(self, event):
        self.next_button.invoke()
    def prev_button_pushed(self, event):
        self.prev_button.invoke()
    def quit_button_pushed(self, event):
        self.quit_button.invoke()
    def keep_button_pushed(self, event):
        self.keep_button.invoke()
    def scroll_cube_left(self, event):
        if self.s_cube.get() == 0:
            self.s_cube.set(len(self.current_cube)-1)
        else:
            self.s_cube.set(np.max([self.s_cube.get()-1,0]))
    def scroll_cube_right(self, event):
        self.s_cube.set((self.s_cube.get()+1)%len(self.current_cube))
    def activate_printing(self, event):
        self.print_good_button.invoke()
        
    def bind_buttons_to_frame(self, frame_ref):
        """
        These are the buttons you want to be recognized by all the frames
        """
        frame_ref.bind("n", self.next_button_pushed)
        frame_ref.bind("p", self.prev_button_pushed)
        frame_ref.bind("k", self.keep_button_pushed)
        frame_ref.bind("q", self.quit_button_pushed)
        frame_ref.bind("<Left>", self.scroll_cube_left) 
        frame_ref.bind("<Right>", self.scroll_cube_right)
        frame_ref.bind("<Up>", self.prev_button_pushed) 
        frame_ref.bind("<Down>", self.next_button_pushed)
        frame_ref.bind("<Shift-P>", self.activate_printing)

        
    @property
    def current_cube(self):
        return self._current_cube
    @current_cube.setter
    def current_cube(self, newval):
        self._current_cube = newval

    #### Commands ####
    def load_selected_cube(self):
        """
        Read the highlighted fitsfile, get the datacube from it, and display it
        This function does the heavy lifting and calls the other update functions
          - check for spot mode
          - set the current file index
          - set the current file using the current file index
          - load the cube from the current file
        """
        # find the new file index
        self.current_file_index = self.fitsfiles.index(self.current_cube_path.get())
        # update the cube and header
        self.current_cube = fits.getdata(self.current_cube_path.get(), 0)
        if self.spot_mode:
            self.current_cube[:,90:160,90:160] = np.nan
        self.current_header = fits.getheader(self.current_cube_path.get(),0)
        # run update functions
        self.update_seeing_and_airmass()
        self.update_cubeax_display()
        self.update_cube_stats()
        
    def update_cube_stats(self):
        header = fits.getheader(self.fitsfiles[self.current_file_index])
        try:
            exptime = header['EXP_TIME']
            airmass = np.mean([header['INIT_AM'], header['FINL_AM']])
            seeing = header['SEEING']
        except KeyError:
            exptime = np.nan
            airmass = np.nan
            seeing = np.nan
        maxval = np.nanmax(self.current_cube)
        minval = np.nanmin(self.current_cube)
        """
        self.curr_exptime.set("Exposure time: {0:.3f}".format(self.current_header['EXP_TIME']))
        self.curr_seeing.set("Seeing: {0:>9.3f}".format(self.current_header['SEEING']))
        self.curr_airmass.set("Airmass: {0:>8.3f}".format(self.current_header['INIT_AM']))
        self.curr_maxval.set("Max val: {0:>8.1f}".format(np.nanmax(self.current_cube)))
        self.curr_minval.set("Min val: {0:>8.1f}".format(np.nanmin(self.current_cube)))
        """
        self.curr_exptime.set("Exposure time: {0:.3f}".format(exptime))
        self.curr_seeing.set("Seeing: {0:>9.3f}".format(seeing))
        self.curr_airmass.set("Airmass: {0:>8.3f}".format(airmass))
        self.curr_maxval.set("Max val: {0:>8.1f}".format(np.nanmax(self.current_cube)))
        self.curr_minval.set("Min val: {0:>8.1f}".format(np.nanmin(self.current_cube)))

    ## Spots ##
    def load_spot_files(self):
        """
        based on the current cube and the spot path, get the spot files
        """
        cubefile_name = os.path.splitext(os.path.basename(self.current_cube_path.get()))[0]
        files = sorted(glob.glob(os.path.join(self.spot_path, cubefile_name)+"*"))
        return files
    
    def draw_spots_on_cube(self):
        spot_files = self.load_spot_files()
        spots = [np.genfromtxt(f, delimiter=',') for f in spot_files]
        chan = self.s_cube.get()
        patches1 = [CirclePolygon(xy=spot[chan][::-1], radius=5,
                                  fill=False, alpha=1, ec='k', lw=2)
                    for spot in spots] # large circles centered on spot
        patches2 = [CirclePolygon(xy=spot[chan][::-1], radius=1,
                                  fill=True, alpha=0.3, ec='k', lw=2)
                    for spot in spots] # dots in location of spot
        patchcoll = PatchCollection(patches1+patches2, match_original=True)
        self.ax_cube.add_collection(patchcoll)
        self.f_cube.canvas.draw_idle()
        
    ## Cube selection shortcuts ##
    def toggle_check(self):
        self.file_select_check[self.current_file_index].toggle()
    def next_cube(self):
        # make sure you're in range
        if self.current_file_index+1 == len(self.file_select_radio):
            pass
        else:
            rad = self.file_select_radio[self.current_file_index+1]
            rad.select()
            rad.invoke()
    def prev_cube(self):
        # make sure you're in range
        if self.current_file_index == 0:
            pass
        else:
            rad = self.file_select_radio[self.current_file_index-1]
            rad.select()
            rad.invoke()

    ## Plotting ##
    def update_cubeax_display(self, event=None):
        img_norm = Normalize
        if self.image_scaling.get() == "log":
            img_norm = LogNorm
        self.ax_cube.clear()

        self.ax_cube.matshow(self.current_cube[self.s_cube.get()], origin='lower', norm=img_norm(),
                             cmap='cubehelix')
        self.ax_cube.xaxis.set_ticks_position("bottom")
        self.ax_cube.set_title(os.path.basename(self.current_cube_path.get()))
        if self.spot_mode is True:
            self.draw_spots_on_cube()
        self.f_cube.canvas.draw_idle()

    def toggle_autoscroll(self):
        len_cube = len(self.current_cube)
        while self.play_cube_var.get() == 1:
            self.s_cube.set((self.s_cube.get()+1)%len_cube)
            time.sleep(0.3) # maybe this needs to be longer

        
    def update_seeing_and_airmass(self):
        self.ax_seeing.clear()
        self.ax_airmass.clear()

        self.ax_seeing.plot(self.seeing,'r')
        self.ax_airmass.plot(self.airmass,'r')

        # axis styling
        self.ax_seeing.set_xlim(xmin=-0.5, xmax=len(self.seeing)+0.5)
        self.ax_seeing.set_yticks(np.sort(self.seeing)[[0,-1]])
        self.ax_seeing.axvline(self.current_file_index, c='k', ls='--')
        self.ax_seeing.axhline(self.seeing[self.current_file_index], c='k', ls='--')
        self.ax_seeing.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        self.ax_seeing.set_title("seeing")
        self.ax_seeing.grid(True)
        self.f_seeing.tight_layout()
        self.f_seeing.canvas.draw_idle()

        self.ax_airmass.set_xlim(xmin=-0.5, xmax=len(self.airmass)+0.5)
        self.ax_airmass.set_yticks(np.sort(self.airmass)[[0,-1]])
        self.ax_airmass.axvline(self.current_file_index, c='k', ls='--')
        self.ax_airmass.axhline(self.airmass[self.current_file_index], c='k', ls='--')
        self.ax_airmass.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        self.ax_airmass.set_title("airmass")
        self.ax_airmass.grid(True)
        self.ax_airmass.set_xlabel("cube #", size='large')
        self.f_airmass.tight_layout()
        self.f_airmass.canvas.draw_idle()

    ## Printing ##
    def print_good_cubes(self):
        good_cubes = [os.path.abspath(i.get()) for i in self.good_files if i.get()]
        # clear the good_cubes frame
        children = self.frame_good_cubes.winfo_children()
        if len(children) > 0:
            for child in children:
                child.destroy()
        
        try:
            max_length = max([len(i) for i in good_cubes])
        except ValueError:
            max_length=0
        width = max([max_length, 20])
        exptimevar = Tk.StringVar()
        exptimevar.set("0 min")
        exptime_label = Tk.Label(self.frame_good_cubes,
                                 textvariable=exptimevar,
                                 font=self.customFont)
        good_cubes_listbox = Tk.Listbox(self.frame_good_cubes,
                                        exportselection=1,
                                        height=len(good_cubes)+2,
                                        selectmode=Tk.EXTENDED,
                                        font=self.customFont,
                                        width=width + 5)
        
        if len(good_cubes) > 0:
            #print("Printing good cubes:")
            for i in good_cubes:
                #print i
                if self.wiki_format_flag.get() == 1:
                    i = "  * " + i
                good_cubes_listbox.insert(Tk.END, i)
            exptime = get_total_exposure_time(good_cubes)
            expmin = np.int(np.floor(exptime.value))
            expsec = exptime.decompose().value % 60
            exptimevar.set("Total exposure time: {0} min {1:0.0f} sec".format(expmin, expsec))
            #print(exptimestr)
        else:
            #print("No good cubes")
            good_cubes_listbox.insert(Tk.END, "No good cubes")
        exptime_label.grid(row=0,column=0)
        good_cubes_listbox.grid(row=1,column=0,rowspan=len(good_cubes)+3)
        self.bind_buttons_to_frame(self.frame_good_cubes)
        # after you push this button, raise the window
        self.frame_good_cubes.lift()

#### Helper functions ####



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


# CLI argument handling
# Command-line option handling
# if switch --config is used, treat the following argument like the path to a config file
# if --config is not present, treat the following arguments like individual datacubes
class ConfigAction(argparse.Action):
    """
    Create a custom action to parse the command line arguments
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
#parser.add_argument('--help', action=usage)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--files', dest='files', nargs='*',
                    help='list of datacube files')
group.add_argument('--config', dest='files', nargs=1, action=ConfigAction,
                    help='config file containing fits files')
spotargs = parser.add_argument_group("Spot fitting", description="Optional arguments, use if you want to check spot locations")
spotargs.add_argument("--spots", dest='spots',action='store_true', default=False,
                      help="use this flag if you want to overplot spot positions")
spotargs.add_argument("--spot_path", dest='spot_path', action='store',
                       default=dnah_spot_directory,
                       help='directory where spot position files are stored')



        
if __name__ == "__main__":

    # some initializations
    spot_mode = False
    spot_directory = dnah_spot_directory
    
    parseobj = parser.parse_args(sys.argv[1:])
    try:
        fitsfiles = parseobj.files
        if parseobj.spots:
            spot_mode=True
            spot_directory = parseobj.spot_path
    except:
        sys.exit()

    try:
        # run the gui
        root = Tk.Tk()
        cubechecker = CubeChecker(root, fitsfiles, spot_mode, spot_directory)
        root.mainloop()
        root.destroy()
    except KeyboardInterrupt as e:
        pass
