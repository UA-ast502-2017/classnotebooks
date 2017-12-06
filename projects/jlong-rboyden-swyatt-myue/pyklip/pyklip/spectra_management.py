__author__ = 'jruffio'

import numpy as np
from scipy.interpolate import interp1d
import astropy.io.fits as pyfits
import platform
import scipy.ndimage as ndimage
from scipy.optimize import leastsq
from scipy.optimize import minimize
import glob
import os
import csv




def find_upper_nearest(array,value):
    """
    Find the upper nearest element to value in array.

    :param array: Array of value
    :param value: Value for which one wants the upper value.
    :return: (up_value, id) with up_value the closest upper value and id its index.
    """
    diff = array-value
    diff[np.where(diff<0.0)] = np.nan
    idx = np.nanargmin(diff)
    return array[idx], idx

def find_lower_nearest(array,value):
    """
    Find the lower nearest element to value in array.

    :param array: Array of value
    :param value: Value for which one wants the lower value.
    :return: (low_value, id) with low_value the closest lower value and id its index.
    """
    diff = array-value
    diff[np.where(diff>0.0)] = np.nan
    idx = np.nanargmax(diff)
    return array[idx], idx


def find_nearest(array,value):
    """
    Find the nearest element to value in array.

    :param array: Array of value
    :param value: Value for which one wants the closest value.
    :return: (closest_value, id) with closest_value the closest lower value and id its index.
    """
    diff = np.array(array)-value
    # diff[np.where(diff>0.0)] = np.nan
    idx = np.nanargmin(np.abs(diff))
    return array[idx], idx

def get_specType(object_name,SpT_file_csv = None):
    """
    Return the spectral type for a target based on Simbad or on the table in SpT_file

    :param object_name: Name of the target: ie "c_Eri"
    :param SpT_file: Filename (.csv) of the table containing the target names and their spectral type.
                    Can be generated from quering Simbad.
                    If None (default), the function directly tries to query Simbad.
    :return: Spectral type
    """
    # Hard-coded spectral type for some targets. Not ideal but I don't want to think about it right now.
    if object_name == "iot_Cen":
        return "A1"
    if object_name == "IK_Peg":
        return "A8"

    if SpT_file_csv is None:
        import urllib

        object_name = object_name.replace('_','+')

        # url = urllib.urlopen("http://cdsweb.u-strasbg.fr/cgi-bin/nph-sesame/-oI?"+object_name)
        url = urllib.urlopen("http://simbad.u-strasbg.fr/simbad/sim-id?output.format=ASCII&output.max=1&\
                              obj.cooN=off&obj.pmsel=off&obj.plxsel=off&obj.rvsel=off&obj.spsel=on&obj.mtsel=off&\
                              obj.sizesel=off&obj.fluxsel=off&obj.messel=off&obj.notesel=off&obj.bibsel=off&Ident="+object_name)
        text = url.read()
        for line in text.splitlines():
            # if line.startswith('Spectral type:'):
            if line.find('Spectral type:') != -1:
                # print(line)
                spec_type =line.split("Spectral type: ")[-1].replace("Spectral type: ","").split(" ")[0]

        try:
            return spec_type
        except:
            print("Couldn't find {0} in Simbad.".format(object_name))
            return None

    with open(SpT_file_csv, 'rb') as csvfile_TID:
        TID_reader = csv.reader(csvfile_TID, delimiter=';')
        TID_csv_as_list = list(TID_reader)
        TID_csv_as_nparr = np.array(TID_csv_as_list)[1:len(TID_csv_as_list),:]
        target_names = np.ndarray.tolist(TID_csv_as_nparr[:,0])
        specTypes = np.ndarray.tolist(TID_csv_as_nparr[:,1])


    try:
        return specTypes[target_names.index(object_name)]
    except:
        import urllib

        print("Couldn't find {0} in spectral type list. Try to retrieve it from Simbad.".format(object_name))

        target_names.append(object_name)
        object_name = object_name.replace('_','+')

        url = urllib.urlopen("http://simbad.u-strasbg.fr/simbad/sim-id?output.format=ASCII&output.max=1&\
                              obj.cooN=off&obj.pmsel=off&obj.plxsel=off&obj.rvsel=off&obj.spsel=on&obj.mtsel=off&\
                              obj.sizesel=off&obj.fluxsel=off&obj.messel=off&obj.notesel=off&obj.bibsel=off&Ident="+object_name)
        text = url.read()
        for line in text.splitlines():
            if line.startswith('Spectral type:'):
                spec_type =line.replace("Spectral type: ","").split(" ")[0]
        specTypes.append(spec_type)


        attrib_name=["name","specType"]
        with open(SpT_file_csv, 'w+') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=';')
            table_to_csv = [attrib_name]+zip(target_names,specTypes)
            csvwriter.writerows(table_to_csv)

        return spec_type

def get_star_spectrum(wvs_or_filter_name,star_type = None, temperature = None,mute = None):
    """
    Get the spectrum of a star with given spectral type interpolating the pickles database.
    The spectrum is normalized to unit mean.
    It assumes type V star.

    Inputs:
        wvs_or_filter_name: array of wavelenths in microns (or string with GPI band 'H', 'J', 'K1', 'K2', 'Y').
                (When using GPI spectral band wavelength samples are linearly spaced between the first and the last
                wavelength of the band.)
        star_type: 'A5','F4',... Is ignored if temperature is defined.
                If star_type is longer than 2 characters it is truncated.
        temperature: temperature of the star. Overwrite star_type if defined.

    Output:
        (wavelengths, spectrum) where
            wavelengths: Sampling in mum.
            spectrum: is the spectrum of the star for the given band.
    """

    if mute is None:
        mute = False

    if isinstance(wvs_or_filter_name, str):
        import pyklip.instruments.GPI as GPI
        sampling_wvs = GPI.get_gpi_wavelength_sampling(wvs_or_filter_name)
    else:
        sampling_wvs = wvs_or_filter_name

    sampling_wvs_unique = np.unique(sampling_wvs)


    if star_type is None:
        return sampling_wvs,None

    if len(star_type) > 2:
        star_type_selec = star_type[0:2]
    else:
        star_type_selec = star_type

    try:
        int(star_type_selec[1])
    except:
        try:
            star_type_selec = star_type[-3:-1]
            int(star_type_selec[1])
        except:
            if not mute:
                print("Returning None. Couldn't parse spectral type.")
            return sampling_wvs,None

    # Sory hard-coded type...
    if star_type_selec == "K8":
        star_type_selec = "K7"

    pykliproot = os.path.dirname(os.path.realpath(__file__))
    filename_temp_lookup = pykliproot+os.path.sep+"pickles"+os.path.sep+"mainseq_colors.txt"
    filename_pickles_lookup = pykliproot+os.path.sep+"pickles"+os.path.sep+"AA_README"

    #a = np.genfromtxt(filename_temp_lookup, names=True, delimiter=' ', dtype=None)

    # The interpolation is based on the temperature of the star
    # If the input was not the temperature then it is taken from the mainseq_colors.txt based on the input spectral type
    if temperature is None:
        #Read pickles list
        dict_temp = dict()
        with open(filename_temp_lookup, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    pass
                else:
                    splitted_line = line.split()
                    # splitted_line[0]: spectral type F5 G0...
                    # splitted_line[2]: Temperature in K
                    dict_temp[splitted_line[0]] = splitted_line[2]

        try:
            target_temp = float(dict_temp[star_type_selec])
        except:
            if not mute:
                print("Returning None. Couldn't find a temperature for this spectral type in pickles mainseq_colors.txt.")
            return sampling_wvs,None
    else:
        target_temp = temperature

    # "AA_README" contains the list of the temperature for which a spectrum is available
    # Read it here
    dict_filename = dict()
    temp_list=[]
    with open(filename_pickles_lookup, 'r') as f:
        for line in f:
            if line.startswith('pickles_uk_'):
                splitted_line = line.split()
                # splitted_line[0]: Filename
                # splitted_line[1]: spectral type F5V G0III...
                # splitted_line[2]: Temperature in K

                #Check that the last character is numeric
                spec_type = splitted_line[1]
                if splitted_line[0][len(splitted_line[0])-1].isdigit() and not (spec_type.endswith('IV') or spec_type.endswith('I')):
                    dict_filename[float(splitted_line[2])] = splitted_line[0]
                    temp_list.append(float(splitted_line[2]))

    #temp_list = np.array(dict_filename.keys())
    temp_list = np.array(temp_list)
    # won't work for the hottest and coldest spectra.
    upper_temp, upper_temp_id = find_upper_nearest(temp_list,target_temp)
    lower_temp, lower_temp_id = find_lower_nearest(temp_list,target_temp)
    #print( upper_temp, upper_temp_id,lower_temp, lower_temp_id)

    upper_filename = dict_filename[upper_temp]
    lower_filename = dict_filename[lower_temp]

    upper_filename = pykliproot+os.path.sep+"pickles"+os.path.sep+upper_filename+".fits"
    lower_filename = pykliproot+os.path.sep+"pickles"+os.path.sep+lower_filename+".fits"


    hdulist = pyfits.open(upper_filename)
    cube = hdulist[1].data
    upper_wave = []
    upper_spec = []
    for wave_value,spec_value in cube:
        upper_wave.append(wave_value) # in angstrom
        upper_spec.append(spec_value)
    delta_wave = upper_wave[1]-upper_wave[0]
    upper_wave = np.array(upper_wave)/10**4 # in mum
    # upper_spec is a density spectrum in flux.A-1 so we need to multiply by delta_wave to integrate and get a flux.
    upper_spec = np.array(upper_spec)*delta_wave

    hdulist = pyfits.open(lower_filename)
    cube = hdulist[1].data
    lower_wave = []
    lower_spec = []
    for wave_value,spec_value in cube:
        lower_wave.append(wave_value) # in angstrom
        lower_spec.append(spec_value)
    lower_wave = np.array(lower_wave)/10**4 # in mum
    # lower_spec is a density spectrum in flux.A-1 so we need to multiply by delta_wave to integrate and get a flux.
    lower_spec = np.array(lower_spec)*delta_wave

    sampling_wvs_unique0 = np.insert(sampling_wvs_unique[:-1],0,sampling_wvs_unique[0])
    sampling_wvs_unique1 = np.insert(sampling_wvs_unique[1::],-1,sampling_wvs_unique[-1])
    upper_spec_unique = np.array([np.mean(upper_spec[np.where((upper_wave>wv0)*(upper_wave<wv1))]) for wv0,wv1 in zip(sampling_wvs_unique0,sampling_wvs_unique1)])
    lower_spec_unique = np.array([np.mean(lower_spec[np.where((lower_wave>wv0)*(lower_wave<wv1))]) for wv0,wv1 in zip(sampling_wvs_unique0,sampling_wvs_unique1)])

    # Sometimes the wavelength sampling is weird and the strategy above yields nans in the spectra.
    # When this happens we don't average out the spectra and takes the nearest available sample
    for k in range(np.size(upper_spec_unique)):
        if np.isnan(upper_spec_unique[k]):
            upper_spec_unique[k]= upper_spec[find_nearest(upper_wave,sampling_wvs_unique[k])[1]]
    for k in range(np.size(lower_spec_unique)):
        if np.isnan(lower_spec_unique[k]):
            lower_spec_unique[k]= lower_spec[find_nearest(lower_wave,sampling_wvs_unique[k])[1]]

    spec_pip_unique = ((target_temp-lower_temp)*upper_spec_unique+(upper_temp-target_temp)*lower_spec_unique)/(upper_temp-lower_temp)

    f = interp1d(sampling_wvs_unique, spec_pip_unique)
    spec_pip = f(sampling_wvs)

    return (sampling_wvs,spec_pip/np.nanmean(spec_pip))

def get_planet_spectrum(spectrum,wavelength,ori_wvs=None):
    """
    Get the normalized spectrum of a planet for a GPI spectral band or any wavelengths array.
    Spectra are extraced from .flx files from Mark Marley et al's models.

    Args:
        spectrum: Path of the .flx file containing the spectrum.
        wavelength: array of wavelenths in microns (or string with GPI band 'H', 'J', 'K1', 'K2', 'Y').
                (When using GPI spectral band wavelength samples are linearly spaced between the first and the last
                wavelength of the band.)

    Return:
        wavelengths: is the gpi sampling of the considered band in micrometer.
        spectrum: is the spectrum of the planet for the given band or wavelength array and normalized to unit mean.
    """


    if isinstance(spectrum, str):
        spec_data = []
        with open(spectrum, 'r') as f:
            for line in f:
                splitted_line = line.split()
                # splitted_line[0]: index
                # splitted_line[1]: wavelength (mum)
                # splitted_line[2]: T_brt
                # splitted_line[2]: flux in units of erg cm-2 sec-1 Hz-1 at the top of the planet's atmosphere

                try:
                    spec_data.append([float(splitted_line[0]),float(splitted_line[1]),float(splitted_line[2]),float(splitted_line[3])])
                except:
                    break

        spec_data = np.array(spec_data)
        N_samp = spec_data.shape[0]
        wave = spec_data[:,1]
        spec = spec_data[:,3]

    # Interpolate the spectrum on GPI sampling and convert F_nu to F_lambda
        spec = spec/wave**2
    else:
        wave = ori_wvs
        spec = spectrum


    # todo: check that it matches the actual sampling
    if isinstance(wavelength, str):
        import pyklip.instruments.GPI as GPI
        sampling_pip = GPI.get_gpi_wavelength_sampling(wavelength)
    else:
        sampling_pip = wavelength

    f = interp1d(wave, spec)
    # Interpolate the spectrum on GPI sampling and convert F_nu to F_lambda
    spec_pip = f(sampling_pip)/(sampling_pip**2)

    if 0:
        import matplotlib.pyplot as plt
        print((sampling_pip,spec_pip/np.nanmean(spec_pip)))
        plt.figure(2)
        wave_range = np.where((wave<sampling_pip[-1]) & (wave>sampling_pip[0]))
        plt.plot(wave[wave_range],spec[wave_range]/np.nanmean(spec[wave_range]),'r')
        plt.plot(sampling_pip,spec_pip/np.nanmean(spec_pip),'b.')
        plt.show()

    return (sampling_pip,spec_pip/np.nanmean(spec_pip))




def place_model_PSF(PSF_template,x_cen,y_cen,output_shape, x_grid = None, y_grid = None):

    ny_template, nx_template = PSF_template.shape
    if x_grid is None and y_grid is None:
        x_grid, y_grid = np.meshgrid(np.arange(0,output_shape[1],1),np.arange(0,output_shape[0],1))

    x_grid = x_grid.astype(np.float)
    y_grid = y_grid.astype(np.float)

    x_grid -= x_cen - nx_template//2
    y_grid -= y_cen - ny_template//2

    return ndimage.map_coordinates(PSF_template, [y_grid,x_grid], mode='constant', cval=0.0)

def LSQ_place_model_PSF(PSF_template,x_cen,y_cen,planet_image, x_grid = None, y_grid = None):
    model = place_model_PSF(PSF_template,x_cen,y_cen,planet_image.shape, x_grid = x_grid, y_grid = y_grid)
    return np.nansum((planet_image-model)**2,axis = (0,1))#/y_model


def extract_planet_centroid(cube, position, PSF_cube):


    nl,ny,nx = cube.shape
    row_id,col_id = position
    nl_PSF,ny_PSF,nx_PSF = PSF_cube.shape

    row_m = np.floor(ny_PSF/2.0)
    row_p = np.ceil(ny_PSF/2.0)
    col_m = np.floor(nx_PSF/2.0)
    col_p = np.ceil(nx_PSF/2.0)
    #print(np.max([0,(row_id-row_m)]),np.min([ny_PSF-1,(row_id+row_p)]),np.max([0,(col_id-col_m)]),np.min([nx_PSF-1,(col_id+col_p)]))
    cube_stamp = cube[:,np.max([0,(row_id-row_m)]):np.min([ny-1,(row_id+row_p)]), np.max([0,(col_id-col_m)]):np.min([nx-1,(col_id+col_p)])]
    flatCube_stamp = np.nansum(cube_stamp,axis=0)
    #plt.figure(5)
    #print(flatCube_stamp.shape,flatCube_stamp)
    #plt.imshow(flatCube_stamp,interpolation="nearest")
    #plt.show()
    flatCube_stamp /= np.nanmax(flatCube_stamp)
    flatPSF = np.nansum(PSF_cube,axis=0)
    flatPSF /= np.nanmax(flatPSF)

    nanargmax_flat_stamp= np.nanargmax(flatCube_stamp)
    max_row_id = np.floor(nanargmax_flat_stamp/nx_PSF)
    max_col_id = nanargmax_flat_stamp-nx_PSF*max_row_id

    param0 = (float(max_col_id),float(max_row_id)+1)

    LSQ_func = lambda para: LSQ_place_model_PSF(flatPSF,para[0],para[1],flatCube_stamp)
    param_fit = minimize(LSQ_func,param0, method="Nelder-Mead").x

    # if 0:
        # plt.figure(1)
        # plt.subplot(2,2,1)
        # plt.imshow(flatCube_stamp,interpolation="nearest")
        # plt.subplot(2,2,2)
        # plt.imshow(flatPSF,interpolation="nearest")
        # plt.subplot(2,2,3)
        # plt.imshow(flatCube_stamp-place_model_PSF(flatPSF,param0[0],param0[1],(ny_PSF,nx_PSF)),interpolation="nearest")
        # plt.subplot(2,2,4)
        # plt.imshow(flatCube_stamp-place_model_PSF(flatPSF,param_fit[0],param_fit[1],(ny_PSF,nx_PSF)),interpolation="nearest")
        # plt.show()

    return (param_fit[1]+row_id-row_m),(param_fit[0]+col_id-col_m)


def LSQ_scale_model_PSF(PSF_template,planet_image,a):
    return np.nansum((planet_image-a*PSF_template)**2,axis = (0,1))#/y_model

def extract_planet_spectrum(cube_para, position, PSF_cube_para, method = None,filter = None, mute = True):

    if isinstance(cube_para, basestring):
        hdulist = pyfits.open(cube_para)
        cube = hdulist[1].data
        exthdr = hdulist[1].header
        prihdr = hdulist[0].header
        hdulist.close()

        try:
            filter = prihdr['IFSFILT'].split('_')[1]
        except:
            if not mute:
                print("Couldn't find IFSFILT keyword in headers.")

    else:
        cube = cube_para

    if isinstance(PSF_cube_para, basestring):
        hdulist = pyfits.open(PSF_cube_para)
        PSF_cube = hdulist[1].data
        exthdr = hdulist[1].header
        prihdr = hdulist[0].header
        hdulist.close()

        sat_spot_spec = np.nanmax(PSF_cube,axis=(1,2))
        for l_id in range(PSF_cube.shape[0]):
            PSF_cube[l_id,:,:] /= sat_spot_spec[l_id]
    else:
        PSF_cube = PSF_cube_para

    row_cen,col_cen = extract_planet_centroid(cube, position, PSF_cube)

    nl,ny,nx = cube.shape
    row_id = np.round(row_cen)
    col_id = np.round(col_cen)
    #row_id,col_id = position
    nl_PSF,ny_PSF,nx_PSF = PSF_cube.shape

    row_m = np.floor(ny_PSF/2.0)
    row_p = np.ceil(ny_PSF/2.0)
    col_m = np.floor(nx_PSF/2.0)
    col_p = np.ceil(nx_PSF/2.0)
    cube_stamp = cube[:,np.max([0,(row_id-row_m)]):np.min([ny-1,(row_id+row_p)]), np.max([0,(col_id-col_m)]):np.min([nx-1,(col_id+col_p)])]
    nl_stamp, ny_stamp,nx_stamp = cube_stamp.shape


    stamp_x_grid, stamp_y_grid = np.meshgrid(np.arange(0,nx_stamp,1),np.arange(0,ny_stamp,1))
    r_stamp = np.sqrt((stamp_x_grid-(col_cen-(col_id-col_m)))**2 +(stamp_y_grid-(row_cen-(row_id-row_m)))**2)
    #stamp_mask = np.ones((stamp_nrow,stamp_ncol))
    #stamp_mask[np.where(r_stamp < 4.0)] = np.nan
    stamp_mask_small = np.ones((ny_stamp,nx_stamp))
    stamp_mask_small[np.where(r_stamp > 2.)] = np.nan
    stamp_cube_small_mask = np.tile(stamp_mask_small[None,:,:],(nl,1,1))


    # if 0:
        # plt.figure(6)
        # plt.subplot(2,2,1)
        # plt.imshow(stamp_mask_small,interpolation="nearest")
        # plt.subplot(2,2,2)
        # plt.imshow(cube_stamp[10,:,:]*stamp_mask_small,interpolation="nearest")
        # plt.show()

    if method is None or method == "max":
        spectrum = np.nanmax(cube_stamp*stamp_cube_small_mask,axis=(1,2))
    elif method == "aperture":
        spectrum = np.nansum(cube_stamp*stamp_cube_small_mask,axis=(1,2))
    elif method == "fit":
        spectrum = np.zeros((nl,))
        for k in range(nl):
            PSF_cube_slice = PSF_cube[k,:,:]/np.nanmax(PSF_cube[k,:,:])
            cube_stamp_slice = cube_stamp[k,:,:]
            param0 = np.nanmax(cube_stamp_slice)
            LSQ_func = lambda para: LSQ_scale_model_PSF(PSF_cube_slice,cube_stamp_slice,para)
            spectrum_fit[k] = minimize(LSQ_func,param0).x #, method="Nelder-Mead"
            # if 0:
                # plt.figure(1)
                # plt.subplot(2,2,1)
                # plt.imshow(cube_stamp_slice,interpolation="nearest")
                # plt.subplot(2,2,2)
                # plt.imshow(cube_stamp_slice*stamp_mask_small,interpolation="nearest")
                # plt.show()

    if 0:
        print(spectrum)
        # plt.figure(3)
        # plt.plot(get_gpi_wavelength_sampling(filter), spectrum)
        # plt.show()

    return get_gpi_wavelength_sampling(filter), spectrum
