
import os
import glob
from copy import copy

import numpy as np
from scipy import optimize
import scipy.ndimage as ndimage
import scipy.interpolate as interp
from scipy.optimize import minimize
from scipy.interpolate import interp1d



import pyklip.spectra_management as spec


def convert_pa_to_image_polar(pa, astr_hdr):
    """
    Given a position angle (angle to North through East), calculate what
    polar angle theta (angle from +X CCW towards +Y) it corresponds to

    Args:
        pa: position angle in degrees
        astr_hdr: wcs astrometry header (astropy.wcs)

    Returns:
        theta: polar angle in degrees
    """
    rot_det = astr_hdr.wcs.cd[0,0] * astr_hdr.wcs.cd[1,1] - astr_hdr.wcs.cd[0,1] * astr_hdr.wcs.cd[1,0]
    if rot_det < 0:
        rot_sgn = -1.
    else:
        rot_sgn = 1.
    # calculate CCW rotation from +Y to North in radians
    rot_YN = np.arctan2(rot_sgn * astr_hdr.wcs.cd[0,1],rot_sgn * astr_hdr.wcs.cd[0,0])
    # now that we know where north it,
    # find the CCW rotation from +Y to find location of planet
    rot_YPA = rot_YN - rot_sgn*pa*np.pi/180. #radians
    # rot_YPA = rot_YN + pa*np.pi/180. #radians

    theta = rot_YPA * 180./np.pi + 90.0 #degrees
    return theta


def convert_polar_to_image_pa(theta, astr_hdr):
    """
    Reversed engineer from covert_pa_to_image_polar by JB. Actually JB doesn't quite understand how it works...

    Args:
        theta: parallactic angle in degrees
        astr_hdr: wcs astrometry header (astropy.wcs)

    Returns:
        theta: polar angle in degrees
    """
    rot_det = astr_hdr.wcs.cd[0,0] * astr_hdr.wcs.cd[1,1] - astr_hdr.wcs.cd[0,1] * astr_hdr.wcs.cd[1,0]
    if rot_det < 0:
        rot_sgn = -1.
    else:
        rot_sgn = 1.
    #calculate CCW rotation from +Y to North in radians
    rot_YN = np.arctan2(rot_sgn * astr_hdr.wcs.cd[0,1],rot_sgn * astr_hdr.wcs.cd[0,0])

    rot_YPA = (theta-90)*np.pi/180.

    pa = rot_sgn*(rot_YN-rot_YPA)* 180./np.pi

    return pa


def _inject_gaussian_planet(frame, xpos, ypos, amplitude, fwhm=3.5, stampsize=None):
    """
    Injects a fake planet with a Gaussian PSF into a dataframe

    Args:
        frame: a 2D data frame
        xpos,ypos: x,y location (in pixels) where the planet should be
        amplitude: peak of the Gaussian PSf (in appropriate units not dictacted here)
        fwhm: fwhm of gaussian
        stampsize: integer specfying the width of the stamp box in which the planet is defined

    Returns:
        frame: the frame with the injected planet
    """
    if stampsize is None:
        stampsize = 3 * fwhm
    # convert boxsize to an integer
    stampsize = int(np.ceil(stampsize))

    # figure out sigma when given FWHM
    sigma = fwhm/(2.*np.sqrt(2*np.log(2)))

    # create a coordinate system for the PSF centered about the closest pixel to the planet
    y, x = np.indices([stampsize, stampsize])
    y -= stampsize // 2 # center about 0
    x -= stampsize // 2  # center about 0
    # find nearest pixel to center coordinate around
    x_int = int(round(xpos))
    y_int = int(round(ypos))
    x += x_int
    y += y_int

    xmin = x[0][0]
    xmax = x[-1][-1]
    ymin = y[0][0]
    ymax = y[-1][-1]

    psf = amplitude * np.exp(-((x - xpos)**2. + (y - ypos)**2.) / (2. * sigma**2))

    frame[ymin:ymax+1, xmin:xmax+1] += psf
    return frame


def inject_planet(frames, centers, inputflux, astr_hdrs, radius, pa, fwhm=3.5, thetas=None, stampsize=None):
    """
    Injects a fake planet into a dataset either using a Gaussian PSF or an input PSF

    Args:
        frames: array of (N,y,x) for N is the total number of frames
        centers: array of size (N,2) of [x,y] coordiantes of the image center
        inputflux: EITHER array of size N of the peak flux of the fake planet in each frame (will inject a Gaussian PSF)
                   OR array of size (N,psfy,psfx) of template PSFs. The brightnesses should be scaled and the PSFs
                   should be centered at the center of each of the template images
        astr_hdrs: array of size N of the WCS headers
        radius: separation of the planet from the star
        pa: position angle (in degrees) of  planet
        fwhm: fwhm (in pixels) of gaussian
        thetas: ignore PA, supply own thetas (CCW angle from +x axis toward +y)
                array of size N
        stampsize: in pixels, the width of the square stamp to inject the image into. Defaults to 3*fwhm if None

    Returns:
        saves result in input "frames" variable
    """

    if thetas is None:
        thetas = np.array([convert_pa_to_image_polar(pa, astr_hdr) for astr_hdr in astr_hdrs])

    if (np.size(inputflux) == 1):
        #input is probably a number and we want an array
        inputflux = np.ones(frames.shape[0]) * inputflux

    for frame, center, inputpsf, theta in zip(frames, centers, inputflux, thetas):
        #calculate the x,y location of the planet for each image
        #theta = covert_pa_to_image_polar(pa, astr_hdr)
        x_pl = radius * np.cos(theta*np.pi/180.) + center[0]
        y_pl = radius * np.sin(theta*np.pi/180.) + center[1]

        ny,nx = frame.shape

        #now that we found the planet location, inject it
        #check whether we are injecting a gaussian of a template PSF
        if isinstance(inputpsf, np.ndarray):
            # stampsize should defautl to minimum dimension of PSF
            if stampsize is None:
                stampsize = int(np.min(inputpsf.shape))
            # convert boxsize to an integer
            stampsize = int(np.ceil(stampsize))

            #shift psf so that center is aligned
            #calculate center of box
            boxsize = inputpsf.shape[0]
            #Using JB's array convention instead.
            boxcent = boxsize // 2

            # create a coordinate system for the PSF centered about the closest pixel to the planet
            ystamp, xstamp = np.indices([stampsize, stampsize])
            ystamp -= stampsize // 2  # center about 0
            xstamp -= stampsize // 2  # center about 0
            # find nearest pixel to center coordinate around
            x_int = int(round(x_pl))
            y_int = int(round(y_pl))
            xstamp += x_int
            ystamp += y_int
            # index bounds
            xmin = xstamp[0][0]
            xmax = xstamp[-1][-1]
            ymin = ystamp[0][0]
            ymax = ystamp[-1][-1]

            if xmin>= nx or ymin>= ny or xmax<= 0 or ymax<= 0:
                continue
            # find corresponding pixels in the PSF
            xpsf = xstamp - x_pl + boxcent
            ypsf = ystamp - y_pl + boxcent

            # Crop the edge if injection at the edge of the image
            if xmin < 0:
                ypsf = ypsf[:,-xmin::]
                xpsf = xpsf[:,-xmin::]
                xmin=0
            if ymin < 0:
                ypsf = ypsf[-ymin::,:]
                xpsf = xpsf[-ymin::,:]
                ymin = 0
            if xmax >= nx:
                ypsf = ypsf[:,:-(xmax-nx + 1)]
                xpsf = xpsf[:,:-(xmax-nx + 1)]
                xmax = nx
            if ymax >= ny:
                ypsf = ypsf[:-(ymax-ny + 1),:]
                xpsf = xpsf[:-(ymax-ny + 1),:]
                ymax = ny

            #inject into frame
            frame[ymin:ymax + 1, xmin:xmax + 1] += ndimage.map_coordinates(inputpsf, [ypsf, xpsf], mode='constant', cval=0.0)
        else:
            if stampsize is None:
                stampsize = 3 * fwhm
            # convert boxsize to an integer
            stampsize = int(np.ceil(stampsize))

            _inject_gaussian_planet(frame, x_pl, y_pl, inputpsf, fwhm=fwhm, stampsize=stampsize)


def generate_dataset_with_fakes(dataset, fake_position_dict, fake_flux_dict, spectrum = None, PSF_cube = None, PSF_cube_wvs = None,
                                star_type = None, mute = False, SpT_file_csv = None, real_planets_pos = None, sep_skip_real_pl = None,
                                pa_skip_real_pl = None,dn_per_contrast=None):
    '''
    Generate spectral datacubes with fake planets.
    It will do a copy of the cubes read in GPIData after having injected fake planets in them.
    This new set of cubes can then be reduced in the same manner as the campaign data.

    Doesn't work with remove slice: assumes that the dataset is made of a list of similar datacubes or images.

    Args:
        dataset: An object of type GPIData.
                The fakes are injected directly into dataset so you should make a copy of dataset prior to running this
                function.
                In order for the function to query simbad for the spectral type of the star, the attribute object_name needs
                to be defined in dataset.
        fake_position_dict:
                Dictionary defining the way the fake planets are positionned
                - fake_position_dict["mode"]="sector": Put a planet in each klip sector. Can actually generate several
                        datasets in which the planets will be shifted in separation and position angle with respect to one
                        another.
                        It can be usefull for fake based contrast curve calculation.
                        Several parameters needs to be defined.
                        - fake_position_dict["annuli"]: Number of annulis in the image
                        - fake_position_dict["subsections"]: Number of angular sections in the image
                        - fake_position_dict["sep_shift"]: separation shift from the center of the sectors
                        - fake_position_dict["pa_shift"]: position angle shift from the center of the sectors
                - fake_position_dict["mode"]="custom": Put planets at given (separation, position angle).
                        The following parameter needs to be defined
                        - fake_position_dict["pa_sep_list"]: List of tuple [(r1,pa1),(r2,pa2),...] with each tuple giving
                                the separation and position angle of each planet to be injected.
                - fake_position_dict["mode"]="ROC": Generate fake for ROC curves calculation. Use hard-coded parameters.
            fake_flux_dict:
                Dictionary defining the way in which the flux of the fake is defined.
                - fake_flux_dict["mode"]="contrast": Defines the contrast value of the fakes.
                        - fake_flux_dict["contrast"]: Contrast of the fake planets
                - fake_flux_dict["mode"]="SNR": Defines the brightness of the fakes relatively to the satellite spots.
                        - fake_flux_dict["SNR"]: SNR wished for the fake planets.
                        - fake_flux_dict["sep_arr"]: Separation sampling of the contrast curve in pixels.
                        - fake_flux_dict["contrast_arr"]: 5 sigma contrast curve (planet to star ratio).
        PSF_cube: the psf of the image. A numpy array with shape (wv, y, x)
        PSF_cube_wvs: the wavelegnths that correspond to the input psfs
        spectrum: spectrum name (string) or array
                    - "host_star_spec": The spectrum from the star or the satellite spots is directly used.
                                        It is derived from the inverse of the calibrate_output() output.
                    - "constant": Use a constant spectrum np.ones(nl).
                    - other strings: name of the spectrum file in #pykliproot#/spectra/*/ with pykliproot the
                                    directory in which pyklip is installed. It that case it should be a spectrum
                                    from Mark Marley or one following the same convention.
                                    Spectrum will be corrected for transmission.
                    - ndarray: 1D array with a user defined spectrum. Spectrum will be corrected for transmission.
        star_type: Spectral type of the current star. If None, Simbad is queried.
        mute: If True prevent printed log outputs.
        suffix: Suffix to be added at the end of the filenames.
        SpT_file_csv: Filename of the table (.csv) contaning the spectral type of the stars.
        real_planets_pos: list of position of real point sources in the dataset that should be avoided when injecting fakes.
                        [(sep1,pa1),(sep2,pa2),...] with the separation in pixels and the position angle in degrees.
        sep_skip_real_pl: Limit in seperation of how close a fake can be injected of a known GOI.
        pa_skip_real_pl: Limit in position angle  of how close a fake can be injected of a known GOI.
        dn_per_contrast: array of the same size as spectrum giving the conversion between the peak flux of a planet in
                        data number and its contrast.

    '''

    if sep_skip_real_pl is None:
        sep_skip_real_pl = 20
    if pa_skip_real_pl is None:
        pa_skip_real_pl = 90

    try:
        star_name = dataset.object_name.replace(" ","_")
    except:
        star_name = "noname"

    nl, ny, nx = dataset.input.shape
    if dn_per_contrast is None:
        dn_per_contrast = 1./dataset.calibrate_output(np.ones((nl,1,1)),spectral=True).squeeze()


    # Make sure the total flux of each PSF is unity for all wavelengths
    # So the peak value won't be unity.
    # print("np.sum(PSF_cube)",np.sum(PSF_cube))
    PSF_cube = PSF_cube/np.nansum(PSF_cube,axis=(1,2))[:,None,None]
    # print("np.sum(PSF_cube)",np.sum(PSF_cube))
    # Get the conversion factor from peak spectrum to aperture based spectrum
    aper_over_peak_ratio = 1/np.nanmax(PSF_cube,axis=(1,2))
    aper_over_peak_ratio_tiled = np.zeros(nl)#wavelengths
    for k,wv in enumerate(dataset.wvs):
        aper_over_peak_ratio_tiled[k] = aper_over_peak_ratio[spec.find_nearest(PSF_cube_wvs,wv)[1]]
    # Summed DN flux of the star in the entire dataset calculated from dn_per_contrast
    host_star_spec = aper_over_peak_ratio_tiled*dn_per_contrast
    star_flux = np.sum(host_star_spec)
    # print(star_flux,aper_over_peak_ratio_tiled[0],dn_per_contrast[0],aper_over_peak_ratio)
    # # exit()
    host_star_spec = host_star_spec/np.mean(host_star_spec)
    nl_psf, ny_psf, nx_psf = PSF_cube.shape
    inputpsfs = np.zeros((nl,ny_psf,nx_psf))
    for k,wv in enumerate(dataset.wvs):
        inputpsfs[k,:,:] = PSF_cube[spec.find_nearest(PSF_cube_wvs,wv)[1],:,:]


    if np.size(np.unique(dataset.wvs)) == 1:
        star_sp = np.ones(dn_per_contrast.shape)
    else:
        if star_type is None:
            star_type = spec.get_specType(star_name,SpT_file_csv)
        # Interpolate a spectrum of the star based on its spectral type/temperature
        wv,star_sp = spec.get_star_spectrum(dataset.wvs,star_type)

    # Define the output Foldername
    if isinstance(spectrum, str):

        # Do the best it can with the spectral information given in inputs.
        if spectrum == "host_star_spec":
            # If spectrum_filename is an empty string the function takes the sat spot spectrum by default.
            if not mute:
                print("Default host star specrum will be used.")
            spectrum_vec = copy(host_star_spec)
            spectrum_name = "host_star_spec"
        elif spectrum == "constant":
            if not mute:
                print("Spectrum is not or badly defined so taking flat spectrum")
            spectrum_vec = np.ones(nl)
            spectrum_name = "constant"
        else :
            pykliproot = os.path.dirname(os.path.realpath(spec.__file__))
            spectrum_filename = os.path.abspath(glob.glob(os.path.join(pykliproot,"spectra","*",spectrum+".flx"))[0])
            spectrum_name = spectrum_filename.split(os.path.sep)
            spectrum_name = spectrum_name[len(spectrum_name)-1].split(".")[0]

            # spectrum_filename is not empty it is assumed to be a valid path.
            if not mute:
                print("Spectrum model: "+spectrum_filename)
            # Interpolate the spectrum of the planet based on the given filename
            wv,planet_sp = spec.get_planet_spectrum(spectrum_filename,dataset.wvs)

            # Correct the ideal spectrum given in spectrum_filename for atmospheric and instrumental absorption.
            spectrum_vec = (host_star_spec/star_sp)*planet_sp

    elif isinstance(spectrum, np.ndarray):
        planet_sp = spectrum
        spectrum_name = "custom"

        # Correct the ideal spectrum given in spectrum_filename for atmospheric and instrumental absorption.
        spectrum_vec = (host_star_spec/star_sp)*planet_sp
    else:
        raise ValueError("Invalid spectrum: {0}".format(spectrum))

    spectrum_vec = spectrum_vec/np.mean(spectrum_vec)

    if real_planets_pos is not None:
        sep_real_object_list = [sep for (sep,pa) in real_planets_pos] # in pixels
        pa_real_object_list = [pa for (sep,pa) in real_planets_pos] # in degrees

    # inputflux_is_def = False
    # Build the list of separation and position angles for the fakes
    if fake_position_dict["mode"] == "custom":
        sep_pa_iter_list = fake_position_dict["pa_sep_list"]

    if fake_position_dict["mode"] == "spirals":
        if "pa_shift" in fake_position_dict:
            pa_shift = fake_position_dict["pa_shift"]
        else:
            pa_shift = 0.0
        # Calculate the radii of the annuli like in klip_adi_plus_sdi using the first image
        # We want to inject one planet per section where klip is independently applied.
        if "annuli" in fake_position_dict:
            annuli = fake_position_dict["annuli"]
        else:
            annuli = 8
        if "dr" in fake_position_dict:
            dr = fake_position_dict["dr"]
        else:
            dr = 15.
        delta_th = 90

        # Get parallactic angle of where to put fake planets
        # PSF_dist = 20 # Distance between PSFs. Actually length of an arc between 2 consecutive PSFs.
        # delta_pa = 180/np.pi*PSF_dist/radius
        pa_list = np.arange(-180.,180.-0.01,delta_th) + pa_shift
        radii_list = np.array([dr * annuli_it + dataset.IWA + 2.5 for annuli_it in range(annuli)])
        pa_grid, radii_grid = np.meshgrid(pa_list,radii_list)
        # for row_id in range(pa_grid.shape[0]):
        #     pa_grid[row_id,:] = pa_grid[row_id,:] + 30
        for col_id in range(radii_grid.shape[1]):
            radii_grid[:,col_id] = radii_grid[:,col_id] + dr/4*np.mod(col_id,4)
        pa_grid[range(1,annuli,3),:] += 30
        pa_grid[range(2,annuli,3),:] += 60
        pa_grid = pa_grid + 5

        #sep_pa_iter_list = zip(np.reshape(radii_grid,np.size(radii_grid)),np.reshape(pa_grid,np.size(pa_grid)))
        sep_pa_iter_list = [(r, p) for r_arr, p_arr in zip(radii_grid, pa_grid) for r, p in zip(r_arr, p_arr)]

    if fake_position_dict["mode"] == "sector":
        annuli = fake_position_dict["annuli"]
        subsections = fake_position_dict["subsections"]
        sep_shift = fake_position_dict["sep_shift"]
        pa_shift = fake_position_dict["pa_shift"]

        # Calculate the radii of the annuli like in klip_adi_plus_sdi using the first image
        # We want to inject one planet per section where klip is independently applied.
        dims = dataset.input.shape
        x_grid, y_grid = np.meshgrid(np.arange(dims[2] * 1.0), np.arange(dims[1] * 1.0))
        nanpix = np.where(np.isnan(dataset.input[0]))
        if np.size(nanpix) == 0:
            OWA = np.sqrt(np.max((x_grid) ** 2 + (y_grid) ** 2))
        else:
            OWA = np.sqrt(np.min((x_grid[nanpix] - dataset.centers[0][0]) ** 2 + (y_grid[nanpix] - dataset.centers[0][1]) ** 2))
        dr = float(OWA - dataset.IWA) / (annuli)
        delta_th = 360./subsections

        # Get parallactic angle of where to put fake planets
        # PSF_dist = 20 # Distance between PSFs. Actually length of an arc between 2 consecutive PSFs.
        # delta_pa = 180/np.pi*PSF_dist/radius
        pa_list = np.arange(-180.,180.-0.01,delta_th) + pa_shift
        radii_list = np.array([dr * annuli_it + dataset.IWA + dr/2.for annuli_it in range(annuli-1)]) + sep_shift
        pa_grid, radii_grid = np.meshgrid(pa_list,radii_list)
        pa_grid[range(1,annuli-1,2),:] += delta_th/2.

        #sep_pa_iter_list = zip(np.reshape(radii_grid,np.size(radii_grid)),np.reshape(pa_grid,np.size(pa_grid)))
        sep_pa_iter_list = [(r, p) for r_arr, p_arr in zip(radii_grid, pa_grid) for r, p in zip(r_arr, p_arr)]

    # fake_flux_dict = dict(mode = "SNR",sep_arr = sep_samples, contrast_arr=Ttype_contrast)
    if (fake_flux_dict["mode"] == "contrast"):
        if isinstance(fake_flux_dict["contrast"], list) or isinstance(fake_flux_dict["contrast"], np.ndarray):
            planets_contrasts = fake_flux_dict["contrast"]
        else:
            planets_contrasts = [fake_flux_dict["contrast"],]*len(sep_pa_iter_list)
    elif (fake_flux_dict["mode"] == "SNR"):
        sep_arr = np.array(fake_flux_dict["sep_arr"])
        cont_arr = np.array(fake_flux_dict["contrast_arr"])
        f = interp1d(sep_arr[np.where(np.isfinite(cont_arr))], cont_arr[np.where(np.isfinite(cont_arr))],
                     bounds_error=False,fill_value=np.nanmin(fake_flux_dict["contrast_arr"]))
        planets_contrasts = [fake_flux_dict["SNR"]*f(sep)/5. for (sep,pa) in sep_pa_iter_list]

    extra_keywords = {}
    # Loop for injecting fake planets. One planet per section of the image.
    for fake_id, ((radius,pa),contrast) in enumerate(zip(sep_pa_iter_list,planets_contrasts)):
        x_max_pos = radius*np.cos(np.radians(90+pa))
        y_max_pos = radius*np.sin(np.radians(90+pa))

        # Not injecting the planet if too close to real object
        if real_planets_pos is not None:
            too_close = False
            for sep_real_object,pa_real_object  in zip(sep_real_object_list,pa_real_object_list):
                delta_angle = np.min([np.abs(np.mod(pa,360)-np.mod(pa_real_object,360)),
                               np.min([np.mod(pa,360),np.mod(pa_real_object,360)])+360-np.max([np.mod(pa,360),np.mod(pa_real_object,360)])])
                if np.abs(sep_real_object-radius) < sep_skip_real_pl and delta_angle < pa_skip_real_pl:
                    too_close = True
                    if not mute:
                        print("Skipping planet. Real object too close.")
                    break
            if too_close:
                continue


        spectrum_corr = spectrum_vec/np.sum(spectrum_vec)*star_flux*contrast
        inputpsfs = inputpsfs/np.nansum(inputpsfs,axis=(1,2))[:,None,None]
        inputpsfs = inputpsfs*spectrum_corr[:,None,None]

        try:
            if not mute:
                print("injecting planet position ("+str(radius)+"pix,"+str(pa)+"degree)")
            # inject fake planet at given radius,pa into dataset.input
            inject_planet(dataset.input, dataset.centers, inputpsfs, dataset.wcs, radius, pa,
                          stampsize=np.min([ny_psf, nx_psf]), thetas=pa+dataset.PAs)

            # Save fake planet position in headers
            extra_keywords["FKPA{0:02d}".format(fake_id)] = pa
            extra_keywords["FKSEP{0:02d}".format(fake_id)] = radius
            extra_keywords["FKCONT{0:02d}".format(fake_id)] = contrast
            extra_keywords["FKPOSX{0:02d}".format(fake_id)] = x_max_pos
            extra_keywords["FKPOSY{0:02d}".format(fake_id)] = y_max_pos
            extra_keywords["FKSPEC{0:02d}".format(fake_id).format(fake_id)] = spectrum_name
        except OverflowError:
            pass
        # except:
        #     if not mute:
        #         print("Failed to inject planet position ("+str(radius)+"pix,"+str(pa)+"degree)")

    return dataset,extra_keywords


def _construct_gaussian_disk(x0,y0, xsize,ysize, intensity, angle, fwhm=3.5):
    """
    Constructs a rectangular slab for a disk with a vertical gaussian profile

    Args:
        x0,y0: center of disk
        xsize, ysize: x and y dimensions of the output image
        intensity: peak intensity of the disk (whatever units you want)
        angle: orientation of the disk plane (CCW from +x axis) [degrees]
        fwhm: FWHM of guassian profile (in pixels)

    Returns:
        disk_img: 2d array of size (ysize,xsize) with the image of the disk
    """

    #construct a coordinate system
    x,y = np.meshgrid(np.arange(xsize*1.0), np.arange(ysize*1.0))

    #center at image center
    x -= x0
    y -= y0

    #rotate so x is parallel to the disk plane, y is vertical cuts through the disk
    #so need to do a CW rotation
    rad_angle = angle * np.pi/180.
    xp = x * np.cos(rad_angle) + y * np.sin(rad_angle) + x0
    yp = -x * np.sin(rad_angle) + y * np.cos(rad_angle) + y0

    sigma = fwhm/(2 * np.sqrt(2*np.log(2)))
    disk_img = intensity / (np.sqrt(2*np.pi) * sigma) * np.exp(-(yp-y0)**2/(2*sigma**2))

    return disk_img


def inject_disk(frames, centers, inputfluxes, astr_hdrs, pa, fwhm=3.5):
    """
    Injects a fake disk into a dataset

    Args:
        frames: array of (N,y,x) for N is the total number of frames
        centers: array of size (N,2) of [x,y] coordiantes of the image center
        intputfluxes: array of size N of the peak flux of the fake disk in each frame OR
                      array of 2-D models (North up East left) to inject into the data.
                            (Disk is assumed to be centered at center of image)
        astr_hdrs: array of size N of the WCS headers
        pa: position angles angle (in degrees) of disk plane
        fwhm: if injecting a Gaussian disk (i.e inputfluxes is an array of floats), fwhm of Gaussian

    Returns:
        saves result in input "frames" variable
    """

    for frame, center, inputpsf, astr_hdr in zip(frames, centers, inputfluxes, astr_hdrs):
        #calculate the rotation angle in the pixel plane
        theta = convert_pa_to_image_polar(pa, astr_hdr)

        if isinstance(inputpsf, np.ndarray):
            # inject real data
            # rotate and grab pixels of disk that can be injected into the image
            # assume disk is centered
            xpsf0 = inputpsf.shape[1]/2
            ypsf0 = inputpsf.shape[0]/2
            #grab the pixel numbers for the data
            ximg, yimg = np.meshgrid(np.arange(frame.shape[1]), np.arange(frame.shape[0]))
            #rotate them to extract the disk at the right angle
            ximg -= center[0]
            yimg -= center[1]
            theta_rad = np.radians(theta)
            ximgp = ximg * np.cos(theta_rad) + yimg * np.sin(theta_rad) + xpsf0
            yimgp = -ximg * np.sin(theta_rad) + yimg * np.cos(theta_rad) + ypsf0
            #interpolate and inject datqa
            frame += ndimage.map_coordinates(inputpsf, [yimgp, ximgp])
        else:
            #inject guassian bar into data
            frame += _construct_gaussian_disk(center[0], center[1], frame.shape[1], frame.shape[0], inputpsf, theta, fwhm=fwhm)


def gauss2d(x0, y0, peak, sigma):
    """
    2d symmetric guassian function for guassfit2d

    Args:
        x0,y0: center of gaussian
        peak: peak amplitude of guassian
        sigma: stddev in both x and y directions
    """
    sigma *= 1.0
    return lambda y,x: peak*np.exp( -(((x-x0)/sigma)**2+((y-y0)/sigma)**2)/2)


def gaussfit2d(frame, xguess, yguess, searchrad=5, guessfwhm=3, guesspeak=1, refinefit=True):
    """
    Fits a 2d gaussian to the data at point (xguess, yguess)

    Args:
        frame: the data - Array of size (y,x)
        xguess,yguess: location to fit the 2d guassian to (should be pretty accurate)
        searchrad: 1/2 the length of the box used for the fit
        guessfwhm: approximate fwhm to fit to
        guesspeak: approximate flux
        refinefit: whether to refine the fit of the position of the guess

    Returns:
        peakflux: the peakflux of the gaussian
        fwhm: fwhm of the PFS in pixels
        xfit: x position (only chagned if refinefit is True)
        yfit: y position (only chagned if refinefit is True)
    """
    if not isinstance(searchrad, int):
        raise ValueError("searchrad needs to be an integer")

    x0 = np.rint(xguess).astype(int)
    y0 = np.rint(yguess).astype(int)
    #construct our searchbox
    fitbox = np.copy(frame[y0-searchrad:y0+searchrad+1, x0-searchrad:x0+searchrad+1])

    #mask bad pixels
    fitbox[np.where(np.isnan(fitbox))] = 0
 
    #fit a least squares gaussian to refine the fit on the source, otherwise just use the guess
    if refinefit:
        #construct the residual to the fit
        errorfunction = lambda p: np.ravel(gauss2d(*p)(*np.indices(fitbox.shape)) - fitbox)
   
        #do a least squares fit. Note that we use searchrad for x and y centers since we're narrowed it to a box of size
        #(2searchrad+1,2searchrad+1)

        guess = (searchrad, searchrad, guesspeak, guessfwhm/(2 * np.sqrt(2*np.log(2))))

        p, success = optimize.leastsq(errorfunction, guess)

        xfit = p[0]
        yfit = p[1]
        peakflux = p[2]
        fwhm = p[3] * (2 * np.sqrt(2*np.log(2)))
    else:
        xfit = xguess-x0 + searchrad
        yfit = yguess-y0 + searchrad
        fwhm = guessfwhm
   
    #ok now, to really calculate fwhm and flux, because we really need that right, we're going
    # to use what's in the GPI DRP pipeline to measure satellite spot fluxes instead of
    # a least squares gaussian fit. Apparently my least squares fit relatively underestimates
    # the flux so it's not consistent.
    # grab a radial profile of the fit
    rs = np.linspace(0, searchrad+1, np.max([15, searchrad+1])) # should always have at least pixel resolution
    thetas = np.arange(0,2*np.pi, 1./searchrad) #divide maximum circumfrence into equal parts
    radprof = [np.mean(ndimage.map_coordinates(fitbox, [thisr*np.sin(thetas)+yfit, thisr*np.cos(thetas)+xfit])) for thisr in rs]
    #now interpolate this radial profile to get fwhm
    try:
        radprof_interp = interp.interp1d(radprof, rs)
        fwhm = 2*radprof_interp(np.max(radprof[0])/2)
    except ValueError:
        # use old FWHM
        pass


    #now calculate flux
    xfitbox, yfitbox = np.meshgrid(np.arange(0,2* searchrad+1, 1.0)-xfit, np.arange(0, 2*searchrad+1, 1.0)-yfit)
    #correlate data with a gaussian to get flux
    sigma = fwhm/(2*np.sqrt(2*np.log(2)))
    ## attempt to calculate sigma using moments
    #sigmax = np.sqrt(np.nansum(xfitbox*xfitbox*fitbox)/np.nansum(fitbox) - (np.nansum(xfitbox*fitbox)/np.nansum(fitbox))**2)
    #sigmay = np.sqrt(np.nansum(yfitbox*yfitbox*fitbox)/np.nansum(fitbox) - (np.nansum(yfitbox*fitbox)/np.nansum(fitbox))**2)
    #sigma = np.nanmean([sigmax, sigmay])
    #print(sigma, sigmax, sigmay)
    gmask = np.exp(-(xfitbox**2+yfitbox**2)/(2.*sigma**2))
    outofaper = np.where(xfitbox**2 + yfitbox**2 > searchrad**2)
    gmask[outofaper] = 0 
    corrflux = np.nansum(fitbox*gmask)/np.sum(gmask*gmask)

    if not np.isfinite(corrflux):
        # if it's infinite, it is bad
        corrflux = np.nan

    # convert xfit, yfit back to image coordinates
    xfit = xfit - searchrad + x0
    yfit = yfit - searchrad + y0

    return corrflux, fwhm, xfit, yfit


def LSQ_gauss2d(planet_image, x_grid, y_grid,a,x_cen,y_cen,sig):
    """
    Calculate the squared norm of the residuals of the model with the data.
    Helper function for least square fit.
    The model is a 2d symmetric gaussian.

    Args:
        planet_image: stamp image (y,x) of the satellite spot.
        x_grid: x samples grid as given by meshgrid.
        y_grid: y samples grid as given by meshgrid.
        a: amplitude of the 2d gaussian
        x_cen: x center of the gaussian
        y_cen: y center of the gaussian
        sig: standard deviation of the gaussian

    Returns:
        Squared norm of the residuals
    """
    #gauss2d(x, y, amplitude = 1.0, xo = 0.0, yo = 0.0, sigma_x = 1.0, sigma_y = 1.0, theta = 0, offset = 0)
    model = gauss2d(x_cen,y_cen, a, sig)(x_grid, y_grid)
    #model = gauss2d(x_grid, y_grid,a,x_cen,y_cen,sig,sig,0.0)
    return np.nansum((planet_image-model)**2,axis = (0,1))#/y_model


def PSFcubefit(frame, xguess, yguess, searchrad=10,psfs_func_list=None,wave_index=None,residuals=False):
    """
    Estimate satellite spot amplitude (peak value) by fitting a symmetric 2d gaussian.
    Fit parameters: x,y position, amplitude, standard deviation (same in x and y direction)

    Args:
        frame: the data - Array of size (y,x)
        xguess: x location to fit the 2d guassian to.
        yguess: y location to fit the 2d guassian to.
        searchrad: 1/2 the length of the box used for the fit
        psfs_func_list: List of spline fit function for the PSF_cube.
        wave_index: Index of the current wavelength. In [0,36] for GPI. Only used when psfs_func_list is not None.
        residuals: If True (Default = False) then calculate the residuals of the sat spot fit (gaussian or PSF cube).

    Returns:
        returned_flux: scalar, Estimation of the peak flux of the satellite spot.
            ie Amplitude of the fitted gaussian.
    """
    x0 = int(np.round(xguess))
    y0 = int(np.round(yguess))
    #construct our searchbox
    fitbox = np.copy(frame[y0-searchrad:y0+searchrad+1, x0-searchrad:x0+searchrad+1])

    xguess_box = xguess-x0 + searchrad
    yguess_box = yguess-y0 + searchrad

    xfitbox, yfitbox = np.meshgrid(np.arange(0,2* searchrad+1, 1.0)-xguess_box, np.arange(0, 2*searchrad+1, 1.0)-yguess_box)
    stamp_r = np.sqrt((xfitbox)**2+(yfitbox)**2)
    small_aper_indices = np.where(stamp_r<3)
    big_aper_indices = np.where(stamp_r<7)

    # try to remove background
    if 1:
        stamp_masked = copy(fitbox)
        stamp_x_masked = copy(xfitbox)
        stamp_y_masked = copy(yfitbox)
        stamp_masked[big_aper_indices] = np.nan
        stamp_x_masked[big_aper_indices] = np.nan
        stamp_y_masked[big_aper_indices] = np.nan
        background_med =  np.nanmedian(stamp_masked)
        stamp_masked = stamp_masked - background_med
        #Solve 2d linear fit to remove background
        xx = np.nansum(stamp_x_masked**2)
        yy = np.nansum(stamp_y_masked**2)
        xy = np.nansum(stamp_y_masked*stamp_x_masked)
        xz = np.nansum(stamp_masked*stamp_x_masked)
        yz = np.nansum(stamp_y_masked*stamp_masked)
        #Cramer's rule
        a = (xz*yy-yz*xy)/(xx*yy-xy*xy)
        b = (xx*yz-xy*xz)/(xx*yy-xy*xy)
        fitbox = fitbox - (a*(xfitbox)+b*(yfitbox) + background_med)

    if isinstance(wave_index,(np.ndarray)):
        # Get a deprecation warning when wave_index = [5] instead of an integer. So this picks the integer...
        new_wave_index = wave_index[0]
    else:
        new_wave_index = wave_index
    model = psfs_func_list[new_wave_index](np.arange(0,2* searchrad+1, 1.0)-xguess_box,np.arange(0, 2*searchrad+1, 1.0)-yguess_box).transpose()
    # model = psfs_func_list[wave_index](np.arange(0,2* searchrad+1, 1.0)+(xguess-x0) - searchrad,np.arange(0, 2*searchrad+1, 1.0)+(yguess-y0) - searchrad)#.transpose()

    returned_flux = np.sum(model[small_aper_indices]*fitbox[small_aper_indices])/np.sum(model[small_aper_indices]**2)*model[searchrad,searchrad]

    if residuals:
        residuals_map = fitbox - returned_flux*model/model[searchrad,searchrad]
        return returned_flux,residuals_map
    else:
        return returned_flux


def gaussfit2dLSQ(frame, xguess, yguess, searchrad=5,fit_centroid = False,residuals=False):
    """
    Estimate satellite spot amplitude (peak value) by fitting a symmetric 2d gaussian.
    Fit parameters: x,y position, amplitude, standard deviation (same in x and y direction)

    Args:
        frame: the data - Array of size (y,x)
        xguess: x location to fit the 2d guassian to.
        yguess: y location to fit the 2d guassian to.
        searchrad: 1/2 the length of the box used for the fit
        fit_centroid: If False (default), disable the centroid fit and only fit the amplitude and the standard deviation
        residuals: If True (Default = False) then calculate the residuals of the sat spot fit (gaussian or PSF cube).

    Returns:
        returned_flux: scalar, estimation of the peak flux of the satellite spot.
            ie Amplitude of the fitted gaussian.
    """
    x0 = int(np.round(xguess))
    y0 = int(np.round(yguess))
    #construct our searchbox
    fitbox = np.copy(frame[y0-searchrad:y0+searchrad+1, x0-searchrad:x0+searchrad+1])

    xguess_box = xguess-x0 + searchrad
    yguess_box = yguess-y0 + searchrad

    xfitbox, yfitbox = np.meshgrid(np.arange(0,2* searchrad+1, 1.0)-xguess_box, np.arange(0, 2*searchrad+1, 1.0)-yguess_box)

    if fit_centroid:
        param0 = [fitbox[searchrad,searchrad],0.,0.,1.5]
        LSQ_func = lambda para: LSQ_gauss2d(fitbox,xfitbox, yfitbox,para[0],para[1],para[2],para[3])

        param_fit = minimize(LSQ_func,param0, method="Nelder-Mead").x

        if (param_fit[1]**2+param_fit[2]**2)>2.**2 or (param_fit[0]/param0[0] >10.) or param_fit[3] > 3. or param_fit[3] < 0.5:
            returned_flux = np.nan
        else:
            returned_flux = param_fit[0]

        if residuals:
            model = gauss2d(param_fit[1],param_fit[2], param_fit[0], param_fit[3])(xfitbox, yfitbox)
            residuals_map = fitbox - model
            return returned_flux,residuals_map
        else:
            return returned_flux
    else:
        param0 = [fitbox[searchrad,searchrad],1.5]
        LSQ_func = lambda para: LSQ_gauss2d(fitbox,xfitbox, yfitbox,para[0],0,0,para[1])

        param_fit = minimize(LSQ_func,param0, method="Nelder-Mead").x

        # print(param_fit[1])
        if abs(param_fit[0] - fitbox[searchrad,searchrad]) > 0.5*fitbox[searchrad,searchrad]:
            returned_flux = np.nan
        else:
            returned_flux = param_fit[0]

        if residuals:
            model = gauss2d(0,0, param_fit[0], param_fit[1])(xfitbox, yfitbox)
            residuals_map = fitbox - model
            return returned_flux,residuals_map
        else:
            return returned_flux


def retrieve_planet_flux(frames, centers, astr_hdrs, sep, pa, searchrad=7, guessfwhm=3.0, guesspeak=1, refinefit=False,
                         thetas=None):
    """
    Retrives the planet flux from a series of frames given a separation and PA

    Args:
        frames: frames of data to retrieve planet. Can be a single 2-D image ([y,x]) for a series/cube ([N,y,x])
        centers: coordiantes of the image center. Can be [2]-element lst or an array that matches array of frames [N,2]
        astr_hdrs: astr_hdrs, can be a single one or an array of N of them
        sep: radial distance in pixels
        PA: parallactic angle in degrees
        searchrad: 1/2 the length of the box used for the fit
        guessfwhm: approximate fwhm to fit to
        guesspeak: approximate flux
        refinefit: whether or not to refine the positioning of the planet
        thetas: ignore PA, supply own thetas (CCW angle from +x axis toward +y)
                single number or array of size N

    Returns:
        peakflux: either a single peak flux or an array depending on whether a single frame or multiple frames
                    where passed in
    """
    measured = retrieve_planet(frames, centers, astr_hdrs, sep, pa, searchrad, guessfwhm, guesspeak, refinefit, thetas)

    if np.ndim(measured) == 1:
        # just one frame, return one number
        return measured[0]
    else:
        # return an array of fluxes
        return measured[:, 0]


def retrieve_planet(frames, centers, astr_hdrs, sep, pa, searchrad=7, guessfwhm=3.0, guesspeak=1, refinefit=True,
                    thetas=None):
    """
    Retrives the planet properties from a series of frames given a separation and PA

    Args:
        frames: frames of data to retrieve planet. Can be a single 2-D image ([y,x]) for a series/cube ([N,y,x])
        centers: coordiantes of the image center. Can be [2]-element lst or an array that matches array of frames [N,2]
        astr_hdrs: astr_hdrs, can be a single one or an array of N of them
        sep: radial distance in pixels
        PA: parallactic angle in degrees
        searchrad: 1/2 the length of the box used for the fit
        guessfwhm: approximate fwhm to fit to
        guesspeak: approximate flux
        refinefit: whether or not to refine the positioning of the planet
        thetas: ignore PA, supply own thetas (CCW angle from +x axis toward +y)
                single number or array of size N

    Returns:
        measured: (peakflux, x, y, fwhm). A single tuple if one frame passed in. Otherwise an array of tuples
    """

    # check to make sure all arguments are the right/consistent dimensions
    # get the number of dimensions of frames as reference for all the other variables
    frames_ndim = np.ndim(frames)
    if frames_ndim == 2:
        # make sure all other arguments are consistent in dimensions
        if np.ndim(centers) != 1:
            raise IndexError("centers needs to be a 2-element list")
        if np.ndim(astr_hdrs) != 0:
            raise IndexError("astr_hdrs cannot be a list because you only passed in one frame")
        if thetas is not None:
            if np.ndim(thetas) != 0:
                raise IndexError("thetas cannot be a list because you only passed in one frame")
                thetas = [thetas]
        # turn them into lists so we can reuse the same code
        frames = [frames]
        centers = [centers]
        astr_hdrs = [astr_hdrs]
    elif frames_ndim == 3:
        # make sure all other arguments are consistent in dimensions
        if np.ndim(centers) != 2:
            raise IndexError("centers needs to an array of [x,y] coordinates")
        if np.ndim(astr_hdrs) != 1:
            raise IndexError("astr_hdrs must be a list because you only passed in multiple frames")
        if thetas is not None:
            if np.ndim(thetas) != 1:
                raise IndexError("thetas must be a list because you only passed in multiple frames")
                thetas = [thetas]
    else:
        raise IndexError("frames is either 2-D or 3-D, not {0)-D".format(frames_ndim))

    measured = []
   
    if thetas is None:
        thetas = np.array([convert_pa_to_image_polar(pa, astr_hdr) for astr_hdr in astr_hdrs])

    # loop over all of them
    for frame, center, theta in zip(frames, centers, thetas):
        # find the pixel location on this image
        # theta = covert_pa_to_image_polar(pa, astr_hdr)
        x = sep*np.cos(np.radians(theta)) + center[0]
        y = sep*np.sin(np.radians(theta)) + center[1]
        # calculate the flux
        flux, fwhm, xfit, yfit = gaussfit2d(frame, x, y, searchrad=searchrad, guessfwhm=guessfwhm, guesspeak=guesspeak, refinefit=refinefit)
        measured.append((flux, xfit, yfit, fwhm))

    # return a single number if onyl one frame passed in
    if frames_ndim == 2:
        measured = measured[0]
    else:
        measured = np.array(measured)

    return measured
