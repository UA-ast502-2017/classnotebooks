__author__ = 'JB'


import numpy as np
from copy import copy
from  glob import glob
import csv
import os

def mask_known_objects(cube,fakeinfohdr,object_name,pix2as,center,MJDOBS=None,OI_list_folder=None,ignore_fakes = False,fakes_only = False,
                          include_speckles = False,IWA=None,OWA=None, mask_radius = 7):
    """

    Mask point sources in cube with an NaN aperture.

    Args:
        cube: Image or cube to be masked.
        fakeinfohdr: fits file header containing the injected planets related keywords.
        object_name: Name of the star being observed.
        pix2as: platescale.
        center: Center of the image.
        MJDOBS: Julian date of the observation. (needed when OI_list_folder is not None)
        OI_list_folder: List of Object of Interest (OI) that should be masked from any standard deviation
                        calculation. See the online documentation for instructions on how to define it.
        xy: Boolean. Returns the planets coordinate with x,y coordinates in pixels
        pa_sep: Boolean. Returns the planets coordinates as position angle (in arcsec), separation (in pix)
        ignore_fakes: Don't return fake planets.
        fakes_only: Returns only fake planets.
        include_speckles: Include speckles from the list of object of interest (OI_list_folder)
        IWA: Inner working angle (in pixels).
        OWA: Outer working angle (in pixels).
        mask_radius: Radius of the mask in pixels. (default = 7 pixels)

    Return: Masked cube.
    """

    row_vec,col_vec = get_pos_known_objects(fakeinfohdr,object_name,pix2as,center=center,MJDOBS=MJDOBS,OI_list_folder=OI_list_folder,
                          xy = False,pa_sep = False,ignore_fakes = ignore_fakes,fakes_only = fakes_only,
                          include_speckles = include_speckles,IWA=IWA,OWA=OWA)

    cube_cpy = copy(cube)

    if np.size(cube_cpy.shape) == 3:
        nl,ny,nx = cube_cpy.shape
    elif np.size(cube_cpy.shape) == 2:
        ny,nx = cube_cpy.shape
        cube_cpy = cube_cpy[None,:]
        nl = 1

    width = 2*mask_radius+1
    stamp_x_grid, stamp_y_grid = np.meshgrid(np.arange(0,width,1)-width/2,np.arange(0,width,1)-width/2)
    stamp_mask = np.ones((width,width))
    r_stamp = abs((stamp_x_grid) +(stamp_y_grid)*1j)
    stamp_mask[np.where(r_stamp < mask_radius)] = np.nan

    row_m = int(np.floor(width/2.0))
    row_p = int(np.ceil(width/2.0))
    col_m = int(np.floor(width/2.0))
    col_p = int(np.ceil(width/2.0))

    for row,col in zip(row_vec,col_vec):
        k = int(round(row))
        l = int(round(col))

        cube_cpy[:,(k-row_m):(k+row_p), (l-col_m):(l+col_p)] = np.tile(stamp_mask,(nl,1,1)) * cube_cpy[:,(k-row_m):(k+row_p), (l-col_m):(l+col_p)]

    return np.squeeze(cube_cpy)


def get_pos_known_objects(fakeinfohdr,object_name,pix2as,center=None,MJDOBS=None,OI_list_folder=None,xy = False,pa_sep = False,ignore_fakes = False,fakes_only = False,
                          include_speckles = False,IWA=None,OWA=None):
    """
    
    Return the position of real and/or simulated planets in an image based on its headers.

    Args:
        fakeinfohdr: fits file header containing the injected planets related keywords.
        object_name: Name of the star being observed.
        pix2as: platescale.
        center: Center of the image (not needed if pa_sep = True).
        MJDOBS: Julian date of the observation. (needed when OI_list_folder is not None)
        OI_list_folder: List of Object of Interest (OI) that should be masked from any standard deviation
                        calculation. See the online documentation for instructions on how to define it.
        xy: Boolean. Returns the planets coordinate with x,y coordinates in pixels
        pa_sep: Boolean. Returns the planets coordinates as position angle (in arcsec), separation (in pix)
        ignore_fakes: Don't return fake planets.
        fakes_only: Returns only fake planets.
        include_speckles: Include speckles from the list of object of interest (OI_list_folder)
        IWA: Inner working angle (in pixels).
        OWA: Outer working angle (in pixels).

    Return: (sep_vec,pa_vec) or (x_vec,y_vec) or (row_vec,col_vec) (default)
        Objects coordinates (real/fakes/others). For e.g., sep_vec,pa_vec are vectors and (sep_vec[0],pa_vec[0]) is the
        coordinate of the first object and so on...
    """
    x_vec = []
    y_vec = []
    col_vec = []
    row_vec = []
    pa_vec = []
    sep_vec = []


    if object_name is not None:
        object_name = object_name.replace(" ","_")

        if not fakes_only and MJDOBS is not None and OI_list_folder is not None:
            object_GOI_filename = OI_list_folder+os.path.sep+object_name+'_GOI.csv'
            if len(glob(object_GOI_filename)) != 0:
                with open(object_GOI_filename, 'rb') as csvfile_GOI_list:
                    GOI_list_reader = csv.reader(csvfile_GOI_list, delimiter=';')
                    GOI_csv_as_list = list(GOI_list_reader)
                    attrib_name = GOI_csv_as_list[0]
                    GOI_list = np.array(GOI_csv_as_list[1:len(GOI_csv_as_list)])

                    pa_id = attrib_name.index("PA")
                    sep_id = attrib_name.index("SEP")
                    MJDOBS_id = attrib_name.index("MJDOBS")
                    STATUS_id = attrib_name.index("STATUS")

                    MJDOBS_arr = np.array([ float(it) for it in GOI_list[:,MJDOBS_id]])
                    MJDOBS_unique = np.unique(MJDOBS_arr)
                    MJDOBS_closest_id = np.argmin(np.abs(MJDOBS_unique-MJDOBS))
                    MJDOBS_closest = MJDOBS_unique[MJDOBS_closest_id]
                    #Check that the closest MJDOBS is closer than 2 hours
                    if abs(MJDOBS_closest-MJDOBS) > 2./24.:
                        # Skip if we couldn't find a matching date.
                        return [],[]

                    for obj_id in np.where(MJDOBS_arr == MJDOBS_closest)[0]:
                        try:
                            pa = float(GOI_list[obj_id,pa_id])
                            radius = float(GOI_list[obj_id,sep_id])
                            if IWA is not None:
                                if 1./pix2as*radius < IWA:
                                    continue
                            if OWA is not None:
                                if 1./pix2as*radius > OWA:
                                    continue
                            status = str(GOI_list[obj_id,STATUS_id])
                            # print(status, include_speckles or (status in ["Planet","Background","Candidate","Unknown","Brown Dwarf"]))
                            if include_speckles or (status in ["Planet","Background","Candidate","Unknown","Brown Dwarf"]):
                                pa_vec.append(pa)
                                sep_vec.append(radius)
                                x_max_pos = float(1./pix2as*radius)*np.cos(np.radians(90+pa))
                                y_max_pos = float(1./pix2as*radius)*np.sin(np.radians(90+pa))
                                x_vec.append(x_max_pos)
                                y_vec.append(y_max_pos)
                                row_vec.append(y_max_pos+center[1])
                                col_vec.append(x_max_pos+center[0])
                        except:
                            print("Missing data in GOI database for {0}".format(object_name))

    if not ignore_fakes:
        for fake_id in range(100):
            try:
                pa = fakeinfohdr["FKPA{0:02d}".format(fake_id)]
                radius = pix2as*fakeinfohdr["FKSEP{0:02d}".format(fake_id)]
                if IWA is not None:
                    if 1./pix2as*radius < IWA:
                        continue
                if OWA is not None:
                    if 1./pix2as*radius > OWA:
                        continue
                pa_vec.append(pa)
                sep_vec.append(radius)
                x_max_pos = float(1./pix2as*radius)*np.cos(np.radians(90+pa))
                y_max_pos = float(1./pix2as*radius)*np.sin(np.radians(90+pa))
                x_vec.append(x_max_pos)
                y_vec.append(y_max_pos)
                row_vec.append(y_max_pos+center[1])
                col_vec.append(x_max_pos+center[0])
            except:
                continue


    if pa_sep:
        return sep_vec,pa_vec
    elif xy:
        return x_vec,y_vec
    else:
        return row_vec,col_vec


def make_GOI_list(outputDir,GOI_list_csv,GPI_TID_csv):
    """
    Generate the GOI files from the GOI table and the TID table (queried from the database).

    outputDir: Output directory in which to save the GOI files.
    GOI_list_csv: Table with the list of GOIs (including separation, PA...)
    GPI_TID_csv: Table giving the TID code for a given object name.
    :return: One .csv file per target for which at list one GOI exists.
            The filename follows: [object]_GOI.csv. For e.g. c_Eri_GOI.csv.
    """
    with open(GOI_list_csv, 'rb') as csvfile_GOI_list:
        GOI_list_reader = csv.reader(csvfile_GOI_list, delimiter=';')
        GOI_csv_as_list = list(GOI_list_reader)
        attrib_name = GOI_csv_as_list[0]
        GOI_list = np.array(GOI_csv_as_list[1:len(GOI_csv_as_list)])

        TID_id = attrib_name.index("TID")
        TID_GOI = GOI_list[:,TID_id]
        TID_unique = np.unique(TID_GOI)

        with open(GPI_TID_csv, 'rb') as csvfile_TID:
            TID_reader = csv.reader(csvfile_TID, delimiter=';')
            TID_csv_as_list = list(TID_reader)
            TID_csv_as_nparr = np.array(TID_csv_as_list)[1:len(TID_csv_as_list),:]
            TID_campaign = np.ndarray.tolist(TID_csv_as_nparr[:,0])
            star_campaign = np.ndarray.tolist(TID_csv_as_nparr[:,1])
            #print(TID_campaign)
            #print(star_campaign)

            dict_matching_TID_to_name = {}
            for TID_it in TID_unique:
                star_raw_name = star_campaign[TID_campaign.index(TID_it)]
                star_name = star_raw_name.replace(" ","_")
                dict_matching_TID_to_name[TID_it] = star_name
            #print(dict_matching_TID_to_name)

        for TID_it in TID_unique:
            where_same_star = np.where(TID_GOI == TID_it)

            with open(outputDir+os.path.sep+dict_matching_TID_to_name[TID_it]+'_GOI.csv', 'w+') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=';')
                table_to_csv = [attrib_name]+np.ndarray.tolist(GOI_list[where_same_star[0],:])#.insert(0,attrib_name)
                #print(table_to_csv)
                csvwriter.writerows(table_to_csv)