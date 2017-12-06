__author__ = 'jruffio'
import os
import astropy.io.fits as pyfits
from glob import glob
import multiprocessing as mp
import numpy as np
from scipy.signal import convolve2d

from pyklip.kpp.utils.oi import *
from pyklip.kpp.utils.GPIimage import *
from pyklip.kpp.kppPerDir import *
import pyklip.kpp.stat.stat_utils as stat_utils


def calculate_contrast(read_func,nofakes_filename,fakes_filename_list,pix2as,
                       OI_list_folder=None,
                       mask_radius=None,
                       IOWA=None,
                       Dr=None,
                       save_dir = None,
                       suffix=None,
                       spec_type=None,
                       fakes_SNR_filename_list=None,
                       conversion_break = None,
                       linfit = False,MJDOBS=None):
    '''

    :param nofakes_filename:
    :param fakes_filename_list:
    :param GOI_list_folder:
    :param mask_radius:
    :param IOWA:
    :param Dr:
    :return:
    '''
    # Read the image using the user defined reading function
    image_obj = read_func([glob(nofakes_filename)[0]])
    metric_image = np.squeeze(image_obj.input)
    try:
        star_name = image_obj.object_name.replace(" ","_")
    except:
        star_name = None
    center = image_obj.centers[0]

    if IOWA is None:
        IWA,OWA,inner_mask,outer_mask = get_occ(metric_image, centroid = center)
        IOWA = (IWA,OWA)
        IOWA_as = (pix2as*IWA,pix2as*OWA)

    real_contrast_list = []
    sep_list = []
    pa_list = []
    metric_fakes_val = []
    metric_nofakes_val = []

    for fakes_filename in fakes_filename_list:
        image_fakes_obj = read_func([glob(fakes_filename)[0]])
        metric_image_fakes = np.squeeze(image_fakes_obj.input)
        hdulist = pyfits.open(glob(fakes_filename)[0])
        fakeinfohdr = None
        try:
            exthdr_fakes = hdulist[1].header
            if np.sum(["FKPA" in key for key in exthdr_fakes.keys()]):
                fakeinfohdr = exthdr_fakes
        except:
            pass
        try:
            prihdr_fakes = hdulist[0].header
            if np.sum(["FKPA" in key for key in prihdr_fakes.keys()]):
                fakeinfohdr = prihdr_fakes
        except:
            pass

        row_real_object_list,col_real_object_list = get_pos_known_objects(fakeinfohdr,star_name,pix2as,MJDOBS=MJDOBS,center=center,fakes_only=True)
        sep,pa = get_pos_known_objects(fakeinfohdr,star_name,pix2as,MJDOBS=MJDOBS,center=center,pa_sep=True,fakes_only=True)
        sep_list.extend(sep)
        pa_list.extend(pa)
        for fake_id in range(100):
            try:
                real_contrast_list.append(fakeinfohdr["FKCONT{0:02d}".format(fake_id)])
            except:
                continue
        for (row_real_object,col_real_object) in zip(row_real_object_list,col_real_object_list):
            try:
                metric_fakes_val.append(metric_image_fakes[int(np.round(row_real_object)),int(np.round(col_real_object))])
                metric_nofakes_val.append(metric_image[int(np.round(row_real_object)),int(np.round(col_real_object))])
            except:
                metric_fakes_val.append(np.nan)
                metric_nofakes_val.append(np.nan)

    metric_fakes_val =  np.array(metric_fakes_val) - np.array(metric_nofakes_val)

    if OI_list_folder is not None:
        metric_image_without_planet = mask_known_objects(metric_image,fakeinfohdr,star_name,MJDOBS=MJDOBS,center=center,
                                                         OI_list_folder=OI_list_folder, mask_radius = mask_radius)
    else:
        metric_image_without_planet = metric_image

    metric_1Dstddev,metric_stddev_rSamp = stat_utils.get_image_stddev(metric_image_without_planet,
                                                                     IOWA,
                                                                     N = None,
                                                                     centroid = center,
                                                                     r_step = Dr/2,
                                                                     Dr=Dr)
    metric_stddev_rSamp = np.array([r_tuple[0] for r_tuple in metric_stddev_rSamp])
    metric_1Dstddev = np.array(metric_1Dstddev)
    from  scipy.interpolate import interp1d
    metric_1Dstddev_func = interp1d(metric_stddev_rSamp,metric_1Dstddev,bounds_error=False, fill_value=np.nan)

    whereNoNans = np.where(np.isfinite(metric_fakes_val))
    metric_fakes_val = metric_fakes_val[whereNoNans]
    sep_list = np.array(sep_list)[whereNoNans]
    pa_list = np.array(pa_list)[whereNoNans]
    real_contrast_list =  np.array(real_contrast_list)[whereNoNans]

    sep_list,pa_list,metric_fakes_val,real_contrast_list = zip(*sorted(zip(sep_list,pa_list,metric_fakes_val,real_contrast_list)))
    metric_fakes_val = np.array(metric_fakes_val)
    sep_list =  np.array(sep_list)
    pa_list =  np.array(pa_list)
    real_contrast_list =  np.array(real_contrast_list)
    if linfit:
        if conversion_break is not None:
            whereInRange = np.where((np.array(sep_list)>IOWA_as[0])*(np.array(sep_list)<conversion_break))
            z1 = np.polyfit(np.array(sep_list)[whereInRange],np.array(real_contrast_list)[whereInRange]/np.array(metric_fakes_val)[whereInRange],1)

            whereInRange = np.where((np.array(sep_list)>conversion_break)*(np.array(sep_list)<IOWA_as[1]))
            z2 = np.polyfit(np.array(sep_list)[whereInRange],np.array(real_contrast_list)[whereInRange]/np.array(metric_fakes_val)[whereInRange],1)

            linfit1 = np.poly1d(z1)
            linfit2 = np.poly1d(z2)
            metric_conversion_func = lambda sep: np.concatenate((linfit1(np.array(sep)[np.where(np.array(sep)<conversion_break)]),
                                                                 linfit2(np.array(sep)[np.where(conversion_break<np.array(sep))])))

        else:
            whereInRange = np.where((np.array(sep_list)>IOWA_as[0])*(np.array(sep_list)<IOWA_as[1]))
            z = np.polyfit(np.array(sep_list)[whereInRange],np.array(real_contrast_list)[whereInRange]/np.array(metric_fakes_val)[whereInRange],1)
            metric_conversion_func = np.poly1d(z)
    else:
        whereInRange = np.where((sep_list>IOWA_as[0])*(sep_list<IOWA_as[1]))
        metric_fakes_in_range = metric_fakes_val[whereInRange]
        cont_in_range = real_contrast_list[whereInRange]
        unique_sep = np.unique(sep_list[whereInRange])
        med_conversion = np.zeros(len(unique_sep))
        std_conversion = np.zeros(len(unique_sep))
        for k,sep_it in enumerate(unique_sep):
            where_sep = np.where(sep_list==sep_it)
            # med_conversion[k] = np.nanmedian(cont_in_range[where_sep]) / np.nanmedian(metric_fakes_in_range[where_sep])
            # std_conversion[k] = np.nanmedian(np.abs(cont_in_range[where_sep]/metric_fakes_in_range[where_sep] - med_conversion[k]))
            med_conversion[k] = np.nanmean(cont_in_range[where_sep]/metric_fakes_in_range[where_sep])
            var_conversion = np.nanvar(cont_in_range[where_sep] / metric_fakes_in_range[where_sep])
            std_conversion[k] = np.sqrt(var_conversion)
        metric_conversion_func = interp1d(unique_sep,med_conversion,bounds_error=False, fill_value=np.nan)
        # std_conversion_func = interp1d(unique_sep,1.4826*std_conversion,bounds_error=False, fill_value=np.nan)
        std_conversion_func = interp1d(unique_sep,std_conversion,bounds_error=False, fill_value=np.nan)


    if 0:
        import matplotlib.pyplot as plt
        plt.figure(2)
        plt.title(suffix)
        plt.plot(sep_list,np.array(metric_fakes_val)/np.array(real_contrast_list),"*")
        plt.plot(sep_list,metric_conversion_func(sep_list),"-")
        plt.xlabel("Separation (arcsec)", fontsize=20)
        plt.ylabel("Throughput (arbritrary units)", fontsize=20)
        ax= plt.gca()
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        plt.show()


    contrast_curve = 5*metric_1Dstddev*metric_conversion_func(pix2as*metric_stddev_rSamp)

    if fakes_SNR_filename_list is not None:
        SNR_real_contrast_list = []
        SNR_sep_list = []
        SNR_fakes=[]

        for fakes_SNR_filename in fakes_SNR_filename_list:
            SNR_map_fakes_obj = read_func([glob(fakes_SNR_filename)[0]])

            SNR_map_fakes = np.squeeze(SNR_map_fakes_obj.input)
            hdulist = pyfits.open(glob(fakes_SNR_filename)[0])
            fakeinfohdr_SNR = None
            try:
                exthdr_fakes = hdulist[1].header
                if np.sum(["FKPA" in key for key in exthdr_fakes.keys()]):
                    fakeinfohdr_SNR = exthdr_fakes
            except:
                pass
            try:
                prihdr_fakes = hdulist[0].header
                if np.sum(["FKPA" in key for key in prihdr_fakes.keys()]):
                    fakeinfohdr_SNR = prihdr_fakes
            except:
                pass

            row_real_object_list,col_real_object_list = get_pos_known_objects(fakeinfohdr_SNR,star_name,pix2as,MJDOBS=MJDOBS,center=center,fakes_only=True)
            sep,pa = get_pos_known_objects(fakeinfohdr_SNR,star_name,pix2as,MJDOBS=MJDOBS,center=center,pa_sep=True,fakes_only=True)
            SNR_sep_list.extend(sep)
            for fake_id in range(100):
                try:
                    SNR_real_contrast_list.append(fakeinfohdr_SNR["FKCONT{0:02d}".format(fake_id)])
                except:
                    continue
            for (row_real_object,col_real_object) in zip(row_real_object_list,col_real_object_list):
                try:
                    SNR_fakes.append(SNR_map_fakes[int(np.round(row_real_object)),int(np.round(col_real_object))])
                except:
                    SNR_fakes.append(np.nan)
            # SNR_fakes.extend([np.nanmax(SNR_map_fakes[(int(np.round(row_real_object))-1):(int(np.round(row_real_object))+2),(int(np.round(col_real_object))-1):(int(np.round(col_real_object))+2)]) \
            #                              for row_real_object,col_real_object in zip(row_real_object_list,col_real_object_list)])
            # # SNR_fakes.extend([SNR_map_fakes[np.round(row_real_object),np.round(col_real_object)] \
            # #                              for row_real_object,col_real_object in zip(row_real_object_list,col_real_object_list)])

        from scipy.interpolate import interp1d
        contrast_curve_interp = interp1d(pix2as*metric_stddev_rSamp,contrast_curve,kind="linear",bounds_error=False)
        SNR_from_contrast = np.array(SNR_real_contrast_list)/(contrast_curve_interp(SNR_sep_list)/5.0)

        with open(os.path.join(save_dir,"contrast-SNR-check-"+suffix+'.csv'), 'w+') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=' ')
            csvwriter.writerows([["Seps","ContSNR","PixSNR","contrast"]])
            csvwriter.writerows(zip(SNR_sep_list,SNR_from_contrast,SNR_fakes,SNR_real_contrast_list))



    if save_dir is not None:
        if suffix is None:
            suffix = "default"
        with open(os.path.join(save_dir,"contrast-"+suffix+'.csv'), 'w+') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=' ')
            csvwriter.writerows([["Seps",spec_type,spec_type+"_met_std",spec_type+"_conv",spec_type+"_conv_std"]])
            contrast_curve[np.where(np.isnan(contrast_curve))] = -1.
            not_neg = np.where(contrast_curve>0)
            csvwriter.writerows(zip(pix2as*metric_stddev_rSamp[not_neg],
                                    contrast_curve[not_neg],
                                    metric_1Dstddev[not_neg],
                                    metric_conversion_func(pix2as*metric_stddev_rSamp[not_neg]),
                                    std_conversion_func(pix2as*metric_stddev_rSamp[not_neg])))

        with open(os.path.join(save_dir,"conversion-"+suffix+'.csv'), 'w+') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=' ')
            csvwriter.writerows([["Seps","PA","conversion","fit","metric","contrast","kMAD"]])
            csvwriter.writerows(zip(sep_list,pa_list,
                                    np.array(real_contrast_list)/np.array(metric_fakes_val),
                                    metric_conversion_func(sep_list),
                                    np.array(metric_fakes_val),
                                    np.array(real_contrast_list),
                                    std_conversion_func(sep_list)))

    conversion_tuple = (sep_list,np.array(metric_fakes_val)/np.array(real_contrast_list),np.array(metric_fakes_val),np.array(real_contrast_list))

    return pix2as*metric_stddev_rSamp,contrast_curve,conversion_tuple