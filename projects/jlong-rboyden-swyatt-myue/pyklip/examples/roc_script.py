#!/home/anaconda/bin/python2.7
__author__ = 'jruffio'

import platform
import os
import sys
import shutil
from time import time
import datetime

import multiprocessing as mp
import astropy.io.fits as pyfits

from pyklip.kpp.metrics.FMMF import FMMF
from pyklip.kpp.stat.statPerPix_utils import *
from pyklip.kpp.stat.stat import Stat
from pyklip.kpp.metrics.crossCorr import CrossCorr
from pyklip.kpp.metrics.matchedfilter import Matchedfilter
from pyklip.kpp.kpop_wrapper import kpop_wrapper
from pyklip.kpp.utils.oi import *
from pyklip.kpp.detection.detection import Detection
from pyklip.kpp.detection.ROC import ROC
import pyklip.fakes as fakes

def roc_dataset(inputDir,dir_fakes,mvt,reduc_spectrum,fakes_spectrum):

    numbasis=[5]
    maxnumbasis = 10
    #contrast of the fake planets
    contrast = 4.0*(10**-6)
    # Less important parameters
    mute = False # Mute print statements
    resolution = 3.5 # FWHM of the PSF used for the small sample statistic correction
    mask_radius = 5 # (Pixels) Radius of the disk used to mask out objects in an image
    overwrite = False # Force rewriting the files even if they already exist

    ###########################################################################################
    ## Generate PSF cube
    ###########################################################################################
    from pyklip.instruments import GPI

    # Generate PSF cube for GPI from the satellite spots
    filenames = glob(os.path.join(inputDir,"S*distorcorr.fits"))
    dataset = GPI.GPIData(filenames,highpass=True)
    dataset.generate_psf_cube(20,same_wv_only=True)
    PSF_cube = inputDir + os.path.sep + "beta_Pic_test"+"-original_PSF_cube.fits"
    # Save the original PSF calculated from combining the sat spots
    dataset.savedata(PSF_cube, dataset.psfs, filetype="PSF Spec Cube")

    ###########################################################################################
    ## Reduce the dataset with FMMF
    ###########################################################################################
    # This section will take for ever due to the FMMF reduction
    print("~~ FMMF and SNR ~~")

    # Define the function to read the dataset files.
    # In order to show a more advanced feature, in this case the satellite spots fluxes
    # are reestimated using the PSF cube calculated previously
    raw_read_func = lambda file_list:GPI.GPIData(file_list,
                                             meas_satspot_flux=True,
                                             numthreads = None,
                                             highpass=True,
                                             PSF_cube="*-original_PSF_cube.fits")
    filename = "S*distorcorr.fits"
    # Define the function to read the PSF cube file.
    # You can choose to directly give an array as the PSF cube and not bother with this.
    PSF_read_func = lambda file_list:GPI.GPIData(file_list,highpass=False)
    FMMFObj = FMMF(raw_read_func,filename = filename,
                   spectrum=reduc_spectrum,
                   PSF_read_func = PSF_read_func,
                   PSF_cube = PSF_cube,
                   PSF_cube_wvs = None,
                   predefined_sectors = "1.0 as",
                   label = "FMMF",
                   overwrite=overwrite,
                   numbasis=numbasis,
                   maxnumbasis = maxnumbasis,
                   mvt=mvt)

    err_list = kpop_wrapper(inputDir,[FMMFObj],spectrum_list=[reduc_spectrum],mute_error = False)

    read_func = lambda file_list:GPI.GPIData(file_list,recalc_centers=False,recalc_wvs=False,highpass=False)

    # Definition of the SNR object
    filename = os.path.join("kpop_FMMF",reduc_spectrum,"*{0:0.2f}-FM*-KL{1}.fits".format(mvt,numbasis[0]))
    FMMF_snr_obj = Stat(read_func,filename,type="pixel based SNR",
                   overwrite=overwrite,mute=mute,resolution=resolution,mask_radius=mask_radius,
                           pix2as = GPI.GPIData.lenslet_scale)
    filename = os.path.join("kpop_FMMF",reduc_spectrum,"*{0:0.2f}-FM*-KL{1}-SNRPerPixDr2.fits".format(mvt,numbasis[0]))
    FMMF_detec_obj = Detection(read_func,filename,mask_radius = None,threshold = 2,overwrite=overwrite,mute=mute,IWA=9,OWA=1./0.01414,
                           pix2as = GPI.GPIData.lenslet_scale)

    err_list = kpop_wrapper(inputDir,[FMMF_snr_obj,FMMF_detec_obj],mute_error=False)

    ###########################################################################################
    ## Add Cross correlation and Matched Filter reductions
    filename = os.path.join("kpop_FMMF",reduc_spectrum,
                                       "*_{0:0.2f}-speccube-KL{1}.fits".format(mvt,numbasis[0]))
    cc_obj = CrossCorr(read_func,filename,kernel_type="gaussian",kernel_para=1.0,
                       collapse=True,spectrum=reduc_spectrum,folderName=reduc_spectrum,overwrite=overwrite)
    mf_obj = Matchedfilter(read_func,filename,kernel_type="gaussian",kernel_para=1.0,label="pyklip",
                       spectrum=reduc_spectrum,folderName=reduc_spectrum,overwrite=overwrite)

    filename = os.path.join("kpop_FMMF",reduc_spectrum,
                                       "*_{0:0.2f}-speccube-KL{1}-*gaussian.fits".format(mvt,numbasis[0]))
    klip_snr_obj = Stat(read_func,filename,type="pixel based SNR",
                   overwrite=overwrite,mute=mute,resolution=resolution,mask_radius=mask_radius,
                           pix2as = GPI.GPIData.lenslet_scale)
    filename = os.path.join("kpop_FMMF",reduc_spectrum,
                                       "*_{0:0.2f}-speccube-KL{1}-*gaussian-SNRPerPixDr2.fits".format(mvt,numbasis[0]))
    klip_detec_obj = Detection(read_func,filename,mask_radius = None,threshold = 2,overwrite=overwrite,mute=mute,IWA=9,OWA=1./0.01414,
                           pix2as = GPI.GPIData.lenslet_scale)

    err_list = kpop_wrapper(inputDir,[cc_obj,mf_obj,klip_snr_obj,klip_detec_obj],mute_error=False)


    ###########################################################################################
    ## Inject fake planet for the fakes
    ###########################################################################################
    # Fixed contrast fakes
    # Load the PSF cube that has been calculated from the sat spots
    PSF_cube = glob(os.path.join(inputDir,"*-original_PSF_cube.fits"))[0]
    PSF_cube_obj = PSF_read_func([PSF_cube])
    PSF_cube_arr = PSF_cube_obj.input
    PSF_cube_wvs = PSF_cube_obj.wvs

    if not os.path.exists(dir_fakes):
        os.makedirs(dir_fakes)
    shutil.copyfile(PSF_cube,os.path.join(dir_fakes,os.path.basename(PSF_cube)))

    fake_flux_dict = dict(mode = "contrast",contrast = contrast)
    fake_position_dict = dict(mode = "spirals")
    spdc_glob = glob(inputDir+os.path.sep+"S*_spdc_distorcorr.fits")

    dataset = GPI.GPIData(spdc_glob,highpass=True,meas_satspot_flux=True,numthreads=mp.cpu_count(),PSF_cube = PSF_cube_arr)

    dataset,extra_keywords = fakes.generate_dataset_with_fakes(dataset,
                                                             fake_position_dict,
                                                             fake_flux_dict,
                                                             spectrum = fakes_spectrum,
                                                             PSF_cube = PSF_cube_arr,
                                                             PSF_cube_wvs=PSF_cube_wvs,
                                                             mute = mute)

    numwaves = np.size(np.unique(dataset.wvs))
    N_cubes = np.size(dataset.wvs)/numwaves
    suffix = fakes_spectrum+"_ROC"
    #Save each cube with the fakes
    for cube_id in range(N_cubes):
        spdc_filename = dataset.filenames[(cube_id*numwaves)].split(os.path.sep)[-1].split(".")[0]
        print("Saving file: "+dir_fakes + os.path.sep + spdc_filename+"_"+suffix+".fits")
        dataset.savedata(dir_fakes + os.path.sep + spdc_filename+"_"+suffix+".fits",
                         dataset.input[(cube_id*numwaves):((cube_id+1)*numwaves),:,:],
                         filetype="raw spectral cube with fakes", more_keywords =extra_keywords,
                         user_prihdr=dataset.prihdrs[cube_id], user_exthdr=dataset.exthdrs[cube_id])

    ###########################################################################################
    ## Reduce dataset with fakes
    ###########################################################################################
    # Object to reduce the fake dataset with FMMF
    raw_nohpf_read_func = lambda file_list:GPI.GPIData(file_list,
                                             meas_satspot_flux=True,
                                             numthreads = None,
                                             highpass=False,
                                             PSF_cube="*-original_PSF_cube.fits")
    filename ="S*_spdc_distorcorr_{0}_ROC.fits".format(fakes_spectrum)
    FMMFObj_fk = FMMF(raw_nohpf_read_func,filename = filename,
                   spectrum=reduc_spectrum,
                   PSF_read_func = PSF_read_func,
                   PSF_cube = PSF_cube,
                   PSF_cube_wvs = None,
                   predefined_sectors = "1.0 as",
                   label = "FMMF_ROC",
                   overwrite=overwrite,
                   numbasis=numbasis,
                   maxnumbasis = maxnumbasis,
                   mvt=mvt,
                   fakes_only=True,
                   pix2as = GPI.GPIData.lenslet_scale)

    err_list = kpop_wrapper(dir_fakes,[FMMFObj_fk],spectrum_list=[reduc_spectrum],mute_error = False)

    read_func = lambda file_list:GPI.GPIData(file_list,recalc_centers=False,recalc_wvs=False,highpass=False)

    # Definition of the SNR object
    filename = os.path.join("kpop_FMMF_ROC",reduc_spectrum,"*{0:0.2f}-FM*-KL{1}.fits".format(mvt,numbasis[0]))
    filename_noPlanets = os.path.join(inputDir,"kpop_FMMF",reduc_spectrum,"*{0:0.2f}-FM*-KL{1}.fits".format(mvt,numbasis[0]))
    FMMF_snr_Obj_fk = Stat(read_func,filename,filename_noPlanets=filename_noPlanets,type="pixel based SNR",
                   overwrite=overwrite,mute=mute,resolution=resolution,mask_radius=mask_radius,N_threads=-1,
                           pix2as = GPI.GPIData.lenslet_scale)
    filename_fits = os.path.join("kpop_FMMF_ROC",reduc_spectrum,"*{0:0.2f}-FM*-KL{1}-SNRPerPixDr2.fits".format(mvt,numbasis[0]))
    filename_csv = os.path.join(inputDir,"kpop_FMMF",reduc_spectrum,"*{0:0.2f}-FM*-KL{1}-SNRPerPixDr2-DetecTh2Mr4.csv".format(mvt,numbasis[0]))
    FMMF_ROC_Obj_fk = ROC(read_func,filename,filename_csv,overwrite=overwrite,mute=mute,IWA = 15,OWA = 70,
                           pix2as = GPI.GPIData.lenslet_scale)

    err_list = kpop_wrapper(dir_fakes,[FMMF_snr_Obj_fk,FMMF_ROC_Obj_fk],mute_error=False)

    ###########################################################################################
    ## Add Cross correlation and Matched Filter reductions
    filename = os.path.join("kpop_FMMF_ROC",reduc_spectrum,
                                       "*_{0:0.2f}-speccube-KL{1}.fits".format(mvt,numbasis[0]))
    cc_Obj_fk = CrossCorr(read_func,filename,kernel_type="gaussian",kernel_para=1.0,
                       collapse=True,spectrum=reduc_spectrum,folderName=reduc_spectrum,overwrite=overwrite)
    mf_Obj_fk = Matchedfilter(read_func,filename,kernel_type="gaussian",kernel_para=1.0,
                       spectrum=reduc_spectrum,folderName=reduc_spectrum,overwrite=overwrite)

    filename = os.path.join("kpop_FMMF_ROC",reduc_spectrum,
                                       "*_{0:0.2f}-speccube-KL{1}-*gaussian.fits".format(mvt,numbasis[0]))
    filename_noPlanets = os.path.join(inputDir,"kpop_FMMF",reduc_spectrum,
                                       "*_{0:0.2f}-speccube-KL{1}-*gaussian.fits".format(mvt,numbasis[0]))
    klip_snr_Obj_fk = Stat(read_func,filename,filename_noPlanets=filename_noPlanets,type="pixel based SNR",
                   overwrite=overwrite,mute=mute,resolution=resolution,mask_radius=mask_radius,
                           pix2as = GPI.GPIData.lenslet_scale)
    filename_csv = os.path.join(inputDir,"kpop_FMMF",reduc_spectrum,
                                "*_{0:0.2f}-speccube-KL{1}-*gaussian-SNRPerPixDr2-DetecTh2Mr4.csv".format(mvt,numbasis[0]))
    filename_fits = os.path.join("kpop_FMMF_ROC",reduc_spectrum,
                                       "*_{0:0.2f}-speccube-KL{1}-*gaussian-SNRPerPixDr2.fits".format(mvt,numbasis[0]))
    klip_ROC_Obj_fk = ROC(read_func,filename_fits,filename_csv,overwrite=overwrite,mute=mute,IWA = 15,OWA = 70,
                           pix2as = GPI.GPIData.lenslet_scale)

    err_list = kpop_wrapper(dir_fakes,[cc_Obj_fk,mf_Obj_fk,klip_snr_Obj_fk,klip_ROC_Obj_fk],mute_error=False)


    ###########################################################################################
    ## Remove the fake datasets because we don't need them anymore
    ###########################################################################################
    spdc_glob = glob(dir_fakes+os.path.sep+"S*_spdc_distorcorr_{0}_ROC.fits".format(fakes_spectrum))
    for filename in spdc_glob:
        print("Removing {0}".format(filename))
        os.remove(filename)

if __name__ == "__main__":
    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass

    print(platform.system())

    OS = platform.system()
    if OS == "Windows":
        print("Using WINDOWS!!")
    else:
        print("I hope you are using a UNIX OS")

    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv))
    print("CPU COUNT: {0}".format(mp.cpu_count()))

    ###########################################################################################
    ## Contrast curve parameters
    ###########################################################################################

    if 0:
        inputDir = sys.argv[1]
        dir_fakes = sys.argv[2]

        mvt = float(sys.argv[3])
        reduc_spectrum = sys.argv[4]
        fakes_spectrum = sys.argv[5]
    else:
        inputDir = "/home/sda/jruffio/pyklip/tests/data/"
        dir_fakes = "/home/sda/jruffio/pyklip/tests/data-Fakes/"

        mvt = 0.7
        reduc_spectrum = "t600g100nc"
        fakes_spectrum = "t1000g100nc"

    roc_dataset(inputDir,dir_fakes,mvt,reduc_spectrum,fakes_spectrum)

