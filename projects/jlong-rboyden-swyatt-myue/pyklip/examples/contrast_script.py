#!/home/anaconda/bin/python2.7
__author__ = 'jruffio'

import platform
import os
import sys
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
from pyklip.kpp.utils.contrast import calculate_contrast
from pyklip.kpp.detection.detection import Detection
from pyklip.kpp.detection.ROC import ROC
from pyklip.instruments import GPI
import pyklip.fakes as fakes
import shutil


def contrast_dataset(inputDir,dir_fakes,mvt,reduc_spectrum,fakes_spectrum,approx_throughput):
    ###########################################################################################
    ## Contrast curve parameters
    ###########################################################################################

    numbasis=[5]
    maxnumbasis = 10
    # Less important parameters
    mute = False # Mute print statements
    mask_radius = 5 # (Pixels) Radius of the disk used to mask out objects in an image
    overwrite = False # Force rewriting the files even if they already exist

    # contrast_range = [0.2,1.2] # Range of separation in arcsec for the contrast curve calculation
    pa_shift_list = [0,180] # Position angle shift between the fakes in the different copies of the dataset

    ###########################################################################################
    ## Generate PSF cube
    ###########################################################################################


    # Generate PSF cube for GPI from the satellite spots
    filenames = glob(os.path.join(inputDir,"S*distorcorr.fits"))
    dataset = GPI.GPIData(filenames,highpass=True)
    dataset.generate_psf_cube(20,same_wv_only=True)
    PSF_cube = inputDir + os.path.sep + os.path.basename(filenames[0]).split(".fits")[0]+"-original_PSF_cube.fits"
    # Save the original PSF calculated from combining the sat spots
    dataset.savedata(PSF_cube, dataset.psfs, filetype="PSF Spec Cube",pyklip_output=False)

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
    w_ann0 = 5
    w_ann1 = 10
    tmp = (np.arange(8.7,30,w_ann0)).tolist()\
           + (np.arange(np.arange(8.7,30,w_ann0)[-1]+w_ann1,140,w_ann1)).tolist()
    annuli = [(rho1,rho2) for rho1,rho2 in zip(tmp[0:-1],tmp[1::])]
    FMMFObj = FMMF(raw_read_func,filename = filename,
                   spectrum=reduc_spectrum,
                   PSF_read_func = PSF_read_func,
                   PSF_cube = PSF_cube,
                   PSF_cube_wvs = None,
                   subsections = 1,
                   annuli = annuli,
                   label = "FMMF",
                   overwrite=overwrite,
                   numbasis=numbasis,
                   maxnumbasis = maxnumbasis,
                   mvt=mvt)
    read_func = lambda file_list:GPI.GPIData(file_list,recalc_centers=False,recalc_wvs=False,highpass=False)

    # Definition of the SNR object
    filename = os.path.join("kpop_FMMF",reduc_spectrum,"*{0:0.2f}-FM*-KL{1}.fits".format(mvt,numbasis[0]))
    FMMF_snr_obj = Stat(read_func,filename,type="pixel based SNR",
                   overwrite=overwrite,mute=mute,mask_radius=mask_radius,
                       pix2as = GPI.GPIData.lenslet_scale)
    filename = os.path.join("kpop_FMMF",reduc_spectrum,"*{0:0.2f}-FM*-KL{1}-SNRPerPixDr2.fits".format(mvt,numbasis[0]))
    FMMF_detec_obj = Detection(read_func,filename,mask_radius = None,threshold = 2,overwrite=overwrite,mute=mute,IWA=9,OWA=1./0.01414,
                       pix2as = GPI.GPIData.lenslet_scale)

    err_list = kpop_wrapper(inputDir,[FMMFObj,FMMF_snr_obj,FMMF_detec_obj],mute_error = False)

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
                   overwrite=overwrite,mute=mute,mask_radius=mask_radius,
                       pix2as = GPI.GPIData.lenslet_scale)
    filename = os.path.join("kpop_FMMF",reduc_spectrum,
                                       "*_{0:0.2f}-speccube-KL{1}-*gaussian-SNRPerPixDr2.fits".format(mvt,numbasis[0]))
    klip_detec_obj = Detection(read_func,filename,mask_radius = None,threshold = 2,overwrite=overwrite,mute=mute,IWA=9,OWA=1./0.01414,
                       pix2as = GPI.GPIData.lenslet_scale)

    err_list = kpop_wrapper(inputDir,[cc_obj,mf_obj,klip_snr_obj,klip_detec_obj],mute_error=False)

    ###########################################################################################
    ## Initial guess for the contrast curve (to determine the contrast of the fakes)
    ###########################################################################################
    # This section can be user defined as long as sep_bins_center and cont_stddev are set.

    FMCont_filename = os.path.join(inputDir,"kpop_FMMF",reduc_spectrum,"*{0:0.2f}-FMCont-KL{1}.fits".format(mvt,numbasis[0]))
    contrast_image_obj = read_func([glob(FMCont_filename)[0]])

    cont_stddev,sep_bins = get_image_stddev(np.squeeze(contrast_image_obj.input),
                                            centroid = contrast_image_obj.centers[0])
    # Separation samples in pixels
    sep_bins_center =  (np.array([r_tuple[0] for r_tuple in sep_bins]))
    # Approximative contrast curve at these separations
    approx_cont_curve = 5*np.array(cont_stddev)/approx_throughput

    # ###########################################################################################
    ## Build fake datasets to be used to calibrate the conversion factor
    ###########################################################################################
    print("~~ Injecting fakes ~~")

    # Load the PSF cube that has been calculated from the sat spots
    PSF_cube = glob(os.path.join(inputDir,"*-original_PSF_cube.fits"))[0]
    PSF_cube_obj = PSF_read_func([PSF_cube])
    PSF_cube_arr = PSF_cube_obj.input
    PSF_cube_wvs = PSF_cube_obj.wvs

    if not os.path.exists(dir_fakes):
        os.makedirs(dir_fakes)
    shutil.copyfile(PSF_cube,os.path.join(dir_fakes,os.path.basename(PSF_cube)))

    for pa_shift in pa_shift_list:
        #Define the fakes position and contrast
        fake_flux_dict = dict(mode = "SNR",SNR=10,sep_arr = sep_bins_center, contrast_arr=approx_cont_curve)
        # fake_flux_dict = dict(mode = "contrast",contrast=5e-6)
        fake_position_dict = dict(mode = "spirals",pa_shift=pa_shift,annuli=10)

        # Inject the fakes
        spdc_glob = glob(inputDir+os.path.sep+"S*_spdc_distorcorr.fits")
        if overwrite or len(glob(os.path.join(dir_fakes,"S*_spdc_distorcorr_{0}_PA*.fits").format(fakes_spectrum))) != len(pa_shift_list)*len(spdc_glob):
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
            suffix = fakes_spectrum+"_PA{0:02d}".format(pa_shift)
            #Save each cube with the fakes
            for cube_id in range(N_cubes):
                spdc_filename = dataset.filenames[(cube_id*numwaves)].split(os.path.sep)[-1].split(".")[0]
                print("Saving file: "+dir_fakes + os.path.sep + spdc_filename+"_"+suffix+".fits")
                dataset.savedata(dir_fakes + os.path.sep + spdc_filename+"_"+suffix+".fits",
                                 dataset.input[(cube_id*numwaves):((cube_id+1)*numwaves),:,:],
                                 filetype="raw spectral cube with fakes", more_keywords =extra_keywords,
                                 user_prihdr=dataset.prihdrs[cube_id], user_exthdr=dataset.exthdrs[cube_id],
                                 pyklip_output=False)

    ###########################################################################################
    ## Reduce the fake dataset
    ###########################################################################################

    # Object to reduce the fake dataset with FMMF
    raw_nohpf_read_func = lambda file_list:GPI.GPIData(file_list,
                                             meas_satspot_flux=True,
                                             numthreads = None,
                                             highpass=False,
                                             PSF_cube="*-original_PSF_cube.fits")
    for pa_shift in pa_shift_list:
        # Object to reduce the fake dataset with FMMF
        filename = "S*_spdc_distorcorr_{0}_PA{1:02d}.fits".format(fakes_spectrum,pa_shift)
        FMMFObj = FMMF(raw_nohpf_read_func,filename = filename,
                       spectrum=reduc_spectrum,
                       PSF_read_func = PSF_read_func,
                       PSF_cube = PSF_cube,
                       PSF_cube_wvs = None,
                       subsections = 1,
                       annuli = annuli,
                       label = "FMMF_PA{0:02d}".format(pa_shift),
                       overwrite=overwrite,
                       numbasis=numbasis,
                       maxnumbasis = maxnumbasis,
                       mvt=mvt,
                       fakes_only=True,
                       pix2as = GPI.GPIData.lenslet_scale)

        # Definition of the SNR object
        filename = os.path.join("kpop_FMMF_PA{0:02d}".format(pa_shift),reduc_spectrum,"*{0:0.2f}-FM*-KL{1}.fits".format(mvt,numbasis[0]))
        filename_noPlanets = os.path.join(inputDir,"kpop_FMMF",reduc_spectrum,"*{0:0.2f}-FM*-KL{1}.fits".format(mvt,numbasis[0]))
        FMMF_snr_Obj_fk = Stat(read_func,filename,filename_noPlanets=filename_noPlanets,type="pixel based SNR",
                       overwrite=overwrite,mute=mute,mask_radius=mask_radius,N_threads=-1,
                       pix2as = GPI.GPIData.lenslet_scale)

        err_list = kpop_wrapper(dir_fakes,[FMMFObj,FMMF_snr_Obj_fk],mute_error=False)

        ###########################################################################################
        ## Add Cross correlation and Matched Filter reductions
        filename = os.path.join("kpop_FMMF_PA{0:02d}".format(pa_shift),reduc_spectrum,
                                           "*_{0:0.2f}-speccube-KL{1}.fits".format(mvt,numbasis[0]))
        cc_Obj_fk = CrossCorr(read_func,filename,kernel_type="gaussian",kernel_para=1.0,
                           collapse=True,spectrum=reduc_spectrum,folderName=reduc_spectrum,overwrite=overwrite)
        mf_Obj_fk = Matchedfilter(read_func,filename,kernel_type="gaussian",kernel_para=1.0,
                           spectrum=reduc_spectrum,folderName=reduc_spectrum,overwrite=overwrite)

        filename = os.path.join("kpop_FMMF_PA{0:02d}".format(pa_shift),reduc_spectrum,
                                           "*_{0:0.2f}-speccube-KL{1}-*gaussian.fits".format(mvt,numbasis[0]))
        filename_noPlanets = os.path.join(inputDir,"kpop_FMMF",reduc_spectrum,
                                           "*_{0:0.2f}-speccube-KL{1}-*gaussian.fits".format(mvt,numbasis[0]))
        klip_snr_Obj_fk = Stat(read_func,filename,filename_noPlanets=filename_noPlanets,type="pixel based SNR",
                       overwrite=overwrite,mute=mute,mask_radius=mask_radius,
                       pix2as = GPI.GPIData.lenslet_scale)

        err_list = kpop_wrapper(dir_fakes,[cc_Obj_fk,mf_Obj_fk,klip_snr_Obj_fk],mute_error=False)

    ###########################################################################################
    ## Combine all the data to build the contrast curves
    ###########################################################################################

    # 2 lines to be removed
    read_func = lambda file_list:GPI.GPIData(file_list,recalc_centers=False,recalc_wvs=False,highpass=False)
    PSF_read_func = lambda file_list:GPI.GPIData(file_list,highpass=False)

    # # Extract Julian date of the file
    # tmp_file = os.path.join(dir_fakes,"kpop_FMMF_PA{0:02d}".format(pa_shift),reduc_spectrum,"*{0:0.2f}-FM*-KL{1}.fits".format(mvt,numbasis[0]))
    # hdulist = pyfits.open(glob(tmp_file)[0])
    # MJDOBS = hdulist[0].header['MJD-OBS']
    MJDOBS = None

    # For the 3 different metric, calculate the contrast curve using the fakes to calibrate the conversion factor.
    separation_list = []
    contrast_curve_list = []
    for FMMF_metric in ["FMMF-KL{0}".format(numbasis[0]),
                        "FMCC-KL{0}".format(numbasis[0]),
                        "FMCont-KL{0}".format(numbasis[0]),
                        "speccube-KL{0}-crossCorrgaussian".format(numbasis[0]),
                        "speccube-KL{0}-MF3Dgaussian".format(numbasis[0])]:
        nofakes_filename = os.path.join(inputDir,"kpop_FMMF",reduc_spectrum,
                                           "*_{0:0.2f}-{1}.fits".format(mvt,FMMF_metric))
        fakes_filename_list = [os.path.join(dir_fakes,"kpop_FMMF_PA{0:02d}".format(pa_shift),reduc_spectrum,
                                               "*_{0:0.2f}-{1}.fits".format(mvt,FMMF_metric).format(pa_shift)) for pa_shift in pa_shift_list]
        fakes_SNR_filename_list = [os.path.join(dir_fakes,"kpop_FMMF_PA{0:02d}".format(pa_shift),reduc_spectrum,
                                               "*_{0:0.2f}-{1}-SNRPerPixDr2.fits".format(mvt,FMMF_metric).format(pa_shift)) for pa_shift in pa_shift_list]
        separation,contrast_curve,throughput_tuple = calculate_contrast(read_func,
                                               nofakes_filename,
                                               fakes_filename_list,
                                               GPI.GPIData.lenslet_scale,
                                               mask_radius=mask_radius,Dr=2,
                                               save_dir = os.path.join(inputDir,"kpop_FMMF",reduc_spectrum),
                                               suffix=FMMF_metric+"-mvt{0:0.2f}".format(mvt),spec_type=fakes_spectrum,
                                               fakes_SNR_filename_list=fakes_SNR_filename_list,MJDOBS=MJDOBS)
        separation_list.append(separation)
        contrast_curve_list.append(contrast_curve)


    if 0:
        import matplotlib.pyplot as plt
        plt.figure(1)
        for separation,contrast_curve,FMMF_metric in zip(separation_list,contrast_curve_list,["FMMF-KL{0}".format(numbasis[0]),
                            "FMCC-KL{0}".format(numbasis[0]),
                            "FMCont-KL{0}".format(numbasis[0]),
                            "speccube-KL{0}-crossCorrgaussian".format(numbasis[0]),
                            "speccube-KL{0}-MF3Dgaussian".format(numbasis[0])]):
            plt.plot(separation,contrast_curve,label=FMMF_metric)
        plt.gca().set_yscale('log')
        plt.legend()
        plt.show()
    ###########################################################################################
    ## Remove the fake datasets because we don't need them anymore
    ###########################################################################################

    spdc_glob = glob(dir_fakes+os.path.sep+"S*_spdc_distorcorr_{0}_*.fits".format(fakes_spectrum))
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
        approx_throughput = sys.argv[6]
    else:
        inputDir = "/home/sda/jruffio/pyklip/tests/data/"
        dir_fakes = "/home/sda/jruffio/pyklip/tests/data-Fakes/"

        mvt = 0.7
        # T-type
        reduc_spectrum = "t600g100nc"
        fakes_spectrum = "t1000g100nc"
        approx_throughput = 0.5
        # L-type
        # reduc_spectrum = "t1300g100f2"
        # fakes_spectrum = "t1300g100f2"
        # approx_throughput = 0.7

    contrast_dataset(inputDir,dir_fakes,mvt,reduc_spectrum,fakes_spectrum,approx_throughput)
