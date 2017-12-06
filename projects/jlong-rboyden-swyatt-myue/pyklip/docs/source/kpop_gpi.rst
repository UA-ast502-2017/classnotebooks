.. _kpop-label:


Klip POst Processing (KPOP)
=====================================================
Klip POst Processing (KPOP) is a module with tools to calculate:

    * matched filter maps,
    * SNR maps,
    * detection,
    * ROC curves,
    * contrast curves.

We will go over these application in this tutorial.
KPOP modules can be used as standalone functions but it its normalized class architecture allows an easy processing of surveys by simplifying some user tasks.

.. note::
    The ipython notebook ``pyklip.examples.kpop_tutorial.ipynb`` go through most of the applications with a GPI example
    based on the beta Pictoris test files in tests/data.

.. note::
    KPOP is the framework developped in the context of `Ruffio et al. (2016) <https://arxiv.org/pdf/1705.05477.pdf>`_.

PyKLIP can be installed following :ref:`install-label`.
It has only been tested with python 2.7 even though it should work for python 3.

KPOP modules
-----------------
In this section we show a tutorial for the KPOP modules used as standalone functions. For this example, we use the GPI
data living in the test directory of pyklip.

FMMF
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In signal processing, a matched filter is the linear filter maximizing the Signal to Noise Ratio (SNR) of a known signal in the presence of additive noise.

Matched filters are used in Direct imaging to detect point sources using the expected shape of the planet Point Spread Function (PSF) as a template.

The distortion makes the definition of the template somewhat challenging,the planet PSF doesn't look like the instrumental PSF, but reasonable results can be obtained by using approximations.

Forward Model `Pueyo (2016) <http://arxiv.org/abs/1604.06097>`_

.. code-block:: python

    import numpy as np
    import pyklip.parallelized as parallelized
    pykliproot = os.path.dirname(os.path.realpath(parallelized.__file__))
    inputDir = os.path.join(pykliproot,"..","tests","data")
    outputDir = os.path.join(inputDir,"fmmf_test")
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)


    ####################
    ## Define the instrument object
    from pyklip.instruments import GPI
    import glob
    dataset = GPI.GPIData(glob.glob(os.path.join(inputDir,"S*distorcorr.fits")), highpass=True)

    ####################
    ## Generate PSF cube for GPI from the satellite spots
    dataset.generate_psf_cube(20,same_wv_only=True)

    ####################
    ## Define the fmlib object
    # flat spectrum here, make sure to define your own and correct it for transmission
    spectrum_vec = np.ones((dataset.input.shape[0],))
    # Number KL modes used for KLIP
    numbasis = [5]
    # Number of images in the reference library
    maxnumbasis = 10
    # Definition of the planet PSF
    PSF_cube_arr = dataset.psfs
    PSF_cube_wvs = np.unique(dataset.wvs)

    # Build the FM class to do matched filter
    import pyklip.fmlib.matchedFilter as mf
    fm_class = mf.MatchedFilter(dataset.input.shape,numbasis,
                                     PSF_cube_arr, PSF_cube_wvs,
                                     [spectrum_vec])
    # run KLIP-FM
    movement = 2.0
    from pyklip.kpp.metrics.FMMF import FMMF
    FMMFObj = FMMF(predefined_sectors = "0.6 as",
                   numbasis=numbasis,
                   maxnumbasis = maxnumbasis,
                   mvt=movement)
    FMMF_map,FMCC_map,contrast_map,final_cube_modes = FMMFObj.calculate(dataset=dataset,spectrum=spectrum_vec,fm_class=fm_class)

    # Still possible but optional to save the data with FMMFObj
    FMMFObj.save(outputDir=outputDir,folderName="",prefix="bet_Pic_test")

Cross-correlation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

    ########################
    ## cross correlation of speccube

    import astropy.io.fits as pyfits
    hdulist = pyfits.open(os.path.join(outputDir,"bet_Pic_test-speccube-KL5.fits"))
    cube = hdulist[1].data
    hdulist.close()

    PSF = np.ones((4,4))
    spectrum = np.ones(cube.shape[0])

    from pyklip.kpp.metrics.crossCorr import CrossCorr
    cc_obj = CrossCorr(collapse=True)
    cc_image = cc_obj.calculate(image=cube, PSF=PSF,spectrum = spectrum)

    hdulist = pyfits.HDUList()
    hdulist.append(pyfits.ImageHDU(data=cc_image))
    hdulist.writeto(os.path.join(outputDir,"bet_Pic_test-speccube-KL5-crossCorr.fits"), overwrite=True)
    hdulist.close()
    # also possible to use the save() method
    # cc_obj.save(dataset=dataset,outputDir=outputDir,folderName="",prefix="bet_Pic_test-speccube-KL5")

Matched filter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

    ########################
    # matched filter of speccube

    import astropy.io.fits as pyfits
    hdulist = pyfits.open(os.path.join(outputDir,"bet_Pic_test-speccube-KL5.fits"))
    cube = hdulist[1].data
    hdulist.close()

    radius = 4
    size = 20
    x, y = np.meshgrid(np.arange(0,size,1)-size//2,np.arange(0,size,1)-size//2)
    r = x**2+y**2
    PSF = np.tile(np.array(r <= radius*radius,dtype=np.int),(cube.shape[0],1,1))
    spectrum = np.ones(cube.shape[0])

    from pyklip.kpp.metrics.matchedfilter import Matchedfilter
    mf_obj = Matchedfilter(sky_aper_radius=2)
    mf_map,cc_map,flux_map = mf_obj.calculate(image=cube, PSF=PSF,spectrum = spectrum)

    hdulist = pyfits.HDUList()
    hdulist.append(pyfits.ImageHDU(data=mf_map))
    hdulist.writeto(os.path.join(outputDir,"bet_Pic_test-speccube-KL5-MF.fits"), overwrite=True)
    hdulist.close()
    # also possible to use the save() method
    # mf_obj.save(dataset=dataset,outputDir=outputDir,folderName="",prefix="bet_Pic_test-speccube-KL5")

SNR
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

    ########################
    # SNRs

    import astropy.io.fits as pyfits
    hdulist = pyfits.open(os.path.join(outputDir,"bet_Pic_test-FMMF-KL5.fits"))
    image = hdulist[1].data
    center = [138.4694028209982,140.3317480866463]
    hdulist.close()

    from pyklip.kpp.stat.stat import Stat

    # Definition of the SNR object
    snr_obj = Stat(type="pixel based SNR")
    snr_image = snr_obj.calculate(image=image,center=center)

    hdulist = pyfits.HDUList()
    hdulist.append(pyfits.ImageHDU(data=snr_image))
    hdulist.writeto(os.path.join(outputDir,"bet_Pic_test-FMMF-KL5-SNR.fits"), overwrite=True)
    hdulist.close()
    # also possible to use the save() method
    # snr_obj.save(dataset=dataset,outputDir=outputDir,folderName="",prefix="bet_Pic_test-speccube-KL5")

Point-source detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

    ########################
    # Detection
    from pyklip.kpp.detection.detection import Detection

    import astropy.io.fits as pyfits
    hdulist = pyfits.open(os.path.join(outputDir,"bet_Pic_test-FMMF-KL5-SNR.fits"))
    image = hdulist[0].data
    center = [138.4694028209982,140.3317480866463]
    hdulist.close()

    detec_obj = Detection(threshold = 3,pix2as = GPI.GPIData.lenslet_scale)
    # get tables of candidates with columns: "index","value","PA","Sep (pix)","Sep (as)","x","y","row","col"
    candidate_table = detec_obj.calculate(image=image,center=center)

    # Possible to use the save() method to save as csv file
    detec_obj.save(outputDir=outputDir,folderName="",prefix="bet_Pic_test-FMMF-KL5-SNR")


KPOP framework
-----------------

Requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some advanced KPOP features put more constraints on the instrument classes than the regular KLIP reduction, which might
not be always implemented for all instruments. For now, both GPI and SPHERE classes have been tested, but remember that
it is always possible to use the KPOP functions by manually defining the inputs.
These constraints are:

    - The instrument class should be able to read processed data saved using its savedata() method.
        - This can involve saving the dn2contrast array in the fits file headers.
    - The calibrate_output() should be properly implemented.
    - A object_name attribute should be defined with the name of the star following Simbad syntax.

Architecture
--------------------------
Each KPOP module is a class ihnerited from :py:class:`pyklip.kpp.utils.kppSuperClass`.
All KPOP inherit from the same object, which normalizes the function calls.

The parameter of the task are defined when instantiating the object.
The :meth:`pyklip.kpp.utils.kppSuperClass.initialize` method will then read the files and update the object's attributes.
Then, :meth:`pyklip.kpp.utils.kppSuperClass.calculate()` will process the file(s) and return the final product.
To finish, :meth:`pyklip.kpp.utils.kppSuperClass.save()` will save the final product following the class convention.
After initialize has been ran, it possible to check if the file has already been reduced by calling :meth:`pyklip.kpp.utils.kppSuperClass.check_existence()`.
The method :meth:`pyklip.kpp.utils.kppSuperClass.init_new_spectrum()` allows to change the reduction spectrum if needed.

In order to simplify the reduction of survey data, the filenames are defined with wild characters.
During the initialization, the object will read the file matching the filename pattern.
When several files match the filename pattern, it is possible to simply call initialize() in sequence and the object will automatically read the matching files one by one.

The function :meth:`pyklip.kpp.kpop_wrapper.kpop_wrapper()` will take a list of objects (ie tasks) and a list of spectra as an input and run all the
task as many time as necessary to reduce all the matching files with all the spectra.

Using KPOP framework
--------------------------
We refer the user to the ipython notebook in the pyklip/examples called kpop_tutorial.py.

ROC Curves
--------------------------
ROC curves can be built following the GPI script ``pyklip.examples.roc_script.py`` and adapting to it any instrument or data reduction.
This might include changing the PSF cube calculation, or the platescale and other details.
This script calculate the ROC curve for a single dataset but using different matched filters.
By running this script on several datasets and by combining the final product one can build a ROC curve for an entire survey.
One should consider modify the script for a different instrument.


Contrast Curves and Completeness
--------------------------
Contrast curves can be built following the GPI script ``pyklip.examples.contrast_script.py`` and adapting to it any instrument or data reduction.


