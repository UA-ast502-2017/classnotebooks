.. _genericdata-label:

For All Else: Generic Data Tutorial
===================================
The :py:class:`pyklip.instruments.Instrument.GenericData` interface allows anyone to pass in any data set into pyKLIP to do basic
KLIP reductions without having to write an insturment module first. It is recommended that eventually there should be an insturment class
to leverage all the features of pyKLIP. However to test out pyKLIP on data from a new instrument or to handle a dataset you will only
see once (e.g. special lab data), then GenericData is the way to go.

Reading in Generic Data
-----------------------
Basically, to read in generic data, you'll need to do most of the file handling yourself. Then the GenericData class will organize
that information to be interfaceable with pyKLIP. For example, you will need to pass in things like the data frames, centers, filenames,
inner working angles, etc. 

Generic data requires you to pass in the data frames and centers. The data frames are passed in as a 3-D datacube with dimensions of
(Nframes, y, x). The centers are passed in as an array of (x,y) coordiantes with dimensions of (Nframes, 2). The rest of the arguments
are optional and depends on your data (e.g. if ou have ADI data, you should pass in the parallactic angles; if you have SDI data you will
need to pass in the wavelengths; if you have RDI data, you will need to pass in the filenames). See the docstring of 
:py:class:`pyklip.instruments.Instrument.GenericData` for the details.

Example with Simulated WFIRST Data
----------------------------------
Here is an example of using GenericData on simulated WFIRST data available 
`publically online <https://wfirst.ipac.caltech.edu/sims/Coronagraph_public_images.html>`_ (Version 3.1, Observing Scenario 5).

.. code-block:: python

    import astropy.io.fits as fits
    import numpy as np
    from pyklip.instruments.Instrument import GenericData
    import pyklip.parallelized as parallelized

    # Read in science images, which are taken at 2 roll angles. For each angle, the files come in as a 3D cube
    # We want to append the two roll angles together 
    with fits.open("data/OS5_adi_3_polx_lowfc_random_47_Uma_roll_m13deg_HLC_sequence_with_planets.fits") as input_hdu_1:
        with fits.open("data/OS5_adi_3_polx_lowfc_random_47_Uma_roll_p13deg_HLC_sequence_with_planets.fits") as input_hdu_2:

            frames_per_roll = input_hdu_1[0].data.shape[0]

            # the input science data is the combination of the two roll angles. 
            input_data = np.append(input_hdu_1[0].data, input_hdu_2[0].data, axis=0) # makes a (2N, y, x) sized cube
            # generate roll angle lengths for each frame
            pas = np.append([13 for _ in range(frames_per_roll)], [-13 for _ in range(frames_per_roll)])
            # for each frame, give it a filename
            input_filenames = np.append(["OS5_adi_3_polx_lowfc_random_47_Uma_roll_m13deg_HLC_sequence_with_planets.fits" for _ in range(frames_per_roll)], 
                                ["OS5_adi_3_polx_lowfc_random_47_Uma_roll_p13deg_HLC_sequence_with_planets.fits" for _ in range(frames_per_roll)])
            # all of the files are yet again at (31,31)
            input_centers = np.array([[xcenter, ycenter] for _ in range(frames_per_roll*2)])

    # set the inner working angle
    IWA = 6 # pixels

    # now let's generate a dataset to reduce for KLIP. This contains data at both roll angles
    dataset = GenericData(input_data, input_centers, IWA=IWA, parangs=pas, filenames=input_filenames)

    # set up the KLIP parameters and run KLIP
    numbasis=[1,5,10,20,50] # number of KL basis vectors to use to model the PSF. We will try several different ones
    maxnumbasis=150 # maximum number of most correlated PSFs to do PCA reconstruction with
    annuli=3
    subsections=4 # break each annulus into 4 sectors
    parallelized.klip_dataset(dataset, outputdir="data/", fileprefix="pyklip_nonoise_k150a3s4m1", annuli=annuli, 
                              subsections=subsections, numbasis=numbasis, maxnumbasis=maxnumbasis, mode="ADI", 
                              movement=1)