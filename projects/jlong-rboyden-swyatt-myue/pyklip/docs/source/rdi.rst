.. _rdi-label:

RDI with a PSF Library
======================
pyKLIP supports RDI with a PSF Library using the :py:class:`pyklip.rdi.PSFLibrary` class. The PSF Library class holds a correlation
matrix of all frames with each other, so it knows which frames are good reference frames for a certain frame. This means that all
the data, science data and reference data, should be in the PSF libary. In this sense, it is good to generate the PSF Library 
class once, spend the time to compute the correlation matrix once, and then save the correlation to file using the 
``save_correlation()`` function inside of PSFLibrary. This way, the PSF library can be applied on multiple science targets, and
the correlation matrix needs to just be computed once. 

Set up a PSF Library
--------------------
Here we will assume you have a 3-D cube of frames from a long series of data (e.g. a night of observing, a full survey) that is
already in the variable ``psflib_imgs``. This could be made from your own code, or from ``dataset.input`` after you read in a 
large list of files into the Data object. All of these images need to be aligned to some defined ``aligned_center``, a [x,y] array. 
If the images haven't been aligned to this common center already, you can use :py:meth:`pyklip.klip.align_and_scale` to register each
frame. You will also need ``psflib_filenames``, an array of filenames, so that each frame has a corresponding filename. 
This again can be taken from ``dataset.filenames`` if you don't have it already, but have read in all the files into a Data object.
You want to make sure that the filenames are accurate, as this is how we figure out which frames to exclude for RDI (i.e. we don't
want to use data in the science sequence as reference images in a RDI reduction). 

Now here we demonstrate how to make the PSFLibrary object, compute the correlation matrix, and save the correlation matrix.
Note that with a lot of files, generating the correlation matrix can take a long time.

.. code-block:: python

    import pyklip.rdi as rdi

    # make the PSF library
    # we need to compute the correlation matrix of all images vs each other since we haven't computed it before
    psflib = rdi.PSFLibrary(psflib_imgs, aligned_center, psflib_filenames, compute_correlation=True)

    # save the correlation matrix to disk so that we also don't need to recomptue this ever again
    # In the future we can just pass in the correlation matrix into the PSFLibrary object rather than having it compute it
    psflib.save_correlation("corr_matrix.fits", clobber=True)

Then, in the future, we don't need to recompute the correlation matrix when setting up the PSF library. Instead we can read it in
and regenerate the PSFLibrary quickly.

.. code-block:: python

    import pyklip.rdi as rdi
    import astropy.io.fits as fits

    # read in the correlation matrix we already saved
    corr_matrix_hdulist = fits.open("corr_matrix.fits")
    corr_matrix = corr_matrix_hdulist[0].data

    # make the PSF library again, this time we have the correlation matrix
    psflib = rdi.PSFLibrary(psflib_imgs, aligned_center, psflib_filenames, correlation_matrix=corr_matrix)


Running RDI on a dataset
------------------------
Now let's assume you have a ``dataset``, a object that implements :py:class:`pyklip.instruments.Instrument.Data`. The input files
of ``dataset`` are also part of the PSF library (e.g. part of the data taken in that night, or as part of the survey). We 
will then prepare the PSF library for this dataset. This basically invalidates the frames in the PSF library that belong to this
target so they aren't used as reference images, so there won't be any self-subtraction. This is where we use the filenames to match
frames, so **it is important that dataset.filenames matches the filenames passed into the PSF library.**

Then, it's as simple as running KLIP. We pass in the same ``aligned_center`` that all the images in the PSF library is aligned to
so that this reduction also aligns the science images to that cetner. 

.. code-block:: python

    # now we need to prepare the PSF library to reduce this dataset
    # what this does is tell the PSF library to not use files from this star in the PSF library for RDI
    psflib.prepare_library(dataset)

    # now we can run RDI klip
    # as all RDI images are aligned to aligned_center, we need to pass in that aligned_center into KLIP
    numbasis=[1,5,10,20,50] # number of KL basis vectors to use to model the PSF. We will try several different ones
    maxnumbasis=150 # maximum number of most correlated PSFs to do PCA reconstruction with
    annuli=3
    subsections=4 # break each annulus into 4 sectors
    parallelized.klip_dataset(dataset, outputdir="data/", fileprefix="pyklip_k150a3s4m1", annuli=annuli, 
                            subsections=subsections, numbasis=numbasis, maxnumbasis=maxnumbasis, mode="RDI", 
                            aligned_center=aligned_center, psf_library=psflib, movement=1)


