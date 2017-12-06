.. _diskfm_gpi-label:

Disk Foward Modelling Tutorial with GPI
=====================================================
Disk forward modelling is intended for use in cases where you would
like to model a variety of different model disks on the same dataset. This
can be used with an MCMC that is fitting for model parameters. It
works by saving the KLIP basis vectors in a file so they do not have
to be recomputed every time. 

Running
--------------------------
How to use::

    import glob
    import pyklip.parallelized.GPI as GPI
    from pyklip.fmlib.diskfm import DiskFM
    import pyklip.fm as FM
    
    filelist = glob.glob("path/to/dataset/*.fits")
    dataset = GPI.GPIData(filelist)
    model = [some 2D image array]


If you would like to forward model multiple models on a dataset, then you will need to save the eigenvalues and eigenvectors::

    diskobj = DiskFM([n_files, data_xshape, data_yshape], numbasis, dataset, model_disk, annuli = 2, subsections = 1, basis_filename = 'klip-basis.p', save_basis = True, load_from_basis = False)


If you only need to forward model once and don't need the eigenvalues and eigenvectors more than once, you can omit the final keywords::

    diskobj = DiskFM([n_files, data_xshape, data_yshape], numbasis, dataset, model_disk, annuli = 2, subsections = 1)

To run the forward modelling, run::

    fmout = fm.klip_dataset(dataset, diskobj, numbasis = numbasis, annuli = 2, subsections = 1, mode = 'ADI')

Note that in the case that annuli = 1, you will need to set padding = 0 in klip_dataset

If you have saved the eigen vectors then you can load them in at any point with::
  
    diskobj = DiskFM([n_files, data_xshape, data_yshape], numbasis, dataset, model_disk, annuli = 2, subsections = 1, basis_filename = 'klip-basis.p', load_from_basis = True, save_basis = False)

Then, you can run new disks with::

    diskobj.update_disk(newmodel)
    fmout = diskobj.fm_parallelized()


Current Works in Progress
------------------------------------
* Does not support SDI mode
* Is not parallized 
