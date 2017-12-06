.. _sphere-label:

SPHERE Tutorials
================
The pyKLIP SPHERE interface supports IFS and IRDIS data produced from 
`Arthur Vigan's SPHERE tools <http://astro.vigan.fr/tools.html>`_ and IFS data from ESO's SPHERE pipeline. 
These tutorials all assume you have processed the raw SPHERE data into a format like what is produced by 
`Arthur Vigan's SPHERE tools <http://astro.vigan.fr/tools.html>`_. That is, you will want the following things:

* Input data cube of dimensions (Nfiles, Nwvs, y, x)
* PSF cube of dimensions (Nwvs, y, x)
* A fits table with information like parallactic angle and pupil offset
* For IFS data, a wavelength array that provides the wavelegth solution

The ESO SPHERE IFS pipeline produces similar files, with a parallactic angle FITS file replacing the FITS table described above.
One should be able to follow the following tutorial, but replacing the four data products mentioend above with the corresponding
products produced by the ESO pipeline.  

IFS Data
--------
For SPHERE IFS data, we just need to pass in the output FITS files from Arthur Vigan's pipeline. Outside of that, the only other thing to 
consider masking pixels into NaNs. Because the output data from the pipeline uses 0 rather than NaN to denote pixels that fall 
outside of the FOV, there can be edge effects (e.g. downweighting pixels at the edge with 0's during mean collapse). Here, we will
trim the edges of the FOV by masking any pixel within 3 pixels of a 0 as a NaN (i.e. using a 9 pixel box to mask things as NaNs).

.. code-block:: python

    datacube = "IFS_YJH_cube_coro.fits"
    psfcube = "IFS_YJH_cube_psf.fits"
    fitsinfo = "IFS_YJH_info.fits"
    wvinfo = "IFS_YJH_wavelength.fits"

    print("read in data")
    dataset = SPHERE.Ifs(datacube, psfcube, fitsinfo, wvinfo, nan_mask_boxsize=9)

    print(dataset.psfs.shape) # instrumental PSF at each wavelength stored here if you need it

    parallelized.klip_dataset(dataset, outputdir="path/to/save/dir/", fileprefix="myobject",
                              annuli=9, subsections=4, movement=1, numbasis=[1,20,50,100],
                              mode="ADI+SDI", spectrum="methane")


IRDIS Data
----------
For IRDIS data, we don't need to specify the wavelength info. Instead we should pass in a string specifying an IRDIS band name
(options are ``"Y2Y3"``, ``"J2J3"``, ``"H2H3"``, ``"H3H4"``, ``"K1K2"``). 

.. code-block:: python

    datacube = "IRDIS_K1K2_cube_coro.fits"
    psfcube = "IRDIS_K1K2_cube_psf.fits"
    fitsinfo = "IRDIS_K1K2_info.fits"

    print("read in data")
    dataset = SPHERE.Irdis(datacube, psfcube, fitsinfo, "K1K2")

    print(dataset.psfs.shape) # instrumental PSF at each wavelength stored here if you need it

    parallelized.klip_dataset(dataset, outputdir="path/to/save/dir/", fileprefix="myobject",
                              annuli=9, subsections=4, movement=1, numbasis=[1,20,50,100],
                              mode="ADI+SDI", spectrum="methane")

