.. _addinstrument-label:

Adding an Instrument Interface
==============================
To add an instrument interface, one needs to implement a subclass of the abstract class
:py:class:`pyklip.instruments.Instrument.Data`, overriding the abstract methods and fields required by the class. Some fields
may not not be relevant a particular instrument (e.g. wavelengths for a broadband instrument with only one channel) and we will
provide suggestions on what to default the value to. 

For a simple example, look at how :py:class:`pyklip.instruments.Instrument.GenericData` implements the interface. 

Now we will discuss what are the necessary steps to make your own instrument class.

Extending Instrument.Data
-------------------------
The first thing we need to do is create a new object that is a subclass of :py:class:`pyklip.instruments.Instrument.Data`. To do
this, we specify our new class inherits :py:class:`pyklip.instruments.Instrument.Data` and we need to call the ``__init__()`` function
of the super class (i.e. :py:class:`pyklip.instruments.Instrument.Data`) as the first step of our new ``__init__()`` function.

.. code-block:: python

    import pyklip.instruments.Instrument.Data as Data

    class MyInstrument(Data):
        """
        This is my new instrument class
        """
        # Constructor
        def __init__(self, args):
            # initialize the super class
            super(MyInstrument, self).__init__()

            # run some more initialization code here

        # the rest of the class goes here


Fields
------
For each required field, we use the following syntax to implement each required field. The field itself is just a wrapper
for an internal field named ``_field``, which should be set by the ``__init__()`` function. This is a little clunky, but 
allows python to ensure these fields are implemented. You should have one of the following code blocks for each required field:

.. code-block:: python

    @property
    def field(self):
        return self._field
    @input.setter
    def field(self, newval):
        self._field = newval


Here are the fields that need to be implemented (some are optional and are marked as such). For the optional fields, you do not
need to use the previous code block and make getter/setter methods; you can just set them like you normally set attributes.

``input``
"""""""""
This is the input data. It should be a numpy array of dimensions (Nframes, y, x). For dual-band or IFS data, the wavelength 
dimension should be merged into the total number of files so that Nframes = Nwvs x Nfiles. Basically, this should always
be a 3-D cube, and we will use other bookkeeping fields to track the wavelengths and filenames for each frame.

``output``
""""""""""
This is where the output data gets stored. For initialization, set this equal to None so that the variable is defined. Othewise,
you can expect that after pyKLIP, the 5-D output cube gets stored here with dimension (KL, Nfiles, Nwvs, y, x) where KL is the 
numbe of KL cutoffs you specified, Nfiles is the number of unique filenames, Nwvs is the number of unique wavelengths.

``fmout`` (Optional)
""""""""""""""""""""
This is where the output of the forward modelling is stored (unless otherwise noted by the specific KLIP-FM library). Refer to each
KLIP-FM feature for how to make sense of this data. 

``centers``
"""""""""""
This is the image centers for each input frame of data. It should be a numpy array of dimensions (Nframes, 2) where the second dimension
is the (x,y) center for that frame. This is required for all datasets.

``filenames``
"""""""""""""
This is an array of filenames that correspond to each frame. Depending on how the data is formatted, filenames can be duplicated so that
more than one frame has the same filename (e.g. each frame in an IFS datacube). **For RDI, filenames are required**, so that the 
PSF library will exclude the science frames from the PSF library that was built. If you really don't care about them, set them
to something generic (e.g. for the first frame, "0.fits")

``filenums``
""""""""""""
This is an array of file numbers so that each filename corresponds to a certain number. This allows for easier manipulating of frames,
since it is easier to sort and slice numbers than strings. For easy implementation, make the first filename corespond to 0, the
second correspond to 1, and so on.

``wvs``
"""""""
This is an array of wavelengths, **required for SDI to figure out how to rescale the speckles**. If you are working with broadband
data or generally wavelength agnostic, make this an array of the same number (e.g. use the central wavelength of the filter or 
set it all to 1). 

``PAs``
"""""""
This is an array of angles for each frame. This is defined as the angle needed to rotate the image north up, which means it is
a combination of the parallactic angle (angle from North to zenith in the direction towards East) and any instrumental angles
(e.g. the angle from the image to zenith). **For ADI, PAs are required to determine field rotation.** If you don't know or don't 
want image rotation, set this to an array of 0 with length equal to the number of frames. 

``flipx`` (Optional)
""""""""""""""""""""
This specifies a boolean that at the end, when derotating and stacking the images, whether to flip the x-axis.
**By default, this is set to True**. In the end we want to rotate images North-up East-left (i.e. East CCW of North). 
If your image starts out with East clockwise of North, then flipx should be set to True. Otherwise, set it to False.

``wcs``
"""""""
This is an array of astropy.wcs.WCS objects that specifies the orientation of each input image. Since pyKLIP primiarily uses ``PAs`` and
``flipx`` to figure out image orientation, this keyword isn't strictly necessary, but could be good to have (e.g. pyklip.fakes
uses it for fake planet injection, and it is generally nice to have in your final PSF subtracted images). Note that WCS objects
have the method ``deepcopy`` which allows you to replicate WCS headers, so if you have multiple frames that share the same WCS,
it is an easy to to give each frame its own WCS info. If you don't have WCS info or don't want to deal with it, set ``wcs`` to an
array of None with length equal to the number of frames.

``IWA``
"""""""
This is the inner working angle (radius, in pixels) of your data. pyKLIP will not reduce this part of your data and instead mask it
as NaNs. If you don't have an inner working angle well defined for your instrument, make this a parameter the user could pass in, 
guess one, or set it to 0. Note that this is a single number and not an array.

``OWA`` (Optional)
""""""""""""""""""
This is the outer working angle for your data. By default this is None, and pyKLIP will use either the closest NaN to the center of
the first image, or (if there are no NaNs) the edge of the image as the outer working angle. This is also a single number.

``output``, ``output_centers``, ``output_wcs`` (Required-ish)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
These fields corresponds to the output data, an array of dimensions (KLcutoffs, Nframes, y, x), the (x,y) center for each output
frame, an array of dimensions (Nframes, 2), and an array of WCS headers corresponding to the output images, which are typically
rotated North-up and East-left. The one you must explicitly define in your class is ``output``, but you should expect the other 
two fields to also get populated after a KLIP reduction, so it could be useful to refer to those fields in ``savedata``. Note that
if you pass an array of None to ``wcs``, ``output_wcs`` will also be an array of None. Also note that ``output_center`` is the same
(x,y) coordinate repeated for each frame since the images are aligned together after KLIP. 

Methods
-------
Here are some required methods that need to be implemented. Of them, you definely want to make sure ``savedata()`` works.


``readdata(self, filepaths)``
"""""""""""""""""""""""""""""
This function should be able to read in files from a list of filenames, compile them together, and set up the fields for this dataset.
Typically this function is called by the ``__init__()`` function. If your data doesn't come in a form where reading it in like this
is a very elegant solution, feel free to skip this function (by implementing this funciton with ``pass`` as the only command) and 
writing a different way to initalize your dataset.

``savedata(self, filepath, data, klipparams=None, filetype="", zaxis=None, more_keywords=None)``
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
This is the most important function to implement since it saves your pyKLIP reductions. The filepath is where to save the 
file to, and the data is what to save. ``klipparams`` is a string listing all of the pyKLIP parameters, and is typically
saved into the histroy of a FITS file. ``filetype`` tells you the type of datacube this data is (i.e. "KL Mode Cube", 
"PSF Subtracted Spectral Cube"). For data with just one wavelength, the data will only be KL Mode cubes where the third
dimension is the KL mode cutoff. ``zaxis`` is used for KL Mode cubes to specify the KL mode cutoffs for each slice. 
``more_keywords`` is additional keywords to save into the header for bookkeeping. 


``calibrate_output(self, img, spectral=False)``
"""""""""""""""""""""""""""""""""""""""""""""""
This handles the flux calibration of the image passed in via ``img``. For spectral data cubes (i.e. the third dimension is 
wavelength), then the spectral flag is set to true.  