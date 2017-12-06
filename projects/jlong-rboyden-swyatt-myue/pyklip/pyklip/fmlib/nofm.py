import multiprocessing as mp
import ctypes

import numpy as np

class NoFM(object):
    """
    Super class for all forward modelling classes. Has fall-back functions for all fm dependent calls so that each FM class does
    not need to implement functions it doesn't want to. Should do no forward modelling and just do regular KLIP by itself
    """
    def __init__(self, inputs_shape, numbasis):
        """
        Initializes the NoFM class

        Args:
            inputs_shape: shape of the inputs numpy array. Typically (N, y, x)
            numbasis: 1d numpy array consisting of the number of basis vectors to use

        Returns:
            None
        """
        self.inputs_shape = inputs_shape
        self.numbasis = numbasis
        self.outputs_shape = inputs_shape + numbasis.shape
        self.need_aux = False
        # Use float64
        # self.data_type = ctypes.c_double
        # Use float32
        self.data_type = ctypes.c_float

    def alloc_output(self):
        """
        Allocates shared memory array for final output

        Only use multiprocessing data structors as we are using the multiprocessing class

        Args:

        Returns:
            outputs: mp.array to store final outputs in (shape of (N*wv, y, x, numbasis))
            outputs_shape: shape of outputs array to convert to numpy arrays
        """

        outputs_size = np.prod(np.array(self.inputs_shape)) * np.size(self.numbasis)

        outputs = mp.Array(self.data_type, outputs_size)
        outputs_shape = self.outputs_shape

        return outputs, outputs_shape


    def alloc_interm(self, max_sector_size, numsciframes):
        """
        Allocates shared memory array for intermediate step

        Intermediate step is allocated for a sector by sector basis

        Args:
            max_sector_size: number of pixels in this sector. Max because this can be variable. Stupid rotating sectors

        Returns:
            interm: mp.array to store intermediate products from one sector in
            interm_shape:shape of interm array (used to convert to numpy arrays)

        """

        return None, None


    def alloc_fmout(self, output_img_shape):
        """
        Allocates shared memory for the output of the forward modelling

        Args:
            output_img_shape: shape of output image (usually N,y,x,b)

        Returns:
            fmout: mp.array to store FM data in
            fmout_shape: shape of FM data array

        """

        return None, None


    def alloc_perturbmag(self, output_img_shape, numbasis):
        """
        Allocates shared memory to store the fractional magnitude of the linear KLIP perturbation

        Args:
            output_img_shape: shape of output image (usually N,y,x,b)
            numbasis: array/list of number of KL basis cutoffs requested

        Returns:
            perturbmag: mp.array to store linaer perturbation magnitude
            perturbmag_shape: shape of linear perturbation magnitude

        """

        return None, None


    def fm_from_eigen(self, **kwargs):
        """
        Generate forward models using the KL modes, eigenvectors, and eigenvectors from KLIP
        This is called immediately after regular KLIP

        """

        return


    def fm_end_sector(selfself, **kwargs):
        """
        Does some forward modelling at the end of a sector after all images have been klipped for that sector.

        """
        return

    def cleanup_fmout(self, fmout):
        """
        After running KLIP-FM, if there's anything to do to the fmout array (such as reshaping it), now's the time
        to do that before outputting it

        Args:
            fmout: numpy array of ouput of FM

        Returns:
            fmout: same but cleaned up if necessary
        """

        return fmout

    def skip_section(self, radstart, radend, phistart, phiend,flipx=None):
        """
        Returns a boolean indicating if the section defined by (radstart, radend, phistart, phiend) should be skipped.
        When True is returned the current section in the loop in klip_parallelized() is skipped.

        Args:
            radstart: minimum radial distance of sector [pixels]
            radend: maximum radial distance of sector [pixels]
            phistart: minimum azimuthal coordinate of sector [radians]
            phiend: maximum azimuthal coordinate of sector [radians]
            flipx: if True, flip x coordinate in final image

        Returns:
            Boolean: False so by default it never skips.
        """
        return False

    def save_fmout(self, dataset, fmout, outputdir, fileprefix, numbasis, klipparams=None, calibrate_flux=False,
                   spectrum=None):
        """
        Saves the fmout data to disk following the instrument's savedata function

        Args:
            dataset: Instruments.Data instance. Will use its dataset.savedata() function to save data
            fmout: the fmout data passed from fm.klip_parallelized which is passed as the output of cleanup_fmout
            outputdir: output directory
            fileprefix: the fileprefix to prepend the file name
            numbasis: KL mode cutoffs used
            klipparams: string with KLIP-FM parameters
            calibrate_flux: if True, flux calibrate the data (if applicable)
            spectrum: if not None, the spectrum to weight the data by. Length same as dataset.wvs
        """
        return