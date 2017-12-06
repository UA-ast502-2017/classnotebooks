import os, subprocess
import astropy.io.fits as fits
from astropy import wcs
import numpy as np
import scipy.ndimage as ndimage
from scipy.ndimage.filters import median_filter

from pyklip.instruments.Instrument import Data
import pyklip.klip as klip

class Ifs(Data):
    """
    A spectral cube of Osiris IFS Data.

    Args:
        data_cube: FITS file list with 3D-cubes (Nwvs, Ny, Nx) with an osiris IFS data
        telluric_cube: single telluric reference FITS file with a 3D-cube (Nwvs, Ny, Nx) with an osiris IFS data.
        psf_cube_size: size of the psf cube to save (length along 1 dimension)
        coaddslices: if not None, combine (addition) slices together to reduce the size of the spectral cube.
                coaddslices should be an integer corresponding to the number of slices to be combined.

    Attributes:
        input: Array of shape (N,y,x) for N images of shape (y,x)
        centers: Array of shape (N,2) for N centers in the format [x_cent, y_cent]
        filenums: Array of size N for the numerical index to map data to file that was passed in
        filenames: Array of size N for the actual filepath of the file that corresponds to the data
        PAs: Array of N for the parallactic angle rotation of the target (used for ADI) [in degrees]
        wvs: Array of N wavelengths of the images (used for SDI) [in microns]. For polarization data, defaults to "None"
        IWA: a floating point scalar (not array). Specifies to inner working angle in pixels
        output: Array of shape (b, len(files), len(uniq_wvs), y, x) where b is the number of different KL basis cutoffs
        psfs: Spectral cube of size (Nwv, psfy, psfx) where psf_cube_size defines the size of psfy, psfx.
        psf_center: [x, y] location of the center of the PSF for a frame in self.psfs 
        flipx: True by default. Determines whether a relfection about the x axis is necessary to rotate image North-up East left
        nfiles: number of datacubes
        nwvs: number of wavelengths

    """
    # class initialization

    # Coonstructor
    def __init__(self, data_cube_list, telluric_cube,
                 guess_center=None,recalculate_center_cadi=False, centers = None,
                 psf_cube_size=21,
                 coaddslices=None, nan_mask_boxsize=0,median_filter_boxsize = 0,badpix2nan=False):
        super(Ifs, self).__init__()

        self.nfiles = len(data_cube_list)

        # read in the data
        self.filenums = []
        self.filenames = []
        self.prihdrs = []
        self.wvs = []
        self.centers = []
        for k,data_cube in enumerate(data_cube_list):
            with fits.open(data_cube) as hdulist:
                print("Reading "+data_cube)
                tmp_input = np.rollaxis(np.rollaxis(hdulist[0].data,2),2,1)
                self.nwvs = tmp_input.shape[0]
                try:
                    self.input = np.concatenate((self.input,tmp_input),axis=0) # 3D cube, Nwvs, Ny, Nx
                except:
                    self.input = tmp_input
                self.prihdrs.append(hdulist[0].header)
                # Move dimensions of input array to match pyklip conventions
                self.filenums.extend(np.ones(self.nwvs)*k)
                self.filenames.extend([os.path.basename(data_cube),]*self.nwvs)
                # centers are at dim/2
                init_wv = self.prihdrs[k]["CRVAL1"]/1000. # wv for first slice in mum
                dwv = self.prihdrs[k]["CDELT1"]/1000. # wv interval between 2 slices in mum
                self.wvs.extend(np.arange(init_wv,init_wv+dwv*self.nwvs,dwv))

                # Plate scale of the spectrograph
                self.platescale = float(self.prihdrs[k]["SSCALE"])

                if guess_center is None:
                    self.centers.extend(np.array([[img.shape[1]/2., img.shape[0]/2.] for img in tmp_input]))
                else:
                    self.centers.extend(np.array([guess_center,]*self.nwvs))

        if centers is not None:
            self.centers = []
            for x,y in centers:
                self.centers.extend([[x,y],]*self.nwvs)

        self.wvs = np.array(self.wvs)
        self.centers = np.array(self.centers)

        # TODO set the PAs right?
        self.PAs = np.zeros(self.wvs.shape)

        if badpix2nan:
            box_w = 3
            smooth_input = median_filter(self.input,size=(box_w,box_w,box_w))
            res_input = np.abs((self.input - smooth_input))
            res_input = res_input/np.nanstd(res_input,axis=(1,2))[:,None,None]
            where_bad = np.where(res_input>5)
            self.input[where_bad] = np.nan

        self.input[np.where(self.input==0)] = np.nan
        # import matplotlib.pyplot as plt
        # for k in range(res_input.shape[0]):
        #     # plt.figure(1)
        #     # plt.imshow(res_input[k,::-1,:],interpolation="nearest")
        #     # plt.colorbar()
        #     plt.figure(2)
        #     plt.imshow(self.input[k,::-1,:],interpolation="nearest")
        #     plt.colorbar()
        #     # plt.figure(3)
        #     # res_input[k,::-1,:][np.where(res_input[k,::-1,:]>5)] = np.nan
        #     # plt.imshow(res_input[k,::-1,:],interpolation="nearest")
        #     # plt.colorbar()
        #     plt.show()

        # import matplotlib.pyplot as plt
        # for k in range(self.nwvs):
        #     plt.imshow(self.input[50*k,::-1,:])
        #     plt.colorbar()
        #     plt.show()
        # read in the psf cube
        with fits.open(telluric_cube) as hdulist:
            psfs = hdulist[0].data # Nwvs, Ny, Nx
            # Move dimensions of input array to match pyklip conventions
            psfs = np.rollaxis(np.rollaxis(psfs,2),2,1)

            # The definition of psfs_wvs requires that no wavelengths has been skipped in the input files
            # But it works with keepslices
            self.psfs_wvs = np.arange(init_wv,init_wv+dwv*self.nwvs,dwv)

            # trim the cube
            pixelsbefore = psf_cube_size//2
            pixelsafter = psf_cube_size - pixelsbefore

            psfs = np.pad(psfs,((0,0),(pixelsbefore,pixelsafter),(pixelsbefore,pixelsafter)),mode="constant",constant_values=0)
            psfs_centers = np.array([np.unravel_index(np.nanargmax(img),img.shape) for img in psfs])
            # Change center index order to match y,x convention
            psfs_centers = [(cent[1],cent[0]) for cent in psfs_centers]
            psfs_centers = np.array(psfs_centers)
            center0 = np.median(psfs_centers,axis=0)

            # TODO Calculate precise centroid
            from pyklip.fakes import gaussfit2d
            psfs_centers = []
            self.star_peaks = []
            self.psfs = np.zeros((psfs.shape[0],psf_cube_size,psf_cube_size))
            for k,im in enumerate(psfs):
                corrflux, fwhm, spotx, spoty = gaussfit2d(im, center0[0], center0[1], searchrad=5, guessfwhm=3, guesspeak=np.nanmax(im), refinefit=True)
                #spotx, spoty = center0
                psfs_centers.append((spotx, spoty))
                self.star_peaks.append(corrflux)

                # Get the closest pixel
                xarr_spot = int(np.round(spotx))
                yarr_spot = int(np.round(spoty))
                # Extract a stamp around the sat spot
                stamp = im[(yarr_spot-pixelsbefore):(yarr_spot+pixelsafter),\
                                (xarr_spot-pixelsbefore):(xarr_spot+pixelsafter)]
                # Define coordinates grids for the stamp
                stamp_x, stamp_y = np.meshgrid(np.arange(psf_cube_size, dtype=np.float32),
                                               np.arange(psf_cube_size, dtype=np.float32))
                # Calculate the shift of the sat spot centroid relative to the closest pixel.
                dx = spotx-xarr_spot
                dy = spoty-yarr_spot

                # The goal of the following section is to remove the local background (or sky) around the sat spot.
                # The plane is defined by 3 constants (a,b,c) such that z = a*x+b*y+c
                # In order to do so we fit a 2D plane to the stamp after having masked the sat spot (centered disk)
                stamp_r = np.sqrt((stamp_x-dx-psf_cube_size//2)**2+(stamp_y-dy-psf_cube_size//2)**2)
                from copy import copy
                stamp_masked = copy(stamp)
                stamp_x_masked = stamp_x-dx
                stamp_y_masked = stamp_y-dy
                stamp_center = np.where(stamp_r<7)
                stamp_masked[stamp_center] = np.nan
                stamp_x_masked[stamp_center] = np.nan
                stamp_y_masked[stamp_center] = np.nan
                background_med =  np.nanmedian(stamp_masked)
                stamp_masked = stamp_masked - background_med
                #Solve 2d linear fit to remove background
                xx = np.nansum(stamp_x_masked**2)
                yy = np.nansum(stamp_y_masked**2)
                xy = np.nansum(stamp_y_masked*stamp_x_masked)
                xz = np.nansum(stamp_masked*stamp_x_masked)
                yz = np.nansum(stamp_y_masked*stamp_masked)
                #Cramer's rule
                a = (xz*yy-yz*xy)/(xx*yy-xy*xy)
                b = (xx*yz-xy*xz)/(xx*yy-xy*xy)
                stamp = stamp - (a*(stamp_x-dx)+b*(stamp_y-dy) + background_med)

                stamp = ndimage.map_coordinates(stamp, [stamp_y+dy, stamp_x+dx])

                self.psfs[k,:,:] = stamp

            # import matplotlib.pyplot as plt
            # plt.figure(1)
            # plt.imshow(np.nanmean(psfs,axis=0))
            # plt.figure(2)
            # plt.imshow(np.nanmean(self.psfs,axis=0))
            # plt.show()

            # Spectrum of the telluric star
            self.star_peaks = np.array(self.star_peaks)
            # TODO include brightness of the telluric star
            self.dn_per_contrast = np.array([self.star_peaks[np.where(self.psfs_wvs==wv)[0]] for wv in self.wvs])

        # we don't need to flip x for North Up East left
        self.flipx = False

        # I have no idea
        self.IWA = 0.0
        # Infinity...
        self.OWA = 10000


        self._output = None

        if coaddslices is not None:

            N_chunks = self.psfs.shape[0]//coaddslices
            self.psfs = np.array([np.nanmean(self.psfs[k*coaddslices:(k+1)*coaddslices,:,:],axis=0) for k in range(N_chunks)])
            self.psfs_wvs = np.array([np.nanmean(self.psfs_wvs[k*coaddslices:(k+1)*coaddslices]) for k in range(N_chunks)])

            new_wvs = []
            new_filenums = []
            new_centers = []
            new_PAs = []
            new_filenames = []
            new_dn_per_contrast = []
            # new_wcs = []
            for k in range(self.nfiles):
                tmp_input = copy(self.input[k*self.nwvs:(k+1)*self.nwvs,:,:])
                tmp_input = np.array([np.nanmean(tmp_input[l*coaddslices:(l+1)*coaddslices,:,:],axis=0) for l in range(N_chunks)])
                try:
                    new_input = np.concatenate((new_input,tmp_input),axis=0) # 3D cube, Nwvs, Ny, Nx
                except:
                    new_input = tmp_input
                new_nwvs = tmp_input.shape[0]

                new_wvs.extend(np.array([np.nanmean(self.wvs[k*self.nwvs:(k+1)*self.nwvs][l*coaddslices:(l+1)*coaddslices]) for l in range(N_chunks)]))
                new_filenums.extend(np.array([self.filenums[k*self.nwvs:(k+1)*self.nwvs][l*coaddslices] for l in range(N_chunks)]))
                new_centers.extend([np.nanmean(self.centers[k*self.nwvs:(k+1)*self.nwvs,:][l*coaddslices:(l+1)*coaddslices,:],axis=0) for l in range(N_chunks)])
                new_PAs.extend([self.PAs[k*self.nwvs:(k+1)*self.nwvs][l*coaddslices] for l in range(N_chunks)])
                new_filenames.extend([self.filenames[k*self.nwvs:(k+1)*self.nwvs][l*coaddslices] for l in range(N_chunks)])
                new_dn_per_contrast.extend([np.nanmean(self.dn_per_contrast[k*self.nwvs:(k+1)*self.nwvs][l*coaddslices:(l+1)*coaddslices]) for l in range(N_chunks)])
                # new_wcs.extend([self.wcs[k*self.nwvs:(k+1)*self.nwvs][l*coaddslices] for l in range(N_chunks)])


            self.input = new_input
            self.wvs = np.array(new_wvs)
            self.nwvs = new_nwvs
            self.filenums = np.array(new_filenums)
            self.centers = np.array(new_centers)
            self.PAs = np.array(new_PAs)
            self.filenames = new_filenames
            self.dn_per_contrast = np.array(new_dn_per_contrast)
            # self.wcs = new_wcs
        self.dn_per_contrast = np.squeeze(self.dn_per_contrast)

        # # TODO: need to check how it works (cf GPI)
        # self.wcs = np.array([None for _ in range(self.nfiles * self.nwvs)])
        # Creating WCS info for OSIRIS
        self.wcs = []
        for vert_angle in self.PAs:
            w = wcs.WCS()
            vert_angle = np.radians(vert_angle)
            pc = np.array([[(-1)*np.cos(vert_angle), (-1)*-np.sin(vert_angle)],[np.sin(vert_angle), np.cos(vert_angle)]])
            cdmatrix = pc * self.platescale /3600.
            w.wcs.cd = cdmatrix
            self.wcs.append(w)
        self.wcs = np.array(self.wcs)

        # import matplotlib.pyplot as plt
        # for k in range(self.nwvs):
        #     plt.imshow(self.input[k,::-1,:])
        #     plt.colorbar()
        #     plt.show()

        if median_filter_boxsize != 0:
            self.input = median_filter(self.input,size=(1,median_filter_boxsize,median_filter_boxsize))
            self.psfs = median_filter(self.psfs,size=(1,median_filter_boxsize,median_filter_boxsize))

        if nan_mask_boxsize != 0:
            # zeros are nans, and anything adjacient to a pixel less than zero is 0.
            input_nans = np.where(np.isnan(self.input))
            self.input[input_nans] = 0
            input_minfilter = ndimage.minimum_filter(self.input, (0, nan_mask_boxsize, nan_mask_boxsize))
            self.input[np.where(input_minfilter <= 0)] = np.nan
            self.input[:,0:nan_mask_boxsize//2,:] = np.nan
            self.input[:,-nan_mask_boxsize//2+1::,:] = np.nan
            self.input[:,:,0:nan_mask_boxsize//2] = np.nan
            self.input[:,:,-nan_mask_boxsize//2+1::] = np.nan

        # for wv_index in range(self.psfs.shape[0]):
        #     model_psf = self.psfs[wv_index, :, :]
        #     import matplotlib.pyplot as plt
        #     plt.imshow(model_psf)
        #     plt.show()
        # import matplotlib.pyplot as plt
        # for k in range(self.nwvs):
        #     plt.imshow(self.input[10*k,::-1,:])
        #     plt.colorbar()
        #     plt.show()

        # Required for automatically querying Simbad for the spectral type of the star.
        self.object_name = "HR8799"#self.prihdr["OBJECT"]

        if recalculate_center_cadi:
            for k in range(self.nfiles):
                tmp_input = copy(self.input[k*self.nwvs:(k+1)*self.nwvs,:,:])
                if guess_center is None:
                    xcen0,ycen0 = tmp_input.shape[2]/2., tmp_input.shape[1]/2.
                else:
                    xcen0,ycen0 = guess_center

                range_list = [100,20,4,1]
                samples = 10
                for it,width in enumerate(range_list):
                    x_list,y_list = np.linspace(xcen0-width/2.,xcen0+width/2.,samples),np.linspace(ycen0-width/2.,ycen0+width/2.,samples)
                    # print(x_list,y_list)
                    xcen_grid,ycen_grid = np.meshgrid(x_list,y_list)
                    cost_func  = np.zeros(xcen_grid.shape)
                    cost_func.shape = [np.size(cost_func)]

                    import multiprocessing as mp
                    import itertools
                    self.N_threads = mp.cpu_count()
                    pool = mp.Pool(processes=self.N_threads)
                    #multitask this
                    outputs_list = pool.map(casdi_residual_star, itertools.izip(xcen_grid.ravel(),
                                                                                ycen_grid.ravel(),
                                                                                itertools.repeat(tmp_input),
                                                                                itertools.repeat(self.wvs[k*self.nwvs:(k+1)*self.nwvs])))

                    for l,out in enumerate(outputs_list):
                        cost_func[l] = out
                    pool.close()

                    xcen0 = xcen_grid.ravel()[np.argmin(cost_func)]
                    ycen0 = ycen_grid.ravel()[np.argmin(cost_func)]

                    # import matplotlib.pyplot as plt
                    # cost_func.shape = (samples,samples)
                    # plt.figure(1)
                    # plt.subplot(2,1,1)
                    # plt.imshow(xcen_grid[::-1,:],interpolation="nearest")
                    # plt.colorbar()
                    # plt.subplot(2,1,2)
                    # plt.imshow(ycen_grid[::-1,:],interpolation="nearest")
                    # plt.colorbar()
                    # plt.figure(2)
                    # plt.imshow(cost_func[::-1,:],interpolation="nearest")
                    # plt.figure(3)
                    # plt.plot(self.wvs[k*self.nwvs:(k+1)*self.nwvs])
                    # plt.show()
                    # print(k,xcen0,ycen0)
                self.centers[k*self.nwvs:(k+1)*self.nwvs,:] = np.array([(xcen0,ycen0),]*tmp_input.shape[0])


        # else:
        #     # from scipy.optimize import leastsq
        #     # LSQ_func = lambda para: casdi_residual(para[0],para[1],self.input,self.wvs,nan2zero=True)
        #     # new_cent = leastsq(LSQ_func,(img.shape[1]/2.-sep_planet/ 0.02, img.shape[0]/2.))
        #     from scipy.optimize import minimize
        #     LSQ_func = lambda para: np.nanvar(casdi_residual(para[0],para[1],self.input,self.wvs,nan2zero=False))
        #     new_cent = minimize(LSQ_func,(img.shape[1]/2.-sep_planet/ 0.02, img.shape[0]/2.),method="nelder-mead",options={'disp':True})
        #     # casdi_residual(cent[0],cent[1],self.input,self.wvs)
        #     print("old",(img.shape[1]/2.-sep_planet/ 0.02, img.shape[0]/2.))
        #     print("new_cent",new_cent.x)
        #     exit()

    ################################
    ### Instance Required Fields ###
    ################################

    @property
    def input(self):
        return self._input
    @input.setter
    def input(self, newval):
        self._input = newval

    @property
    def centers(self):
        return self._centers
    @centers.setter
    def centers(self, newval):
        self._centers = newval

    @property
    def filenums(self):
        return self._filenums
    @filenums.setter
    def filenums(self, newval):
        self._filenums = newval

    @property
    def filenames(self):
        return self._filenames
    @filenames.setter
    def filenames(self, newval):
        self._filenames = newval

    @property
    def PAs(self):
        return self._PAs
    @PAs.setter
    def PAs(self, newval):
        self._PAs = newval

    @property
    def wvs(self):
        return self._wvs
    @wvs.setter
    def wvs(self, newval):
        self._wvs = newval

    @property
    def wcs(self):
        return self._wcs
    @wcs.setter
    def wcs(self, newval):
        self._wcs = newval


    @property
    def IWA(self):
        return self._IWA
    @IWA.setter
    def IWA(self, newval):
        self._IWA = newval

    @property
    def output(self):
        return self._output
    @output.setter
    def output(self, newval):
        self._output = newval


    ###############
    ### Methods ###
    ###############

    def readdata(self, filepaths):
        """
        Reads in the data from the files in the filelist and writes them to fields
        """
        pass


    def savedata(self, filepath, data,center=None, klipparams=None, filetype="", zaxis=None , more_keywords=None,
                 pyklip_output=True):
        """
        Save SPHERE Data.

        Args:
            filepath: path to file to output
            data: 2D or 3D data to save
            center: center of the image to be saved in the header as the keywords PSFCENTX and PSFCENTY in pixels.
                The first pixel has coordinates (0,0)
            klipparams: a string of klip parameters
            filetype: filetype of the object (e.g. "KL Mode Cube", "PSF Subtracted Spectral Cube")
            zaxis: a list of values for the zaxis of the datacub (for KL mode cubes currently)
            more_keywords (dictionary) : a dictionary {key: value, key:value} of header keywords and values which will
                                         written into the primary header
            pyklip_output: (default True) If True, indicates that the attributes self.output_wcs and self.output_centers
                            have been defined.
        
        """
        hdulist = fits.HDUList()
        hdulist.append(fits.PrimaryHDU(data=data,header=self.prihdrs[0]))

        # save all the files we used in the reduction
        # we'll assume you used all the input files
        # remove duplicates from list
        filenames = np.unique(self.filenames)
        nfiles = np.size(filenames)
        hdulist[0].header["DRPNFILE"] = (nfiles, "Num raw files used in pyKLIP")
        for i, filename in enumerate(filenames):
            hdulist[0].header["FILE_{0}".format(i)] = filename + '.fits'

        # write out psf subtraction parameters
        # get pyKLIP revision number
        pykliproot = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        # the universal_newline argument is just so python3 returns a string instead of bytes
        # this will probably come to bite me later
        try:
            pyklipver = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=pykliproot, universal_newlines=True).strip()
        except:
            pyklipver = "unknown"
        hdulist[0].header['PSFSUB'] = ("pyKLIP", "PSF Subtraction Algo")
        hdulist[0].header.add_history("Reduced with pyKLIP using commit {0}".format(pyklipver))
        hdulist[0].header['CREATOR'] = "pyKLIP-{0}".format(pyklipver)

        # store commit number for pyklip
        hdulist[0].header['pyklipv'] = (pyklipver, "pyKLIP version that was used")

        if klipparams is not None:
            hdulist[0].header['PSFPARAM'] = (klipparams, "KLIP parameters")
            hdulist[0].header.add_history("pyKLIP reduction with parameters {0}".format(klipparams))

        # write z axis units if necessary
        if zaxis is not None:
            # Writing a KL mode Cube
            if "KL Mode" in filetype:
                hdulist[0].header['CTYPE3'] = 'KLMODES'
                # write them individually
                for i, klmode in enumerate(zaxis):
                    hdulist[0].header['KLMODE{0}'.format(i)] = (klmode, "KL Mode of slice {0}".format(i))
                hdulist[0].header['CUNIT3'] = "N/A"
                hdulist[0].header['CRVAL3'] = 1
                hdulist[0].header['CRPIX3'] = 1.
                hdulist[0].header['CD3_3'] = 1.


        #use the dataset center if none was passed in
        if not pyklip_output:
            center = self.centers[0]
        else:
            center = self.output_centers[0]
        if center is not None:
            hdulist[0].header.update({'PSFCENTX': center[0], 'PSFCENTY': center[1]})
            hdulist[0].header.update({'CRPIX1': center[0], 'CRPIX2': center[1]})
            hdulist[0].header.add_history("Image recentered to {0}".format(str(center)))

        hdulist.writeto(filepath, overwrite=True)
        hdulist.close()

    def calibrate_output(self, img, spectral=False, units="contrast"):
        """
       Calibrates the flux of an output image. Can either be a broadband image or a spectral cube depending
        on if the spectral flag is set.

        Assumes the broadband flux calibration is just multiplication by a single scalar number whereas spectral
        datacubes may have a separate calibration value for each wavelength

        Args:
            img: unclaibrated image.
                 If spectral is not set, this can either be a 2-D or 3-D broadband image
                 where the last two dimensions are [y,x]
                 If specetral is True, this is a 3-D spectral cube with shape [wv,y,x]
            spectral: if True, this is a spectral datacube. Otherwise, it is a broadband image.
            units: currently only support "contrast" w.r.t central star

        Return:
            img: calibrated image of the same shape (this is the same object as the input!!!)
        """
        if units == "contrast":
            if spectral:
                # spectral cube, each slice needs it's own calibration
                numwvs = img.shape[0]
                img /= self.dn_per_contrast[:numwvs, None, None]
            else:
                # broadband image
                img /= np.nanmean(self.dn_per_contrast)
            self.flux_units = "contrast"

        return img


def casdi_residual_star(params):
    """
    Convert `f([1,2])` to `f(1,2)` call.
    It allows one to call casdi_residual() with a tuple of parameters.
    """
    return np.nanvar(casdi_residual(*params))

def casdi_residual(xcen,ycen,input,wvs,nan2zero = False):
    input_scaled = np.zeros(input.shape)
    ref_wv = np.mean(wvs)

    for k,wv in enumerate(wvs):
        input_scaled[k,:,:] = klip.align_and_scale(input[k,:,:],(xcen,ycen),(xcen,ycen),ref_wv/wv)

    # input_sub = np.zeros(input.shape)
    # lib_size = np.max([np.size(wvs)/10,10])
    # for k,wv in enumerate(wvs):
    #     # print(k,np.size(wvs))
    #     input_sub[k,:,:] = input_scaled[k,:,:] - np.nanmedian(input_scaled[np.max([0,k-lib_size]):np.min([np.size(wvs),k+lib_size]),:,:],axis=0)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        input_sub = input_scaled - np.nanmedian(input_scaled,axis=0)[None,:,:]

    if nan2zero:
        input_sub[np.where(np.isnan(input_sub))] = 0

    # for k,wv in enumerate(wvs):
    #     input_sub[k,:,:] = klip.align_and_scale(input_sub[k,:,:],(xcen,ycen),(xcen,ycen),wv/ref_wv)
    # print(xcen,ycen,np.nansum(input_sub**2),np.nanvar(input_sub.ravel()))
    # import matplotlib.pyplot as plt
    # plt.figure(1)
    # plt.imshow(np.nanmedian(input,axis=0))
    # print(np.nanvar(np.nanmedian(input,axis=0).ravel()))
    # plt.colorbar()
    # plt.figure(2)
    # plt.imshow(np.nanmedian(input_scaled,axis=0))
    # print(np.nanvar(np.nanmedian(input_scaled,axis=0).ravel()))
    # plt.colorbar()
    # plt.figure(3)
    # plt.imshow(np.nanmedian(input_sub,axis=0))
    # print(np.nanvar(np.nanmedian(input_sub,axis=0).ravel()))
    # plt.colorbar()
    # plt.show()

    return input_sub.ravel()