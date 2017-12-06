__author__ = 'JB'


import warnings
import itertools

from scipy.optimize import leastsq
from astropy.modeling import models, fitting
from matplotlib import rcParams
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

import numpy as np
from copy import copy

from pyklip.kpp.utils.mathfunc import *
from pyklip.kpp.utils.multiproc import *
from pyklip.kpp.utils.GPIimage import *


def get_image_stat_map(image,
                        image_without_planet=None,
                        IOWA = None,
                        N = 3000,
                        centroid = None,
                        r_step = 5,
                        mute = True,
                        Dr = None,
                        type = "SNR",
                        image_wide = None):
    """
    Calculate the SNR, the standard deviation or the probability (tail distribution) of a given image using annuli.

    Args:
        image: The image or cubes for which one wants the statistic.
        image_without_planet: Same as image but where real signal has been masked out. The code will actually use
                                    map to calculate the standard deviation or the PDF.
        IOWA: (IWA,OWA) inner working angle, outer working angle. It defines boundary to the zones in which the
                    statistic is calculated.
                    If None, kpp.utils.GPIimage.get_IOWA() is used.
        N: Defines the width of the ring by the number of pixels it has to include.
                The width of the annuli will therefore vary with sepration. Default is N=3000.
        centroid: (x_cen,y_cen) Define the center of the image.
                Default is x_cen = (nx-1)//2 ; y_cen = (ny-1)//2
        r_step: Distance between two consecutive annuli mean separation. Not available if "pixel based" is defined,
        mute: Won't print any logs.
        Dr: If not None defines the width of the ring as Dr. N is then ignored if Dth is defined.
        type: Indicate the type of statistic to be calculated.
                    If "SNR" (default) simple stddev calculation and returns SNR.
                    If "stddev" returns the pure standard deviation map.
                    If "proba" triggers proba calculation with pdf fitting.
        image_wide: Don't divide the image in annuli or sectors when computing the statistic.
                    Use the entire image directly.

    Return:
        The statistic map for image.
    """

    if image_without_planet is None:
        image_without_planet = image
    ny,nx = image.shape


    if centroid is None :
        centroid = ((nx-1)//2 ,(ny-1)//2)

    if image_wide is None:
        image_wide = False

    if IOWA is None:
        IWA,OWA = get_IOWA(image_without_planet, centroid = centroid)
    else:
        IWA,OWA = IOWA

    if type == "proba":
        pdf_list, cdf_list, sampling_list, annulus_radii_list = get_image_PDF(image_without_planet,(IWA,OWA),N,centroid,r_step=r_step,Dr=Dr,image_wide = image_wide)

        pdf_radii = np.array(annulus_radii_list)[:,0]

        stat_map = np.zeros(image.shape) + np.nan
        # Build the x and y coordinates grids
        x_grid, y_grid = np.meshgrid(np.arange(nx)-centroid[0], np.arange(ny)-centroid[1])

        # Calculate the radial distance of each pixel
        r_grid = abs(x_grid +y_grid*1j)

        image_finite = np.where(np.isfinite(image))

        #Build the cdf_models from interpolation
        cdf_interp_list = []
        for sampling,cdf_sampled in zip(sampling_list,cdf_list):
            cdf_interp_list.append(interp1d(sampling,cdf_sampled,kind = "linear",bounds_error = False, fill_value=1.0))

            #f = interp1d(sampling,cdf_sampled,kind = "linear",bounds_error = False, fill_value=1.0)
            #plt.plot(np.arange(-10,10,0.1),f(np.arange(-10,10,0.1)))
            #plt.show()

        for k,l in zip(image_finite[0],image_finite[1]):
            #stdout.flush()
            #stdout.write("\r%d" % k)
            r = r_grid[k,l]

            if r < OWA:
                r_closest_id, r_closest = min(enumerate(pdf_radii), key=lambda x: abs(x[1]-r))

                if (r-r_closest) < 0:
                    r_closest_id2 = r_closest_id - 1
                else:
                    r_closest_id2 = r_closest_id + 1

                if (r_closest_id2 < 0) or (r_closest_id2 > (pdf_radii.size-1)):
                    stat_map[k,l] = 1-cdf_interp_list[r_closest_id](image[k,l])
                    #plt.plot(np.arange(-10,10,0.1),cdf(np.arange(-10,10,0.1)))
                    #plt.show()
                else:
                    r_closest2 = pdf_radii[r_closest_id2]
                    stat_map[k,l] = 1-(cdf_interp_list[r_closest_id](image[k,l])*abs(r-r_closest2)+cdf_interp_list[r_closest_id2](image[k,l])*abs(r-r_closest))/abs(r_closest-r_closest2)
            else:
                    stat_map[k,l] = 1-cdf_interp_list[pdf_radii.size-1](image[k,l])

        if 0:
            import matplotlib.pyplot as plt
            plt.figure(1)
            plt.subplot(1,3,1)
            plt.imshow(np.log10(stat_map),interpolation="nearest")
            plt.colorbar()
            plt.subplot(1,3,2)
            plt.imshow(image,interpolation="nearest")
            plt.subplot(1,3,3)
            plt.imshow(image_without_planet,interpolation="nearest")
            plt.show()

        return -np.log10(stat_map)
    else:
        stddev_list, annulus_radii_list = get_image_stddev(image_without_planet,(IWA,OWA),N,centroid,r_step=r_step,Dr=Dr,
                                                           image_wide=image_wide)

        radii = np.array(annulus_radii_list)[:,0]
        #print(radii,stddev_list)

        if not image_wide:
            stddev_func = interp1d(radii,stddev_list,kind = "linear",bounds_error = False, fill_value=np.nan)
        else:
            stddev_func = lambda x: stddev_list[0]

        #plt.figure()
        #plt.plot(np.linspace(0,140,200),stddev_func(np.linspace(0,140,200)))
        #plt.show()

        stat_map = np.zeros(image.shape) + np.nan
        ny,nx = image.shape

        # Build the x and y coordinates grids
        x_grid, y_grid = np.meshgrid(np.arange(nx)-centroid[0], np.arange(ny)-centroid[1])

        # Calculate the radial distance of each pixel
        r_grid = abs(x_grid +y_grid*1j)

        image_finite = np.where(np.isfinite(image))

        for k,l in zip(image_finite[0],image_finite[1]):
            #stdout.flush()
            #stdout.write("\r%d" % k)
            r = r_grid[k,l]
            if type == "SNR":
                stat_map[k,l] = image[k,l]/stddev_func(r)
            elif type == "stddev":
                stat_map[k,l] = stddev_func(r)

        return stat_map


def get_image_PDF(image,IOWA=None,N = 2000,centroid = None, r_step = None,Dr=None,image_wide = None):
    """
    Calculate the PDF of a given image using annuli.

    Args:
        image: The image or cubes for which one wants the statistic.
        IOWA: (IWA,OWA) inner working angle, outer working angle. It defines boundary to the zones in which the
                    statistic is calculated.
                    If None, kpp.utils.GPIimage.get_IOWA() is used.
        N: Defines the width of the ring by the number of pixels it has to include.
                The width of the annuli will therefore vary with sepration. Default is N=3000.
        centroid: (x_cen,y_cen) Define the center of the image.
                Default is x_cen = (nx-1)//2 ; y_cen = (ny-1)//2
        r_step: Distance between two consecutive annuli mean separation. Not available if "pixel based" is defined,
        Dr: If not None defines the width of the ring as Dr. N is then ignored if Dth is defined.
        image_wide: Don't divide the image in annuli or sectors when computing the statistic.
                    Use the entire image directly. Not available if "pixel based: is defined,

    Return:
        pdf_list: List of PDF values for each annulus. The sampling of each PDF can be found in sampling_list.
        cdf_list: CDF values for each annulus. The sampling of each CDF can be found in sampling_list.
        sampling_list: Sampling for the PDF and the CDF.
        annulus_radii_list: List of ((r_min+r_max)/2.,r_min,r_max) with r_min,r_max the boundaries of an annulus.
    """
    if image_wide is None:
        image_wide = False
    if IOWA is None:
        IWA,OWA = get_IOWA(image, centroid = centroid)
    else:
        IWA,OWA = IOWA
    ny,nx = image.shape

    if 0:
        import matplotlib.pyplot as plt
        fig = 1
        plt.figure(fig,figsize=(16,8))
        plt.subplot(121)
        plt.imshow(image,interpolation="nearest")
        plt.colorbar()

        data = image[np.where(np.isfinite(image))]
        im_std = np.std(data)
        bins = np.arange(np.min(data),np.max(data),im_std/10.)
        im_histo = np.histogram(data, bins=bins)[0]

        N_bins = bins.size-1
        center_bins = 0.5*(bins[0:N_bins]+bins[1:N_bins+1])
        plt.subplot(122)
        plt.plot(center_bins,np.array(im_histo,dtype="double"),'bx-', markersize=5,linewidth=3)
        plt.grid(True)
        ax = plt.gca()
        ax.set_yscale('log')
        plt.show()


    image_mask = np.ones((ny,nx))
    image_mask[np.where(np.isnan(image))] = 0

    if centroid is None :
        x_cen = (nx-1)//2 ; y_cen = (ny-1)//2
    else:
        x_cen, y_cen = centroid

    # Build the x and y coordinates grids
    x, y = np.meshgrid(np.arange(nx)-x_cen, np.arange(ny)-y_cen)
    # Calculate the radial distance of each pixel
    r_grid = abs(x +y*1j)
    #th_grid = np.arctan2(x,y)

    if not image_wide:
        # Define the radii intervals for each annulus
        if Dr is None:
            r0 = IWA
            annuli_radii = []
            if r_step is None:
                while np.sqrt(N/np.pi+r0**2) < OWA:
                    annuli_radii.append((r0,np.sqrt(N/np.pi+r0**2)))
                    r0 = np.sqrt(N/np.pi+r0**2)
            else:
                while np.sqrt(N/np.pi+r0**2) < OWA:
                    annuli_radii.append((r0,np.sqrt(N/np.pi+r0**2)))
                    r0 += r_step

            annuli_radii.append((r0,np.max([ny,nx])))
        else:
            annuli_radii = []
            for r in np.arange(IWA+Dr,OWA-Dr,Dr):
                annuli_radii.append((r-Dr,r+Dr))
    else:
        annuli_radii = [(IWA,OWA)]

    N_annuli = len(annuli_radii)


    pdf_list = []
    cdf_list = []
    sampling_list = []
    annulus_radii_list = []
    if 0:
        rings = np.zeros((ny,nx))+np.nan
    for it, rminmax in enumerate(annuli_radii):
        r_min,r_max = rminmax
        #print(rminmax)

        where_ring = np.where((r_min< r_grid) * (r_grid < r_max) * image_mask)
        #print(np.size(where_ring[0]))
        if 0:
            import matplotlib.pyplot as plt
            image_tmp = copy(image)
            image_tmp[where_ring] = np.nan
            plt.figure(2)
            plt.imshow(image_tmp,interpolation="nearest")
            plt.show()
        if 0:
            rings[where_ring] = it

        data = image[where_ring]
        cdf_model, pdf_model, sampling, im_histo, center_bins  = get_cdf_model(data)

        pdf_list.append(pdf_model)
        cdf_list.append(cdf_model)
        sampling_list.append(sampling)
        annulus_radii_list.append(((r_min+r_max)/2.,r_min,r_max))
        if 0:
            import matplotlib.pyplot as plt
            fig = 1
            plt.figure(fig,figsize=(8,8))
            plt.subplot(np.ceil(np.sqrt(N_annuli)),np.ceil(np.sqrt(N_annuli)),it)
            plt.plot(sampling,pdf_model,'b-',linewidth=3)
            plt.plot(sampling,1.-cdf_model,'r-',linewidth=3)
            plt.xlabel('criterion value', fontsize=20)
            plt.ylabel('Probability of the value', fontsize=20)
            plt.grid(True)
            ax = plt.gca()
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            ax.legend(['flat cube histogram','flat cube histogram (Gaussian fit)','planets'], loc = 'upper right', fontsize=12)
            ax.set_yscale('log')
            plt.ylim((10**-7,10))

    if 0:
        import matplotlib.pyplot as plt
        plt.figure(2,figsize=(8,8))
        plt.imshow(rings,interpolation="nearest")
        plt.show()

    return pdf_list, cdf_list, sampling_list, annulus_radii_list


def get_image_stddev(image,
                     IOWA = None,
                     N = None,
                     centroid = None,
                     r_step = 2,
                     Dr=2,
                     image_wide = None,
                     resolution = None):
    """
    Calculate the standard deviation of a given image using annuli.

    Args:
        image: The image or cubes for which one wants the statistic.
        IOWA: (IWA,OWA) inner working angle, outer working angle. It defines boundary to the zones in which the
                    statistic is calculated.
                    If None, kpp.utils.GPIimage.get_IOWA() is used.
        N: Defines the width of the ring by the number of pixels it has to include.
                The width of the annuli will therefore vary with sepration. Default is N=3000.
        centroid: (x_cen,y_cen) Define the center of the image.
                Default is x_cen = (nx-1)//2 ; y_cen = (ny-1)//2
        r_step: Distance between two consecutive annuli mean separation. Not available if "pixel based" is defined,
        Dr: If not None defines the width of the ring as Dr. N is then ignored if Dth is defined.
        image_wide: Don't divide the image in annuli or sectors when computing the statistic.
                    Use the entire image directly. Not available if "pixel based: is defined,
        resolution: Diameter of the resolution elements (in pix) used to do do the small sample statistic.
                For e.g., FWHM of the PSF.
                /!\ I am not sure the implementation is correct. We should probably do better.

    Return:
        stddev_list: standard deviation values at the center of each
        annulus_radii_list: List of ((r_min+r_max)/2.,r_min,r_max) with r_min,r_max the boundaries of an annulus.
    """
    if image_wide is None:
        image_wide = False


    if IOWA is None:
        IWA,OWA,inner_mask,outer_mask = get_occ(image, centroid = centroid)
    else:
        IWA,OWA = IOWA

    ny,nx = image.shape

    image_mask = np.ones((ny,nx))
    image_mask[np.where(np.isnan(image))] = 0

    if centroid is None :
        x_cen = (nx-1)//2 ; y_cen = (ny-1)//2
    else:
        x_cen, y_cen = centroid

    # Build the x and y coordinates grids
    x, y = np.meshgrid(np.arange(nx)-x_cen, np.arange(ny)-y_cen)
    # Calculate the radial distance of each pixel
    r_grid = abs(x +y*1j)
    #th_grid = np.arctan2(x,y)

    if not image_wide:
        # Define the radii intervals for each annulus
        if Dr is None:
            r0 = IWA
            annuli_radii = []
            if r_step is None:
                while np.sqrt(N/np.pi+r0**2) < OWA:
                    annuli_radii.append((r0,np.sqrt(N/np.pi+r0**2)))
                    r0 = np.sqrt(N/np.pi+r0**2)
            else:
                while np.sqrt(N/np.pi+r0**2) < OWA:
                    annuli_radii.append((r0,np.sqrt(N/np.pi+r0**2)))
                    r0 += r_step

            annuli_radii.append((r0,np.max([ny,nx])))
        else:
            annuli_radii = []
            # for r in np.arange(IWA+Dr,nx/2-Dr,Dr):
            for r in np.arange(IWA+Dr,OWA,Dr):
                annuli_radii.append((r-Dr,r+Dr))
    else:
        annuli_radii = [(IWA,OWA)]
    #N_annuli = len(annuli_radii)


    stddev_list = []
    annulus_radii_list = []
    for it, rminmax in enumerate(annuli_radii):
        r_min,r_max = rminmax

        where_ring = np.where((r_min< r_grid) * (r_grid < r_max) * image_mask)

        data = image[where_ring]
        sigma = np.nanstd(data)

        if resolution is not None and np.size(data) != 0:
            N_res_elt = np.size(data)/(np.pi*(resolution/2.)**2)
            sigma = sigma*np.sqrt(1+1./N_res_elt)
        stddev_list.append(sigma)
        annulus_radii_list.append(((r_min+r_max)/2.,r_min,r_max))

    return stddev_list, annulus_radii_list


def get_cdf_model(data,interupt_plot = False,pure_gauss=False):
    """
    Calculate a model CDF for some data.

    /!\ This function is for some reason still a work in progress. JB could never decide what the best option was.
    But it should work even if the code is a mess.

    Args:
        data: arrays of samples from a random variable
        interupt_plot: Plot the histogram and model fit. It
        pure_gauss: Assume gaussian statistic. Do not fit exponential tails.

    Return: (cdf_model,new_sampling,im_histo, center_bins) with:
                cdf_model: The cdf model = np.cumsum(pdf_model)
                pdf_model: The pdf model
                sampling: sampling of pdf/cdf_model
                im_histo: histogram from original data
                center_bins: bin centers for im_histo
    """
    pdf_model,sampling,im_histo,center_bins = get_pdf_model(data,interupt_plot=interupt_plot,pure_gauss=pure_gauss)
    return np.cumsum(pdf_model),pdf_model,sampling,im_histo,center_bins


def get_pdf_model(data,interupt_plot = False,pure_gauss = False):
    """
    Calculate a model PDF for some data.

    /!\ This function is for some reason still a work in progress. JB could never decide what the best option was.
    But it should work even if the code is a mess.

    Args:
        data: arrays of samples from a random variable
        interupt_plot: Plot the histogram and model fit. It
        pure_gauss: Assume gaussian statistic. Do not fit exponential tails.

    Return: (pdf_model,new_sampling,im_histo, center_bins) with:
                pdf_model: The pdf model
                new_sampling: sampling of pdf_model
                im_histo: histogram from original data
                center_bins: bin centers for im_histo
    """
    im_std = np.std(data)
    #print(im_std)
    bins = np.arange(np.min(data),np.max(data),im_std/5.)
    im_histo = np.histogram(data, bins=bins)[0]


    N_bins = bins.size-1
    center_bins = 0.5*(bins[0:N_bins]+bins[1:N_bins+1])

    g_init = models.Gaussian1D(amplitude=np.max(im_histo), mean=0.0, stddev=im_std)
    fit_g = fitting.LevMarLSQFitter()
    warnings.simplefilter('ignore')
    g = fit_g(g_init, center_bins, im_histo)#, weights=1/im_histo)
    g.stddev = abs(g.stddev)

    right_side_noZeros = np.where((center_bins > (g.mean+2*g.stddev))*(im_histo != 0))
    N_right_bins_noZeros = len(right_side_noZeros[0])
    left_side_noZeros = np.where((center_bins < (g.mean-2*g.stddev))*(im_histo != 0))
    N_left_bins_noZeros = len(left_side_noZeros[0])

    right_side = np.where((center_bins > (g.mean+2*g.stddev)))
    left_side = np.where((center_bins < (g.mean-2*g.stddev)))

    if not pure_gauss:
        if N_right_bins_noZeros < 5:
            where_pos_zero = np.where((im_histo == 0) * (center_bins > g.mean))
            if len(where_pos_zero[0]) != 0:
                right_side_noZeros = (range(where_pos_zero[0][0]-5,where_pos_zero[0][0]),)
                right_side = (range(where_pos_zero[0][0]-5,center_bins.size),)
            else:
                right_side_noZeros = (range(center_bins.size-5,center_bins.size),)
                right_side = right_side_noZeros
            N_right_bins_noZeros = 5

        if N_left_bins_noZeros < 5:
            where_neg_zero = np.where((im_histo == 0) * (center_bins < g.mean))
            if len(where_neg_zero[0]) != 0:
                left_side_noZeros = (range(where_neg_zero[0][len(where_neg_zero[0])-1]+1,where_neg_zero[0][len(where_neg_zero[0])-1]+6),)
                left_side = (range(0,where_neg_zero[0][len(where_neg_zero[0])-1]+6),)
            else:
                left_side_noZeros = (range(0,5),)
                left_side = left_side_noZeros
            N_left_bins_noZeros = 5

        #print(left_side,right_side)
        #print(im_histo[left_side],im_histo[right_side])
        #print(right_side_noZeros,left_side_noZeros)
        #print(im_histo[right_side_noZeros],im_histo[left_side_noZeros])



        #print(N_right_bins_noZeros,N_left_bins_noZeros)
        if N_right_bins_noZeros >= 2:
            alpha0 = (np.log(im_histo[right_side_noZeros[0][N_right_bins_noZeros-1]])-np.log(im_histo[right_side_noZeros[0][0]]))/(center_bins[right_side_noZeros[0][0]]-center_bins[right_side_noZeros[0][N_right_bins_noZeros-1]])
            m_alpha0 = -np.log(im_histo[right_side_noZeros[0][0]])-alpha0*center_bins[right_side_noZeros[0][0]]
            param0_rightExp = (m_alpha0,alpha0)

            LSQ_func = lambda para: LSQ_model_exp((bins[0:bins.size-1])[right_side], im_histo[right_side],para[0],para[1])
            param_fit_rightExp = leastsq(LSQ_func,param0_rightExp)
        else:
            param_fit_rightExp = None
        #print(param0_rightExp,param_fit_rightExp)

        if N_left_bins_noZeros >= 2:
            alpha0 = (np.log(im_histo[left_side_noZeros[0][N_left_bins_noZeros-1]])-np.log(im_histo[left_side_noZeros[0][0]]))/(center_bins[left_side_noZeros[0][0]]-center_bins[left_side_noZeros[0][N_left_bins_noZeros-1]])
            m_alpha0 = -np.log(im_histo[left_side_noZeros[0][0]])-alpha0*center_bins[left_side_noZeros[0][0]]
            param0_leftExp = (m_alpha0,alpha0)

            LSQ_func = lambda para: LSQ_model_exp((bins[0:bins.size-1])[left_side], im_histo[left_side],para[0],para[1])
            param_fit_leftExp = leastsq(LSQ_func,param0_leftExp)
        else:
            param_fit_leftExp = None
        #print(param0_leftExp,param_fit_leftExp)


    new_sampling = np.arange(2*np.min(data),4*np.max(data),im_std/100.)

    if pure_gauss:
        pdf_model = g(new_sampling)
        pdf_model_exp = new_sampling*0
    else:
        pdf_model_gaussian = interp1d(center_bins,np.array(im_histo,dtype="double"),kind = "cubic",bounds_error = False, fill_value=0.0)(new_sampling)


    if not pure_gauss:
        right_side2 = np.where((new_sampling >= g.mean))
        left_side2 = np.where((new_sampling < g.mean))

        #print(g.mean+0.0,g.stddev+0.0)
        pdf_model_exp = np.zeros(new_sampling.size)
        weights = np.zeros(new_sampling.size)
        if param_fit_rightExp is not None:
            pdf_model_exp[right_side2] = model_exp(new_sampling[right_side2],*param_fit_rightExp[0])
            weights[right_side2] = np.tanh((new_sampling[right_side2]-(g.mean+2*g.stddev))/(0.1*g.stddev))
        else:
            weights[right_side2] = -1.

        if param_fit_leftExp is not None:
            pdf_model_exp[left_side2] = model_exp(new_sampling[left_side2],*param_fit_leftExp[0])
            weights[left_side2] = np.tanh(-(new_sampling[left_side2]-(g.mean-2*g.stddev))/(0.1*g.stddev))
        else:
            weights[left_side2] = -1.


        weights = 0.5*(weights+1.0)

        #weights[np.where(weights > 1-10^-3)] = 1


        pdf_model = weights*pdf_model_exp + (1-weights)*pdf_model_gaussian
        #pdf_model[np.where(weights > 1-10^-5)] = pdf_model_exp[np.where(pdf_model > 1-10^-5)]

    if 0:
        import matplotlib.pyplot as plt
        fig = 2
        plt.figure(fig,figsize=(8,8))
        plt.plot(new_sampling, weights, "r")
        #plt.plot(new_sampling, (1-weights), "--r")
        #plt.plot(new_sampling, pdf_model_exp, "g")
        #plt.plot(new_sampling, pdf_model_gaussian, "b")
        #plt.plot(new_sampling, pdf_model, "c") #/np.sum(pdf_model)
        #plt.plot(new_sampling, 1-np.cumsum(pdf_model/np.sum(pdf_model)), "--.")
        ax = plt.gca()
        #ax.set_yscale('log')
        plt.grid(True)
        #plt.ylim((10**-15,100000))
        #plt.xlim((1*np.min(data),2*np.max(data)))
        plt.show()

    if interupt_plot:
        import matplotlib.pyplot as plt
        rcParams.update({'font.size': 20})
        fig = 2
        plt.close(2)
        plt.figure(fig,figsize=(16,8))
        plt.subplot(121)
        plt.plot(new_sampling,pdf_model,'r-',linewidth=5)
        plt.plot(center_bins,g(center_bins),'c--',linewidth=3)
        plt.plot(new_sampling,pdf_model_exp,'g--',linewidth=3)
        plt.plot(center_bins,np.array(im_histo,dtype="double"),'b.', markersize=10,linewidth=3)
        #plt.plot(new_sampling,np.cumsum(pdf_model),'g.')
        plt.xlabel('Metric value')
        plt.ylabel('Number per bin')
        plt.xlim((2*np.min(data),2*np.max(data)))
        plt.grid(True)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax = plt.gca()
        ax.tick_params(axis='x')
        ax.tick_params(axis='y')
        ax.legend(['PDF Model Fit','Central Gaussian Fit','Tails Exponential Fit','Histogram'], loc = 'lower left', fontsize=15)
        ax.set_yscale('log')
        plt.ylim((10**-1,10000))

    pdf_model /= np.sum(pdf_model)

    if interupt_plot:
        host = host_subplot(122, axes_class=AA.Axes)
        par1 = host.twinx()
        p1, = host.plot(new_sampling,pdf_model/(new_sampling[1]-new_sampling[0]),'r-',linewidth=5)
        host.tick_params(axis='x', labelsize=20)
        host.tick_params(axis='y', labelsize=20)
        host.set_ylim((10**-3,10**2))
        host.set_yscale('log')
        p2, = par1.plot(new_sampling,1-np.cumsum(pdf_model),'g-',linewidth=5)
        par1.set_ylabel("False positive rate")
        par1.set_yscale('log')
        par1.set_ylim((10**-4,10.))
        host.axis["left"].label.set_color(p1.get_color())
        par1.axis["right"].label.set_color(p2.get_color())
        plt.xlabel('Metric value')
        plt.ylabel('Probability density')
        plt.xlim((2*np.min(data),2*np.max(data)))
        plt.grid(True)
        plt.legend(['PDF model','Tail distribution'], loc = 'lower left', fontsize=15)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.show()

    return pdf_model,new_sampling,np.array(im_histo,dtype="double"), center_bins


def get_cube_stddev(cube,IOWA,N = 2000,centroid = None, r_step = None,Dr=None):
    # Not tested
    nl,ny,nx = cube.shape

    stddev_table = []
    annulus_radii_table = []
    for k in range(nl):
        stddev_list, annulus_radii_list = get_image_stddev(cube[k,:,:],IOWA,N = N,centroid = centroid, r_step = r_step,Dr=Dr)
        stddev_table.append(stddev_list)
        annulus_radii_table.append(annulus_radii_list)

    return stddev_table,annulus_radii_table