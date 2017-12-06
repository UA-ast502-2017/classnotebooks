
#KLIP Forward Modelling
import os
from sys import stdout
from time import time
import itertools
import multiprocessing as mp
import ctypes

import numpy as np
import scipy.linalg as la
from scipy.stats import norm
import scipy.ndimage as ndimage
import scipy.interpolate as sinterp

import pyklip.klip as klip
from pyklip.parallelized import _arraytonumpy, high_pass_filter_imgs


#Logic to test mkl exists
try:
    import mkl
    mkl_exists = True
except ImportError:
    mkl_exists = False

# for debugging purposes
parallel = True

def find_id_nearest(array, value):
    """
    Find index of the closest value in input array to input value
    Args:
        array: 1D array
        value: scalar value
    Returns:
        Index of the nearest value in array
    """
    index = (np.abs(array-value)).argmin()
    return index

def klip_math(sci, refs, numbasis, covar_psfs=None, model_sci=None, models_ref=None, spec_included=False, spec_from_model=False):
    """
    linear algebra of KLIP with linear perturbation
    disks and point source

    Args:
        sci: array of length p containing the science data
        refs: N x p array of the N reference images that
                  characterizes the extended source with p pixels
        numbasis: number of KLIP basis vectors to use (can be an int or an array of ints of length b)
                If numbasis is [None] the number of KL modes to be used is automatically picked based on the eigenvalues.
        covar_psfs: covariance matrix of reference images (for large N, useful). Normalized following numpy normalization in np.cov documentation
        # The following arguments must all be passed in, or none of them for klip_math to work
        models_ref: N x p array of the N models corresponding to reference images. Each model should be normalized to unity (no flux information)
        model_sci: array of size p corresponding to the PSF of the science frame
        Sel_wv: wv x N array of the the corresponding wavelength for each reference PSF
        input_spectrum: array of size wv with the assumed spectrum of the model


    Returns:
        sub_img_rows_selected: array of shape (p,b) that is the PSF subtracted data for each of the b KLIP basis
                               cutoffs. If numbasis was an int, then sub_img_row_selected is just an array of length p
        KL_basis: array of KL basis (shape of [numbasis, p])
        If models_ref is passed in (not None):
            delta_KL_nospec: array of shape (b, wv, p) that is the almost perturbed KL modes just missing spectral info
        Otherwise:
            evals: array of eigenvalues (size of max number of KL basis requested aka nummaxKL)
            evecs: array of corresponding eigenvectors (shape of [p, nummaxKL])
    """

    # remove means and nans
    sci_mean_sub = sci - np.nanmean(sci)
    sci_nanpix = np.where(np.isnan(sci_mean_sub))
    sci_mean_sub[sci_nanpix] = 0
    refs_mean_sub = refs - np.nanmean(refs, axis=1)[:, None]
    refs_mean_sub[np.where(np.isnan(refs_mean_sub))] = 0

    # calculate the covariance matrix for the reference PSFs
    # note that numpy.cov normalizes by p-1 to get the NxN covariance matrix
    # we have to correct for that since that's not part of the equation in the KLIP paper
    if covar_psfs is None:
        # call np.cov to make the covariance matrix
        covar_psfs = np.cov(refs_mean_sub)
    # fix normalization of covariance matrix
    covar_psfs *= (np.size(sci)-1)

    # calculate the total number of KL basis we need based on the number of reference PSFs and number requested
    tot_basis = covar_psfs.shape[0]

    if numbasis[0] is None:
        evals, evecs = la.eigh(covar_psfs, eigvals = (tot_basis-np.min([100,tot_basis-1]), tot_basis-1))
        evals = np.copy(evals[::-1])
        evecs = np.copy(evecs[:,::-1])
        # import matplotlib.pyplot as plt
        # plt.plot(np.log10(evals))
        # plt.show()

        max_basis = find_id_nearest(evals/evals[2],10**-1.25)+1
        print(max_basis)
        evals = evals[:max_basis]
        evecs = evecs[:,:max_basis]
    else:
        numbasis = np.clip(numbasis - 1, 0, tot_basis-1)
        max_basis = np.max(numbasis) + 1

        # calculate eigenvectors/values of covariance matrix
        evals, evecs = la.eigh(covar_psfs, eigvals = (int(tot_basis-max_basis), int(tot_basis-1)))
        evals = np.copy(evals[::-1])
        evecs = np.copy(evecs[:,::-1])

    # project on reference PSFs to generate KL modes
    KL_basis = np.dot(refs_mean_sub.T,evecs)
    KL_basis = KL_basis * (1. / np.sqrt(evals))[None,:]
    KL_basis = KL_basis.T # flip dimensions to be consistent with Laurent's paper

    # If we are interested in only one numbasis there is no need to use the triangular matrices.
    if np.size(numbasis) == 1:
        N_pix = np.size(sci_mean_sub)
        sci_rows_selected = np.reshape(sci_mean_sub, (1,N_pix))

        # sci_nanpix = np.where(np.isnan(sci_rows_selected))
        # sci_rows_selected[sci_nanpix] = 0

        # run KLIP on this sector and subtract the stellar PSF
        inner_products = np.dot(sci_rows_selected, KL_basis.T)
        inner_products[0,int(max_basis)::]=0

        klip_reconstruction = np.dot(inner_products, KL_basis)

    else:
        # prepare science frame for KLIP subtraction
        sci_mean_sub_rows = np.tile(sci_mean_sub, (max_basis,1))
        sci_rows_selected = np.tile(sci_mean_sub, (np.size(numbasis),1))

        # sci_nanpix = np.where(np.isnan(sci_mean_sub_rows))
        # sci_mean_sub_rows[sci_nanpix] = 0
        # sci_nanpix = np.where(np.isnan(sci_rows_selected))
        # sci_rows_selected[sci_nanpix] = 0

        # run KLIP on this sector and subtract the stellar PSF
        inner_products = np.dot(sci_mean_sub_rows, KL_basis.T)
        lower_tri = np.tril(np.ones([max_basis,max_basis]))
        inner_products = inner_products * lower_tri

        if numbasis[0] is None:
            klip_reconstruction = np.dot(inner_products[[max_basis-1],:], KL_basis)
        else:
            klip_reconstruction = np.dot(inner_products[numbasis,:], KL_basis)


    sub_img_rows_selected = sci_rows_selected - klip_reconstruction
    sub_img_rows_selected[:, sci_nanpix[0]] = np.nan


    if models_ref is not None:


        if spec_included:
            delta_KL = perturb_specIncluded(evals, evecs, KL_basis, refs_mean_sub, models_ref)
            return sub_img_rows_selected.transpose(), KL_basis,  delta_KL
        elif spec_from_model:
            delta_KL_nospec = perturb_nospec_modelsBased(evals, evecs, KL_basis, refs_mean_sub, models_ref)
            return sub_img_rows_selected.transpose(), KL_basis,  delta_KL_nospec
        else:
            delta_KL_nospec = pertrurb_nospec(evals, evecs, KL_basis, refs_mean_sub, models_ref)
            return sub_img_rows_selected.transpose(), KL_basis,  delta_KL_nospec


    else:

        return sub_img_rows_selected.transpose(), KL_basis, evals, evecs

# @profile
def perturb_specIncluded(evals, evecs, original_KL, refs, models_ref, return_perturb_covar=False):
    """
    Perturb the KL modes using a model of the PSF but with the spectrum included in the model. Quicker than the others

    Args:
        evals: array of eigenvalues of the reference PSF covariance matrix (array of size numbasis)
        evecs: corresponding eigenvectors (array of size [p, numbasis])
        orignal_KL: unpertrubed KL modes (array of size [numbasis, p])
        refs: N x p array of the N reference images that
                  characterizes the extended source with p pixels
        models_ref: N x p array of the N models corresponding to reference images.
                    Each model should contain spectral informatoin
        model_sci: array of size p corresponding to the PSF of the science frame

    Returns:
        delta_KL_nospec: perturbed KL modes. Shape is (numKL, wv, pix)
    """

    max_basis = original_KL.shape[0]
    N_ref = refs.shape[0]
    N_pix = original_KL.shape[1]

    refs_mean_sub = refs - np.nanmean(refs, axis=1)[:, None]
    refs_mean_sub[np.where(np.isnan(refs_mean_sub))] = 0

    models_mean_sub = models_ref # - np.nanmean(models_ref, axis=1)[:,None] should this be the case?
    models_mean_sub[np.where(np.isnan(models_mean_sub))] = 0

    #print(evals.shape,evecs.shape,original_KL.shape,refs.shape,models_ref.shape)

    evals_tiled = np.tile(evals,(max_basis,1))
    np.fill_diagonal(evals_tiled,np.nan)
    evals_sqrt = np.sqrt(evals)
    evalse_inv_sqrt = 1./evals_sqrt
    evals_ratio = (evalse_inv_sqrt[:,None]).dot(evals_sqrt[None,:])
    beta_tmp = 1./(evals_tiled.transpose()- evals_tiled)
    #print(evals)
    beta_tmp[np.diag_indices(np.size(evals))] = -0.5/evals
    beta = evals_ratio*beta_tmp

    C_partial = models_mean_sub.dot(refs_mean_sub.transpose())
    C = C_partial+C_partial.transpose()
    #C =  models_mean_sub.dot(refs_mean_sub.transpose())+refs_mean_sub.dot(models_mean_sub.transpose())
    alpha = (evecs.transpose()).dot(C).dot(evecs)

    delta_KL = (beta*alpha).dot(original_KL)+(evalse_inv_sqrt[:,None]*evecs.transpose()).dot(models_mean_sub)

    if return_perturb_covar:
        return delta_KL, C
    else:
        return delta_KL


def perturb_nospec_modelsBased(evals, evecs, original_KL, refs, models_ref_list):
    """
    Perturb the KL modes using a model of the PSF but with no assumption on the spectrum. Useful for planets.

    By no assumption on the spectrum it means that the spectrum has been factored out of Delta_KL following equation (4)
    of Laurent Pueyo 2016 noted bold "Delta Z_k^lambda (x)". In order to get the actual perturbed KL modes one needs to
    multpily it by a spectrum.

    Effectively does the same thing as pertrurb_nospec() but in a different way. It injects models with dirac spectrum
    (all but one vanishing wavelength) and because of the linearity of the problem allow one de get reconstruct the
    perturbed KL mode for any spectrum.
    The difference however in the pertrurb_nospec() case is that the spectrum here is the asummed to be the same for all
     cubes while pertrurb_nospec() fit each cube independently.

    Args:
        evals:
        evecs:
        original_KL:
        refs:
        models_ref:
    Returns:
        delta_KL_nospec
    """

    max_basis = original_KL.shape[0]
    N_wv,N_ref,N_pix = models_ref_list.shape

    refs_mean_sub = refs - np.nanmean(refs, axis=1)[:, None]
    refs_mean_sub[np.where(np.isnan(refs_mean_sub))] = 0

    # perturbed KL modes
    delta_KL_nospec = np.zeros([max_basis, N_wv, N_pix]) # (numKL,N_ref,N_pix)

    for k,models_ref in enumerate(models_ref_list):
        models_mean_sub = models_ref # - np.nanmean(models_ref, axis=1)[:,None] should this be the case?
        models_mean_sub[np.where(np.isnan(models_mean_sub))] = 0

        evals_tiled = np.tile(evals,(N_ref,1))
        np.fill_diagonal(evals_tiled,np.nan)
        evals_sqrt = np.sqrt(evals)
        evalse_inv_sqrt = 1./evals_sqrt
        evals_ratio = (evalse_inv_sqrt[:,None]).dot(evals_sqrt[None,:])
        beta_tmp = 1./(evals_tiled.transpose()- evals_tiled)
        beta_tmp[np.diag_indices(N_ref)] = -0.5/evals
        beta = evals_ratio*beta_tmp

        C =  models_mean_sub.dot(refs.transpose())+refs.dot(models_mean_sub.transpose())
        alpha = (evecs.transpose()).dot(C).dot(evecs)

        delta_KL = (beta*alpha).dot(original_KL)+(evalse_inv_sqrt[:,None]*evecs.transpose()).dot(models_mean_sub)
        delta_KL_nospec[:,k,:] = delta_KL[:,:]


    return delta_KL_nospec

def pertrurb_nospec(evals, evecs, original_KL, refs, models_ref):
    """
    Perturb the KL modes using a model of the PSF but with no assumption on the spectrum. Useful for planets.

    By no assumption on the spectrum it means that the spectrum has been factored out of Delta_KL following equation (4)
    of Laurent Pueyo 2016 noted bold "Delta Z_k^lambda (x)". In order to get the actual perturbed KL modes one needs to
    multpily it by a spectrum.

    This function fits each cube's spectrum independently. So the effective spectrum size is N_wavelengths * N_cubes.


    Args:
        evals: array of eigenvalues of the reference PSF covariance matrix (array of size numbasis)
        evecs: corresponding eigenvectors (array of size [p, numbasis])
        orignal_KL: unpertrubed KL modes (array of size [numbasis, p])
        Sel_wv: wv x N array of the the corresponding wavelength for each reference PSF
        refs: N x p array of the N reference images that
                  characterizes the extended source with p pixels
        models_ref: N x p array of the N models corresponding to reference images. Each model should be normalized to unity (no flux information)
        model_sci: array of size p corresponding to the PSF of the science frame

    Returns:
        delta_KL_nospec: perturbed KL modes but without the spectral info. delta_KL = spectrum x delta_Kl_nospec.
                         Shape is (numKL, wv, pix)
    """

    max_basis = original_KL.shape[0]
    N_ref = refs.shape[0]
    N_pix = original_KL.shape[1]

    refs_mean_sub = refs - np.nanmean(refs, axis=1)[:, None]
    refs_mean_sub[np.where(np.isnan(refs_mean_sub))] = 0

    models_mean_sub = models_ref # - np.nanmean(models_ref, axis=1)[:,None] should this be the case?
    models_mean_sub[np.where(np.isnan(models_mean_sub))] = 0

    # science PSF models
    #model_sci_mean_sub = model_sci # should be subtracting off the mean?
    #model_nanpix = np.where(np.isnan(model_sci_mean_sub))
    #model_sci_mean_sub[model_nanpix] = 0

    # perturbed KL modes
    delta_KL_nospec = np.zeros([max_basis, N_ref, N_pix]) # (numKL,N_ref,N_pix)

    #plt.figure(1)
    #plt.plot(evals)
    #ax = plt.gca()
    #ax.set_yscale('log')
    #plt.show()

    models_mean_sub_X_refs_mean_sub_T = models_mean_sub.dot(refs_mean_sub.transpose())
    # calculate perturbed KL modes. TODO: make this NOT a freaking for loop
    for k in range(max_basis):
        Zk = np.reshape(original_KL[k,:],(1,original_KL[k,:].size))
        Vk = (evecs[:,k])[:,None]

        DeltaZk_noSpec = -(1/np.sqrt(evals[k]))*(Vk*models_mean_sub_X_refs_mean_sub_T).dot(Vk).dot(Zk)+Vk*models_mean_sub
        # TODO: Make this NOT a for loop
        diagVk_X_models_mean_sub_X_refs_mean_sub_T = Vk*models_mean_sub_X_refs_mean_sub_T
        models_mean_sub_X_refs_mean_sub_T_X_Vk = models_mean_sub_X_refs_mean_sub_T.dot(Vk)
        for j in range(k):
            Zj = original_KL[j, :][None,:]
            Vj = evecs[:, j][:,None]
            DeltaZk_noSpec += np.sqrt(evals[j])/(evals[k]-evals[j])*(diagVk_X_models_mean_sub_X_refs_mean_sub_T.dot(Vj) + Vj*models_mean_sub_X_refs_mean_sub_T_X_Vk).dot(Zj)
        for j in range(k+1, max_basis):
            Zj = original_KL[j, :][None,:]
            Vj = evecs[:, j][:,None]
            DeltaZk_noSpec += np.sqrt(evals[j])/(evals[k]-evals[j])*(diagVk_X_models_mean_sub_X_refs_mean_sub_T.dot(Vj) + Vj*models_mean_sub_X_refs_mean_sub_T_X_Vk).dot(Zj)

        delta_KL_nospec[k] = DeltaZk_noSpec/np.sqrt(evals[k])

    return delta_KL_nospec


def calculate_fm(delta_KL_nospec, original_KL, numbasis, sci, model_sci, inputflux = None):
    r"""
    Calculate what the PSF looks up post-KLIP using knowledge of the input PSF, assumed spectrum of the science target,
    and the partially calculated KL modes (\Delta Z_k^\lambda in Laurent's paper). If inputflux is None,
    the spectral dependence has already been folded into delta_KL_nospec (treat it as delta_KL).

    Note: if inputflux is None and delta_KL_nospec has three dimensions (ie delta_KL_nospec was calculated using
    pertrurb_nospec() or perturb_nospec_modelsBased()) then only klipped_oversub and klipped_selfsub are returned.
    Besides they will have an extra first spectral dimension.

    Args:
        delta_KL_nospec: perturbed KL modes but without the spectral info. delta_KL = spectrum x delta_Kl_nospec.
                         Shape is (numKL, wv, pix). If inputflux is None, delta_KL_nospec = delta_KL
        orignal_KL: unpertrubed KL modes (array of size [numbasis, numpix])
        numbasis: array of KL mode cutoffs
                If numbasis is [None] the number of KL modes to be used is automatically picked based on the eigenvalues.
        sci: array of size p representing the science data
        model_sci: array of size p corresponding to the PSF of the science frame
        input_spectrum: array of size wv with the assumed spectrum of the model

    If delta_KL_nospec does NOT include a spectral dimension or if inputflux is not None:
    Returns:
        fm_psf: array of shape (b,p) showing the forward modelled PSF
                Skipped if inputflux = None, and delta_KL_nospec has 3 dimensions.
        klipped_oversub: array of shape (b, p) showing the effect of oversubtraction as a function of KL modes
        klipped_selfsub: array of shape (b, p) showing the effect of selfsubtraction as a function of KL modes
        Note: psf_FM = model_sci - klipped_oversub - klipped_selfsub to get the FM psf as a function of K Lmodes
              (shape of b,p)

    If inputflux = None and if delta_KL_nospec include a spectral dimension:
    Returns:
        klipped_oversub: Sum(<S|KL>KL) with klipped_oversub.shape = (size(numbasis),Npix)
        klipped_selfsub: Sum(<N|DKL>KL) + Sum(<N|KL>DKL) with klipped_selfsub.shape = (size(numbasis),N_lambda or N_ref,N_pix)
    """
    if np.size(numbasis) == 1:
        return calculate_fm_singleNumbasis(delta_KL_nospec, original_KL, numbasis, sci, model_sci, inputflux = inputflux)

    max_basis = original_KL.shape[0]
    if numbasis[0] is None:
        numbasis_index = [max_basis-1]
    else:
        numbasis_index = np.clip(numbasis - 1, 0, max_basis-1)

    # remove means and nans from science image
    sci_mean_sub = np.copy(sci - np.nanmean(sci))
    sci_nanpix = np.where(np.isnan(sci_mean_sub))
    sci_mean_sub[sci_nanpix] = 0
    sci_mean_sub_rows = np.tile(sci_mean_sub, (max_basis,1))
    #sci_rows_selected = np.tile(sci_mean_sub, (np.size(numbasis),1))


    # science PSF models, ready for FM
    # /!\ JB: If subtracting the mean. It should be done here. not in klip_math since we don't use model_sci there.
    model_sci_mean_sub = model_sci # should be subtracting off the mean?
    model_nanpix = np.where(np.isnan(model_sci_mean_sub))
    model_sci_mean_sub[model_nanpix] = 0
    model_sci_mean_sub_rows = np.tile(model_sci_mean_sub, (max_basis,1))
    # model_rows_selected = np.tile(sci_mean_sub, (np.size(numbasis),1)) # don't need this because of python behavior where I don't need to duplicate rows


    # calculate perturbed KL modes based on spectrum
    if inputflux is not None:
        # delta_KL_nospec.shape = (max_basis,N_lambda,N_pix) or (max_basis,N_ref,N_pix)
        delta_KL = np.dot(inputflux, delta_KL_nospec) # this will take the last dimension of input_spectrum (wv) and sum over the second to last dimension of delta_KL_nospec (wv)
    else:
        delta_KL = delta_KL_nospec

    # Forward model the PSF
    # 3 terms: 1 for oversubtracton (planet attenauted by speckle KL modes),
    # and 2 terms for self subtraction (planet signal leaks in KL modes which get projected onto speckles)
    #
    # Klipped = N-Sum(<N|KL>KL) + S-Sum(<S|KL>KL) - Sum(<N|DKL>KL) - Sum(<N|KL>DKL)
    # With  N = noise/speckles (science image)
    #       S = signal/planet model
    #       KL = KL modes
    #       DKL = perturbation of the KL modes/Delta_KL
    #
    # sci_mean_sub_rows.shape = (max_basis,N_pix)  (tiled)
    # model_sci_mean_sub_rows.shape = (max_basis,N_pix) (tiled)
    # original_KL.shape = (max_basis,N_pix)
    # delta_KL.shape = (max_basis,N_pix)
    oversubtraction_inner_products = np.dot(model_sci_mean_sub_rows, original_KL.T)
    if np.size(delta_KL.shape) == 2:
        selfsubtraction_1_inner_products = np.dot(sci_mean_sub_rows, delta_KL.T)
        # selfsubtraction_1_inner_products.shape = (max_basis,N_pix,max_basis)
    else:
        Nlambda = delta_KL.shape[1]
        #Before delta_KL.shape = (max_basis,N_lambda or N_ref,N_pix)
        delta_KL = np.rollaxis(delta_KL,1,0)
        #Now delta_KL.shape = (N_lambda or N_ref,max_basis,N_pix)
        # np.rollaxis(delta_KL,2,1).shape = (N_lambda or N_ref,N_pix,max_basis)
        # np.dot() takes the last dimension of first array and sum over the second to last dimension of second array
        selfsubtraction_1_inner_products = np.dot(sci_mean_sub_rows, np.rollaxis(delta_KL,2,1))
        # selfsubtraction_1_inner_products.shape = (N_lambda or N_ref,max_basis,max_basis)
    selfsubtraction_2_inner_products = np.dot(sci_mean_sub_rows, original_KL.T)

    # lower_tri is a matrix with all the element below and one the diagonal equal to unity. The upper part of the matrix
    #  is full of zeros.
    # lower_tri,shape = (max_basis,max_basis)
    # oversubtraction_inner_products = (N_ref=max_basis,max_basis)
    lower_tri = np.tril(np.ones([max_basis,max_basis]))
    oversubtraction_inner_products = oversubtraction_inner_products * lower_tri
    klipped_oversub = np.dot(np.take(oversubtraction_inner_products, numbasis_index, axis=0), original_KL)
    if np.size(delta_KL.shape) == 2:
        # selfsubtraction_1_inner_products = (max_basis,max_basis)
        # selfsubtraction_2_inner_products = (max_basis,max_basis)
        selfsubtraction_1_inner_products = selfsubtraction_1_inner_products * lower_tri
        selfsubtraction_2_inner_products = selfsubtraction_2_inner_products * lower_tri
        klipped_selfsub = np.dot(np.take(selfsubtraction_1_inner_products, numbasis_index, axis=0), original_KL) + \
                          np.dot(np.take(selfsubtraction_2_inner_products,numbasis_index, axis=0), delta_KL)

        return model_sci - klipped_oversub - klipped_selfsub, klipped_oversub, klipped_selfsub
    else:
        selfsubtraction_1_inner_products = np.array([selfsubtraction_1_inner_products[:,k,:] * lower_tri for k in range(Nlambda)])
        selfsubtraction_2_inner_products = selfsubtraction_2_inner_products * lower_tri
        # selfsubtraction_1_inner_products = (N_lambda or N_ref,max_basis,max_basis)
        # selfsubtraction_2_inner_products = (N_ref=max_basis,max_basis)
        # original_KL.shape = (max_basis,N_pix)
        # delta_KL.shape = (N_lambda or N_ref,max_basis,N_pix)
        klipped_selfsub1 = np.dot(np.take(selfsubtraction_1_inner_products, numbasis_index, axis=1), original_KL)
        klipped_selfsub2 = np.dot(np.take(selfsubtraction_2_inner_products,numbasis_index, axis=0), delta_KL)
        klipped_selfsub = np.rollaxis(klipped_selfsub1,1,0) + klipped_selfsub2

        # klipped_oversub.shape = (size(numbasis),Npix)
        # klipped_selfsub.shape = (size(numbasis),N_lambda or N_ref,N_pix)
        # klipped_oversub = Sum(<S|KL>KL)
        # klipped_selfsub = Sum(<N|DKL>KL) + Sum(<N|KL>DKL)
        return klipped_oversub, klipped_selfsub


def calculate_fm_singleNumbasis(delta_KL_nospec, original_KL, numbasis, sci, model_sci, inputflux = None):
    r"""
    Same function as calculate_fm() but faster when numbasis has only one element. It doesn't do the mutliplication with
    the triangular matrix.

    Calculate what the PSF looks up post-KLIP using knowledge of the input PSF, assumed spectrum of the science target,
    and the partially calculated KL modes (\Delta Z_k^\lambda in Laurent's paper). If inputflux is None,
    the spectral dependence has already been folded into delta_KL_nospec (treat it as delta_KL).

    Note: if inputflux is None and delta_KL_nospec has three dimensions (ie delta_KL_nospec was calculated using
    pertrurb_nospec() or perturb_nospec_modelsBased()) then only klipped_oversub and klipped_selfsub are returned.
    Besides they will have an extra first spectral dimension.

    Args:
        delta_KL_nospec: perturbed KL modes but without the spectral info. delta_KL = spectrum x delta_Kl_nospec.
                         Shape is (numKL, wv, pix). If inputflux is None, delta_KL_nospec = delta_KL
        orignal_KL: unpertrubed KL modes (array of size [numbasis, numpix])
        numbasis: array of (ONE ELEMENT ONLY) KL mode cutoffs
                If numbasis is [None] the number of KL modes to be used is automatically picked based on the eigenvalues.
        sci: array of size p representing the science data
        model_sci: array of size p corresponding to the PSF of the science frame
        input_spectrum: array of size wv with the assumed spectrum of the model

    If delta_KL_nospec does NOT include a spectral dimension or if inputflux is not None:
    Returns:
        fm_psf: array of shape (b,p) showing the forward modelled PSF
                Skipped if inputflux = None, and delta_KL_nospec has 3 dimensions.
        klipped_oversub: array of shape (b, p) showing the effect of oversubtraction as a function of KL modes
        klipped_selfsub: array of shape (b, p) showing the effect of selfsubtraction as a function of KL modes
        Note: psf_FM = model_sci - klipped_oversub - klipped_selfsub to get the FM psf as a function of K Lmodes
              (shape of b,p)

    If inputflux = None and if delta_KL_nospec include a spectral dimension:
    Returns:
        klipped_oversub: Sum(<S|KL>KL) with klipped_oversub.shape = (size(numbasis),Npix)
        klipped_selfsub: Sum(<N|DKL>KL) + Sum(<N|KL>DKL) with klipped_selfsub.shape = (size(numbasis),N_lambda or N_ref,N_pix)

    """
    max_basis = original_KL.shape[0]
    if numbasis[0] is None:
        numbasis_index = [max_basis-1]
    else:
        numbasis_index = np.clip(numbasis - 1, 0, max_basis-1)

    N_pix = np.size(sci)

    # remove means and nans from science image
    sci_mean_sub = sci - np.nanmean(sci)
    sci_nanpix = np.where(np.isnan(sci_mean_sub))
    sci_mean_sub[sci_nanpix] = 0
    sci_mean_sub_rows = np.reshape(sci_mean_sub,(1,N_pix))


    # science PSF models, ready for FM
    # /!\ JB: If subtracting the mean. It should be done here. not in klip_math since we don't use model_sci there.
    model_sci_mean_sub = model_sci # should be subtracting off the mean?
    model_nanpix = np.where(np.isnan(model_sci_mean_sub))
    model_sci_mean_sub[model_nanpix] = 0
    model_sci_mean_sub_rows = np.reshape(model_sci_mean_sub,(1,N_pix))


    # calculate perturbed KL modes based on spectrum
    if inputflux is not None:
        # delta_KL_nospec.shape = (max_basis,N_lambda,N_pix) or (max_basis,N_ref,N_pix)
        delta_KL = np.dot(inputflux, delta_KL_nospec) # this will take the last dimension of input_spectrum (wv) and sum over the second to last dimension of delta_KL_nospec (wv)
    else:
        delta_KL = delta_KL_nospec

    # Forward model the PSF
    # 3 terms: 1 for oversubtracton (planet attenauted by speckle KL modes),
    # and 2 terms for self subtraction (planet signal leaks in KL modes which get projected onto speckles)
    #
    # Klipped = N-Sum(<N|KL>KL) + S-Sum(<S|KL>KL) - Sum(<N|DKL>KL) - Sum(<N|KL>DKL)
    # With  N = noise/speckles (science image)
    #       S = signal/planet model
    #       KL = KL modes
    #       DKL = perturbation of the KL modes/Delta_KL
    #
    # sci_mean_sub_rows.shape = (1,N_pix)
    # model_sci_mean_sub_rows.shape = (1,N_pix)
    # original_KL.shape = (max_basis,N_pix)
    # delta_KL.shape = (max_basis,N_pix)
    oversubtraction_inner_products = np.dot(model_sci_mean_sub_rows, original_KL.T)
    if np.size(delta_KL.shape) == 2:
        selfsubtraction_1_inner_products = np.dot(sci_mean_sub_rows, delta_KL.T)
        # selfsubtraction_1_inner_products.shape = (max_basis,N_pix,max_basis)
    else:
        Nlambda = delta_KL.shape[1]
        #Before delta_KL.shape = (max_basis,N_lambda or N_ref,N_pix)
        delta_KL = np.rollaxis(delta_KL,1,0)
        #Now delta_KL.shape = (N_lambda or N_ref,max_basis,N_pix)
        # np.rollaxis(delta_KL,2,1).shape = (N_lambda or N_ref,N_pix,max_basis)
        # np.dot() takes the last dimension of first array and sum over the second to last dimension of second array
        selfsubtraction_1_inner_products = np.dot(sci_mean_sub_rows, np.rollaxis(delta_KL,2,1))
        # selfsubtraction_1_inner_products.shape = (N_lambda or N_ref,max_basis,max_basis)
    selfsubtraction_2_inner_products = np.dot(sci_mean_sub_rows, original_KL.T)

    # oversubtraction_inner_products = (1,max_basis)
    oversubtraction_inner_products[max_basis::] = 0
    klipped_oversub = np.dot(oversubtraction_inner_products, original_KL)
    if np.size(delta_KL.shape) == 2:
        # selfsubtraction_1_inner_products = (1,max_basis)
        # selfsubtraction_2_inner_products = (1,max_basis)
        selfsubtraction_1_inner_products[0,max_basis::] = 0
        selfsubtraction_2_inner_products[0,max_basis::] = 0
        klipped_selfsub = np.dot(selfsubtraction_1_inner_products, original_KL) + \
                          np.dot(selfsubtraction_2_inner_products, delta_KL)

        # secondorder_inner_products = np.dot(model_sci_mean_sub_rows, delta_KL.T)
        # klipped_secondOrder = np.dot(selfsubtraction_1_inner_products, delta_KL) + \
        #                      np.dot(oversubtraction_inner_products, delta_KL) + \
        #                      np.dot(secondorder_inner_products, original_KL) + \
        #                      np.dot(secondorder_inner_products, delta_KL)
        # # print(oversubtraction_inner_products.shape,selfsubtraction_1_inner_products.shape,selfsubtraction_2_inner_products.shape,secondorder_inner_products.shape)
        # # print(sci_mean_sub_rows.shape,model_sci_mean_sub_rows.shape,delta_KL.shape,original_KL.shape)
        # return model_sci[None,:] - klipped_oversub - klipped_selfsub - klipped_secondOrder, klipped_oversub, klipped_selfsub
        return model_sci[None,:] - klipped_oversub - klipped_selfsub, klipped_oversub, klipped_selfsub
        #return model_sci[None,:], klipped_oversub, klipped_selfsub
    else:
        for k in range(Nlambda):
            selfsubtraction_1_inner_products[:,k,max_basis::] = 0
        selfsubtraction_1_inner_products = np.rollaxis(selfsubtraction_1_inner_products,0,1)
        selfsubtraction_2_inner_products[:,max_basis::] = 0
        # selfsubtraction_1_inner_products = (N_lambda or N_ref,max_basis,max_basis)
        # selfsubtraction_2_inner_products = (N_ref=max_basis,max_basis)
        # original_KL.shape = (max_basis,N_pix)
        # delta_KL.shape = (N_lambda or N_ref,max_basis,N_pix)
        klipped_selfsub1 = np.dot(selfsubtraction_1_inner_products, original_KL)
        klipped_selfsub2 = np.dot(selfsubtraction_2_inner_products, delta_KL)
        klipped_selfsub = klipped_selfsub1 + klipped_selfsub2

        # klipped_oversub.shape = (size(numbasis),Npix)
        # klipped_selfsub.shape = (size(numbasis),N_lambda or N_ref,N_pix)
        # klipped_oversub = Sum(<S|KL>KL)
        # klipped_selfsub = Sum(<N|DKL>KL) + Sum(<N|KL>DKL)
        return klipped_oversub, klipped_selfsub



def calculate_validity(covar_perturb, models_ref, numbasis, evals_orig, covar_orig, evecs_orig, KL_orig, delta_KL):
    """
    Calculate the validity of the perturbation based on the eigenvalues or the 2nd order term compared
     to the 0th order term of the covariance matrix expansion

    Args:
        evals_perturb: linear expansion of the perturbed covariance matrix (C_AS). Shape of N x N
        models_ref: N x p array of the N models corresponding to reference images.
                    Each model should contain spectral information
        numbasis: array of KL mode cutoffs
        evevs_orig: size of [N, maxKL]

    Returns:
        delta_KL_nospec: perturbed KL modes. Shape is (numKL, wv, pix)
    """

    # calculate the C_AA matrix, covariance of the model psf with itself in the sequence
    covars_model = np.dot(models_ref, models_ref.T)
    tot_basis = covars_model.shape[0]

    numbasis = np.clip(numbasis - 1, 0, tot_basis-1)
    max_basis = np.max(numbasis) + 1

    ## calculate eigenvectors/values including 1st order term in covariance matrix expansion
    #evals_linear, evecs_linear = la.eigh(covar_orig + covar_perturb, eigvals = (tot_basis-max_basis, tot_basis-1))
    #evals_linear = np.copy(evals_linear[::-1]) 
    ## calculate eigenvectors/values including first and 2nd order term in covariance matrix expansion
    #evals_full, evecs_full = la.eigh(covar_orig + covar_perturb + covars_model, eigvals = (tot_basis-max_basis, tot_basis-1))
    #evals_full = np.copy(evals_full[::-1])
    #
    #linear_perturb = evals_linear - evals_orig
    #quad_perturb = evals_full - evals_linear

    # Calculate ~second order of \delta KL


    evals_tiled = np.tile(evals_orig,(max_basis,1))
    np.fill_diagonal(evals_tiled,np.nan)
    evals_sqrt = np.sqrt(evals_orig)
    evalse_inv_sqrt = 1./evals_sqrt
    evals_ratio = (evalse_inv_sqrt[:,None]).dot(evals_sqrt[None,:])
    beta_tmp = 1./(evals_tiled.transpose()- evals_tiled)
    #print(evals)
    beta_tmp[np.diag_indices(np.size(evals_orig))] = -0.5/evals_orig
    beta = evals_ratio*beta_tmp

    alpha = (evecs_orig.transpose()).dot(covars_model).dot(evecs_orig)

    quad_delta_KL = (beta*alpha).dot(KL_orig)

    linear_perturb = np.std(delta_KL, axis=1)
    quad_perturb = np.std(quad_delta_KL, axis=1)


    perturb_mag = np.abs(quad_perturb/linear_perturb)

    perturb_mag[np.where(linear_perturb == 0)] = 0



    validity = np.zeros(np.size(numbasis))
    for i, basis_cutoff in enumerate(numbasis):
        validity[i] = np.mean(perturb_mag[:basis_cutoff+1])

    return validity




#####################################################################
################# Begin Parallelized Framework ######################
#####################################################################

def _tpool_init(original_imgs, original_imgs_shape, aligned_imgs, aligned_imgs_shape, output_imgs, output_imgs_shape,
                output_imgs_numstacked,
                pa_imgs, wvs_imgs, centers_imgs, interm_imgs, interm_imgs_shape, fmout_imgs, fmout_imgs_shape,
                perturbmag_imgs, perturbmag_imgs_shape):
    """
    Initializer function for the thread pool that initializes various shared variables. Main things to note that all
    except the shapes are shared arrays (mp.Array) - output_imgs does not need to be mp.Array and can be anything

    Args:
        original_imgs: original images from files to read and align&scale.
        original_imgs_shape: (N,y,x), N = number of frames = num files * num wavelengths
        aligned: aligned and scaled images for processing.
        aligned_imgs_shape: (wv, N, y, x), wv = number of wavelengths per datacube
        output_imgs: PSF subtraceted images
        output_imgs_shape: (N, y, x, b)
        output_imgs_numstacked: number of images stacked together for each pixel due to geometry overlap. Shape of
                                (N, y x). Output without the b dimension
        pa_imgs, wvs_imgs: arrays of size N with the PA and wavelength
        centers_img: array of shape (N,2) with [x,y] image center for image frame
        interm_imgs: intermediate data product shape - what is saved on a sector to sector basis before combining to
                     form the output of that sector. The first dimention should be N (i.e. same thing for each science
                     image)
        interm_imgs_shape: shape of interm_imgs. The first dimention should be N.
        fmout_imgs: array for output of forward modelling. What's stored in here depends on the class
        fmout_imgs_shape: shape of fmout
        perturbmag_imgs: array for output of size of linear perturbation to assess validity
        perturbmag_imgs_shape: shape of perturbmag_imgs
    """
    global original, original_shape, aligned, aligned_shape, outputs, outputs_shape, outputs_numstacked, img_pa, \
        img_wv, img_center, interm, interm_shape, fmout, fmout_shape, perturbmag, perturbmag_shape
    # original images from files to read and align&scale. Shape of (N,y,x)
    original = original_imgs
    original_shape = original_imgs_shape
    # aligned and scaled images for processing. Shape of (wv, N, y, x)
    aligned = aligned_imgs
    aligned_shape = aligned_imgs_shape
    # output images after KLIP processing
    outputs = output_imgs
    outputs_shape = output_imgs_shape
    outputs_numstacked = output_imgs_numstacked
    # parameters for each image (PA, wavelegnth, image center)
    img_pa = pa_imgs
    img_wv = wvs_imgs
    img_center = centers_imgs

    #intermediate and FM arrays
    interm = interm_imgs
    interm_shape = interm_imgs_shape
    fmout = fmout_imgs
    fmout_shape = fmout_imgs_shape
    perturbmag = perturbmag_imgs
    perturbmag_shape = perturbmag_imgs_shape


def _align_and_scale_subset(thread_index, aligned_center,numthreads = None,dtype=float):
    """
    Aligns and scales a subset of images

    Args:
        thread_index: index of thread, break-up algin and scale equally among threads
        algined_center: center to align things to
        numthreads: Number of threads to be used. if none mp.cpu_count() is used.
        dtype: data type of the arrays for numpy (Should match the type used for the shared multiprocessing arrays)

    Returns:
        None
    """
    original_imgs = _arraytonumpy(original, original_shape,dtype=dtype)
    wvs_imgs = _arraytonumpy(img_wv,dtype=dtype)
    centers_imgs = _arraytonumpy(img_center, (np.size(wvs_imgs),2),dtype=dtype)
    aligned_imgs = _arraytonumpy(aligned, aligned_shape,dtype=dtype)

    unique_wvs = np.unique(wvs_imgs)

    # calculate all possible combinations of images and wavelengths to scale to
    # this ordering should hopefully have better cache optimization?
    combos = [combo for combo in itertools.product(np.arange(original_imgs.shape[0]), np.arange(np.size(unique_wvs)))]

    if numthreads is None:
        numthreads = mp.cpu_count()

    # figure out which ones this thread should do
    numframes_todo = int(np.round(len(combos)/numthreads))
    leftovers = len(combos) % numthreads
    # the last thread needs to finish all of them
    if thread_index == numthreads - 1:
        combos_todo = combos[leftovers + thread_index*numframes_todo:]
        #print(len(combos), len(combos_todo), leftovers + thread_index*numframes_todo)
    else:
        if thread_index < leftovers:
            leftovers_completed = thread_index
            plusone = 1
        else:
            leftovers_completed = leftovers
            plusone = 0

        combos_todo = combos[leftovers_completed + thread_index*numframes_todo:(thread_index+1)*numframes_todo + leftovers_completed + plusone]
        #print(len(combos), len(combos_todo), leftovers_completed + thread_index*numframes_todo, (thread_index+1)*numframes_todo + leftovers_completed + plusone)

    #print(len(combos), len(combos_todo), leftovers, thread_index)

    for img_index, ref_wv_index in combos_todo:
        #aligned_imgs[ref_wv_index,img_index,:,:] = np.ones(original_imgs.shape[1:])
        aligned_imgs[ref_wv_index,img_index,:,:] = klip.align_and_scale(original_imgs[img_index], aligned_center,
                                                                        centers_imgs[img_index], unique_wvs[ref_wv_index]/wvs_imgs[img_index])
    return


def _get_section_indicies(input_shape, img_center, radstart, radend, phistart, phiend, padding, parang, IOWA,
                          flatten=True, flipx=False):
    """
    Gets the pixels (via numpy.where) that correspond to this section

    Args:
        input_shape: shape of the image [ysize, xsize] [pixels]
        img_center: [x,y] image center [pxiels]
        radstart: minimum radial distance of sector [pixels]
        radend: maximum radial distance of sector [pixels]
        phistart: minimum azimuthal coordinate of sector [radians]
        phiend: maximum azimuthal coordinate of sector [radians]
        padding: number of pixels to pad to the sector [pixels]
        parang: how much to rotate phi due to field rotation [IN DEGREES]
        IOWA: tuple (IWA,OWA) where IWA = Inner working angle and OWA = Outer working angle both in pixels.
                It defines the separation interva in which klip will be run.

    Returns:
        sector_ind: the pixel coordinates that corespond to this sector
    """
    IWA,OWA = IOWA

    # create a coordinate system.
    x, y = np.meshgrid(np.arange(input_shape[1] * 1.0), np.arange(input_shape[0] * 1.0))
    if flatten:
        x.shape = (x.shape[0] * x.shape[1]) # Flatten
        y.shape = (y.shape[0] * y.shape[1])
    if flipx:
        x = img_center[0] - (x - img_center[0])        
    r = np.sqrt((x - img_center[0])**2 + (y - img_center[1])**2)
    phi = np.arctan2(y - img_center[1], x - img_center[0])

    if phistart < phiend:
        deltaphi = phiend - phistart + 2 * padding/np.mean([radstart, radend])
    else:
        deltaphi = (2*np.pi - (phistart - phiend))  + 2 * padding/np.mean([radstart, radend])

    # If the length or the arc is higher than 2*pi, simply pick the entire circle.
    if deltaphi >= 2*np.pi:
        phistart = 0
        phiend = 2*np.pi
    else:
        phistart = ((phistart)- padding/np.mean([radstart, radend])) % (2.0 * np.pi)
        phiend = ((phiend) + padding/np.mean([radstart, radend])) % (2.0 * np.pi)

    radstart = np.max([radstart-padding,IWA])
    if OWA is not None:
        radend = np.min([radend+padding,OWA])
    else:
        radend = radend+padding

    # grab the pixel location of the section we are going to anaylze
    phi_rotate = ((phi + np.radians(parang)) % (2.0 * np.pi))
    # normal case where there's no 2 pi wrap
    if phistart < phiend:
        section_ind = np.where((r >= radstart) & (r < radend) & (phi_rotate >= phistart) & (phi_rotate < phiend))
    # 2 pi wrap case
    else:
        section_ind = np.where((r >= radstart) & (r < radend) & ((phi_rotate >= phistart) | (phi_rotate < phiend)))

    return section_ind



def _save_rotated_section(input_shape, sector, sector_ind, output_img, output_img_numstacked, angle, radstart, radend, phistart, phiend, padding,IOWA, img_center, flipx=True,
                          new_center=None):
    """
    Rotate and save sector in output image at desired ranges

    Args:
        input_shape: shape of input_image
        sector: data in the sector to save to output_img
        sector_ind: index into input img (corresponding to input_shape) for the original sector
        output_img: the array to save the data to
        output_img_numstacked: array to increment region where we saved output to to bookkeep stacking. None for
                               skipping bookkeeping
        angle: angle that the sector needs to rotate (I forget the convention right now)

        The next 6 parameters define the sector geometry in input image coordinates
        radstart: radius from img_center of start of sector
        radend: radius from img_center of end of sector
        phistart: azimuthal start of sector
        phiend: azimuthal end of sector
        padding: amount of padding around each sector
        IOWA: tuple (IWA,OWA) where IWA = Inner working angle and OWA = Outer working angle both in pixels.
                It defines the separation interva in which klip will be run.
        img_center: center of image in input image coordinate

        flipx: if true, flip the x coordinate to switch coordinate handiness
        new_center: if not none, center of output_img. If none, center stays the same
    """
    # convert angle to radians
    angle_rad = np.radians(angle)

    #wrap phi
    phistart %= 2 * np.pi
    phiend %= 2 * np.pi

    #incorporate padding
    IWA,OWA = IOWA
    radstart_padded = np.max([radstart-padding,IWA])
    if OWA is not None:
        radend_padded = np.min([radend+padding,OWA])
    else:
        radend_padded = radend+padding
    phistart_padded = (phistart - padding/np.mean([radstart, radend])) % (2 * np.pi)
    phiend_padded = (phiend + padding/np.mean([radstart, radend])) % (2 * np.pi)

    # create the coordinate system of the image to manipulate for the transform
    dims = input_shape
    x, y = np.meshgrid(np.arange(dims[1], dtype=np.float32), np.arange(dims[0], dtype=np.float32))

    # if necessary, move coordinates to new center
    if new_center is not None:
        dx = new_center[0] - img_center[0]
        dy = new_center[1] - img_center[1]
        x -= dx
        y -= dy

    # flip x if needed to get East left of North
    if flipx is True:
        x = img_center[0] - (x - img_center[0])

    # do rotation. CW rotation formula to get a CCW of the image
    xp = (x-img_center[0])*np.cos(angle_rad) + (y-img_center[1])*np.sin(angle_rad) + img_center[0]
    yp = -(x-img_center[0])*np.sin(angle_rad) + (y-img_center[1])*np.cos(angle_rad) + img_center[1]

    if new_center is None:
        new_center = img_center

    rot_sector_pix = _get_section_indicies(input_shape, new_center, radstart, radend, phistart, phiend,
                                           padding, 0, IOWA, flatten=False, flipx=flipx)


    # do NaN detection by defining any pixel in the new coordiante system (xp, yp) as a nan
    # if any one of the neighboring pixels in the original image is a nan
    # e.g. (xp, yp) = (120.1, 200.1) is nan if either (120, 200), (121, 200), (120, 201), (121, 201)
    # is a nan
    dims = input_shape
    blank_input = np.zeros(dims[1] * dims[0])
    blank_input[sector_ind] = sector
    blank_input.shape = [dims[0], dims[1]]

    xp_floor = np.clip(np.floor(xp).astype(int), 0, xp.shape[1]-1)[rot_sector_pix]
    xp_ceil = np.clip(np.ceil(xp).astype(int), 0, xp.shape[1]-1)[rot_sector_pix]
    yp_floor = np.clip(np.floor(yp).astype(int), 0, yp.shape[0]-1)[rot_sector_pix]
    yp_ceil = np.clip(np.ceil(yp).astype(int), 0, yp.shape[0]-1)[rot_sector_pix]
    rotnans = np.where(np.isnan(blank_input[yp_floor.ravel(), xp_floor.ravel()]) | 
                       np.isnan(blank_input[yp_floor.ravel(), xp_ceil.ravel()]) |
                       np.isnan(blank_input[yp_ceil.ravel(), xp_floor.ravel()]) |
                       np.isnan(blank_input[yp_ceil.ravel(), xp_ceil.ravel()]))

    # resample image based on new coordinates, set nan values as median
    nanpix = np.where(np.isnan(blank_input))
    medval = np.median(blank_input[np.where(~np.isnan(blank_input))])
    input_copy = np.copy(blank_input)
    input_copy[nanpix] = medval
    rot_sector = ndimage.map_coordinates(input_copy, [yp[rot_sector_pix], xp[rot_sector_pix]], cval=np.nan)

    # mask nans
    rot_sector[rotnans] = np.nan
    sector_validpix = np.where(~np.isnan(rot_sector))

    # need to define only where the non nan pixels are, so we can store those in the output image
    blank_output = np.zeros([dims[0], dims[1]]) * np.nan
    blank_output[rot_sector_pix] = rot_sector
    blank_output.shape = (dims[0], dims[1])
    rot_sector_validpix_2d = np.where(~np.isnan(blank_output))

    # save output sector. We need to reshape the array into 2d arrays to save it
    output_img.shape = [outputs_shape[1], outputs_shape[2]]
    output_img[rot_sector_validpix_2d] = np.nansum([output_img[rot_sector_pix][sector_validpix], rot_sector[sector_validpix]], axis=0)
    output_img.shape = [outputs_shape[1] * outputs_shape[2]]

    # Increment the numstack counter if it is not None
    if output_img_numstacked is not None:
        output_img_numstacked.shape = [outputs_shape[1], outputs_shape[2]]
        output_img_numstacked[rot_sector_validpix_2d] += 1
        output_img_numstacked.shape = [outputs_shape[1] *  outputs_shape[2]]


def klip_parallelized(imgs, centers, parangs, wvs, IWA, fm_class, OWA=None, mode='ADI+SDI', annuli=5, subsections=4,
                      movement=None, flux_overlap=0.1,PSF_FWHM=3.5, numbasis=None,maxnumbasis=None, aligned_center=None, numthreads=None, minrot=0, maxrot=360,
                      spectrum=None, padding=3, save_klipped=True, flipx=True,
                      N_pix_sector = None,mute_progression = False):
    """
    multithreaded KLIP PSF Subtraction

    Args:
        imgs: array of 2D images for ADI. Shape of array (N,y,x)
        centers: N by 2 array of (x,y) coordinates of image centers
        parangs: N length array detailing parallactic angle of each image
        wvs: N length array of the wavelengths
        IWA: inner working angle (in pixels)
        fm_class: class that implements the the forward modelling functionality
        OWA: if defined, the outer working angle for pyklip. Otherwise, it will pick it as the cloest distance to a
            nan in the first frame
        mode: one of ['ADI', 'SDI', 'ADI+SDI'] for ADI, SDI, or ADI+SDI
        anuuli: Annuli to use for KLIP. Can be a number, or a list of 2-element tuples (a, b) specifying
                the pixel bondaries (a <= r < b) for each annulus
        subsections: Sections to break each annuli into. Can be a number [integer], or a list of 2-element tuples (a, b)
                     specifying the positon angle boundaries (a <= PA < b) for each section [radians]
        N_pix_sector: Rough number of pixels in a sector. Overwriting subsections and making it sepration dependent.
                  The number of subsections is defined such that the number of pixel is just higher than N_pix_sector.
                  I.e. subsections = floor(pi*(r_max^2-r_min^2)/N_pix_sector)
                  Warning: There is a bug if N_pix_sector is too big for the first annulus. The annulus is defined from
                            0 to 2pi which create a bug later on. It is probably in the way pa_start and pa_end are
                            defined in fm_from_eigen().
        movement: minimum amount of movement (in pixels) of an astrophysical source
                  to consider using that image for a refernece PSF
        flux_overlap: Maximum fraction of flux overlap between a slice and any reference frames included in the
                    covariance matrix. Flux_overlap should be used instead of "movement" when a template spectrum is used.
                    However if movement is not None then the old code is used and flux_overlap is ignored.
                    The overlap is estimated for 1D gaussians with FWHM defined by PSF_FWHM. So note that the overlap is
                    not exactly the overlap of two real 2D PSF for a given instrument but it will behave similarly.
        PSF_FWHM: FWHM of the PSF used to calculate the overlap (cf flux_overlap). Default is FWHM = 3.5 corresponding
                to sigma ~ 1.5.
        numbasis: number of KL basis vectors to use (can be a scalar or list like). Length of b
                If numbasis is [None] the number of KL modes to be used is automatically picked based on the eigenvalues.
        maxnumbasis: Number of KL modes to be calculated from whcih numbasis modes will be taken.
        aligned_center: array of 2 elements [x,y] that all the KLIP subtracted images will be centered on for image
                        registration
        numthreads: number of threads to use. If none, defaults to using all the cores of the cpu

        minrot: minimum PA rotation (in degrees) to be considered for use as a reference PSF (good for disks)
        maxrot: maximum PA rotation (in degrees) to be considered for use as a reference PSF (temporal variability)

        spectrum: if not None, a array of length N with the flux of the template spectrum at each wavelength. Uses
                    minmove to determine the separation from the center of the segment to determine contamination and
                    the size of the PSF (TODO: make PSF size another quanitity)
                    (e.g. minmove=3, checks how much containmination is within 3 pixels of the hypothetical source)
                    if smaller than 10%, (hard coded quantity), then use it for reference PSF
        padding: for each sector, how many extra pixels of padding should we have around the sides.
        save_klipped: if True, will save the regular klipped image. If false, it wil not and sub_imgs will return None
        flipx: if True, flips x axis after rotation to get North up East left
        mute_progression: Mute the printing of the progression percentage. Indeed sometimes the overwriting feature
                        doesn't work and one ends up with thousands of printed lines. Therefore muting it can be a good
                        idea.


    Returns:
        sub_imgs: array of [array of 2D images (PSF subtracted)] using different number of KL basis vectors as
                    specified by numbasis. Shape of (b,N,y,x).
                  Note: this will be None if save_klipped is False
        fmout_np: output of forward modelling.
        perturbmag: output indicating the magnitude of the linear perturbation to assess validity of KLIP FM
        aligned_center: (x, y) location indicating the star center for all images and FM after PSF subtraction
    """

    ################## Interpret input arguments ####################

    # defaullt numbasis if none
    totalimgs = imgs.shape[0]
    if numbasis is None:
        maxbasis = np.min([totalimgs, 100]) #only going up to 100 KL modes by default
        numbasis = np.arange(1, maxbasis + 5, 5)
        print("KL basis not specified. Using default.", numbasis)
    else:
        if hasattr(numbasis, "__len__"):
            numbasis = np.array(numbasis)
        else:
            numbasis = np.array([numbasis])

    if movement is None:
        if spectrum is None:
            movement = 3

    if numbasis[0] is None:
        if np.size(numbasis)>1:
            print("numbasis should have only one element if numbasis[0] = 0.")
            return None

    if maxnumbasis is None and numbasis[0] is not None:
        maxnumbasis = np.max(numbasis)
    elif maxnumbasis is None and numbasis[0] is None:
        maxnumbasis = 100

    if numthreads is None:
        numthreads = mp.cpu_count()

    # default aligned_center if none:
    if aligned_center is None:
        aligned_center = [np.mean(centers[:,0]), np.mean(centers[:,1])]

    # save all bad pixels
    allnans = np.where(np.isnan(imgs))


    dims = imgs.shape
    if isinstance(annuli, int):
        # use first image to figure out how to divide the annuli
        # TODO: what to do with OWA
        # need to make the next 10 lines or so much smarter

        x, y = np.meshgrid(np.arange(dims[2] * 1.0), np.arange(dims[1] * 1.0))
        nanpix = np.where(np.isnan(imgs[0]))
        # need to define OWA if one wasn't passed. Try to use NaNs to figure out where it should be
        if OWA is None:
            if np.size(nanpix) == 0:
                OWA = np.sqrt(np.max((x - centers[0][0]) ** 2 + (y - centers[0][1]) ** 2))
            else:
                # grab the NaN from the 1st percentile (this way we drop outliers)
                OWA = np.sqrt(np.percentile((x[nanpix] - centers[0][0]) ** 2 + (y[nanpix] - centers[0][1]) ** 2, 1))
        dr = float(OWA - IWA) / (annuli)
        # calculate the annuli
        rad_bounds = [(dr * rad + IWA, dr * (rad + 1) + IWA) for rad in range(annuli)]
        # last annulus should mostly emcompass everything
        # rad_bounds[annuli - 1] = (rad_bounds[annuli - 1][0], imgs[0].shape[0])
    else:
        rad_bounds = annuli

    if N_pix_sector is None:
        if isinstance(subsections, int):
            # divide annuli into subsections
            dphi = 2 * np.pi / subsections
            phi_bounds = [[dphi * phi_i, dphi * (phi_i + 1)] for phi_i in range(subsections)]
            phi_bounds[-1][1] = 2 * np.pi - 0.0001
        else:
            sign = -1
            if not flipx:
                sign = 1
            phi_bounds = [[((sign*pa + np.pi/2)) % (2*np.pi) for pa in pa_tuple[::sign]] for pa_tuple in subsections]

        iterator_sectors = itertools.product(rad_bounds, phi_bounds)
        tot_sectors = len(rad_bounds)*len(phi_bounds)
    else:
        iterator_sectors = []
        for [r_min,r_max] in rad_bounds:
            curr_sep_N_subsections = np.max([int(np.pi*(r_max**2-r_min**2)/N_pix_sector),1]) # equivalent to using floor but casting as well
            # divide annuli into subsections
            dphi = 2 * np.pi / curr_sep_N_subsections
            phi_bounds_list = [[dphi * phi_i, dphi * (phi_i + 1)] for phi_i in range(curr_sep_N_subsections)]
            phi_bounds_list[-1][1] = 2 * np.pi
            # for phi_bound in phi_bounds_list:
            #     print(((r_min,r_max),phi_bound) )
            iterator_sectors.extend([((r_min,r_max),phi_bound) for phi_bound in phi_bounds_list ])
        tot_sectors = len(iterator_sectors)

    global tot_iter
    tot_iter = np.size(np.unique(wvs)) * tot_sectors

    sectors_area = []
    iterator_sectors = list(iterator_sectors)
    for (r0,r1),(phi0,phi1) in iterator_sectors:
        phi0_mod = phi0 % (2*np.pi)
        phi1_mod = phi1 % (2*np.pi)
        if phi1_mod > phi0_mod:
            dphi = phi1_mod-phi0_mod
        else:
            dphi = phi1_mod+2*np.pi-phi0_mod
        sectors_area.append((dphi/2.)*(r1**2-r0**2))
    sectors_area = np.array(sectors_area)
    tot_area = np.sum(sectors_area)

    ########################### Create Shared Memory ###################################

    # implement the thread pool
    # make a bunch of shared memory arrays to transfer data between threads
    # make the array for the original images and initalize it
    original_imgs = mp.Array(fm_class.data_type, np.size(imgs))
    original_imgs_shape = imgs.shape
    original_imgs_np = _arraytonumpy(original_imgs, original_imgs_shape,dtype=fm_class.data_type)
    original_imgs_np[:] = imgs
    # make array for recentered/rescaled image for each wavelength
    unique_wvs = np.unique(wvs)
    recentered_imgs = mp.Array(fm_class.data_type, np.size(imgs)*np.size(unique_wvs))
    recentered_imgs_shape = (np.size(unique_wvs),) + imgs.shape

    # remake the PA, wv, and center arrays as shared arrays
    pa_imgs = mp.Array(fm_class.data_type, np.size(parangs))
    pa_imgs_np = _arraytonumpy(pa_imgs,dtype=fm_class.data_type)
    pa_imgs_np[:] = parangs
    wvs_imgs = mp.Array(fm_class.data_type, np.size(wvs))
    wvs_imgs_np = _arraytonumpy(wvs_imgs,dtype=fm_class.data_type)
    wvs_imgs_np[:] = wvs
    centers_imgs = mp.Array(fm_class.data_type, np.size(centers))
    centers_imgs_np = _arraytonumpy(centers_imgs, centers.shape,dtype=fm_class.data_type)
    centers_imgs_np[:] = centers

    # make output array which also has an extra dimension for the number of KL modes to use
    if save_klipped:
        output_imgs = mp.Array(fm_class.data_type, np.size(imgs)*np.size(numbasis))
        output_imgs_np = _arraytonumpy(output_imgs,dtype=fm_class.data_type)
        output_imgs_np[:] = np.nan
        output_imgs_numstacked = mp.Array(ctypes.c_int, np.size(imgs))
    else:
        output_imgs = None
        output_imgs_numstacked = None

    output_imgs_shape = imgs.shape + numbasis.shape
    # make an helper array to count how many frames overlap at each pixel


    # Create Custom Shared Memory array fmout to save output of forward modelling
    fmout_data, fmout_shape = fm_class.alloc_fmout(output_imgs_shape)
    # Create shared memory to keep track of validity of perturbation
    perturbmag, perturbmag_shape = fm_class.alloc_perturbmag(output_imgs_shape, numbasis)

    # align and scale the images for each image. Use map to do this asynchronously]
    tpool = mp.Pool(processes=numthreads, initializer=_tpool_init,
                    initargs=(original_imgs, original_imgs_shape, recentered_imgs, recentered_imgs_shape, output_imgs,
                              output_imgs_shape, output_imgs_numstacked, pa_imgs, wvs_imgs, centers_imgs, None, None,
                              fmout_data, fmout_shape,perturbmag,perturbmag_shape), maxtasksperchild=50)

    # # SINGLE THREAD DEBUG PURPOSES ONLY
    if not parallel:
        _tpool_init(original_imgs, original_imgs_shape, recentered_imgs, recentered_imgs_shape, output_imgs,
                    output_imgs_shape, output_imgs_numstacked, pa_imgs, wvs_imgs, centers_imgs, None, None,
                    fmout_data, fmout_shape,perturbmag,perturbmag_shape)



    print("Begin align and scale images for each wavelength")
    aligned_outputs = []
    for threadnum in range(numthreads):
        #multitask this
        aligned_outputs += [tpool.apply_async(_align_and_scale_subset, args=(threadnum, aligned_center,numthreads,fm_class.data_type))]

        #save it to shared memory
    for aligned_output in aligned_outputs:
        aligned_output.wait()

    print("Align and scale finished")

    # list to store each threadpool task
    tpool_outputs = []
    sector_job_queued = np.zeros(tot_sectors) # count for jobs in the tpool queue for each sector

    # as each is finishing, queue up the aligned data to be processed with KLIP
    N_it = 0
    N_tot_it = totalimgs*tot_sectors
    time_spent_per_sector_list = []
    time_spent_last_sector=0
    for sector_index, ((radstart, radend),(phistart,phiend)) in enumerate(iterator_sectors):
        t_start_sector = time()
        print("Starting KLIP for sector {0}/{1} with an area of {2} pix^2".format(sector_index+1,tot_sectors,sectors_area[sector_index]))
        if len(time_spent_per_sector_list)==0:
            print("Time spent on last sector: {0:.0f}s".format(0))
            print("Time spent since beginning: {0:.0f}s".format(0))
            print("First sector: Can't predict remaining time")
        else:
            print("Time spent on last sector: {0:.0f}s".format(time_spent_last_sector))
            print("Time spent since beginning: {0:.0f}s".format(np.sum(time_spent_per_sector_list)))
            print("Estimated remaining time: {0:.0f}s".format((tot_area-np.sum(sectors_area[0:sector_index]))*\
                                      (np.sum(time_spent_per_sector_list)/np.sum(sectors_area[0:sector_index]))))
            print("Average time per pixel: {0} during last sector, {1} since begining"\
                  .format(time_spent_last_sector/sectors_area[sector_index-1],
                          (np.sum(time_spent_per_sector_list)/np.sum(sectors_area[0:sector_index]))))
        # calculate sector size
        section_ind = _get_section_indicies(original_imgs_shape[1:], aligned_center, radstart, radend, phistart, phiend,
                                            padding, 0,[IWA,OWA])
        #print(np.shape(section_ind))
        #print(radstart, radend, phistart, phiend)

        if fm_class.skip_section(radstart, radend, phistart, phiend,flipx=flipx):
            print("SKIPPING")
            continue

        sector_size = np.size(section_ind) #+ 2 * (radend- radstart) # some sectors are bigger than others due to boundary
        interm_data, interm_shape = fm_class.alloc_interm(sector_size, original_imgs_shape[0])

        for wv_index, wv_value in enumerate(unique_wvs):

            # pick out the science images that need PSF subtraction for this wavelength
            scidata_indicies = np.where(wvs == wv_value)[0]

            # perform KLIP asynchronously for each group of files of a specific wavelength and section of the image
            sector_job_queued[sector_index] += scidata_indicies.shape[0]
            if parallel:
                tpool_outputs += [tpool.apply_async(_klip_section_multifile_perfile,
                                                    args=(file_index, sector_index, radstart, radend, phistart, phiend,
                                                          parang, wv_value, wv_index, (radstart + radend) / 2., padding,(IWA,OWA),
                                                          numbasis,maxnumbasis,
                                                          movement,flux_overlap,PSF_FWHM, aligned_center, minrot, maxrot, mode, spectrum,
                                                          flipx, fm_class))
                                  for file_index,parang in zip(scidata_indicies, pa_imgs_np[scidata_indicies])]

            # # SINGLE THREAD DEBUG PURPOSES ONLY
            if not parallel:
                tpool_outputs += [_klip_section_multifile_perfile(file_index, sector_index, radstart, radend, phistart, phiend,
                                                                  parang, wv_value, wv_index, (radstart + radend) / 2., padding,(IWA,OWA),
                                                                  numbasis,maxnumbasis,
                                                                  movement,flux_overlap,PSF_FWHM, aligned_center, minrot, maxrot, mode, spectrum,
                                                                  flipx, fm_class)
                                  for file_index,parang in zip(scidata_indicies, pa_imgs_np[scidata_indicies])]

        # Run post processing on this sector here
        # Can be multithreaded code using the threadpool defined above
        # Check tpool job outputs. It there is stuff, go do things with it
        N_it_perSector = 0
        if parallel:
            while len(tpool_outputs) > 0:
                tpool_outputs.pop(0).wait()
                N_it = N_it+1
                N_it_perSector = N_it_perSector+1
                if not mute_progression:
                    stdout.write("\r {0:.2f}% of sector, {1:.2f}% of total completed".format(100*float(N_it_perSector)/float(totalimgs),100*float(N_it)/float(N_tot_it)))
                    stdout.flush()
                #JB debug
                #print("outputs klip_section_multifile_perfile",tpool_outputs)

            # if this is the last job finished for this sector,
            # do something here?

        # newline for next sector
        stdout.write("\n")


        # run custom function to handle end of sector post-processing analysis
        interm_data_np = _arraytonumpy(interm_data, interm_shape,dtype=fm_class.data_type)
        fmout_np = _arraytonumpy(fmout_data, fmout_shape,dtype=fm_class.data_type)
        fm_class.fm_end_sector(interm_data=interm_data_np, fmout=fmout_np, sector_index=sector_index,
                               section_indicies=section_ind)

        # Add time spent on last sector to the list
        time_spent_last_sector = time() - t_start_sector
        time_spent_per_sector_list.append(time_spent_last_sector)



    #close to pool now and make sure there's no processes still running (there shouldn't be or else that would be bad)
    print("Closing threadpool")
    tpool.close()
    tpool.join()

    # finished!
    # Mean the output images if save_klipped is True
    if save_klipped:
        # Let's take the mean based on number of images stacked at a location
        sub_imgs = _arraytonumpy(output_imgs, output_imgs_shape,dtype=fm_class.data_type)
        sub_imgs_numstacked = _arraytonumpy(output_imgs_numstacked, original_imgs_shape, dtype=ctypes.c_int)
        sub_imgs = sub_imgs / sub_imgs_numstacked[:,:,:,None]

        # Let's reshape the output images
        # move number of KLIP modes as leading axis (i.e. move from shape (N,y,x,b) to (b,N,y,x)
        sub_imgs = np.rollaxis(sub_imgs.reshape((dims[0], dims[1], dims[2], numbasis.shape[0])), 3)

        #restore bad pixels
        # sub_imgs[:, allnans[0], allnans[1], allnans[2]] = np.nan
    else:
        sub_imgs = None

    # put any finishing touches on the FM Output
    fmout_np = _arraytonumpy(fmout_data, fmout_shape,dtype=fm_class.data_type)
    fmout_np = fm_class.cleanup_fmout(fmout_np)

    # convert pertrubmag to numpy
    perturbmag_np = _arraytonumpy(perturbmag, perturbmag_shape,dtype=fm_class.data_type)

    # Output for the sole PSFs
    return sub_imgs, fmout_np, perturbmag_np, aligned_center



def _klip_section_multifile_perfile(img_num, sector_index, radstart, radend, phistart, phiend, parang, wavelength,
                                    wv_index, avg_rad, padding,IOWA,
                                    numbasis,maxnumbasis, minmove,flux_overlap,PSF_FWHM, ref_center, minrot, maxrot,
                                    mode, spectrum, flipx,
                                    fm_class):
    """
    Imitates the rest of _klip_section for the multifile code. Does the rest of the PSF reference selection

    Args:
        img_num: file index for the science image to process
        sector: index for the section of the image. Used for return purposes only
        radstart: radial distance of inner edge of annulus
        radend: radial distance of outer edge of annulus
        phistart: start of azimuthal sector (in radians)
        phiend: end of azimuthal sector (in radians)
        parang: PA of science iamge
        wavelength: wavelength of science image
        wv_index: array index of the wavelength of the science image
        avg_rad: average radius of this annulus
        padding: number of pixels to pad the sector by
        IOWA: tuple (IWA,OWA) where IWA = Inner working angle and OWA = Outer working angle both in pixels.
                It defines the separation interva in which klip will be run.
        numbasis: number of KL basis vectors to use (can be a scalar or list like). Length of b
                If numbasis is [None] the number of KL modes to be used is automatically picked based on the eigenvalues.
        maxnumbasis: Number of KL modes to be calculated from whcih numbasis modes will be taken.
        minmove: minimum movement between science image and PSF reference image to use PSF reference image (in pixels)
        flux_overlap: Maximum fraction of flux overlap between a slice and any reference frames included in the
                    covariance matrix. Flux_overlap should be used instead of "movement" when a template spectrum is used.
                    However if movement is not None then the old code is used and flux_overlap is ignored.
                    The overlap is estimated for 1D gaussians with FWHM defined by PSF_FWHM. So note that the overlap is
                    not exactly the overlap of two real 2D PSF for a given instrument but it will behave similarly.
        PSF_FWHM: FWHM of the PSF used to calculate the overlap (cf flux_overlap). Default is FWHM = 3.5 corresponding
                to sigma ~ 1.5.
        maxmove:minimum movement (opposite of minmove) - CURRENTLY NOT USED
        mode: one of ['ADI', 'SDI', 'ADI+SDI'] for ADI, SDI, or ADI+SDI
        spectrum: if not None, a array of length N with the flux of the template spectrum at each wavelength. Uses
                    minmove to determine the separation from the center of the segment to determine contamination and
                    the size of the PSF (TODO: make PSF size another quanitity)
                    (e.g. minmove=3, checks how much containmination is within 3 pixels of the hypothetical source)
                    if smaller than 10%, (hard coded quantity), then use it for reference PSF
        flipx: if True, flips x axis after rotation to get North up East left

    Returns:
        sector_index: used for tracking jobs
        Saves image to output array defined in _tpool_init()
    """

    # get the indicies in the aligned data that correspond to the section and the section without padding
    #print(img_num)
    IWA,OWA = IOWA
    section_ind = _get_section_indicies(original_shape[1:], ref_center, radstart, radend, phistart, phiend,
                                        padding, parang,IOWA)
    section_ind_nopadding = _get_section_indicies(original_shape[1:], ref_center, radstart, radend, phistart, phiend,
                                                  0, parang,IOWA)

    if np.size(section_ind) <= 1:
        print("section is too small ({0} pixels), skipping...".format(np.size(section_ind)))
        return False
    #print(np.size(section_ind), np.min(phi_rotate), np.max(phi_rotate), phistart, phiend)

    #load aligned images for this wavelength
    aligned_imgs = _arraytonumpy(aligned, (aligned_shape[0], aligned_shape[1], aligned_shape[2] * aligned_shape[3]),dtype=fm_class.data_type)[wv_index]
    ref_psfs = aligned_imgs[:,  section_ind[0]]

    if np.sum(np.isfinite(aligned_imgs[img_num, section_ind[0]])) == 0:
        print("section is full of NaNs ({0} pixels), skipping...".format(np.size(section_ind)))
        return False

    #do the same for the reference PSFs
    #playing some tricks to vectorize the subtraction of the mean for each row
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref_psfs_mean_sub = ref_psfs - np.nanmean(ref_psfs, axis=1)[:, None]
    ref_nanpix = np.where(np.isnan(ref_psfs_mean_sub))
    ref_psfs_mean_sub[ref_nanpix] = 0

    #calculate the covariance matrix for the reference PSFs
    #note that numpy.cov normalizes by p-1 to get the NxN covariance matrix
    #we have to correct for that in the klip.klip_math routine when consturcting the KL
    #vectors since that's not part of the equation in the KLIP paper
    covar_psfs = np.cov(ref_psfs_mean_sub)
    #also calculate correlation matrix since we'll use that to select reference PSFs
    covar_diag_sqrt = np.sqrt(np.diag(covar_psfs))
    covar_diag_sqrt_inverse = np.zeros(covar_diag_sqrt.shape)
    where_zeros = np.where(covar_diag_sqrt != 0)
    covar_diag_sqrt_inverse[where_zeros] = 1./covar_diag_sqrt[where_zeros]
    # any image where the diagonal is 0 is all NaNs and shouldn't be infinity
    # covar_diag_sqrt_inverse[np.where(covar_diag_sqrt == 0)] = 0
    covar_diag = np.diagflat(covar_diag_sqrt_inverse)
    
    corr_psfs = np.dot( np.dot(covar_diag, covar_psfs ), covar_diag)


    # grab the files suitable for reference PSF
    # load shared arrays for wavelengths and PAs
    wvs_imgs = _arraytonumpy(img_wv,dtype=fm_class.data_type)
    pa_imgs = _arraytonumpy(img_pa,dtype=fm_class.data_type)
    # calculate average movement in this section for each PSF reference image w.r.t the science image
    moves = klip.estimate_movement(avg_rad, parang, pa_imgs, wavelength, wvs_imgs, mode)
    # check all the PSF selection criterion
    # enough movement of the astrophyiscal source
    if spectrum is None:
        goodmv = (moves >= minmove)
    else:
        if minmove is not None:
            if minmove > 0:
                # optimize the selection based on the spectral template rather than just an exclusion principle
                goodmv = (spectrum * norm.sf(moves-minmove/2.355, scale=minmove/2.355) <= 0.1 * spectrum[wv_index])
            else:
                # handle edge case of minmove == 0
                goodmv = (moves >= minmove) # should be all true
        else:
            # Calculate the flux overlap between the current slice and all the other based on moves and PSF_FWHM.
            overlaps = (spectrum * norm.sf(moves-PSF_FWHM/2.355, scale=PSF_FWHM/2.355))/spectrum[wv_index]
            # optimize the selection based on the spectral template rather than just an exclusion principle
            goodmv = overlaps <= flux_overlap

    # enough field rotation
    if minrot > 0:
        goodmv = (goodmv) & (np.abs(pa_imgs - parang) >= minrot)

    # if no SDI, don't use other wavelengths
    if "SDI" not in mode.upper():
        goodmv = (goodmv) & (wvs_imgs == wavelength)
    # if no ADI, don't use other parallactic angles
    if "ADI" not in mode.upper():
        goodmv = (goodmv) & (pa_imgs == parang)

    # if minrot > 0:
    #     file_ind = np.where((moves >= minmove) & (np.abs(pa_imgs - parang) >= minrot))
    # else:
    #     file_ind = np.where(moves >= minmove)
    # select the good reference PSFs
    file_ind = np.where(goodmv)
    if np.size(file_ind[0]) < 2:
        print("less than 2 reference PSFs available for minmove={0}, skipping...".format(minmove))
        return False
    # pick out a subarray. Have to play around with indicies to get the right shape to index the matrix
    covar_files = covar_psfs[file_ind[0].reshape(np.size(file_ind), 1), file_ind[0]]

    # pick only the most correlated reference PSFs if there's more than enough PSFs
    maxbasis_requested = maxnumbasis
    maxbasis_possible = np.size(file_ind)
    if maxbasis_possible > maxbasis_requested:
        xcorr = corr_psfs[img_num, file_ind[0]]  # grab the x-correlation with the sci img for valid PSFs
        sort_ind = np.argsort(xcorr)
        closest_matched = sort_ind[-maxbasis_requested:]  # sorted smallest first so need to grab from the end
        # grab the new and smaller covariance matrix
        covar_files = covar_files[closest_matched.reshape(np.size(closest_matched), 1), closest_matched]
        # grab smaller set of reference PSFs
        ref_psfs_selected = ref_psfs[file_ind[0][closest_matched], :]
        ref_psfs_indicies = file_ind[0][closest_matched]
    else:
        # else just grab the reference PSFs for all the valid files
        ref_psfs_selected = ref_psfs[file_ind[0], :]
        ref_psfs_indicies = file_ind[0]

    # create a selection matrix for selecting elements
    unique_wvs = np.unique(wvs_imgs)
    numwv = np.size(unique_wvs)
    numcubes = np.size(wvs_imgs)/numwv
    numpix = np.shape(section_ind)[1]
    numref = np.shape(ref_psfs_indicies)[0]

    # restore NaNs
    ref_psfs_mean_sub[ref_nanpix] = np.nan

    aligned_imgs = _arraytonumpy(aligned, (aligned_shape[0], aligned_shape[1], aligned_shape[2] * aligned_shape[3]),dtype=fm_class.data_type)[wv_index]

    # convert to numpy array if we are saving outputs
    output_imgs = _arraytonumpy(outputs, (outputs_shape[0], outputs_shape[1]*outputs_shape[2], outputs_shape[3]),dtype=fm_class.data_type)
    output_imgs_numstacked = _arraytonumpy(outputs_numstacked, (outputs_shape[0], outputs_shape[1]*outputs_shape[2]), dtype=ctypes.c_int)

    # convert to numpy array if fmout is defined
    fmout_np = _arraytonumpy(fmout, fmout_shape,dtype=fm_class.data_type)
    # convert to numpy array if pertrubmag is defined
    perturbmag_np = _arraytonumpy(perturbmag, perturbmag_shape,dtype=fm_class.data_type)
    # run regular KLIP and get the klipped img along with KL modes and eigenvalues/vectors of covariance matrix
    klip_math_return = klip_math(aligned_imgs[img_num, section_ind[0]], ref_psfs_selected, numbasis,
                                 covar_psfs=covar_files,)
    klipped, original_KL, evals, evecs = klip_math_return

    # write standard klipped image to output if we are saving outputs
    if output_imgs is not None:
        for thisnumbasisindex in range(klipped.shape[1]):
            if thisnumbasisindex == 0:
                # only increment the numstack counter for the first KL mode
                _save_rotated_section([original_shape[1], original_shape[2]], klipped[:, thisnumbasisindex], section_ind,
                                      output_imgs[img_num,:,thisnumbasisindex], output_imgs_numstacked[img_num], parang,
                                      radstart, radend, phistart, phiend, padding,IOWA, ref_center, flipx=flipx)
            else:
                _save_rotated_section([original_shape[1], original_shape[2]], klipped[:, thisnumbasisindex], section_ind,
                                      output_imgs[img_num,:,thisnumbasisindex], None, parang,
                                      radstart, radend, phistart, phiend, padding,IOWA, ref_center, flipx=flipx)


    # call FM Class to handle forward modelling if it wants to. Basiclaly we are passing in everything as a variable
    # and it can choose which variables it wants to deal with using **kwargs
    # result is stored in fmout
    fm_class.fm_from_eigen(klmodes=original_KL, evals=evals, evecs=evecs,
                           input_img_shape=[original_shape[1], original_shape[2]], input_img_num=img_num,
                           ref_psfs_indicies=ref_psfs_indicies, section_ind=section_ind,section_ind_nopadding=section_ind_nopadding, aligned_imgs=aligned_imgs,
                           pas=pa_imgs[ref_psfs_indicies], wvs=wvs_imgs[ref_psfs_indicies], radstart=radstart,
                           radend=radend, phistart=phistart, phiend=phiend, padding=padding,IOWA = IOWA, ref_center=ref_center,
                           parang=parang, ref_wv=wavelength, numbasis=numbasis,maxnumbasis=maxnumbasis,
                           fmout=fmout_np,perturbmag = perturbmag_np,klipped=klipped, covar_files=covar_files, flipx=flipx)

    return sector_index


def klip_dataset(dataset, fm_class, mode="ADI+SDI", outputdir=".", fileprefix="pyklipfm", annuli=5, subsections=4,
                 OWA=None, N_pix_sector=None, movement=None, flux_overlap=0.1, PSF_FWHM=3.5, minrot=0, padding=3,
                 numbasis=None, maxnumbasis=None, numthreads=None, calibrate_flux=False, aligned_center=None,
                 spectrum=None, highpass=False, save_klipped=True, mute_progression=False):
    """
    Run KLIP-FM on a dataset object

    Args:
        dataset: an instance of Instrument.Data (see instruments/ subfolder)
        fm_class: class that implements the the forward modelling functionality
        mode: one of ['ADI', 'SDI', 'ADI+SDI'] for ADI, SDI, or ADI+SDI
        anuuli: Annuli to use for KLIP. Can be a number, or a list of 2-element tuples (a, b) specifying
                the pixel bondaries (a <= r < b) for each annulus
        subsections: Sections to break each annuli into. Can be a number [integer], or a list of 2-element tuples (a, b)
                     specifying the positon angle boundaries (a <= PA < b) for each section [radians]
        OWA: if defined, the outer working angle for pyklip. Otherwise, it will pick it as the cloest distance to a
            nan in the first frame
        N_pix_sector: Rough number of pixels in a sector. Overwriting subsections and making it sepration dependent.
                  The number of subsections is defined such that the number of pixel is just higher than N_pix_sector.
                  I.e. subsections = floor(pi*(r_max^2-r_min^2)/N_pix_sector)
                  Warning: There is a bug if N_pix_sector is too big for the first annulus. The annulus is defined from
                            0 to 2pi which create a bug later on. It is probably in the way pa_start and pa_end are
                            defined in fm_from_eigen(). (I am taking about matched filter by the way)
        movement: minimum amount of movement (in pixels) of an astrophysical source
                  to consider using that image for a refernece PSF
        flux_overlap: Maximum fraction of flux overlap between a slice and any reference frames included in the
                    covariance matrix. Flux_overlap should be used instead of "movement" when a template spectrum is used.
                    However if movement is not None then the old code is used and flux_overlap is ignored.
                    The overlap is estimated for 1D gaussians with FWHM defined by PSF_FWHM. So note that the overlap is
                    not exactly the overlap of two real 2D PSF for a given instrument but it will behave similarly.
        PSF_FWHM: FWHM of the PSF used to calculate the overlap (cf flux_overlap). Default is FWHM = 3.5 corresponding
                to sigma ~ 1.5.
        minrot: minimum PA rotation (in degrees) to be considered for use as a reference PSF (good for disks)
        padding: for each sector, how many extra pixels of padding should we have around the sides.
        numbasis: number of KL basis vectors to use (can be a scalar or list like). Length of b
                If numbasis is [None] the number of KL modes to be used is automatically picked based on the eigenvalues.
        maxnumbasis: Number of KL modes to be calculated from whcih numbasis modes will be taken.

        numthreads: number of threads to use. If none, defaults to using all the cores of the cpu
        calibrate_flux: if true, flux calibrates the regular KLIP subtracted data. DOES NOT CALIBRATE THE FM
        aligned_center: array of 2 elements [x,y] that all the KLIP subtracted images will be centered on for image
                        registration
        spectrum: if not None, a array of length N with the flux of the template spectrum at each wavelength. Uses
                    minmove to determine the separation from the center of the segment to determine contamination and
                    the size of the PSF (TODO: make PSF size another quanitity)
                    (e.g. minmove=3, checks how much containmination is within 3 pixels of the hypothetical source)
                    if smaller than 10%, (hard coded quantity), then use it for reference PSF
        highpass:       if True, run a Gaussian high pass filter (default size is sigma=imgsize/10)
                            can also be a number specifying FWHM of box in pixel units
        save_klipped: if True, will save the regular klipped image. If false, it wil not and sub_imgs will return None
        mute_progression: Mute the printing of the progression percentage. Indeed sometimes the overwriting feature
                        doesn't work and one ends up with thousands of printed lines. Therefore muting it can be a good
                        idea.

    """

    ########### Sanitize input arguments ###########

    # numbasis default, needs to be array
    if numbasis is None:
        totalimgs = dataset.input.shape[0]
        maxbasis = np.min([totalimgs, 100]) # only going up to 100 KL modes by default
        numbasis = np.arange(1, maxbasis + 5, 5)
        print("KL basis not specified. Using default.", numbasis)
    else:
        if hasattr(numbasis, "__len__"):
            numbasis = np.array(numbasis)
        else:
            numbasis = np.array([numbasis])

    # high pass filter?
    if isinstance(highpass, bool):
        if highpass:
            dataset.input = high_pass_filter_imgs(dataset.input, numthreads=numthreads)
    else:
        # should be a number
        if isinstance(highpass, (float, int)):
            highpass = float(highpass)
            fourier_sigma_size = (dataset.input.shape[1]/(highpass)) / (2*np.sqrt(2*np.log(2)))
            dataset.input = high_pass_filter_imgs(dataset.input, numthreads=numthreads, filtersize=fourier_sigma_size)

    # output dir edge case
    if outputdir == "":
        outputdir = "."

    # spectral template
    if spectrum is not None:
        if spectrum.lower() == "methane":
            pykliproot = os.path.dirname(os.path.realpath(__file__))
            spectrum_dat = np.loadtxt(os.path.join(pykliproot,"spectra","t800g100nc.flx"))[:160] #skip wavelegnths longer of 10 microns
            spectrum_wvs = spectrum_dat[:,1]
            spectrum_fluxes = spectrum_dat[:,3]
            spectrum_interpolation = sinterp.interp1d(spectrum_wvs, spectrum_fluxes, kind='cubic')

            spectra_template = spectrum_interpolation(dataset.wvs)
        else:
            raise ValueError("{0} is not a valid spectral template. Only currently supporting 'methane'"
                             .format(spectrum))
    else:
        spectra_template = None

    # default to instrument specific OWA?
    if OWA is None:
        OWA = dataset.OWA

    # save klip parameters as a string
    klipparams = "fmlib={fmclass}, mode={mode},annuli={annuli},subsect={subsections},sector_N_pix={sector_N_pix}," \
                 "fluxoverlap={fluxoverlap}, psf_fwhm={psf_fwhm}, minmove={movement}, " \
                 "numbasis={numbasis}/{maxbasis},minrot={minrot},calibflux={calibrate_flux},spectrum={spectrum}," \
                 "highpass={highpass}".format(mode=mode, annuli=annuli, subsections=subsections, movement=movement,
                                              numbasis="{numbasis}", maxbasis=np.max(numbasis), minrot=minrot,
                                              calibrate_flux=calibrate_flux, spectrum=spectrum, highpass=highpass,
                                              sector_N_pix=N_pix_sector, fluxoverlap=flux_overlap, psf_fwhm=PSF_FWHM,
                                              fmclass=fm_class)
    dataset.klipparams = klipparams

    # run WCS rotation on output WCS, which we'll copy from the input ones
    # TODO: wcs rotation not yet implemented. 
    dataset.output_wcs = np.array([w.deepcopy() if w is not None else None for w in dataset.wcs])

    # Set MLK parameters
    if mkl_exists:
        old_mkl = mkl.get_max_threads()
        mkl.set_num_threads(1)

    klip_outputs = klip_parallelized(dataset.input, dataset.centers, dataset.PAs, dataset.wvs, dataset.IWA, fm_class,
                                     OWA=OWA, mode=mode, annuli=annuli, subsections=subsections, movement=movement,
                                     flux_overlap=flux_overlap, PSF_FWHM=PSF_FWHM, numbasis=numbasis,
                                     maxnumbasis=maxnumbasis, aligned_center=aligned_center, numthreads=numthreads,
                                     minrot=minrot, spectrum=spectra_template, padding=padding, save_klipped=True,
                                     flipx=dataset.flipx,
                                     N_pix_sector=N_pix_sector, mute_progression=mute_progression)

    klipped, fmout, perturbmag, klipped_center = klip_outputs # images are already rotated North up East left

    dataset.fmout = fmout
    dataset.perturbmag = perturbmag
    # save output centers here
    dataset.output_centers = np.array([klipped_center for _ in range(klipped.shape[1])])

    # write fmout
    fm_class.save_fmout(dataset, fmout, outputdir, fileprefix, numbasis, klipparams=klipparams,
                        calibrate_flux=calibrate_flux, spectrum=spectra_template)

    # if we want to save the klipped image
    if save_klipped:
        # store it in the dataset object
        dataset.output = klipped

        # write to disk. Filepath
        outputdirpath = os.path.realpath(outputdir)
        print("Writing KLIPed Images to directory {0}".format(outputdirpath))


        # collapse in time and wavelength to examine KL modes
        if spectrum is None:
            KLmode_cube = np.nanmean(dataset.output, axis=1)
        else:
            #do the mean combine by weighting by the spectrum
            KLmode_cube = np.nanmean(dataset.output * spectra_template[None,:,None,None], axis=1)\
                          / np.mean(spectra_template)

        # broadband flux calibration for KL mode cube
        if calibrate_flux:
            KLmode_cube = dataset.calibrate_output(KLmode_cube, spectral=False)
        dataset.savedata(outputdirpath + '/' + fileprefix + "-klipped-KLmodes-all.fits", KLmode_cube,
                         klipparams=klipparams.format(numbasis=str(numbasis)), filetype="KL Mode Cube",
                         zaxis=numbasis)

        # if there is more than one wavelength, save spectral cubes
        if np.size(np.unique(dataset.wvs)) > 1:
            numwvs = np.size(np.unique(dataset.wvs))
            klipped_spec = klipped.reshape([klipped.shape[0], klipped.shape[1]//numwvs, numwvs,
                                            klipped.shape[2], klipped.shape[3]]) # (b, N_cube, wvs, y, x) 5-D cube

            # for each KL mode, collapse in time to examine spectra
            KLmode_spectral_cubes = np.nanmean(klipped_spec, axis=1)
            for KLcutoff, spectral_cube in zip(numbasis, KLmode_spectral_cubes):
                # calibrate spectral cube if needed
                if calibrate_flux:
                    spectral_cube = dataset.calibrate_output(spectral_cube, spectral=True)
                dataset.savedata(outputdirpath + '/' + fileprefix + "-klipped-KL{0}-speccube.fits".format(KLcutoff),
                                 spectral_cube, klipparams=klipparams.format(numbasis=KLcutoff),
                                 filetype="PSF Subtracted Spectral Cube")

    #Restore old setting
    if mkl_exists:
        mkl.set_num_threads(old_mkl)
