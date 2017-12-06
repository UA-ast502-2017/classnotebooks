.. _install-label:

Installation
==============

Dependencies
-------------
Before you install pyKLIP, you will need to install the following packages, which are useful for most astronomical
data analysis situations anyways. The main pyKLIP code is cross-compatible with both python2.7 and python3.5.

* numpy
* scipy
* astropy
* Optional: matplotlib, mkl-service

For the optional packages, matplotlib is useful to actually plot the images. For mkl-service, pyKLIP autmoatically
toggles off MKL parallelism during parallelized KLIP if the mkl-service package is installed. Otherwise, you
will need to toggle them off yourself for optimal performance. See notes on parallelized performance below.

For :ref:`bka-label` specifically, you'll also want to install the following packages:

* emcee
* corner

As pyKLIP is computationally expensive, we recommend a powerful computer to optimize the computation. As direct imaging
data comes in many different forms, we cannot say
right here what the hardware requirements are for your data reduction needs. For data from the Gemini Planet Imager
(GPI), a computer with 20+ GB of memory is optimal for an 1 hour sequence taken with the integral field spectrograph and
reduced using ADI+SDI. For broadband polarimetry data from GPI, any laptop can reduce the data.

Install
-------

Due to the continually developing nature of pyKLIP, we recommend you use the current version of the code on
`Bitbucket <https://bitbucket.org/pyKLIP/pyklip>`_ and keep it updated.
To install the most up to date developer version, clone this repository if you haven't already::

    $ git clone git@bitbucket.org:pyKLIP/pyklip.git

This clones the repoistory using SSH authentication. If you get an authentication error, you will want to follow `this guide <https://confluence.atlassian.com/bitbucket/set-up-ssh-for-git-728138079.html>`_ to setup SSH authentication, or `clone using the HTTPS option instead <https://confluence.atlassian.com/bitbucket/clone-a-repository-223217891.html>`_, which just requires a password.

Once the repository is cloned onto your computer, ``cd`` into it and run the setup file::

    $ python setup.py develop

If you use multiple versions of python, you will need to run ``setup.py`` with each version of python
(this should not apply to most people).

Note on parallelized performance
--------------------------------


Due to the fact that numpy compiled with BLAS and MKL also parallelizes linear algebra routines across multiple cores,
performance can actually sharply decrease when multiprocessing and BLAS/MKL both try to parallelize the KLIP math.
If you are noticing your load averages greatly exceeding the number of threads/CPUs,
try disabling the BLAS/MKL optimization when running pyKLIP.

To disable OpenBLAS, just set the following environment variable before running pyKLIP::

    $ export OPENBLAS_NUM_THREADS=1

`A recent update to anaconda <https://www.continuum.io/blog/developer-blog/anaconda-25-release-now-mkl-optimizations>`_
included some MKL optimizations which may cause load averages to greatly exceed the number of threads specified in pyKLIP.
As with the OpenBLAS optimizations, this can be avoided by setting the maximum number of threads the MKL-enabled processes can use::

    $ export MKL_NUM_THREADS=1

As these optimizations may be useful for other python tasks, you may only want MKL_NUM_THREADS=1 only when pyKLIP is called,
rather than on a system-wide level. By defaulf in ``parallelized.py``, if ``mkl-service`` is installed, the original
maximum number of threads for MKL is saved, and restored to its original value after pyKLIP has finished. You can also
modify the number of threads MKL uses on a per-code basis by running the following piece of code (assuming ``mkl-service`` is installed)::

    import mkl
    mkl.set_num_threads(1)

