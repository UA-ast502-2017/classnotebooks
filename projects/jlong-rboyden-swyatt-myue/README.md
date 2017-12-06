# Applying Dimensionality Reduction Techniques to PSF Estimation in Astronomy

## Install required libraries

The usual suspects should already be installed if you use them regularly:

  * numpy
  * scipy
  * matplotlib
  * astropy (for FITS IO)
  * the Jupyter Notebook

(The command `conda install numpy scipy matplotlib astropy notebook` should do it, if not.)

### [pyKLIP](http://pyklip.readthedocs.io/en/latest/install.html)

First:

    git clone https://bitbucket.org/pyKLIP/pyklip.git

Then:

    cd pyklip
    python setup.py develop

See notebooks in `pyklip/examples`.

### [scikit-learn](http://scikit-learn.org/stable/install.html)

    pip install -U scikit-learn

or

    conda install scikit-learn

### [pandas](https://pandas.pydata.org/pandas-docs/stable/install.html)

    pip install -U pandas

or

    conda install pandas

### [SWarp](https://www.astromatic.net/software/swarp)

See the instructions on the website above.
