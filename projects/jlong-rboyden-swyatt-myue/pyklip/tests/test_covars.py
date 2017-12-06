import numpy as np
import pyklip.covars as covars


def test_matern32_numpy():
    """
    Tests the Matern Kernel (numpy implementation)
    """
    x = np.arange(10.)
    y = np.arange(15., 15 + x.shape[0])
    sigmas = np.random.rand(x.shape[0])
    corr_len = 3.5

    # test numpy version
    cov = covars.matern32(x, y, sigmas, corr_len)

    for j in range(x.shape[0]):
        for i in range(x.shape[0]):
            if i == j:
                np.testing.assert_almost_equal(cov[j, i], sigmas[i] * sigmas[j])
            else:
                dist = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
                arg = np.sqrt(3) * dist / corr_len
                val = sigmas[i] * sigmas[j] * (1 + arg) * np.exp(-arg)
                np.testing.assert_almost_equal(cov[i, j], val)
                np.testing.assert_almost_equal(cov[j, i], val)



def test_sq_exp_numpy():
    """
    Tests the Square Exponential Kernel (numpy implementation)
    """
    x = np.arange(10.)
    y = np.arange(15., 15 + x.shape[0])
    sigmas = np.random.rand(x.shape[0])
    corr_len = 3.5

    # test numpy version
    cov = covars.sq_exp(x, y, sigmas, corr_len)

    for j in range(x.shape[0]):
        for i in range(x.shape[0]):
            if i == j:
                np.testing.assert_almost_equal(cov[j, i], sigmas[i] * sigmas[j])
            else:
                dist = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
                val = sigmas[i] * sigmas[j] * np.exp(-dist ** 2 / (2 * corr_len ** 2))
                np.testing.assert_almost_equal(cov[i, j], val)
                np.testing.assert_almost_equal(cov[j, i], val)