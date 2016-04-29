import numpy as np
from uncertainties import unumpy as unp
from scipy.special import gammaln

def llh(k, l):
    """
    Computes the log-likelihood (llh) of the pseudo_data from the
    template, where each input is expected to be a numpy
    array
    """
    # replace 0 or negative entries in template with very small
    # numbers, to avoid inf in log
    k = unp.nominal_values(k)
    l = unp.nominal_values(l)
    l[l <= 0] = 10e-10
    return np.sum(k * np.log(l) - l - gammaln(k+1))
