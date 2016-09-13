# author : P.Eller, J.L.Lanfranchi
#
# date   : March 25, 2016


import numpy as np
from uncertainties import unumpy as unp
from scipy.special import gammaln

from pisa.utils.comparisons import isbarenumeric
from pisa.utils.log import logging


# A small positive number with which to replace numbers smaller than it
SMALL_POS = 1e-10


def maperror_logmsg(m):
    with np.errstate(invalid='ignore'):
        msg = ''
        msg += '    min val : %s\n' %np.nanmin(m)
        msg += '    max val : %s\n' %np.nanmax(m)
        msg += '    mean val: %s\n' %np.nanmean(m)
        msg += '    num < 0 : %s\n' %np.sum(m < 0)
        msg += '    num == 0: %s\n' %np.sum(m == 0)
        msg += '    num > 0 : %s\n' %np.sum(m > 0)
        msg += '    num nan : %s\n' %np.sum(np.isnan(m))
    return msg


# TODO: chi2
def chi2(actual_values, expected_values):
    """Compute the chi-square between each value in `actual_values` and
    `expected_values`.

    Parameters
    ----------
    actual_values, expected_values : numpy.ndarrays of same shape

    Returns
    -------
    chi2 : numpy.ndarray of same shape as inputs
        chi-squared values corresponding to each pair of elements in the inputs

    Notes
    -----
    * Uncertainties are not propagated through this calculation.
    * Values in each input are clipped to the range [SMALL_POS, inf] prior to
      the calculation to avoid infinities due to the divide function.

    """
    assert actual_values.shape == expected_values.shape

    # TODO: Check isinstance before doing this?
    actual_values = unp.nominal_values(actual_values)
    expected_values = unp.nominal_values(expected_values)

    # TODO: this check (and the same for `actual_values`) should probably be done
    # elsewhere... maybe?
    if not np.all(actual_values >= 0.0):
        msg = '`actual_values` must all be >= 0...\n' \
                + maperror_logmsg(actual_values)
        raise ValueError(msg)

    if not np.all(expected_values >= 0.0):
        msg = '`expected_values` must all be >= 0...\n' \
                + maperror_logmsg(expected_values)
        raise ValueError(msg)

    # TODO: is this okay to do?
    # replace 0's with small positive numbers to avoid inf in division
    np.clip(actual_values, a_min=SMALL_POS, a_max=np.inf, out=actual_values)
    np.clip(expected_values, a_min=SMALL_POS, a_max=np.inf, out=expected_values)

    delta = actual_values - expected_values
    return (delta * delta) / actual_values


def llh(actual_values, expected_values):
    """Compute the log-likelihoods (llh) that each count in `actual_values`
    came from the the corresponding expected value in `expected_values`.

    Parameters
    ----------
    actual_values, expected_values : numpy.ndarrays of same shape

    Returns
    -------
    llh : numpy.ndarray of same shape as the inputs
        llh corresponding to each pair of elements in `actual_values` and
        `expected_values`.

    Notes
    -----
    * Uncertainties are not propagated through this calculation.
    * Values in `expected_values` are clipped to the range [SMALL_POS, inf]
      prior to the calculation to avoid infinities due to the log function.

    """
    assert actual_values.shape == expected_values.shape

    # Convert to simple numpy arrays containing floats
    if not isbarenumeric(actual_values):
        actual_values = unp.nominal_values(actual_values)
    if not isbarenumeric(expected_values):
        expected_values = unp.nominal_values(expected_values)

    with np.errstate(invalid='ignore'):
        # Mask off any nan expected values (these are assumed to be ok)
        actual_values = np.ma.masked_invalid(actual_values)
        expected_values = np.ma.masked_invalid(expected_values)

        # Check that new array contains all valid entries
        if not np.all(actual_values >= 0.0):
            msg = '`actual_values` must all be >= 0...\n' \
                    + maperror_logmsg(actual_values)
            raise ValueError(msg)

        # TODO: How should we handle nan / masked values in the "data"
        # (actual_values) distribution? How about negative numbers?

        ## Make sure actual values (aka "data") are valid -- no infs, no nans, etc.
        #if not np.all((actual_values < 0) | ~np.isfinite(actual_values)):
        #    msg = '`actual_values` must be >= 0 and neither inf nor nan...\n' \
        #            + maperror_logmsg(actual_values)
        #    raise ValueError(msg)

        # Check that new array contains all valid entries
        if not np.all(expected_values >= 0.0):
            msg = '`expected_values` must all be >= 0...\n' \
                    + maperror_logmsg(expected_values)
            raise ValueError(msg)

    # replace 0's with small positive numbers to avoid inf in log
    np.clip(expected_values, a_min=SMALL_POS, a_max=np.inf, out=expected_values)

    return (actual_values*np.log(expected_values) - expected_values -
            gammaln(actual_values + 1))


def log_poisson(k,l):
    return k*np.log(l) -l - gammaln(k+1)


def log_smear(x,sigma):
    return-np.log(sigma)-0.5*np.log(2*np.pi)-np.square(x)/(2*np.square(sigma))


def conv_poisson(k,l,s,nsigma=3,steps=100.):
    st = 2*(steps+1)
    conv_x = np.linspace(-nsigma*s,+nsigma*s,st)[:-1]+nsigma*s/(st-1.)
    conv_y = log_smear(conv_x,s)
    f_x = conv_x + l
    # avoid zero values for lambda
    idx = np.argmax(f_x>0)
    f_y = log_poisson(k,f_x[idx:])
    if np.isnan(f_y).any():
        logging.error('`NaN values`:')
        logging.error('idx = ', idx)
        logging.error('s = ', s)
        logging.error('l = ', l)
        logging.error('f_x = ', f_x)
        logging.error('f_y = ', f_y)
    f_y = np.nan_to_num(f_y)
    conv = np.exp(conv_y[idx:] + f_y)
    return conv.sum()*(conv_x[1]-conv_x[0])


def conv_llh(actual_values, expected_values):
    """Compute the convolution llh using the uncertainty on the expected values
    to smear out the poisson PDFs.

    """
    actual_values = unp.nominal_values(actual_values).ravel()
    sigma = unp.std_devs(expected_values).ravel()
    expected_values = unp.nominal_values(expected_values).ravel()
    triplets = np.array([actual_values, expected_values, sigma]).T
    sum = 0
    for i in xrange(len(triplets)):
        sum += np.log(max(SMALL_POS,conv_poisson(*triplets[i])))
    return sum


def mod_chi2(actual_values, expected_values):
    actual_values = unp.nominal_values(actual_values).ravel()
    sigma = unp.std_devs(expected_values).ravel()
    expected_values = unp.nominal_values(expected_values).ravel()
    chi2 = np.square(actual_values - expected_values)/(np.square(sigma)+expected_values)
    return np.sum(chi2)
