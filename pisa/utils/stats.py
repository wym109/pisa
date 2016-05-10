# author : P.Eller, J.L.Lanfranchi
#
# date   : March 25, 2016


import numpy as np
from uncertainties import unumpy as unp
from scipy.special import gammaln

from pisa.utils.log import logging


# A small positive number with which to replace numbers smaller than it
SMALL_POS = 1e-10


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
        logging.error('`actual_values`:')
        logging.error('    min val : %s' %np.min(actual_values))
        logging.error('    max val : %s' %np.max(actual_values))
        logging.error('    mean val: %s' %np.mean(actual_values))
        logging.error('    num < 0 : %s' %np.sum(actual_values < 0))
        logging.error('    num == 0: %s' %np.sum(actual_values == 0))
        logging.error('    num > 0 : %s' %np.sum(actual_values > 0))
        raise ValueError('`actual_values` must all be >= 0.0.')

    if not np.all(expected_values >= 0.0):
        logging.error('`expected_values`:')
        logging.error('    min val : %s' %np.min(expected_values))
        logging.error('    max val : %s' %np.max(expected_values))
        logging.error('    mean val: %s' %np.mean(expected_values))
        logging.error('    num < 0 : %s' %np.sum(expected_values < 0))
        logging.error('    num == 0: %s' %np.sum(expected_values == 0))
        logging.error('    num > 0 : %s' %np.sum(expected_values > 0))
        raise ValueError('`expected_values` must all be >= 0.0.')

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

    # TODO: Check isinstance before doing this?
    actual_values = unp.nominal_values(actual_values)
    expected_values = unp.nominal_values(expected_values)

    # TODO: this check (and the same for `actual_values`) should probably be done
    # elsewhere... maybe?
    if not np.all(actual_values >= 0.0):
        logging.error('`actual_values`:')
        logging.error('    min val : %s' %np.min(actual_values))
        logging.error('    max val : %s' %np.max(actual_values))
        logging.error('    mean val: %s' %np.mean(actual_values))
        logging.error('    num < 0 : %s' %np.sum(actual_values < 0))
        logging.error('    num == 0: %s' %np.sum(actual_values == 0))
        logging.error('    num > 0 : %s' %np.sum(actual_values > 0))
        raise ValueError('`actual_values` must all be >= 0.0.')

    if not np.all(expected_values >= 0.0):
        logging.error('`expected_values`:')
        logging.error('    min val : %s' %np.min(expected_values))
        logging.error('    max val : %s' %np.max(expected_values))
        logging.error('    mean val: %s' %np.mean(expected_values))
        logging.error('    num < 0 : %s' %np.sum(expected_values < 0))
        logging.error('    num == 0: %s' %np.sum(expected_values == 0))
        logging.error('    num > 0 : %s' %np.sum(expected_values > 0))
        raise ValueError('`expected_values` must all be >= 0.0.')

    # replace 0's with small positive numbers to avoid inf in log
    np.clip(expected_values, a_min=SMALL_POS, a_max=np.inf, out=expected_values)

    return (actual_values*np.log(expected_values) - expected_values -
            gammaln(actual_values + 1))
