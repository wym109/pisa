# author : P.Eller, J.L.Lanfranchi
#
# date   : March 25, 2016


import numpy as np
from uncertainties import unumpy as unp
from scipy.special import gammaln

from pisa.utils.log import logging
from pisa.utils.barlow import likelihoods


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

    llh = actual_values*np.log(expected_values) - expected_values 
    # to center around 0
    llh -= actual_values*np.log(actual_values) - actual_values
    #return (actual_values*np.log(expected_values) - expected_values - gammaln(actual_values + 1))
    return llh


def log_poisson(k,l):
    return k*np.log(l) -l - gammaln(k+1)

def log_smear(x,sigma):
    return-np.log(sigma)-0.5*np.log(2*np.pi)-np.square(x)/(2*np.square(sigma))

def conv_poisson(k,l,s,nsigma=3,steps=50.):
    st = 2*(steps+1)
    conv_x = np.linspace(-nsigma*s,+nsigma*s,st)[:-1]+nsigma*s/(st-1.)
    conv_y = log_smear(conv_x,s)
    f_x = conv_x + l
    #f_x = conv_x + k
    # avoid zero values for lambda
    idx = np.argmax(f_x>0)
    f_y = log_poisson(k,f_x[idx:])
    #f_y = log_poisson(f_x[idx:],l)
    if np.isnan(f_y).any():
	logging.error('`NaN values`:')
        logging.error("idx = ", idx)
        logging.error("s = ", s)
        logging.error("l = ", l)
        logging.error("f_x = ", f_x)
        logging.error("f_y = ", f_y)
    f_y = np.nan_to_num(f_y)
    conv = np.exp(conv_y[idx:] + f_y)
    norm = np.sum(np.exp(conv_y))
    return conv.sum()/norm

def norm_conv_poisson(k,l,s,nsigma=3,steps=50.):
    cp = conv_poisson(k,l,s,nsigma=nsigma,steps=steps)
    n1 = np.exp(log_poisson(l,l))
    n2 = conv_poisson(l,l,s,nsigma=nsigma,steps=steps)
    return cp*n1/n2

def conv_llh(actual_values, expected_values):
    """
    compute the convolution llh using the uncertainty on the expected values to smear out the poisson pdfs
    """
    actual_values = unp.nominal_values(actual_values).ravel()
    sigma = unp.std_devs(expected_values).ravel()
    expected_values = unp.nominal_values(expected_values).ravel()
    triplets = np.array([actual_values, expected_values, sigma]).T
    norm_triplets = np.array([actual_values, actual_values, sigma]).T
    sum = 0
    for i in xrange(len(triplets)):
        sum += np.log(max(SMALL_POS,conv_poisson(*triplets[i])))
        sum -= np.log(max(SMALL_POS,conv_poisson(*norm_triplets[i])))
    return sum

def barlow_llh(actual_values, expected_values):
    '''
    compute the barlow LLH taking into account finite stats
    '''
    l = likelihoods()
    actual_values = unp.nominal_values(actual_values).ravel()
    sigmas = [unp.std_devs(ev.ravel()) for ev in expected_values]
    expected_values = [unp.nominal_values(ev).ravel() for ev in expected_values]
    uws = [np.square(ev)/np.square(s) for ev, s in zip(expected_values, sigmas)]
    ws = [np.square(s)/ev for ev, s in zip(expected_values, sigmas)]
    l.SetData(actual_values)
    l.SetMC(np.array(ws))
    l.SetUnweighted(np.array(uws))
    llh =  l.GetLLH('barlow') 
    del l
    return -llh

def mod_chi2(actual_values, expected_values):
    actual_values = unp.nominal_values(actual_values).ravel()
    sigma = unp.std_devs(expected_values).ravel()
    expected_values = unp.nominal_values(expected_values).ravel()
    chi2 = np.square(actual_values - expected_values)/(np.square(sigma)+expected_values)
    # wrong def.
    #chi2 = np.square(actual_values - expected_values)/(np.square(sigma)+actual_values)
    return np.sum(chi2)
