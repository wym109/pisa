#
# LLHStatistics.py
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# Generic module that has the helper functions for performing an LLH
# calculations on a set of templates. Operates on templates, which are
# numpy arrays. No input/output operations or reading from customized
# files in this module.
#


import re
import numpy as np
from scipy.stats import poisson

from pisa.utils.jsons import from_json


def get_binwise_llh(pseudo_data, template):
    """Return log-likelihood (llh) that `template` came from `pseudo_data`.

    Parameters
    ----------
    pseudo_data, template : arrays of same shape

    Raises
    ------
    ValueError if llh

    """
    llh = np.sum(poisson.logpmf(pseudo_data, expectation))

    if not np.isfinite(llh):
        msg = '`llh` is not finite.'
        msg += '\nllh = %s' % llh
        msg += '\npseudo_data = %s' % pseudo_data
        msg += '\nexpectation = %s' % expectation
        raise ValueError(msg)

    return llh

def get_binwise_chisquare(pseudo_data, template):
    '''
    Computes the chisquare between the pseudo_data
    and the template, where each input is expected to be a 2d numpy array
    '''
    if not np.all(template >= 0.0):
        logging.error('template error...')
        logging.error('min val : %s' % np.min(data))
        logging.error('max val : %s' % np.max(data))
        logging.error('mean val: %s' % np.mean(data))
        logging.error('num < 0 : %s' % np.sum(data < 0))
        logging.error('num == 0: %s' % np.sum(data == 0))
        raise ValueError('Template must have all bins >= 0.0! Template generation bug?')

    total_chisquare = np.sum(np.divide(np.power((pseudo_data - template), 2), pseudo_data))

    return total_chisquare

def get_random_map(template, seed=None):
    """
    Gets an event map with integer entries from non-integer entries
    (in general) in the template, varied according to Poisson
    statistics.
    """
    # Set the seed if given
    if not seed is None:
        np.random.seed(seed=seed)

    return poisson.rvs(template)
