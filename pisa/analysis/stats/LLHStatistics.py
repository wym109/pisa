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
from scipy.special import multigammaln
from scipy.stats import poisson

from pisa.utils.jsons import from_json


def generalized_ln_poisson(data, expectation):
    """
    When the data set is not integer based, we need a different way to
    calculate the poisson likelihood, so we'll use this version, which
    is appropriate for float data types (using the continuous version
    of the poisson pmf) as well as the standard integer data type for
    the discrete Poisson pmf.

    Returns: the natural logarithm of the value of the continuous form
    of the poisson probability mass function, given detected counts,
    'data' from expected counts 'expectation'.
    """

    if not np.all(data >= 0.0):
        print 'data error...'
        print 'min val :', np.min(data)
        print 'max val :', np.max(data)
        print 'mean val:', np.mean(data)
        print 'num < 0 :', np.sum(data < 0)
        print 'num == 0:', np.sum(data == 0)
        raise ValueError(
            "Data must have all bins >= 0.0! Template generation bug?"
        )

    if np.issubdtype(data.dtype, np.int):
        return poisson.logpmf(data, expectation)
    elif np.issubdtype(data.dtype, np.float):
        vals = (data*np.log(expectation) - expectation - multigammaln(data+1.0, 1))
        if not np.all(np.isfinite(vals)):
            print 'data:', data
            print 'expectation:', expectation
            print 'log(expectation):', np.log(expectation)
            print 'multigammaln(data+1.0, 1):', multigammaln(data+1.0, 1)
            print 'vals:', vals
            raise ValueError()
        return vals
    else:
        raise ValueError('Unhandled data dtype: %s. Must be float or'
                         ' int.' % psuedo_data.dtype)

def get_binwise_llh(pseudo_data, template):
    """
    Computes the log-likelihood (llh) of the pseudo_data from the
    template, where each input is expected to be a 2d numpy array
    """
    if not np.all(template >= 0.0):
        print 'template error...'
        print 'min val :', np.min(template)
        print 'max val :', np.max(template)
        print 'mean val:', np.mean(template)
        print 'num < 0 :', np.sum(template < 0)
        print 'num == 0:', np.sum(template == 0)
        raise ValueError("Template must have all bins >= 0.0! Template generation bug?")

    totalLLH = np.sum(generalized_ln_poisson(pseudo_data,template))

    return totalLLH

def get_binwise_chisquare(pseudo_data, template):
    '''
    Computes the chisquare between the pseudo_data
    and the template, where each input is expected to be a 2d numpy array
    '''
    if not np.all(template >= 0.0):
        print 'template error...'
        print 'min val :', np.min(data)
        print 'max val :', np.max(data)
        print 'mean val:', np.mean(data)
        print 'num < 0 :', np.sum(data < 0)
        print 'num == 0:', np.sum(data == 0)
        raise ValueError("Template must have all bins >= 0.0! Template generation bug?")

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
