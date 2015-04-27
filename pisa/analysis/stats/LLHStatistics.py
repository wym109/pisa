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


def generalized_ln_poisson(data,expectation):
    """
    When the data set is not integer based, we need a different way to
    calculate the poisson likelihood, so we'll use this version, which
    is appropriate for float data types (using the continuous version
    of the poisson pmf) as well as the standard integer data type for
    the discrete Poisson pmf
    """

    if not np.alltrue(data >= 0.0):
        raise ValueError(
            "Template must have all bins >= 0.0! Template generation bug?")

    ln_poisson = 0.0
    if bool(re.match('^int',data.dtype.name)):
        return np.log(poisson.pmf(data,expectation))
    elif bool(re.match('^float',data.dtype.name)):
        return (data*np.log(expectation) - expectation - multigammaln(data+1.0,1))
    else:
        raise ValueError(
            "Unknown data dtype: %s. Must be float or int!"%psuedo_data.dtype)

def get_binwise_llh(pseudo_data,template):
    """
    Computes the log-likelihood (llh) of the pseudo_data from the
    template, where each input is expected to be a 2d numpy array
    """

    if not np.alltrue(template >= 0.0):
        raise ValueError("Template must have all bins >= 0.0! Template generation bug?")

    totalLLH = np.sum(generalized_ln_poisson(pseudo_data,template))

    return totalLLH

def get_random_map(template, seed=None):
    """
    Gets an event map with integer entries from non-integer entries
    (in general) in the template, varied according to Poisson
    statistics.
    """
    #Set the seed if given
    if not seed is None:
        np.random.seed(seed=seed)

    return poisson.rvs(template)


