#
# LLHStatistics.py
#
# Generic module that has the helper functions for performing an LLH
# calculations on a set of templates. Operates on templates, which are
# numpy arrays. No input/output operations or reading from customized
# files in this module.
#

import re
import numpy as np
from scipfy.special import gamma
from scipy.stats import poisson

from pisa.utils.jsons import from_json

def poisson_pmf_float(data,expectation):
    """
    When the data set is not integer based, we need a different way to
    calculate the poisson likelihood, so we'll use this version, which
    is appropriate for float data types.
    """
    return (expectation**data)*np.exp(-expectation)/gamma(data+1)

def get_binwise_llh(pseudo_data,template):
    """
    Computes the log-likelihood (llh) of the pseudo_data from the
    template, where each input is expected to be a 2d numpy array
    """

    if not np.alltrue(template >= 0.0):
        raise ValueError("Template must have all bins >= 0.0! Template generation bug?")

    if bool(re.match('^int',pseudo_data.dtype.name)):
        totalLLH = np.sum(np.log(poisson.pmf(pseudo_data,template)))
    elif bool(re.match('^float',pseudo_data.dtype.name)):
        totalLLH = np.sum(np.log(poisson_pmf_float(pseudo_data,template)))
    else:
        raise ValueError(
            "Unknown pseudo_data dtype: %s. Must be float or int!"%psuedo_data.dtype)

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


