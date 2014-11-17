#
# LLHStatistics.py
#
# Generic module that has the helper functions for performing an LLH
# calculations on a set of templates. Operates on templates, which are
# numpy arrays. No input/output operations or reading from customized
# files in this module.
#

from pisa.utils.jsons import from_json
import numpy as np
from scipy.stats import poisson

def get_binwise_llh(pseudo_data,template):
    '''
    computes the log-likelihood (llh) of the pseudo_data from the
    template, where each input is expected to be a 2d numpy array
    '''

    if not np.alltrue(template >= 0.0):
        raise ValueError("Template must have all bins >= 0.0! Template generation bug?")

    totalLLH = np.sum(np.log(poisson.pmf(pseudo_data,np.float64(template))))

    return totalLLH

def get_random_map(template):
    '''
    Gets an event map with integer entries from non-integer entries
    (in general) in the template, varied according to Poisson
    statistics.
    '''
    return poisson.rvs(template)


