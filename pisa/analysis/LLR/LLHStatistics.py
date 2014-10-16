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

def get_binwise_llh(pseudo_data,true_template):
    '''
    computes the log-likelihood (llh) of the pseudo_data from the
    true_template, where each input is expected to be a 2d numpy array
    '''

    if not np.alltrue(true_template >= 0.0):
        raise Exception("true_template must have all bins >= 0.0! Template generation bug?")

    totalLLH = np.sum(np.log(poisson.pmf(pseudo_data,np.float64(true_template))))

    return totalLLH

def get_random_map(template):
    '''
    Gets an event map with integer entries from non-integer entries
    (in general) in the template, varied according to Poisson
    statistics.
    '''
    return poisson.rvs(template)

def add_prior(param,prior):
    '''
    Returns the log(prior) for a gaussian prior probability. Ignores
    the constant term proportional to log(sigma_prior).

    param - specific parameter of likelihood hypothesis
    prior - a pair of (sigma_param,<best_fit_value>_param)
    '''

    sigma = prior[0]
    best = prior[1]
    prior_val = 0
    if prior[0] is None: return 0.0
    else: return -((param - best)**2/(2.0*sigma**2))
    
    
