#
# LLHStatistics.py 
#
# Generic module that has the helper functions for performing an LLH
# calculations on a set of templates. Operates on templates, which are
# 1d (flattened) numpy arrays. No input/output operations or reading
# from customized files in this module.
#
#

from utils.jsons import from_json
import numpy as np
from scipy.stats import poisson

def get_binwise_llh(pseudo_data,true_template):
    '''
    computes the log-likelihood (llh) of the pseudo_data from the
    true_template, where each input is expected to be a 2d numpy array
    '''

    if not np.alltrue(true_template >= 0.0):
        raise Exception("true_template must have all bins >= 0.0! Template generation bug?")
    
    totalLLH = np.sum(np.log10(poisson.pmf(pseudo_data,true_template,dtype=np.float64)))
    
    return totalLLH

def get_random_map(template):
    '''
    Gets an event map with integer entries from non-integer entries
    (in general) in the true template, varied according to Poisson
    statistics.
    '''
    return poisson.rvs(template)
