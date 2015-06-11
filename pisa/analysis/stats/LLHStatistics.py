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
from scipy.special import iv, multigammaln
from scipy.stats import poisson, skellam, norm

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

    if not np.alltrue(data >= 0.0):
        raise ValueError(
            "Template must have all bins >= 0.0! Template generation bug?")

    ln_poisson = 0.0
    if bool(re.match('^int',data.dtype.name)):
        return np.log(poisson.pmf(data,expectation))
    elif bool(re.match('^float',data.dtype.name)):
        return (data*np.log(expectation) - expectation - multigammaln(data+1.0, 1))
    else:
        raise ValueError(
            "Unknown data dtype: %s. Must be float or int!"%psuedo_data.dtype)

def get_binwise_llh(pseudo_data, template, template_params):
    """
    Computes the log-likelihood (llh) of the pseudo_data from the
    template, where each input is expected to be a 2d numpy array
    """

    if not np.alltrue(template >= 0.0):
        raise ValueError("Template must have all bins >= 0.0! Template generation bug?")

    if 'residual_up_down' in template_params and template_params['residual_up_down']:
        if len(template)!=2:
            raise ValueError("Under current template settings, template must be an array of two arrays(i.e. up-going and down-going array)!")
        mu1 = np.float64(template[0])
        mu2 = np.float64(template[1])
        #totalLLH = np.sum(skellam.logpmf(pseudo_data,np.float64(template[0]),np.float64(template[1])))    
                                              # causes overflow when mu1,mu2 gets too big, it calculates pmf first, then take the log.
        totalLLH = np.sum(-mu1-mu2+0.5*pseudo_data*np.log(mu1/mu2)+np.log(iv(pseudo_data,2*np.sqrt(mu1*mu2)))) 
                                              # better, but could also end up inf if (mu1*mu2) is too large, e.g. iv(0,2*np.sqrt(127444)) ~ 1.7961e+308

    elif 'ratio_up_down' in template_params and template_params['ratio_up_down']:
        if len(template)!=2 or len(pseudo_data)!=2:
            raise ValueError("Under current template settings, template and pseudo_data must be an array of two arrays(i.e. up-going and down-going array)!")
        t_N_up = np.float64(template[0])
        t_N_down = np.float64(template[1])
        d_N_up = np.float64(pseudo_data[0])
        d_N_down = np.float64(pseudo_data[1])
        #print "t_N_up = ", t_N_up
        #print "t_N_down = ", t_N_down
        #print "d_N_up = ", d_N_up
        #print "d_N_down = ", d_N_down
        print "min(d_N) = ", min(min(d_N_down),min(d_N_up))
        cut_zero = np.logical_or(np.logical_or(d_N_up == 0, d_N_down == 0), np.logical_or(t_N_up == 0,t_N_down == 0))
        cut_nonzero = np.logical_and(np.logical_and(d_N_up != 0, d_N_down != 0), np.logical_and(t_N_up != 0,t_N_down != 0))
        
        d_zero_N_up = d_N_up[cut_zero]
        d_zero_N_down = d_N_down[cut_zero]
        d_zero = np.append(d_zero_N_up,d_zero_N_down)
        t_zero_N_up = t_N_up[cut_zero]
        t_zero_N_down = t_N_down[cut_zero]
        t_zero = np.append(t_zero_N_up,t_zero_N_down)
        print "d_zero = ", d_zero
        print "t_zero = ", t_zero

        d_N_up = d_N_up[cut_nonzero]
        d_N_down = d_N_down[cut_nonzero]
        t_N_up = t_N_up[cut_nonzero]
        t_N_down = t_N_down[cut_nonzero]
        #print "d_N_down = ", d_N_down

        t_R = np.nan_to_num(t_N_up/t_N_down)
        d_R = np.nan_to_num(d_N_up/d_N_down)
        #t_R = np.nan_to_num(t_N_down/t_N_up)
        #d_R = np.nan_to_num(d_N_down/d_N_up)

        #t_R_err_sqr = np.nan_to_num(t_N_up*(t_N_up+t_N_down)/(t_N_down**3))
        #d_R_err_sqr = np.nan_to_num(d_N_up*(d_N_up+d_N_down)/(d_N_down**3))
        #t_R_err_inv = np.nan_to_num(t_N_down*np.sqrt(t_N_down)/np.sqrt(t_N_up*(t_N_up+t_N_down)))

        # inverse of the error when using ratio of up to down
        d_R_err_inv = np.nan_to_num(d_N_down*np.sqrt(d_N_down)/np.sqrt(d_N_up*(d_N_up+d_N_down)))

        # inverse of the error when using ratio of down to up
        #d_R_err_inv = np.nan_to_num(d_N_up*np.sqrt(d_N_up)/np.sqrt(d_N_down*(d_N_down+d_N_up)))

        # this definition is the new test statistic for ratio_analysis, it's not a likelihood function, its value should increase when two maps get more different, that's why the negative sign ( definition 1)
        #totalLLH = -np.nan_to_num(np.sqrt(np.sum(np.square((t_R-d_R)*d_R_err_inv))))
        #if d_zero !=[] and t_zero !=[]:
        #    totalLLH += np.sum(np.log(poisson.pmf(d_zero,t_zero)))

        #definition 2
        #totalLLH = -np.nan_to_num(np.sum(np.square(t_R-d_R)))      
        #if d_zero !=[] and t_zero !=[]:
        #    totalLLH += np.sum(np.log(poisson.pmf(d_zero,t_zero)))

        #definition 3, normal approximation
        a2 = d_R*d_R/t_N_up + 1/t_N_down
        a = np.sqrt(a2)
        b = d_R + 1
        c = t_N_up + t_N_down
        d = np.exp(0.5*(b*b-c*a2)/a2)
        pdf = b*d*(norm.cdf(b/a)-norm.cdf(-b/a))/a2/np.sqrt(2*t_N_up*t_N_down*np.pi) + np.exp(-0.5*c)/a2/np.pi/np.sqrt(t_N_up*t_N_down)
        totalLLH = np.sum(np.log(pdf))
        if d_zero !=[] and t_zero !=[]:
            totalLLH += np.sum(np.log(poisson.pmf(d_zero,t_zero)))

    else:
        totalLLH = np.sum(generalized_ln_poisson(pseudo_data, template))

    return totalLLH

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
