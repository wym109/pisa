#
# params.py
#
# Functions for accessing and formatting parameters taken from .json
# template settings file for analysis.
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# date:   16 October 2014
#

import scipy as sp
import numpy as np

def get_values(params):
    """
    Takes the params dict which is of the form:
      {'param1': {
        'value': val,
        'fixed': bool
       },
       'param2': {...},...
      }
    and returns a dictionary of names/values as:
      {'param1': val,
       'param2': val,...
      }
    """
    return { key: param['value'] for key, param in sorted(params.items()) }

def select_hierarchy(params, normal_hierarchy):
    """
    Corrects for a key in params being defined for both 'nh' and 'ih',
    and returns a modified dict with one value for the given hierarchy.
    """

    if not isinstance(normal_hierarchy, bool):
        raise ValueError('Hierarchy selection must be boolean value')

    newparams = {}
    for key, value in params.items():
        if key.endswith('_nh'):
            #Don't use if doesn't match request
            if not normal_hierarchy: continue
            #Use simplified key
            key = key.rsplit('_',1)[0]
        if key.endswith('_ih'):
            if normal_hierarchy: continue
            key = key.rsplit('_',1)[0]

        newparams[key] = value

    return newparams

def get_fixed_params(params):
    """
    Finds all fixed parameters in params dict and returns them in a
    new dictionary.
    """

    return { key: value for key, value in params.items() if value['fixed']}

def get_free_params(params):
    """
    Finds all free parameters in params dict and returns them in a new
    dictionary.
    """

    return { key: value for key, value in params.items() if not value['fixed']}

def get_prior_llh(value, p1=None, p2=None):
    '''
    Value of the log prior at a hypothesized parameter value.

    Can specify no log prior, a parabolic log prior (i.e., a Gaussian prior),
    or an arbitrary continuous, piecewise-linearly-interpolated log prior. Note
    that the latter is linearly interpolated in log space.

    Parameters
    ----------
    value : float
        Parameter's (hypothesized) value

    p1 : None, float or sequence of floats
        If None: Return 0.0 (no prior)
        If float: Width (sigma) of the parabolic log prior (Gaussian prior)
        If sequence of floats: Parameter values at which the piecewise-linear
            log prior is defined
        Default: None

    p2 : None, float or sequence of floats (must match that of p1)
        If float: Parameter's best-fit value (i.e., parameter value
            corresponding to parabolic log prior's vertex or, equivalently,
            Gaussian prior's mu)
        If sequence of floats: Log prior values corresponding to parameter
            values p1
        Default: None

    Returns
    -------
    log_prior : float
        If a log prior is specified and the hypothesized value lies within the
        specified domain in the case of a piecewise-linear log prior, returns
        the corresponding log prior's value for the hypothesis; otherwise,
        returns 0.0.

    Note
    ----
    Additive constants are irrelevant, and thus the implementation here of a
    parabolic log prior (Gaussian prior) ignores the normalization constant.

    '''
    # No log prior
    if p1 is None:
        log_prior = np.zeros_like(value)

    # Piecewise-linear log prior (returns 0.0 if hypothesis lies outside specified
    # range)
    elif hasattr(p1, '__len__'):
        interpolant = sp.interpolate.interp1d(x=p1, y=p2,
                                              kind='linear',
                                              copy=True,
                                              bounds_error=False,
                                              fill_value=0.0)
        log_prior = float(interpolant(value))
        
    # Parabolic log prior (Gaussian prior)
    else:
        sigma = p1
        fiducial = p2
        log_prior = -((value - fiducial)**2/(2.0*sigma**2))

    return log_prior

def get_param_values(params):
    """
    Returns a list of parameter values
    """
    return [ val['value'] for key,val in sorted(params.items()) ]

def get_param_scales(params):
    """
    Returns a list of parameter scales
    """
    return [ val['scale'] for key,val in sorted(params.items()) ]

def get_param_bounds(params):
    """
    Returns a list of parameter bounds where elements are (min,max) pairs
    """
    return [ val['range'] for key,val in sorted(params.items()) ]

def get_param_priors(params):
    """
    Returns a list of [(prior,value),...] for each param
    """
    return [ [val['prior'],val['value']] for key,val in sorted(params.items()) ]

def get_atm_params(params):
    """
    Returns dictionary of just the atmospheric parameters
    """
    atm_params = ['deltam31','theta23']
    return { key: value for key, value in params.items()
             for p in atm_params if p in key}

def fix_osc_params(params):
    """
    Returns dict identical to params dict but with all oscillation
    parameters set to "fixed"=True
    """
    new_params = {}
    # or initialize with new copy by dict(params)
    osc_params = ['deltam31','deltam21','theta23','theta13','theta12','deltacp']
    for key,value in params.items():
        new_params[key] = value.copy()
        for okey in osc_params:
            if okey in key:
                new_params[key]['fixed'] = True
    return new_params

def fix_atm_params(params):
    """
    Returns dict identical to params dict but with all atmospheric
    oscillation parameters fixed.
    """
    new_params = {}
    # or initialize with new copy by dict(params)
    atm_params = ['deltam31','theta23']
    for key,value in params.items():
        new_params[key] = value.copy()
        for akey in atm_params:
            if akey in key:
                new_params[key]['fixed'] = True
    return new_params
