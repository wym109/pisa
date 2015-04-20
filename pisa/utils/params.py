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

def get_prior_llh(value,sigma,fiducial):
    """
    Returns the log(prior) for a gaussian prior probability, unless it
    has not been defined, in which case 0.0 is returned.. Ignores the
    constant term proportional to log(sigma_prior).

    value - specific value of free parameter in likelihood hypothesis
    sigma - (gaussian) prior on free parameter.
    fiducial - best fit value of free parameter.
    """
    return 0.0 if sigma is None else -((value - fiducial)**2/(2.0*sigma**2))

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
