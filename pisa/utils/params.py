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
    '''
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
    '''
    newparams = {}
    for key,param in params.items():
        newparams[key] = param['value']

    return newparams


def select_hierarchy(params, normal_hierarchy=True):
    '''
    Corrects for a key in params being defined for both 'nh' and 'ih',
    and returns a modified dict with one value for the given hierarchy.
    '''
    newparams = {}
    for key, value in params.items():
        if key.endswith('_nh'):
            #Don't use if doesn't match request
            if not normal_hierarchy: continue
            #Use simplified key
            key = key.split('_')[0]
        if key.endswith('_ih'):
            if normal_hierarchy: continue
            key = key.split('_')[0]

        newparams[key] = value

    return newparams

def get_fixed_params(params, normal_hierarchy=True):
    '''
    Finds all fixed parameters in params dict and returns them in a
    new dictionary.
    '''

    fixed_params = {}
    for key, value in select_hierarchy(params,normal_hierarchy).items():
        if not value['fixed']: continue
        fixed_params[key] = value

    return fixed_params

def get_free_params(params, normal_hierarchy=True):
    '''
    Finds all free parameters in params dict and returns them in a new
    dictionary.
    '''

    free_params = {}
    for key, value in select_hierarchy(params,normal_hierarchy).items():
        if value['fixed']: continue
        free_params[key] = value

    return free_params

def get_prior_llh(value,fiducial,sigma):
    '''
    Returns the log(prior) for a gaussian prior probability, unless it
    has not been defined, in which case 0.0 is returned.. Ignores the
    constant term proportional to log(sigma_prior).

    value - specific parameter of likelihood hypothesis
    prior - a pair of (sigma_param,<best_fit_value>_param)
    '''

    return 0.0 if prior[0] is None else -((value - fiducial)**2/(2.0*sigma**2))
