#
# Scan.py
#
# Helper module for brute-force scanning analysis over the whole parameter space
#
# author: Sebastian Boeser <sboeser@uni-mainz.de>
#

import sys
import numpy as np
from itertools import product

from pisa.utils.log import logging, physics, profile
from pisa.utils.params import get_values, select_hierarchy, get_fixed_params, get_free_params, get_prior_llh, get_param_values, get_param_scales, get_param_bounds, get_param_priors
from pisa.analysis.stats.Maps import flatten_map
from pisa.analysis.stats.LLHStatistics import get_binwise_llh

def calc_steps(params, settings):
    '''
    Get the actual grid values for each key. If settings is a list of
    values, use these directly as steps. If settings has single value,
    generate this many steps within the bounds given for the parameter.
    Parameters are identified by names, or "*" which is the default for all
    parameters
    '''

    #Collect the steps settings for each parameter
    for key in params:

        #If specific steps are given, use these
        if key in settings:
            params[key]['steps'] = settings[key]
        else:
            params[key]['steps'] = settings['*']

    #Now convert number of steps to actual steps
    for key in params:
        #ignore if we already have those
        if isinstance(params[key]['steps'],np.ndarray): continue

        #calculate the steps
        lower, upper = params[key]['range']
        nsteps = params[key]['steps']
        params[key]['steps'] = np.linspace(lower,upper,nsteps)

    #report for all
    for name, steps in [ (k,v['steps']) for k,v in params.items()]:
       logging.debug("Using %u steps for %s from %.5f to %.5f" %
                          (len(steps), name, steps[0], steps[-1]))


def find_max_grid(fmap,template_maker,params,grid_settings,save_steps=True,
                                                     normal_hierarchy=True):
    '''
    Finds the template (and free systematic params) that maximize
    likelihood that the data came from the chosen template of true
    params, using a brute force grid scan over the whole parameter space.

    returns a dictionary of llh data and best fit params, in the format:
      {'llh': [...],
       'param1': [...],
       'param2': [...],
       ...}
    where 'param1', 'param2', ... are the free params that are varied in the
    scan. If save_steps is False, all lists only contain the best-fit parameters
    and llh values.
    '''

    #print "NOW INSIDE find_max_grid:"
    #print "After fixing to their true values, params dict is now: "
    #for key in params.keys():
    #    try: print "  >>param: %s value: %s"%(key,str(params[key]['best']))
    #    except: continue


    # Get params dict which will be optimized (free_params) and which
    # won't be (fixed_params) but are still needed for get_template()
    fixed_params = get_fixed_params(select_hierarchy(params,normal_hierarchy))
    free_params = get_free_params(select_hierarchy(params,normal_hierarchy))

    #Obtain just the priors
    priors = get_param_priors(free_params)

    #Calculate steps for all free parameters
    calc_steps(free_params, grid_settings['steps'])

    #Build a list from all parameters that holds a list of (name, step) tuples
    steplist = [ [(name,step) for step in param['steps']] for name, param in sorted(free_params.items())]

    #Prepare to store all the steps
    steps = {key:[] for key in free_params.keys()}
    steps['llh'] = []

    #Iterate over the cartesian product
    for pos in product(*steplist):

        #Get a dict with all parameter values at this position
        #including the fixed parameters
        template_params = dict(list(pos) + get_values(fixed_params).items())

        #print "   >> NOW IN LOOP: "
        #for key in template_params.keys():
        #    try: print "  >>param: %s value: %s"%(key,str(template_params[key]['value']))
        #    except: continue

        # Now get true template
        profile.info('start template calculation')
        true_template = template_maker.get_template(template_params)
        profile.info('stop template calculation')
        true_fmap = flatten_map(true_template)

        # and calculate the likelihood
        llh = -get_binwise_llh(fmap,true_fmap)

        # get sorted vals to match with priors
        vals = [ v for k,v in sorted(pos) ]
        llh -= sum([get_prior_llh(v, *p) for v,p in zip(vals, priors)])

        # Save all values to steps and report
        steps['llh'].append(llh)
        physics.debug("LLH is %.2f at: "%llh)
        for key, val in pos:
            steps[key].append(val)
            physics.debug(" %20s = %6.4f" %(key, val))

    #Find best fit value
    maxllh = min(steps['llh'])
    maxpos = steps['llh'].index(maxllh)

    #Report best fit
    physics.info('Found best LLH = %.2f in %d calls at:'
                 %(maxllh,len(steps['llh'])))
    for name, vals in steps.items():
        physics.info('  %20s = %6.4f'%(name,vals[maxpos]))

        #only save this maximum if asked for
        if not save_steps:
            steps[name]=vals[maxpos]

    return steps
