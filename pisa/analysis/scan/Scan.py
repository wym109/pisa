#
# ScanAnalysis.py
#
# Helper module for brute-force scanning analysis over the whole parameter space
#
# author: Sebastian Böser <sboeser@uni-mainz.de>
#

import sys
import numpy as np

from pisa.utils.log import logging, physics, profile
from pisa.utils.params import get_values, select_hierarchy, get_fixed_params, get_free_params, get_prior_llh, get_param_values, get_param_scales, get_param_bounds, get_param_priors
from pisa.analysis.stats.LLHStatistics import get_binwise_llh

def find_max_grid(fmap,template_maker,params,grid_settings,normal_hierarchy,save_steps=True):
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

    # Get params dict which will be optimized (free_params) and which
    # won't be (fixed_params) but are still needed for get_template()
    fixed_params = get_fixed_params(select_hierarchy(params,normal_hierarchy))
    free_params = get_free_params(select_hierarchy(params,normal_hierarchy))

    #Create the grid from the bounds and/or grid settings
    for key in free_params:
        #Use the default if none is specified
        if not key in grid_settings:
            nsteps = grid_settings["*"]
            logging.trace("Using %u steps for %s from %.5f to %.5f" % 
                            (nsteps, key, *free_params[key]['range']))

    sys.exit(1)


    #Store all the steps
    steps_dict = {key:[] for key in fixed_params.keys()}
    steps_dict['llh'] = []

    
    template_params = dict(get_values(free_params).items() + get_values(fixed_params).items())

    # Now get true template
    profile.info('start template calculation')
    true_template = template_maker.get_template(template_params)
    profile.info('stop template calculation')
    true_fmap = flatten_map(true_template)

    #and calculate the likelihood
    llh = -get_binwise_llh(fmap,true_fmap)
    llh -= sum([ get_prior_llh(opt_val,sigma,value) for (opt_val,(sigma,value)) in zip(opt_vals,priors)])

    # Save all optimizer-tested values to opt_steps_dict, to see
    # optimizer history later
    for key in names:
        opt_steps_dict[key].append(template_params[key])
    opt_steps_dict['llh'].append(llh)

   physics.debug("LLH is %.2f at: "%llh) 
    for name, val in zip(names, opt_vals):
        physics.debug(" %20s = %6.4f" %(name,val))
    

    #Only return best fit value?
    #use np.array maximum here...
    #llh_max = -sys.float_info.max
    #Report best fit
    #physics.info('Found best LLH = %.2f in %d calls at:'
    #    %(llh,dict_flags['funcalls']))
    #for name, val in best_fit_params.items():
    #    physics.info('  %20s = %6.4f'%(name,val))

    return steps_dict

