#
# LLHAnalysis.py
#
# Helper module for running the Optimizer-based LLR analysis.
#
# author: Tim Arlen
#         tca3@psu.edu
#
# date:   2 July 2014
#

import sys
import numpy as np
import scipy.optimize as opt

from pisa.utils.jsons import to_json
from pisa.utils.log import logging, physics, tprofile
from pisa.utils.params import get_values, select_hierarchy, get_fixed_params, get_free_params, get_prior_llh, get_param_values, get_param_scales, get_param_bounds, get_param_priors
from pisa.utils.utils import Timer
from pisa.analysis.stats.LLHStatistics import get_binwise_llh
from pisa.analysis.stats.Maps import flatten_map

#@profile
def find_alt_hierarchy_fit(asimov_data_set, template_maker,hypo_params,hypo_normal,
                           minimizer_settings,only_atm_params=True,check_octant=False):
    """
    For the hypothesis of the mass hierarchy being NMH
    ('normal_hierarchy'=True) or IMH ('normal_hierarchy'=False), finds the
    best fit for the free params in 'param_values' for the alternative
    (opposite) hierarchy that maximizes the LLH of the Asimov data set.

    \params:
      * asimov_data_set - asimov data set to find best fit llh
      * template_maker - instance of class pisa.analysis.TemplateMaker, used to
        generate the asimov data set.
      * hypo_params - parameters for template generation
      * hypo_normal - boolean for NMH (True) or IMH (False)
      * minimizer_settings - settings for bfgs minimization
      * only_atm_params - boolean to denote whether the fit will be over the
        atmospheric oscillation parameters only or over all the free params
        in params
    """

    # Find best fit of the alternative hierarchy to the
    # hypothesized asimov data set.
    #hypo_types = [('hypo_IMH',False)] if data_normal else  [('hypo_NMH',True)]
    hypo_params = select_hierarchy(hypo_params,normal_hierarchy=hypo_normal)

    with Timer() as t:
        llh_data = find_max_llh_bfgs(
            asimov_data_set,template_maker,hypo_params,minimizer_settings,
            normal_hierarchy=hypo_normal,check_octant=check_octant)
    tprofile.info("==> elapsed time for optimizer: %s sec"%t.secs)

    return llh_data

def display_optimizer_settings(free_params, names, init_vals, bounds, priors,
                               bfgs_settings):
    """
    Displays parameters and optimization settings that minimizer will run.
    """
    physics.info('%d parameters to be optimized'%len(free_params))
    for name,init,(down,up),(prior, best) in zip(names, init_vals, bounds, priors):
        physics.info(('%20s : init = %6.4f, bounds = [%6.4f,%6.4f], '
                     'best = %6.4f, prior = '+
                     ('%6.4f' if prior else "%s"))%
                     (name, init, down, up, best, prior))

    physics.debug("Optimizer settings:")
    for key,item in bfgs_settings.items():
        physics.debug("  %s -> `%s` = %.2e"%(item['desc'],key,item['value']))

    return

#@profile
def find_max_llh_bfgs(fmap, template_maker, params, bfgs_settings, save_steps=False,
                      normal_hierarchy=None, check_octant=False):
    """
    Finds the template (and free systematic params) that maximize
    likelihood that the data came from the chosen template of true
    params, using the limited memory BFGS algorithm subject to bounds
    (l_bfgs_b).

    returns a dictionary of llh data and best fit params, in the format:
      {'llh': [...],
       'param1': [...],
       'param2': [...],
       ...}
    where 'param1', 'param2', ... are the free params varied by
    optimizer, and they hold a list of all the values tested in
    optimizer algorithm, unless save_steps is False, in which case
    they are one element in length-the best fit params and best fit llh.
    """

    # Get params dict which will be optimized (free_params) and which
    # won't be (fixed_params) but are still needed for get_template()
    fixed_params = get_fixed_params(select_hierarchy(params,normal_hierarchy))
    free_params = get_free_params(select_hierarchy(params,normal_hierarchy))

    if len(free_params) == 0:
        logging.warn("NO FREE PARAMS, returning LLH")
        true_template = template_maker.get_template(get_values(fixed_params))
        channel = params['channel']['value']
        true_fmap = flatten_map(true_template,chan=channel)
        return {'llh': [-get_binwise_llh(fmap,true_fmap)]}

    init_vals = get_param_values(free_params)
    scales = get_param_scales(free_params)
    bounds = get_param_bounds(free_params)
    priors = get_param_priors(free_params)
    names  = sorted(free_params.keys())

    # Scale init-vals and bounds to work with bfgs opt:
    init_vals = np.array(init_vals)*np.array(scales)
    bounds = [bounds[i]*scales[i] for i in range(len(bounds))]

    opt_steps_dict = {key:[] for key in names}
    opt_steps_dict['llh'] = []

    const_args = (names,scales,fmap,fixed_params,template_maker,opt_steps_dict,priors)

    display_optimizer_settings(free_params, names, init_vals, bounds, priors,
                               bfgs_settings)

    best_fit_vals,llh,dict_flags = opt.fmin_l_bfgs_b(
        llh_bfgs, init_vals, args=const_args, approx_grad=True, iprint=0,
        bounds=bounds, **get_values(bfgs_settings))

    # If needed, run optimizer again, checking for second octant solution:
    if check_octant and ('theta23' in free_params.keys()):
        physics.info("Checking alternative octant solution")
        old_th23_val = free_params['theta23']['value']

        # Reflect across pi/4.0:
        delta = (np.pi/4.0) - old_th23_val
        free_params['theta23']['value'] = (np.pi/4.0) + delta

        init_vals = get_param_values(free_params)

        alt_opt_steps_dict = {key:[] for key in names}
        alt_opt_steps_dict['llh'] = []
        const_args = (names,scales,fmap,fixed_params,template_maker,
                      alt_opt_steps_dict,priors)
        display_optimizer_settings(free_params, names, init_vals, bounds, priors,
                                   bfgs_settings)
        alt_fit_vals,alt_llh,alt_dict_flags = opt.fmin_l_bfgs_b(
            llh_bfgs, init_vals, args=const_args, approx_grad=True, iprint=0,
            bounds=bounds, **get_values(bfgs_settings))

        # Alternative octant solution is optimal:
        print "ALT LLH: ",alt_llh
        print "llh: ",llh
        if alt_llh < llh:
            print "  >>TRUE..."
            best_fit_vals = alt_fit_vals
            llh = alt_llh
            dict_flags = alt_dict_flags
            opt_steps_dict = alt_opt_steps_dict


    best_fit_params = { name: value for name, value in zip(names, best_fit_vals) }

    #Report best fit
    physics.info('Found best LLH = %.2f in %d calls at:'
        %(llh,dict_flags['funcalls']))
    for name, val in best_fit_params.items():
        physics.info('  %20s = %6.4f'%(name,val))

    #Report any warnings if there are
    lvl = logging.WARN if (dict_flags['warnflag'] != 0) else logging.DEBUG
    for name, val in dict_flags.items():
        physics.log(lvl," %s : %s"%(name,val))

    if not save_steps:
        # Do not store the extra history of opt steps:
        for key in opt_steps_dict.keys():
            opt_steps_dict[key] = [opt_steps_dict[key][-1]]

    return opt_steps_dict


def llh_bfgs(opt_vals,*args):
    """
    Function that the bfgs algorithm tries to minimize. Essentially,
    it is a wrapper function around get_template() and
    get_binwise_llh().

    This fuction is set up this way, because the fmin_l_bfgs_b
    algorithm must take a function with two inputs: params & *args,
    where 'params' are the actual VALUES to be varied, and must
    correspond to the limits in 'bounds', and 'args' are arguments
    which are not varied and optimized, but needed by the
    get_template() function here. Thus, we pass the arguments to this
    function as follows:

    --opt_vals: [param1,param2,...,paramN] - systematics varied in the optimization.
    --args: [names,scales,fmap,fixed_params,template_maker,opt_steps_dict,priors]
      where
        names: are the dict keys corresponding to param1, param2,...
        scales: the scales to be applied before passing to get_template
          [IMPORTANT! In the optimizer, all parameters must be ~ the same order.
          Here, we keep them between 0.1,1 so the "epsilon" step size will vary
          the parameters in roughly the same precision.]
        fmap: pseudo data flattened map
        fixed_params: dictionary of other paramters needed by the get_template()
          function
        template_maker: template maker object
        opt_steps_dict: dictionary recording information regarding the steps taken
          for each trial of the optimization process.
        priors: gaussian priors corresponding to opt_vals list.
          Format: [(prior1,best1),(prior2,best2),...,(priorN,bestN)]
    """


    names,scales,fmap,fixed_params,template_maker,opt_steps_dict,priors = args

    # free parameters being "optimized" by minimizer re-scaled to their true values.
    unscaled_opt_vals = [opt_vals[i]/scales[i] for i in xrange(len(opt_vals))]

    unscaled_free_params = { names[i]: val for i,val in enumerate(unscaled_opt_vals) }
    template_params = dict(unscaled_free_params.items() + get_values(fixed_params).items())

    # Now get true template, and compute LLH
    with Timer() as t:
        if template_params['theta23'] == 0.0:
            logging.info("Zero theta23, so generating no oscillations template...")
            true_template = template_maker.get_template_no_osc(template_params)
        else:
            true_template = template_maker.get_template(template_params)
    tprofile.info("==> elapsed time for template maker: %s sec"%t.secs)
    true_fmap = flatten_map(true_template,chan=template_params['channel'])

    # NOTE: The minus sign is present on both of these next two lines
    # to reflect the fact that the optimizer finds a minimum rather
    # than maximum.
    llh = -get_binwise_llh(fmap,true_fmap)
    llh -= sum([ get_prior_llh(opt_val,sigma,value)
                 for (opt_val,(sigma,value)) in zip(unscaled_opt_vals,priors)])

    # Save all optimizer-tested values to opt_steps_dict, to see
    # optimizer history later
    for key in names:
        opt_steps_dict[key].append(template_params[key])
    opt_steps_dict['llh'].append(llh)

    physics.debug("LLH is %.2f at: "%llh)
    for name, val in zip(names, opt_vals):
        physics.debug(" %20s = %6.4f" %(name,val))

    return llh

