#
# LLRAnalysis.py
#
# Helper module for running the Optimizer-based LLR analysis.
#
# author: Tim Arlen
#         tca3@psu.edu
#
# date:   2 July 2014
#

import logging,sys
from datetime import datetime
import numpy as np
import scipy.optimize as opt

from pisa.utils.params import get_values, select_hierarchy, get_fixed_params, get_free_params, get_prior_llh
from pisa.analysis.LLR.LLHStatistics import get_random_map, get_binwise_llh

def get_pseudo_data_fmap(template_maker,fiducial_params):
    '''
    Creates a true template from fiducial_params, then uses Poisson statistics
    to vary the expected counts per bin to create a pseudo data set.

    IMPORTANT: returns a SINGLE flattened map of trck/cscd combined
    '''

    true_template = template_maker.get_template(fiducial_params)
    true_fmap = flatten_map(true_template)
    fmap = get_random_map(true_fmap)

    return fmap

def flatten_map(template):
    '''
    Takes a final level true (expected) template of trck/cscd, and returns a
    single flattened map of trck appended to cscd, with all zero bins
    removed.
    '''
    cscd = template['cscd']['map'].flatten()
    trck = template['trck']['map'].flatten()
    fmap = np.append(cscd,trck)
    fmap = np.array(fmap)[np.nonzero(fmap)]

    return fmap


def find_max_llh_grid(fmap_data,true_templates):
    '''
    Find the template from true_templates that maximizes likelihood
    that fmap_data came from the given template. Loops over all
    true_templates to find maximum.

    returns
      llh - value of max log(likelihood) found
      best_fit_params - dict of best fit values of the systematic
          params marginalized over.
      trial_data - dict of (llh,deltam31,theta23) for each point
          tested in the grid search.
    '''


    trial_data = {'llh':[],'deltam31':[],'theta23':[]}

    llh_max = -sys.float_info.max
    best_fit = {}; best_fit['deltam31'] = 0.0; best_fit['theta23'] = 0.0
    #print "  num true templates: ",len(true_templates)
    for true_template in true_templates:
        true_fmap = flatten_map(true_template)
        llh_val = get_binwise_llh(fmap_data,true_fmap)

        trial_data['llh'].append(llh_val)
        trial_data['deltam31'].append(true_template['params']['deltam31'])
        trial_data['theta23'].append(true_template['params']['theta23'])

        if llh_val > llh_max:
            llh_max = llh_val
            best_fit['deltam31'] = true_template['params']['deltam31']
            best_fit['theta23']  = true_template['params']['theta23']

    return llh_max,best_fit,trial_data

def find_max_llh_powell(fmap,settings,assume_nmh=True):
    '''
    Uses Powell's method implementation in scipy.optimize to fine the
    maximum likelihood for a given pseudo data fmap.
    '''

    fixed_params = get_fixed_params(params,normal_hierarchy)
    free_params = get_free_params(params,normal_hierarchy)

    init_vals = []
    const_args = []
    names = []
    scales = []
    priors = []
    for key in free_params.keys():
        init_vals.append(free_params[key]['value'])
        names.append(key)
        scales.append(free_params[key]['scale'])
        priors.append((free_params[key]['prior'],free_params[key]['value']))

    opt_steps_dict = {key:[] for key in names}
    opt_steps_dict['llh'] = []
    const_args = (names,scales,fmap,fixed_params,opt_steps_dict,priors)

    init_vals = np.array(init_vals)*np.array(scales)

    print "init_vals: ",init_vals
    print "priors: ",priors

    #
    # PUT THIS INTO SETTINGS FILE LATER!
    #
    print "opt_settings: "
    ftol = 1.0e-7
    print "  ftol: %.2e"%ftol
    vals = opt.fmin_powell(llh_bfgs,init_vals,args=const_args,ftol=ftol,full_output=1)

    print "\n        CONVERGENCE!\n"
    print "returns: \n",vals

    return vals

def find_max_llh_opt(fmap,temp_maker,params,llr_settings,save_opt_steps=False,
                     normal_hierarchy=True):
    '''
    Finds the template (and free systematic params) that maximize likelihood
    that the data came from the chosen template of true params.

    returns a dictionary of llh data and best fit params, in the format:

      {'llh': [...],
       'param1': [...],
       'param2': [...],
       ...}

    where 'param1', 'param2', ... are the free params varied by
    optimizer, and they hold a list of all the values tested in
    optimizer algorithm, unless save_opt_steps is False, in which case
    they are one element in length-the best fit params and best fit llh.
    '''

    # Get params dict which will be optimized (free_params) and which
    # won't be (fixed_params) but are still needed for get_template()
    fixed_params = get_fixed_params(params,normal_hierarchy)
    free_params = get_free_params(params,normal_hierarchy)

    init_vals = []
    bounds = []
    const_args = []
    names = []
    scales = []
    priors = []
    for key in free_params.keys():
        init_vals.append(free_params[key]['value'])
        bound = list(free_params[key]['range'])
        bounds.append(bound)
        names.append(key)
        scales.append(free_params[key]['scale'])
        priors.append((free_params[key]['prior'],free_params[key]['value']))

    opt_steps_dict = {key:[] for key in names}
    opt_steps_dict['llh'] = []

    const_args = [names,scales,fmap,fixed_params,temp_maker,opt_steps_dict,priors]

    init_vals = np.array(init_vals)*np.array(scales)

    print "bounds: ",bounds
    print "init_vals: ",init_vals
    print "priors: ",priors

    factr = float(llr_settings['factr'])
    epsilon = float(llr_settings['epsilon'])
    pgtol = float(llr_settings['pgtol'])
    m_corr = float(llr_settings['m'])
    maxfun = float(llr_settings['maxfun'])
    maxiter = float(llr_settings['maxiter'])

    print "opt_settings: "
    print "  factr: %.2e"%factr
    print "  epsilon: %.2e"%epsilon
    print "  pgtol: %.2e"%pgtol
    print "  m: %d"%m_corr
    print "  maxfun: %d"%maxfun
    print "  maxiter: %d"%maxiter

    vals = opt.fmin_l_bfgs_b(llh_bfgs,init_vals,args=const_args,approx_grad=True,
                             iprint=0,bounds=bounds,epsilon=epsilon,factr=factr,
                             pgtol=pgtol,m=m_corr,maxfun=maxfun,maxiter=maxiter)

    best_fit_vals = vals[0]
    llh = vals[1]
    dict_flags = vals[2]

    print "  llh: ",llh
    print "  best_fit_vals: ",best_fit_vals
    print "  dict_flags: ",dict_flags

    best_fit_params = {}
    for i in xrange(len(best_fit_vals)):
        best_fit_params[names[i]] = best_fit_vals[i]
    print "  best_fit_params: ",best_fit_params

    if not save_opt_steps:
        # Do not store the extra history of opt steps:
        for key in opt_steps_dict.keys():
            opt_steps_dict[key] = [opt_steps_dict[key][-1]]

    return opt_steps_dict


def llh_bfgs(opt_vals,*args):
    '''
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
    '''

    print "\n  opt_vals: ",opt_vals

    names = args[0]
    scales = args[1]
    fmap = args[2]
    fixed_params = args[3]
    temp_maker = args[4]
    opt_steps_dict = args[5]
    priors = args[6]

    template_params = {}
    for i in xrange(len(opt_vals)):
        template_params[names[i]] = opt_vals[i]/scales[i]

    template_params = dict(template_params.items() + get_values(fixed_params).items())

    # Now get true template, and compute LLH
    true_template = temp_maker.get_template(template_params)
    true_fmap = flatten_map(true_template)

    # NOTE: The minus sign is present on both of these next two lines
    # to reflect the fact that the optimizer finds a minimum rather
    # than maximum.
    llh = -get_binwise_llh(fmap,true_fmap)
    for i,prior in enumerate(priors):
        prior_val = get_prior_llh(opt_vals[i],prior[1],prior[0])
        llh -= prior_val

    # Opt steps dict, to see what values optimizer is testing:
    for key in names:
        opt_steps_dict[key].append(template_params[key])
    opt_steps_dict['llh'].append(llh)

    print "  llh: ",llh

    return llh

