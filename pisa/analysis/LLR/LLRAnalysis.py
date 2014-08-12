#
# LLRAnalysis.py
#
# Helper module for running the Optimizer-based LLR analysis.
#
# author: Tim Arlen
#         tca3@psu.edu
#
# date:   02-July-2014
#

import logging
from datetime import datetime
import numpy as np
import scipy.optimize as opt

from pisa.analysis.template import get_template
from LLHStatistics import get_random_map, get_binwise_llh

# FUNCTIONS
def get_template_params(settings,use_best=True,use_nmh=True):
    '''
    From a settings file, merge all parameters needed by the template
    generator into one 'params' dict.
    
    NOTE: For NOW, use_best is not used, but when we get more
    sophistocated (put a prior on osc params, e.g.), we will need it.
    '''

    params = settings['template'].copy()
    #params['ebins'] = settings['ebins']
    #params['czbins'] = settings['czbins']
    
    for osc_param in settings['bounds']['osc'].keys():
        params[osc_param] = settings['bounds']['osc'][osc_param]['best']
    for det_param in settings['bounds']['detector'].keys():
        params[det_param] = settings['bounds']['detector'][det_param]['best']
        
    if use_nmh: params['deltam31'] = params['deltam31_nh']
    else: params['deltam31'] = params['deltam31_ih']
    
    params['deltam31'] = params['deltam31_nh'] if use_nmh else params['deltam31_ih']
    params.pop('deltam31_nh')
    params.pop('deltam31_ih')
    
    return params


def get_pseudo_data_fmap(**params):
    '''
    Creates a true template from params, then uses Poisson statistics
    to vary the expected counts per bin to create a pseudo data set.

    IMPORTANT: returns a SINGLE flattened map of trck/cscd combined
    '''
    
    true_template = get_template(params)

    # Must make sure the templates are non-zero in case we use the
    # full sky, or in case it is binned too finely for the MC to fill
    # an individual bin, but we still want to analyze this way.
    true_cscd = true_template['cscd']['map'].flatten()
    true_trck = true_template['trck']['map'].flatten()
    true_fmap = np.append(true_cscd,true_trck)
    true_fmap = np.array(true_fmap)[np.nonzero(true_fmap)]
    
    fmap = get_random_map(true_fmap)
    
    return fmap

def get_correct_deltam31(params,assume_nmh):
    '''
    Since we must define deltam31 differently for each hierarchy, need
    to correct for the name in the param dict.
    '''
    params['deltam31'] = params['deltam31_nh'] if assume_nmh else params['deltam31_ih']
    params.pop('deltam31_nh')
    params.pop('deltam31_ih')
    
    return params
    
def get_param_types(settings,assume_nmh):
    '''
    From the settings dict, define the parameters which will be
    optimized over and which will be constant.
    '''
    
    syst_params = {}
    const_params = settings['template']
    #const_params['ebins'] = settings['ebins']
    #const_params['czbins'] = settings['czbins']

    # get both 'osc' and 'detector' types
    for pkey in ['osc','detector']:
        for key in settings['bounds'][pkey].keys():
            if settings['bounds'][pkey][key]['vary']:
                syst_params[key] = settings['bounds'][pkey][key]
            else:
                const_params[key] = settings['bounds'][pkey][key]['best']

    # Now make sure you have the right deltam31:
    if 'deltam31_nh' in syst_params.keys(): 
        syst_params = get_correct_deltam31(syst_params,assume_nmh)
    else:
        const_params = get_correct_deltam31(const_params,assume_nmh)

    logging.warn("Systematic params: %s"%syst_params.keys())
        
    return syst_params,const_params
    
def find_max_llh_opt(fmap,settings,assume_nmh=True):
    '''
    Finds the template (and syst params) that maximize likelihood that
    the data came from the true given template.

    opt_steps_dict - If defined, it will create and store the output of the
    steps taken by the optimizer (whethere it's in calculating
    gradient or function value).
    
    returns 
      LLH - value of the max Log(likelihood) found
      best_fit_params - dict of the best fit values of the 
        systematic params marginalized over.
    '''
    
    # Get params dict which will be optimized (syst_params) and which
    # won't be (const_params) but are still needed for get_template()
    syst_params,const_params = get_param_types(settings,assume_nmh)
    
    init_vals = []
    bounds = []
    const_args = []
    names = []
    scales = []
    for key in syst_params.keys():
        #init_vals.append(syst_params[key]['best'])
        init_vals.append( (syst_params[key]['min']+syst_params[key]['max'])/2.0 )
        bound = (syst_params[key]['min']*syst_params[key]['scale'],
                 syst_params[key]['max']*syst_params[key]['scale'])
        bounds.append(bound)
        names.append(key)
        scales.append(syst_params[key]['scale'])

    opt_steps_dict = {key:[] for key in names}
    opt_steps_dict['llh'] = []
    const_args = [names,scales,fmap,const_params,opt_steps_dict]

    init_vals = np.array(init_vals)*np.array(scales)

    print "bounds: ",bounds
    print "init_vals: ",init_vals

    factr = float(settings['opt_settings']['factr'])
    epsilon = float(settings['opt_settings']['epsilon'])
    pgtol = float(settings['opt_settings']['pgtol'])
    m_corr = float(settings['opt_settings']['m'])
    maxfun = float(settings['opt_settings']['maxfun'])
    maxiter = float(settings['opt_settings']['maxiter'])

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

    return llh,best_fit_params,opt_steps_dict


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
    --args: [names,scales,fmap,other_params,fh]
      where 
        names: are the dict keys corresponding to param1, param2,...
        scales: the scales to be applied before passing to get_template 
          [IMPORTANT! In the optimizer, all parameters must be ~ the same order.
          Here, we keep them between 0.1,1 so the "epsilon" step size will vary
          the parameters in roughly the same precision.]
        fmap: pseudo data flattened map
        other_params: dictionary of other paramters needed by the get_template()
          function
        fh: filehandle of the optimizer output file to write steps taken in search.
    '''

    print "  opt_vals: ",opt_vals

    names = args[0]
    scales = args[1]
    fmap = args[2]
    const_params = args[3]
    opt_steps_dict = args[4]

    template_params = {}
    for i in xrange(len(opt_vals)):
        template_params[names[i]] = opt_vals[i]/scales[i]
    
    template_params = dict(template_params.items() + const_params.items())

    # Now get true template, and compute LLH
    true_template = get_template(template_params)
    true_cscd = true_template['cscd']['map'].flatten()
    true_trck = true_template['trck']['map'].flatten()
    true_fmap = np.append(true_cscd,true_trck)
    true_fmap = np.array(true_fmap)[np.nonzero(true_fmap)]

    llh = -get_binwise_llh(fmap,true_fmap)
        
    # Opt steps dict, to see what values optimizer is testing:
    for key in names:
        opt_steps_dict[key].append(template_params[key])
    opt_steps_dict['llh'].append(llh)
    print "  llh: ",llh
    
    return llh

