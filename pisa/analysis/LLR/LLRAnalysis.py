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

import logging,sys
from datetime import datetime
import numpy as np
import scipy.optimize as opt

from LLHStatistics import get_random_map, get_binwise_llh, add_prior

# FUNCTIONS
def get_template_params(settings,use_best=True,use_nmh=True):
    '''
    From a settings file, merge all parameters from  needed by the template
    generator into one 'params' dict.

    For fiducial parameters, we are getting the 'best' field for the value.
    '''

    params = settings['template'].copy()
    params['ebins'] = settings['binning']['ebins']
    params['czbins'] = settings['binning']['czbins']
    
    for param in settings['fiducial'].keys():
        params[param] = settings['fiducial'][param]['best']
            
    params['deltam31'] = params['deltam31_nh'] if use_nmh else params['deltam31_ih']
    params.pop('deltam31_nh')
    params.pop('deltam31_ih')
    
    return params

def get_fiducial_params(settings,use_best=True,use_nmh=True):
    '''
    Get parameters of fiducial model (systematic + nuisance params)
    '''
    

    params = {}
    for param in settings['fiducial'].keys():
        params[param] = settings['fiducial'][param]['best']
        
    params['deltam31'] = params['deltam31_nh'] if use_nmh else params['deltam31_ih']
    params.pop('deltam31_nh')
    params.pop('deltam31_ih')
    
    return params

def get_pseudo_data_fmap(temp_maker,fiducial_params):
    '''
    Creates a true template from fiducial_params, then uses Poisson statistics
    to vary the expected counts per bin to create a pseudo data set.

    IMPORTANT: returns a SINGLE flattened map of trck/cscd combined
    '''
    
    true_template = temp_maker.get_template(fiducial_params)
    true_fmap = get_true_fmap(true_template)
    fmap = get_random_map(true_fmap)
    
    return fmap

def get_true_fmap(true_template):
    '''
    Takes a final level true template of trck/cscd, and returns a
    single flattened map of trck appended to cscd, with all zero bins
    removed.
    '''
    true_cscd = true_template['cscd']['map'].flatten()
    true_trck = true_template['trck']['map'].flatten()
    true_fmap = np.append(true_cscd,true_trck)
    true_fmap = np.array(true_fmap)[np.nonzero(true_fmap)]
    
    return true_fmap

def correct_mh_params(key,params,use_nmh):
    '''
    Corrects for a key in params being defined for both 'nh' and 'ih',
    and modifies the params dict accordingly.
    '''

    params[key] = params[key+'_nh'] if use_nmh else params[key+'_ih']
    params.pop(key+'_nh')
    params.pop(key+'_ih')
    return 

def get_param_types(fid_settings,assume_nmh):
    '''
    From the settings dict, define the parameters which will be
    optimized over and which will be constant.
    '''

    syst_params = {}
    const_params = {}    
    for key in fid_settings.keys():
        if fid_settings[key]['fixed']:
            const_params[key] = fid_settings[key]['best']
        else:
            syst_params[key] = fid_settings[key]

    if 'deltam31_nh' in const_params.keys():
        correct_mh_params('deltam31',const_params,assume_nmh)
    else:
        correct_mh_params('deltam31',syst_params,assume_nmh)
    
    logging.warn("Systematic params: %s"%syst_params.keys())
        
    return syst_params,const_params


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
        true_fmap = get_true_fmap(true_template)
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
    
    # Get params dict which will be optimized (syst_params) and which
    # won't be (const_params) but are still needed for get_template()
    syst_params,const_params = get_param_types(settings,assume_nmh)

    init_vals = []
    const_args = []
    names = []
    scales = []
    priors = []
    for key in syst_params.keys(): 
        init_vals.append( (syst_params[key]['min']+syst_params[key]['max'])/2.0 )
        names.append(key)
        scales.append(syst_params[key]['scale'])
        priors.append((syst_params[key]['prior'],syst_params[key]['best']))
        
    opt_steps_dict = {key:[] for key in names}
    opt_steps_dict['llh'] = []
    const_args = (names,scales,fmap,const_params,opt_steps_dict,priors)
    
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
    
def find_max_llh_opt(fmap,temp_maker,fiducial_params,llr_settings,save_opt_steps=False,
                     assume_nmh=True):
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
    syst_params,const_params = get_param_types(fiducial_params,assume_nmh) 
   
    init_vals = []
    bounds = []
    const_args = []
    names = []
    scales = []
    priors = []
    for key in syst_params.keys():
        init_vals.append(syst_params[key]['best'])
        #init_vals.append( (syst_params[key]['min']+syst_params[key]['max'])/2.0 )
        bound = (syst_params[key]['min']*syst_params[key]['scale'],
                 syst_params[key]['max']*syst_params[key]['scale'])
        bounds.append(bound)
        names.append(key)
        scales.append(syst_params[key]['scale'])
        priors.append((syst_params[key]['prior'],syst_params[key]['best']))
    
    if save_opt_steps:
        opt_steps_dict = {key:[] for key in names}
        opt_steps_dict['llh'] = []
    else:
        opt_steps_dict = None
    const_args = [names,scales,fmap,const_params,temp_maker,opt_steps_dict,priors]

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
    --args: [names,scales,fmap,const_params,opt_steps_dict]
      where 
        names: are the dict keys corresponding to param1, param2,...
        scales: the scales to be applied before passing to get_template 
          [IMPORTANT! In the optimizer, all parameters must be ~ the same order.
          Here, we keep them between 0.1,1 so the "epsilon" step size will vary
          the parameters in roughly the same precision.]
        fmap: pseudo data flattened map
        const_params: dictionary of other paramters needed by the get_template()
          function
        opt_steps_dict: dictionary recording information regarding the steps taken
          for each trial of the optimization process.
        priors: gaussian priors corresponding to opt_vals list. 
          Format: [(prior1,best1),(prior2,best2),...,(priorN,bestN)]
    '''

    print "\n  opt_vals: ",opt_vals

    names = args[0]
    scales = args[1]
    fmap = args[2]
    const_params = args[3]
    temp_maker = args[4]
    opt_steps_dict = args[5]
    priors = args[6]

    template_params = {}
    for i in xrange(len(opt_vals)):
        template_params[names[i]] = opt_vals[i]/scales[i]
    
    template_params = dict(template_params.items() + const_params.items())
    
    # Now get true template, and compute LLH
    true_template = temp_maker.get_template(template_params)
    true_cscd = true_template['cscd']['map'].flatten()
    true_trck = true_template['trck']['map'].flatten()
    true_fmap = np.append(true_cscd,true_trck)
    true_fmap = np.array(true_fmap)[np.nonzero(true_fmap)]

    # NOTE: The minus sign is present on both of these next two lines
    # to reflect the fact that the optimizer finds a minimum rather
    # than maximum.
    llh = -get_binwise_llh(fmap,true_fmap)
    for i,prior in enumerate(priors): 
        prior_val = add_prior(opt_vals[i],priors[i])
        llh -= prior_val
        
    # Opt steps dict, to see what values optimizer is testing:
    if opt_steps_dict is not None:
        for key in names:
            opt_steps_dict[key].append(template_params[key])
        opt_steps_dict['llh'].append(llh)
    
    print "  llh: ",llh
    
    return llh

