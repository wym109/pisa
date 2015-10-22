#! /usr/bin/env python
#
# LLROptimizerAnalysis.py
#
# Runs the LLR optimizer-based LLR analysis
#
# author: Tim Arlen - tca3@psu.edu
#         Sebatian Boeser - sboeser@uni-mainz.de
#
# date:   02-July-2014
#
# revision: 07-May-2015
# Re-wrote how the alternative hierarchy (null hypothesis) is handled.
# Rather than assuming the alternative hypothesis best fit of the
# oscillation parameters is unchanged,  we allow the asimov data set to
# find the best fit null hypothesis to be changed.
#

import numpy as np
from copy import deepcopy
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from pisa.analysis.llr.LLHAnalysis import find_opt_scipy, find_alt_hierarchy_fit
from pisa.analysis.stats.LLHStatistics import get_random_map
from pisa.analysis.stats.Maps import get_pseudo_data_fmap, get_seed, get_asimov_fmap
from pisa.analysis.TemplateMaker import TemplateMaker
from pisa.utils.log import logging, tprofile, physics, set_verbosity
from pisa.utils.jsons import from_json,to_json
from pisa.utils.params import get_values, select_hierarchy, fix_all_params, fix_non_atm_params
from pisa.utils.utils import Timer

def check_scipy_version(minimizer_settings):
    #Workaround for old scipy versions
    import scipy
    if scipy.__version__ < '0.12.0':
        logging.warn('Detected scipy version %s < 0.12.0'%scipy.__version__)
        if 'maxiter' in minimizer_settings:
            logging.warn('Optimizer settings for \"maxiter\" will be ignored')
            minimizer_settings.pop('maxiter')
    return

def getAsimovData(template_maker, params, data_normal):
    """
    Generates the asimov data set (expected counts distribution) at
    parameters assuming hierarchy of data_normal

    \Params:
      * template_maker - instance of class TemplateMaker service.
      * params - parameters with values, fixed, range, etc. of systematics
      * data_normal - bool for Mass hierarchy being Normal (True)
        or inverted (False)
    """

    fiducial_param_vals = get_values(select_hierarchy(
        params, normal_hierarchy=data_normal))
    return get_asimov_fmap(
        template_maker=template_maker,
        fiducial_params=fiducial_param_vals,
        channel=fiducial_param_vals['channel'])


def getAltHierarchyBestFit(asimov_data, template_maker, params, minimizer_settings,
                           hypo_normal, check_octant):
    """
    Finds the best fit value of alternative hierarchy to that which
    was used to produce the asimov data set.

    \Params:
      * asimov_data - array of values of asimov data set (float)
      * template_maker - instance of class TemplateMaker service.
      * params - parameters with values, fixed, range, etc. of systematics
      * minimizer_settings - used with bfgs_b minimizer
      * hypo_normal - bool for Mass hierarchy being Normal (True)
        or inverted (False)
      * check_octant - bool to check the opposite octant for a solution
        to the minimization of the LLH.
    """

    llh_data = find_alt_hierarchy_fit(
        asimov_data, template_maker, params, hypo_normal,
        minimizer_settings, only_atm_params=True, check_octant=check_octant)

    alt_params = get_values(select_hierarchy(params, normal_hierarchy=hypo_normal))
    for key in llh_data.keys():
        if key == 'llh': continue
        alt_params[key] = llh_data[key][-1]

    return alt_params, llh_data


def get_llh_hypothesis(
        data_tag, asimov_data, ntrials, template_maker, template_params,
        minimizer_settings, save_steps, check_octant):
    """
    Runs the llh fitter ntrials number of times, pulling pseudo data sets from
    asimov_data.

    \Params:
      * data_tag - hierarchy type running for assumed true.
      * asimov_data - asimov (unfluctuated) data from which to generate poisson
        fluctuated pseudo data
      * ntrials - number of trials to run for each hierarchy hypothesis
      * template_maker - instance of TemplateMaker class, from which to fit pseudo
        data to
      * template_params - dictionary of parameters at which to test the pseudo
        data and find the best match llh
      * minimizer_settings - settings for bfgs minimizer in llh fit
      * save_steps - flag to save the optimizer steps
      * check_octant - boolean to check both octants of theta23

    \returns - trials list that holds the dictionaries of llh results.
    """

    trials = []
    for itrial in xrange(1,ntrials+1):
        results = {} # one trial of results

        tprofile.info("start trial %d"%itrial)
        logging.info(">"*10 + "Running trial: %05d"%itrial + "<"*10)

        results['seed'] = get_seed()
        logging.info("  RNG seed: %ld"%results['seed'])
        # Get random map generated from asimov data (or from data_tag).
        fmap = get_random_map(asimov_data, seed=results['seed'])

        for hypo_tag, hypo_normal in [('hypo_NMH',True),('hypo_IMH',False)]:

            physics.info(
                "Finding best fit for %s under %s assumption"%(data_tag,hypo_tag))
            with Timer() as t:
                llh_data = find_opt_scipy(
                    fmap, template_maker, template_params,
                    minimizer_settings, save_steps,
                    normal_hierarchy=hypo_normal, check_octant=check_octant)
            tprofile.info("==> elapsed time for optimizer: %s sec"%t.secs)

            # Store the LLH data
            results[hypo_tag] = llh_data

        trials += [results]
        tprofile.info("stop trial %d"%itrial)

    return trials

parser = ArgumentParser(
    description='''Runs the LLR optimizer-based analysis varying a number of systematic
    parameters defined in settings input .json file and saves the likelihood values for
    a combination of hierarchies. Standard output will provide 8 LLH values per trial-
    4 LLH values for hypothesized Normal, and 4 LLH values for hypothesized Inverted.
    If run in --no_alt_fit mode, 4 LLH values per trial are produced.''',
    formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-t','--template_settings',type=str,
                    metavar='JSONFILE', required = True,
                    help='''Settings related to the template generation & systematics''')
parser.add_argument('-m','--minimizer_settings',type=str,
                    metavar='JSONFILE', required = True,
                    help='''Settings related to the optimizer used in the LLR
                    analysis.''')
parser.add_argument('-n','--ntrials',type=int, default = 1,
                    help="Number of trials to run")
parser.add_argument('--gpu_id',type=int,default=None,
                    help="GPU ID if available.")
parser.add_argument('-s','--save-steps',action='store_true',default=False,
                    dest='save_steps',
                    help="Save all steps the optimizer takes.")
parser.add_argument('--no_alt_fit',action='store_true',default=False,
                    help='''Does not fit for the oscillation parameters that best match
                    the alternative hierarchy hypothesis, but instead uses the fiducial
                    values of the alternative hierarchy to produce the LLR distribution.
                    (Not recommended for values of theta23 close to maximal).''')
parser.add_argument('--single_octant',action='store_true',default=False,
                    help='''Checks opposite octant for a minimum llh solution.
                    Especially important for theta23 close to maximal.''')
parser.add_argument('-o','--outfile',type=str,default='llh_data.json',metavar='JSONFILE',
                    help="Output filename.")
parser.add_argument('-v', '--verbose', action='count', default=None,
                    help='set verbosity level')
args = parser.parse_args()

set_verbosity(args.verbose)

# Read in the settings
template_settings = from_json(args.template_settings)
minimizer_settings  = from_json(args.minimizer_settings)

# Change this throughout code later?
check_octant = not args.single_octant
check_scipy_version(minimizer_settings)

if args.gpu_id is not None:
    template_settings['params']['gpu_id'] = {}
    template_settings['params']['gpu_id']['value'] = args.gpu_id
    template_settings['params']['gpu_id']['fixed'] = True

template_maker = TemplateMaker(get_values(template_settings['params']),
                               **template_settings['binning'])

# Assemble output dict
output = {'template_settings' : template_settings,
          'minimizer_settings' : minimizer_settings}

for data_tag, data_normal in [('true_NMH',True),('true_IMH',False)]:
    tprofile.info("Assuming: %s"%data_tag)

    output[data_tag] = {"true_h_fiducial": {}}

    # Always do the true hierarchy fiducial LLR distribution:
    true_h_fiducial = {}

    # Get Asimov data set for assuming true: data_tag
    asimov_data = getAsimovData(
        template_maker, template_settings['params'], data_normal)

    trials = get_llh_hypothesis(
        data_tag, asimov_data, args.ntrials, template_maker,
        template_settings["params"], minimizer_settings,
        args.save_steps, check_octant)

    output[data_tag]["true_h_fiducial"] = trials

    # If we do not run the alt_fit, then we simply continue in the for
    # loop and the LLR distributions will be interpreted as the
    # ability to discriminate between hierarchies, without fitting for
    # the false hierarchy parameters that match best to the True
    # hierarchy fiducial model.

    if not args.no_alt_fit:
        logging.info("Running false hierarchy best fit...")
        output[data_tag]["false_h_best_fit"] = {}

        false_h_params = fix_non_atm_params(template_settings['params'])
        false_h_settings, llh_data = getAltHierarchyBestFit(
            asimov_data, template_maker, false_h_params, minimizer_settings,
            (not data_normal), check_octant)

        asimov_data_null = get_asimov_fmap(
            template_maker=template_maker,
            fiducial_params=false_h_settings,
            channel=false_h_settings['channel'])
        
        # Store all data tag related inputs:
        output[data_tag]['false_h_best_fit']['false_h_settings'] = false_h_settings
        output[data_tag]['false_h_best_fit']['llh_null'] = llh_data

        trials = get_llh_hypothesis(
            data_tag, asimov_data_null, args.ntrials, template_maker,
            template_settings["params"], minimizer_settings,
            args.save_steps, check_octant)

        output[data_tag]["false_h_best_fit"] = trials

to_json(output,args.outfile)
