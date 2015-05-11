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

from pisa.analysis.llr.LLHAnalysis import find_max_llh_bfgs, find_alt_hierarchy_fit
from pisa.analysis.stats.LLHStatistics import get_random_map
from pisa.analysis.stats.Maps import get_pseudo_data_fmap, get_seed, get_asimov_fmap
from pisa.analysis.TemplateMaker import TemplateMaker
from pisa.utils.log import logging, tprofile, physics, set_verbosity
from pisa.utils.jsons import from_json,to_json
from pisa.utils.params import get_values, select_hierarchy, fix_all_params, fix_non_atm_params
from pisa.utils.utils import Timer

def getAsimovData(template_maker, params, data_normal):
    """
    Gets the asimov data set (expected counts distribution) at params assuming
    hierarchy given by data_normal
    """

    fiducial_params = get_values(select_hierarchy(
        params,normal_hierarchy=data_normal))
    return get_asimov_fmap(template_maker,fiducial_params,
                           chan=fiducial_params['channel'])


def getAltHierarchyBestFit(asimov_data, template_maker, params, minimizer_settings,
                           hypo_normal,check_octant):

    llh_data = find_alt_hierarchy_fit(
        asimov_data,template_maker, params, hypo_normal,
        minimizer_settings, only_atm_params=True, check_octant=check_octant)

    alt_params = get_values(select_hierarchy(params,normal_hierarchy=hypo_normal))
    for key in llh_data.keys():
        if key == 'llh': continue
        alt_params[key] = llh_data[key][-1]

    return alt_params


parser = ArgumentParser(
    description='''Runs the LLR optimizer-based analysis varying a number of systematic
    parameters defined in settings.json file and saves the likelihood values for all
    combination of hierarchies.''',
    formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-t','--template_settings',type=str,
                    metavar='JSONFILE', required = True,
                    help='''Settings related to the template generation and systematics.''')
parser.add_argument('-m','--minimizer_settings',type=str,
                    metavar='JSONFILE', required = True,
                    help='''Settings related to the optimizer used in the LLR analysis.''')
#parser.add_argument('-pd','--pseudo_data_settings',type=str,
#                    metavar='JSONFILE',default=None,
#                    help='''Settings for pseudo data templates, if desired to be different from
#                    template_settings.''')
parser.add_argument('-n','--ntrials',type=int, default = 1,
                    help="Number of trials to run")
parser.add_argument('--gpu_id',type=int,default=None,
                    help="GPU ID if available.")
parser.add_argument('-s','--save-steps',action='store_true',default=False,
                    dest='save_steps',
                    help="Save all steps the optimizer takes.")
parser.add_argument('--no_alt_fit',action='store_true',default=False,
                    help='''Fix all parameters in the alternative MH fit, so just uses the Fiducial
                    for opposite hierarchy''')
parser.add_argument('--check_octant',action='store_true',default=False,
                    help='''Checks opposite octant for a minimum llh solution''')
parser.add_argument('-o','--outfile',type=str,default='llh_data.json',metavar='JSONFILE',
                    help="Output filename.")
parser.add_argument('-v', '--verbose', action='count', default=None,
                    help='set verbosity level')
args = parser.parse_args()

set_verbosity(args.verbose)

#Read in the settings
template_settings = from_json(args.template_settings)
minimizer_settings  = from_json(args.minimizer_settings)

#Workaround for old scipy versions
import scipy
if scipy.__version__ < '0.12.0':
    logging.warn('Detected scipy version %s < 0.12.0'%scipy.__version__)
    if 'maxiter' in minimizer_settings:
      logging.warn('Optimizer settings for \"maxiter\" will be ignored')
      minimizer_settings.pop('maxiter')

if args.gpu_id is not None:
    template_settings['params']['gpu_id'] = {}
    template_settings['params']['gpu_id']['value'] = args.gpu_id
    template_settings['params']['gpu_id']['fixed'] = True

template_maker = TemplateMaker(get_values(template_settings['params']),
                               **template_settings['binning'])

# Assemble output dict
output = {'template_settings' : template_settings,
          'minimizer_settings' : minimizer_settings}


asimov_data = {}
asimov_data_null = {}
alt_mh_settings = {}
for data_tag, data_normal in [('true_NMH',True),('true_IMH',False)]:
    tprofile.info("Assuming: %s"%data_tag)

    output[data_tag] = {}

    # Get Asimov data set for assuming true: data_tag
    asimov_data = getAsimovData(
        template_maker, template_settings['params'], data_normal)
    # Find best fit atm parameters (theta23, deltam31) for alternative hierarchy hypothesis
    #print "template_settings params: ",get_values(template_settings['params'])
    print "Template settings for asimov: "
    print "  theta23: ",get_values(select_hierarchy(
        template_settings['params'],normal_hierarchy=data_normal))['theta23']
    print "  deltam31: ",get_values(select_hierarchy(
        template_settings['params'],normal_hierarchy=data_normal))['deltam31']
    if args.no_alt_fit:
        # Fix all so no fit - should recover close to what we had with joint llr distributions.
        alt_params = fix_all_params(template_settings['params'])
    else:
        # Fit only for atm params - so that we find best fit
        # atmospheric oscillation parameters
        alt_params = fix_non_atm_params(template_settings['params'])
    
    alt_mh_settings = getAltHierarchyBestFit(
        asimov_data, template_maker, alt_params, minimizer_settings,
        (not data_normal), args.check_octant)
    # Get asimov data set at null hypothesis
    #print "alt_mh_settings: ",alt_mh_settings.items()
    #raw_input("PAUSED...")
    print "Alt settings for asimov: "
    print "  theta23: ",alt_mh_settings['theta23']
    print "  deltam31: ",alt_mh_settings['deltam31']
    raw_input("PAUSED")
    asimov_data_null = get_asimov_fmap(template_maker, alt_mh_settings,
                                       chan=alt_mh_settings['channel'])

    # Store all data tag related inputs:
    output[data_tag]['asimov_data'] = asimov_data
    output[data_tag]['asimov_data_null'] = asimov_data_null
    output[data_tag]['alt_mh_settings'] = alt_mh_settings

    trials = []
    for itrial in xrange(1,args.ntrials+1):
        results = {} # one trial of results

        tprofile.info("start trial %d"%itrial)
        logging.info(">"*10 + "Running trial: %05d"%itrial + "<"*10)

        results['seed'] = get_seed()
        logging.info("  RNG seed: %ld"%results['seed'])

        # 1) get random pseudo data map from asimov alternative hypothesis:
        fmap = get_random_map(asimov_data_null, seed=results['seed'])

        for hypo_tag, hypo_normal in [('hypo_NMH',True),('hypo_IMH',False)]:

            physics.info("Finding best fit for %s under %s assumption"%(data_tag,hypo_tag))
            with Timer() as t:
                llh_data = find_max_llh_bfgs(
                    fmap,template_maker, template_settings['params'], minimizer_settings,
                    args.save_steps, normal_hierarchy=hypo_normal,
                    check_octant=args.check_octant)
            tprofile.info("==> elapsed time for optimizer: %s sec"%t.secs)

            # Store the LLH data
            results[hypo_tag] = llh_data

        trials += [results]
        tprofile.info("stop trial %d"%itrial)

    output[data_tag]['trials'] = trials

to_json(output,args.outfile)
