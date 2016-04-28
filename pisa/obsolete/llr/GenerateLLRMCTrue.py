#! /usr/bin/env python
#
# GenerateLLRMCTrue.py
#
# Generates the LLR distributions at the MC True values of all
# parameters, so that we can find the mean for the MC True case and
# use it to determine the average Test Statistic of LLR for the MC
# True parameters. (This can be compared to the Asimov mean to see
# which is more accurate).
#
# author: Tim Arlen - tca3@psu.edu
#
# date:   19-May-2015
#
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

    fiducial_params = get_values(select_hierarchy(
        params, normal_hierarchy=data_normal))
    return get_asimov_fmap(template_maker, fiducial_params,
                           channel=fiducial_params['channel'])


parser = ArgumentParser(
    description='''Runs the LLR optimizer-based analysis varying a number of systematic
    parameters defined in settings.json file and saves the likelihood values for all
    combination of hierarchies.''',
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
                    help='''Fix all parameters in the alternative MH fit, so just uses
                    the Fiducial
                    for opposite hierarchy''')
parser.add_argument('--single_octant',action='store_true',default=False,
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


for data_tag, data_normal in [('data_NMH',True),('data_IMH',False)]:
    tprofile.info("Assuming: %s"%data_tag)

    output[data_tag] = {}

    # Get Asimov data set for assuming true: data_tag, and store for
    # later comparison
    asimov_data = getAsimovData(
        template_maker, template_settings['params'], data_normal)
    output[data_tag]['asimov_data'] = asimov_data

    trials = []
    for itrial in xrange(1,args.ntrials+1):
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
                    fmap, template_maker, template_settings['params'],
                    minimizer_settings, args.save_steps,
                    normal_hierarchy=hypo_normal, check_octant=check_octant)
            tprofile.info("==> elapsed time for optimizer: %s sec"%t.secs)

            # Store the LLH data
            results[hypo_tag] = llh_data

        trials += [results]
        tprofile.info("stop trial %d"%itrial)

    output[data_tag]['trials'] = trials

to_json(output,args.outfile)
