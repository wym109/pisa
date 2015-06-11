#! /usr/bin/env python
#
# ScanAnalysisNutau_ratio.py
#
# Runs a brute-force scan LLR analysis
#
# authors: Feifei Huang - fxh140@psu.edu
#          Sebatian Boeser - sboeser@uni-mainz.de
#

import numpy as np
import copy
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from pisa.utils.log import logging, profile, physics, set_verbosity
from pisa.utils.jsons import from_json,to_json
from pisa.analysis.stats.Maps import get_pseudo_data_fmap, get_seed,get_asimov_data_fmap_up_down 
from pisa.analysis.scan.Scan import find_max_grid
from pisa.analysis.TemplateMaker import TemplateMaker
from pisa.utils.params import get_values, select_hierarchy_and_nutau_norm, change_nutau_norm_settings
import random as rnd

parser = ArgumentParser(
    description='''Runs a brute-force scan analysis varying a number of systematic parameters
    defined in settings.json file and saves the likelihood values for all
    combination of hierarchies.''',
    formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '-t', '--template_settings',
    type=str,
    metavar='JSONFILE',
    required = True,
    help='''Settings (f floating) related to the template generation and systematics.''')
parser.add_argument(
    '-pd', '--pseudo_data_settings',
    type=str,
    metavar='JSONFILE',
    required = False,
    help='''Settings for pseudo data templates(tau,f=0), if desired to be different from template_settings.''')
parser.add_argument(
    '-g', '--grid_settings',type=str,
    metavar='JSONFILE',
    required=True,
    help='''Settings for the grid search.''')
parser.add_argument(
    '-n', '--ntrials',
    type=int,
    default=1,
    help="Number of trials to run")
sselect = parser.add_mutually_exclusive_group(required=False)
sselect.add_argument(
    '--save-steps', action='store_true',
    default=True,
    dest='save_steps',
    help="Save all steps")
sselect.add_argument(
    '--no-save-steps',
    action='store_false',
    default=False,
    dest='save_steps',
   help="Save just the maximum")
parser.add_argument(
    '-o', '--outfile',
    type=str,
    default='llh_data.json',
    metavar='JSONFILE',
    help="Output filename.")
parser.add_argument(
    '-v', '--verbose',
    action='count',
    default=None,
   help='set verbosity level')
args = parser.parse_args()

set_verbosity(args.verbose)

# Read in the settings
template_settings = from_json(args.template_settings)
czbins = template_settings['binning']['czbins']

up_template_settings = copy.deepcopy(template_settings)
up_template_settings['binning']['czbins'] = czbins[czbins <= 0]

down_template_settings = copy.deepcopy(template_settings)
down_template_settings['binning']['czbins'] = czbins[czbins >= 0]
down_template_settings['params']['pid_paramfile'] = {u'fixed': True, u'value': 'pid/DC12/DC12_down_pid.json'}

if args.pseudo_data_settings is not None:
    pseudo_data_settings = from_json(args.pseudo_data_settings)
else:
    pseudo_data_settings = template_settings
grid_settings  = from_json(args.grid_settings)

# List to store results from all the trials
trials = []

template_maker_down = TemplateMaker(get_values(down_template_settings['params']),
                                    **down_template_settings['binning'])
template_maker_up = TemplateMaker(get_values(up_template_settings['params']),
                                  **up_template_settings['binning'])
template_maker = [template_maker_up,template_maker_down]

if args.pseudo_data_settings:
    pseudo_data_template_maker = TemplateMaker(get_values(pseudo_data_settings['params']),
                                               **pseudo_data_settings['binning'])
else:
    pseudo_data_template_maker = template_maker

# Make sure that both pseudo data and template are using the same channel.
# Raise Exception and quit otherwise
channel = template_settings['params']['channel']['value']
if channel != pseudo_data_settings['params']['channel']['value']:
    error_msg = "template_settings and pseudo_data_settings must have the same channel!\n"
    error_msg += " template_settings: '%s', pseudo_data_settings: '%s' " %(channel,pseudo_data_settings['params']['channel']['value'])
    raise ValueError(error_msg)

for itrial in xrange(1, args.ntrials+1):
    profile.info("start trial %d"%itrial)
    logging.info(">"*10 + "Running trial: %05d"%itrial + "<"*10)


    # //////////////////////////////////////////////////////////////////////
    # For each trial, generate two pseudo-data experiemnts (one for each
    # hierarchy), and for each find the best matching template in each of the
    # hierarchy hypothesis.
    # //////////////////////////////////////////////////////////////////////
    results = {}
    data_normal = True
    hypo_normal = True
    #for data_tag, data_nutau_norm in [('data_notau',0.0)]:
    for data_tag, data_nutau_norm in [('data_tau',1.0)]:
    #for data_tag, data_nutau_norm in [('data_tau',1.0),('data_notau',0.0)]:

        results[data_tag] = {}
        # 0) get a random seed and store with the data
        results[data_tag]['seed'] = get_seed()
        #results[data_tag]['seed'] = 100
        logging.info("  RNG seed: %ld"%results[data_tag]['seed'])
        # 1) get a pseudo data fmap from fiducial model (best fit vals of params).
        fiducial_param_values = get_values(
            select_hierarchy_and_nutau_norm(pseudo_data_settings['params'],
                                            normal_hierarchy=data_normal,
                                            nutau_norm_value=data_nutau_norm)
        )
        fmap = get_pseudo_data_fmap(template_maker=pseudo_data_template_maker,
                                    fiducial_params=fiducial_param_values,
                                    channel=channel,
                                    seed=results[data_tag]['seed'])
        print pseudo_data_settings['params']
        #fmap = get_asimov_data_fmap_up_down(pseudo_data_template_maker,
        #                get_values(select_hierarchy_and_nutau_norm(pseudo_data_settings['params'],
        #                           normal_hierarchy=data_normal,nutau_norm_value=data_nutau_norm)),
        #                            chan=channel)

        # 2) find max llh (and best fit free params) from matching pseudo data
        #    to templates.
        rnd.seed(get_seed())
        init_nutau_norm = rnd.uniform(-0.7,3)
        #for hypo_tag, hypo_nutau_norm, nutau_norm_fix in [('hypo_free',init_nutau_norm, False),('hypo_notau',0, True)]:
        for hypo_tag, hypo_nutau_norm, nutau_norm_fix in [('hypo_free',1.0, True)]:
            physics.info("Finding best fit for %s under %s assumption"%(data_tag,hypo_tag))
            profile.info("start scan")
            hypo_params = change_nutau_norm_settings(
                template_settings['params'],
                hypo_nutau_norm,nutau_norm_fix
            )
            llh_data = find_max_grid(fmap=fmap,
                                     template_maker=template_maker,
                                     params=hypo_params,
                                     grid_settings=grid_settings,
                                     save_steps=args.save_steps,
                                    normal_hierarchy=hypo_normal)
            profile.info("stop scan")

            # Store the LLH data
            results[data_tag][hypo_tag] = llh_data


    # Store this trial
    trials += [results]
    profile.info("stop trial %d"%itrial)

# Assemble output dict
output = {'trials' : trials,
          'template_settings_up' : up_template_settings,
          'template_settings_down' : down_template_settings,
          'grid_settings' : grid_settings}
output['pseudo_data_settings'] = pseudo_data_settings
# And write to file
to_json(output, args.outfile)
