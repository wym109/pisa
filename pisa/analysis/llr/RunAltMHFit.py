#! /usr/bin/env python
#
# RunAltMHFit.py
#
# Runs the Optimizer analysis for the Asimov data set for the hypothesized MH
# template and finds the values of the systematics that maximize the LLH of
# the alternative Hierarchy hypothesis, as a function of theta23.
#
# author: Tim Arlen - tca3@psu.edu
#
# date:   05-May-2015
#

import numpy as np
from itertools import product
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from pisa.analysis.llr.LLHAnalysis import find_max_llh_bfgs, find_alt_hierarchy_fit
from pisa.analysis.stats.Maps import get_asimov_fmap
from pisa.analysis.scan.Scan import calc_steps
from pisa.analysis.TemplateMaker import TemplateMaker
from pisa.utils.log import logging, profile, physics, set_verbosity
from pisa.utils.jsons import from_json,to_json
from pisa.utils.params import get_values, select_hierarchy, get_atm_params, get_free_params, get_fixed_params
from pisa.utils.utils import Timer


def fixAllButTheta23(params):
    new_params = {}
    free_param = 'theta23'
    for key, value in params.items():
        new_params[key] = value.copy()
        new_params[key]['fixed'] = False if free_param in key else True

    return new_params

def createStepList(params,true_normal,grid_settings):
    """
    No matter how many params are listed as free, only force theta23 to be free.
    """

    new_params = fixAllButTheta23(params)

    free_params = select_hierarchy(get_free_params(new_params),
                                   normal_hierarchy=true_normal)
    fixed_params = select_hierarchy(get_fixed_params(new_params),
                                    normal_hierarchy=true_normal)
    calc_steps(free_params, grid_settings['steps'])

    # Form list from all parameters that holds a list of (name,step) tuples.

    #steplist = [(name,step) for step in param['steps'] for name, param in free_params.items()]
    steplist = free_params['theta23']['steps']

    #steplist = [(name, step) for step in param['steps']
    #            for name, param in sorted(free_params.items()) ]

    return steplist

def getAsimovParams(params,true_normal,th23_val):
    asimov_params = select_hierarchy(params,normal_hierarchy=true_normal)
    asimov_params['theta23']['value'] = th23_val

    return asimov_params


parser = ArgumentParser(
    description='''Runs the Optimizer analysis for the Asimov data set for the hypothesized MH
    template and finds the values of the systematics that maximize the LLH of
    the alternative Hierarchy hypothesis.''',
    formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-t','--template_settings',type=str,
                    metavar='JSONFILE', required = True,
                    help='''Settings related to the template generation and systematics.''')
parser.add_argument('-m','--minimizer_settings',type=str,
                    metavar='JSONFILE', required = True,
                    help='''Settings related to the optimizer used in the LLR analysis.''')
parser.add_argument('-g','--grid_settings',type=str,metavar='JSONFILE', required = True,
                    help='''Get llh value at defined oscillation parameter grid values,
                    according to these input settigs to.''')
parser.add_argument('--gpu_id',type=int,default=None,
                    help="GPU ID if available.")
parser.add_argument('--check_octant',action='store_true', default=False,
                    help='''Checks alternative octant llh to see if optimum is there.''')
parser.add_argument('-o','--outfile',type=str,default='alt_hypo_study.json',metavar='JSONFILE',
                    help="Output filename.")
parser.add_argument('-v', '--verbose', action='count', default=None,
                    help='set verbosity level')
args = parser.parse_args()

set_verbosity(args.verbose)

#Read in the settings
template_settings = from_json(args.template_settings)
minimizer_settings  = from_json(args.minimizer_settings)
grid_settings = from_json(args.grid_settings)

if args.gpu_id is not None:
    template_settings['params']['gpu_id'] = {}
    template_settings['params']['gpu_id']['value'] = args.gpu_id
    template_settings['params']['gpu_id']['fixed'] = True

with Timer() as t:
    template_maker = TemplateMaker(get_values(template_settings['params']),
                                   **template_settings['binning'])
profile.info("==> elapsed time to initialize templates: %s sec"%t.secs)


#Get the parameters
params = template_settings['params']

mctrue_types = [('true_NMH',True),('true_IMH',False)]

results = {}
# Store for future checking:
results['template_settings'] = template_settings
results['minimizer_settings'] = minimizer_settings
results['grid_settings'] = grid_settings

for true_tag, true_normal in mctrue_types:
    results[true_tag] = {}

    result = {}

    steplist = createStepList(params,true_normal,grid_settings)
    print steplist

    free_params = select_hierarchy(get_free_params(params),
                                   normal_hierarchy=true_normal)

    # Set up the arrays to store the true/fit values in:
    for key in free_params.keys():
        result['true_'+key] = []
        result['fit_'+key] = []
    result['asimov_data'] = []

    # This will actually only iterate over theta23 (for now):
    for step in steplist:
        print "step: ",step

        logging.info("Running at asimov parameters: %s"%step)
        asimov_params = get_values(getAsimovParams(params,true_normal,step))
        print "asimov_params: ",asimov_params
        asimov_data_set = get_asimov_fmap(
            template_maker, asimov_params,
            chan=asimov_params['channel'])

        # Store injected true values in result:
        result['true_theta23'].append(step)
        result['true_deltam31'].append(asimov_params['deltam31'])

        result['asimov_data'].append(asimov_data_set)

        # now get fitted values of opposite hierarchy:
        hypo_normal = False if true_normal else True
        hypo_tag = 'hypo_IMH' if true_normal else 'hypo_NMH'
        llh_data = find_alt_hierarchy_fit(
            asimov_data_set,template_maker, params, hypo_normal,
            minimizer_settings, only_atm_params=False, check_octant=args.check_octant)

        for key in free_params.keys(): result['fit_'+key].append(llh_data[key][-1])

    results[true_tag] = result

logging.warn("FINISHED. Saving to file: %s"%args.outfile)
to_json(results,args.outfile)
