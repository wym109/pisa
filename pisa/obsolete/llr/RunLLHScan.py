#! /usr/bin/env python
#
# RunLLHScan.py
#
# Runs the LLH Scan analysis, which scans over the space of
# atmospheric oscillation parameters, minimizing the LLH value over
# the nuisance parameters by using the bfgs minimizer algorithm.
#
# author: Tim Arlen - tca3@psu.edu
#
# date:   16-December-2014
#

import sys
import numpy as np
from itertools import product
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from pisa.analysis.llr.LLHAnalysis import find_opt_scipy
from pisa.analysis.stats.Maps import get_asimov_fmap
from pisa.analysis.scan.Scan import calc_steps
from pisa.analysis.TemplateMaker import TemplateMaker
from pisa.utils.log import logging, tprofile, physics, set_verbosity
from pisa.utils.jsons import from_json,to_json
from pisa.utils.params import get_values, select_hierarchy, fix_atm_params, get_atm_params
from pisa.utils.utils import Timer

parser = ArgumentParser(
    description='''Runs the LLR optimizer-based analysis varying a number of systematic
    parameters defined in settings.json file and saves the likelihood values for all
    combination of hierarchies. Does not compute any pseudo data sets, but rather takes
    the Asimov data set (or expected counts template) at given value of atm params''',
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
parser.add_argument('--data_nmh',action='store_true',default=False,
                    help='''If true, takes data sets from NMH in the fit, otherwise takes data from IMH.''')
parser.add_argument('--hypo_nmh',action='store_true',default=False,
                    help='''If true: fitting data to assumed NMH, otherwise assumes IMH''')
parser.add_argument('-s','--save-steps',action='store_true',default=False,
                    dest='save_steps',
                    help='''Save all steps the optimizer takes.''')
parser.add_argument('--gpu_id',type=int,default=None,
                    help="GPU ID if available.")
parser.add_argument('-o','--outfile',type=str,default='llh_data.json',metavar='JSONFILE',
                    help='''Output filename.''')
parser.add_argument('-v', '--verbose', action='count', default=None,
                    help='''set verbosity level''')
args = parser.parse_args()

set_verbosity(args.verbose)

#Read in the settings
template_settings = from_json(args.template_settings)
minimizer_settings  = from_json(args.minimizer_settings)
grid_settings = from_json(args.grid_settings)

channel = template_settings['params']['channel']['value']
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

#Get the parameters
params = template_settings['params']

# Make sure that atmospheric parameters are fixed:
logging.warn("Ensuring that atmospheric parameters are fixed for this analysis")
params = fix_atm_params(params)

with Timer() as t:
    template_maker = TemplateMaker(get_values(params),**template_settings['binning'])
tprofile.info("==> elapsed time to initialize templates: %s sec"%t.secs)

results = {}
# Store for future checking:
results['template_settings'] = template_settings
results['minimizer_settings'] = minimizer_settings
results['grid_settings'] = grid_settings


# Set up data/hypo nmh or imh
if args.data_nmh:
    data_tag = 'data_NMH'
    data_normal = True
else:
    data_tag = 'data_IMH'
    data_normal = False
if args.hypo_nmh:
    hypo_tag = 'hypo_NMH'
    hypo_normal = True
else:
    hypo_tag = 'hypo_IMH'
    hypo_normal = False

logging.info("Running with data: %s and hypo: %s"%(data_tag, hypo_tag))
results[data_tag] = {}

data_params = select_hierarchy(params,normal_hierarchy=data_normal)
asimov_data_set = get_asimov_fmap(template_maker,get_values(data_params),
                                  channel=channel)
results[data_tag]['asimov_data'] = asimov_data_set


hypo_params = select_hierarchy(params,normal_hierarchy=hypo_normal)
# Now scan over theta23,deltam31 values and fix params to
# these values:
# Calculate steps for all free parameters
atm_params = get_atm_params(hypo_params)
calc_steps(atm_params, grid_settings['steps'])

# Build a list from all parameters that holds a list of (name, step) tuples
steplist = [ [(name,step) for step in param['steps']]
             for name, param in sorted(atm_params.items())]

print "steplist: ",steplist
print "atm_params: ",atm_params

# Prepare to store all the steps
steps = {key:[] for key in atm_params.keys()}
steps['llh'] = []

# Iterate over the cartesian product, and set fixed parameter to value
for pos in product(*steplist):
    pos_dict = dict(list(pos))
    print "Running at params-pos dict: ",pos_dict
    for k,v in pos_dict.items():
        hypo_params[k]['value'] = v
        steps[k].append(v)

    with Timer() as t:
        llh_data = find_opt_scipy(asimov_data_set,template_maker,hypo_params,
                                  minimizer_settings,args.save_steps,
                                  normal_hierarchy=hypo_normal)
    tprofile.info("==> elapsed time for optimizer: %s sec"%t.secs)

    steps['llh'].append(llh_data['llh'][-1])

    #Store the LLH data
    results[data_tag][hypo_tag] = steps

logging.warn("FINISHED. Saving to file: %s"%args.outfile)
to_json(results,args.outfile)
