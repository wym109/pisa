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

print "importing..."
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np


#from pisa.analysis.llr.LLHAnalysis import find_max_llh_bfgs, find_llh_no_osc, find_llh_mc_true
#from pisa.analysis.stats.Maps import get_pseudo_data_fmap, get_seed
print "again...1"
from pisa.analysis.llr.LLHAnalysis import find_max_llh_bfgs
print "again...2"
from pisa.analysis.TemplateMaker import TemplateMaker
print "again...3"
from pisa.utils.log import logging, profile, physics, set_verbosity
print "again...4"
from pisa.utils.jsons import from_json,to_json
from pisa.utils.params import get_values, select_hierarchy, fix_atm_params, get_atm_params

print "done..."

parser = ArgumentParser(
    description='''Runs the LLR optimizer-based analysis varying a number of systematic
    parameters defined in settings.json file and saves the likelihood values for all
    combination of hierarchies. Does not compute any pseudo data sets, but rather takes
    the Asimov data set (or expected counts template) at the given value of atm params.''',
    formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-t','--template_settings',type=str,
                    metavar='JSONFILE', required = True,
                    help='''Settings related to the template generation and systematics.''')
parser.add_argument('-m','--minimizer_settings',type=str,
                    metavar='JSONFILE', required = True,
                    help='''Settings related to the optimizer used in the LLR analysis.''')
parser.add_argument('-g','--grid_settings',action='store_true',default=None,
                    help='''Get llh value at defined oscillation parameter grid values,
                    according to these input settigs to.''')
parser.add_argument('-s','--save-steps',action='store_true',default=False,
                    dest='save_steps',
                    help='''Save all steps the optimizer takes.''')
parser.add_argument('-c','--chan',type=str,default='all',
                    choices=['trck','cscd','all','no_pid'],
                    help='''which channel to use in the fit.''')
parser.add_argument('-o','--outfile',type=str,default='llh_data.json',metavar='JSONFILE',
                    help='''Output filename.''')
parser.add_argument('-v', '--verbose', action='count', default=None,
                    help='''set verbosity level''')
args = parser.parse_args()

#cmd_str = ""
#for arg in sys.argv: cmd_str+=arg+" "

set_verbosity(args.verbose)

#Read in the settings
template_settings = from_json(args.template_settings)
minimizer_settings  = from_json(args.minimizer_settings)
grid_settings = from_json(args.grid_settings)

#Workaround for old scipy versions
import scipy
if scipy.__version__ < '0.12.0':
    logging.warn('Detected scipy version %s < 0.12.0'%scipy.__version__)
    if 'maxiter' in minimizer_settings:
      logging.warn('Optimizer settings for \"maxiter\" will be ignored')
      minimizer_settings.pop('maxiter')

#for itrial in xrange(1,args.ntrials+1):
#    profile.info("start trial %d"%itrial)
#    logging.info(">"*10 + "Running trial: %05d"%itrial + "<"*10)

    # //////////////////////////////////////////////////////////////////////
    # For each trial, generate pseudo-data experiments. Then do a LLH scan
    # over the atmospheric oscillation parameters, maximizing the likelihood
    # by varyaing the free (nuisance) parameters.
    # //////////////////////////////////////////////////////////////////////

#Get the parameters
params = template_settings['params']

# Make sure that atmospheric parameters are fixed:
logging.warn("Ensuring that atmospheric parameters are fixed for this analysis")
params = fix_atm_params(params)

template_maker = TemplateMaker(get_values(params),**template_settings['binning'])

data_types = [('data_NMH',True),('data_IMH',False)]
results = {}
for data_tag, data_normal in data_types:
    results[data_tag] = {}

    params_hierarchy = select_hierarchy(params,normal_hierarchy=data_normal)
    #fmap = get_pseudo_data_fmap(template_maker,get_values(params_hierarchy),chan=args.chan)
    asimov_data_set = get_asimov_fmap(template_maker,get_values(params_hierarchy),
                                      chan=args.chan)

    hypo_types = [('hypo_NMH',True),('hypo_IMH',False)]
    for hypo_tag, hypo_normal in hypo_types:

        # Now scan over theta23,deltam31 values and fix params to
        # these values:
        #Calculate steps for all free parameters
        atm_params = get_atm_params(select_hierarchy(params,hypo_normal))
        calc_steps(atm_params, grid_settings['steps'])

        #Build a list from all parameters that holds a list of (name, step) tuples
        steplist = [ [(name,step) for step in param['steps']]
                     for name, param in sorted(atm_params.items())]

        #Prepare to store all the steps
        steps = {key:[] for key in free_params.keys()}
        steps['llh'] = []

        #Iterate over the cartesian product
        for pos in product(*steplist):

            #Get a dict with all parameter values at this position
            #including the fixed parameters
            template_params = dict(list(pos) + get_values(fixed_params).items())

            print "pos dict: ",dict(list(pos))
            raw_input("PAUSED...right before optimizer...")
            llh_data = find_max_llh_bfgs(asimov_data_set,template_maker,params,
                                         minimizer_settings,args.save_steps,
                                         normal_hierarchy=hypo_normal,
                                         chan=args.chan)
            print "llh_data: ",llh_data
            raw_input("PAUSED...")
        profile.info("stop optimizer")

        #Store the LLH data
        results[data_tag][hypo_tag] = llh_data
