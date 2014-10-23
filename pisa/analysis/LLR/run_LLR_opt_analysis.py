#! /usr/bin/env python
#
# run_LLR_opt_analysis.py
#
# Runs the LLR optimizer-based LLR analysis
#
# author: Tim Arlen - tca3@psu.edu
#         Sebatian Boeser - sboeser@uni-mainz.de
#
# date:   02-July-2014
#

import numpy as np
from argparse import ArgumentParser,RawTextHelpFormatter

from pisa.utils.log import logging, profile, physics, set_verbosity
from pisa.utils.jsons import from_json,to_json
from pisa.analysis.LLR.LLRAnalysis import get_pseudo_data_fmap, find_max_llh_bfgs
from pisa.analysis.TemplateMaker import TemplateMaker
from pisa.utils.params import get_values, select_hierarchy

parser = ArgumentParser(description='''Runs the LLR optimizer-based analysis varying a number of systematic parameters
defined in settings.json file and saves the likelihood ratios, which will be
later converted to a significance of the measurement.''',
                        formatter_class=RawTextHelpFormatter)
parser.add_argument('-t','--template_settings',type=str,
                    metavar='JSONFILE', required = True,
                    help='''Settings related to the template generation and systematics.''')
parser.add_argument('-m','--minimizer_settings',type=str,
                    metavar='JSONFILE', required = True,
                    help='''Settings related to the optimizer used in the LLR analysis.''')
parser.add_argument('-n','--ntrials',type=int, required = True,
                    help="Number of trials to run")
parser.add_argument('-s','--save_steps',action='store_true',default=False,
                    help="Save all steps the optimizer takes.")
parser.add_argument('-o','--outfile',type=str,default='llh_data.json',metavar='JSONFILE',
                    help="Output filename.")
parser.add_argument('-v', '--verbose', action='count', default=0,
                    help='set verbosity level')
args = parser.parse_args()

set_verbosity(args.verbose)

#Read in the settings
template_settings = from_json(args.template_settings)
minimizer_settings  = from_json(args.minimizer_settings)

#Get the parameters
params = template_settings['params']

#store results from all the trials
trials = []

template_maker = TemplateMaker(get_values(params),**template_settings['binning'])

for itrial in xrange(1,args.ntrials+1):
    profile.info("start trial %d"%itrial)
    logging.info(">"*10 + "Running trial: %05d"%itrial + "<"*10)


    # //////////////////////////////////////////////////////////////////////
    # For each trial, generate two pseudo-data experiemnts (one for each
    # hierarchy), and for each find the best matching template in each of the
    # hierarchy hypothesis.
    # //////////////////////////////////////////////////////////////////////
    results = {}
    for data_tag, data_normal in [('data_NMH',True),('data_IMH',False)]:

        results[data_tag] = {}
        # 1) get a pseudo data fmap from fiducial model (best fit vals of params).
        fmap = get_pseudo_data_fmap(template_maker,
                                    get_values(select_hierarchy(params,
                                                                normal_hierarchy=data_normal)))

        # 2) find max llh (and best fit free params) from matching pseudo data
        #    to templates.
        for hypo_tag, hypo_normal in [('hypo_NMH',True),('hypo_IMH',False)]:

            physics.info("Finding best fit for %s under %s assumption"%(data_tag,hypo_tag))
            profile.info("start optimizer")
            llh_data = find_max_llh_bfgs(fmap,template_maker,params,
                                        minimizer_settings,args.save_steps,normal_hierarchy=hypo_normal)
            profile.info("stop optimizer")

            #Store the LLH data
            results[data_tag][hypo_tag] = llh_data


    #Store this trial
    trials += [results]
    profile.info("stop trial %d"%itrial)

#Assemble output dict
output = {'trials' : trials,
          'template_settings' : template_settings,
          'minimizer_settings' : minimizer_settings}
#And write to file
to_json(output,args.outfile)
