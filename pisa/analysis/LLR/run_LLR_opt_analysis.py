#! /usr/bin/env python
#
# run_LLR_opt_analysis.py
#
# Runs the LLR optimizer-based analysis
#
# author: Tim Arlen
#         tca3@psu.edu
#
# date:   02-July-2014
#

import logging,os
from datetime import datetime
import numpy as np
from argparse import ArgumentParser,RawTextHelpFormatter

from pisa.utils.utils import set_verbosity
from pisa.utils.jsons import from_json,to_json
from LLRAnalysis import get_fiducial_params, get_pseudo_data_fmap, find_max_llh_opt, get_true_fmap
from pisa.analysis.TemplateMaker import TemplateMaker

def fill_llh_dict(llh_dict,data,trial):
    '''
    Fills the LLH_dict with the data from 1 trial of data MH assume MH
    '''
    llh_dict['llh'].append(data[0])
    llh_dict['best'].append(data[1])
    if llh_dict['opt_steps']:
        llh_dict['opt_steps']['trial'+str(trial)] = dict(data[2].items())

    return

    
parser = ArgumentParser(description='''Runs the LLR optimizer-based analysis varying a number of systematic 
parameters defined in settings.json file and saves the likelihood ratios, 
which will be later converted to a significance of the measurement.''',
                        formatter_class=RawTextHelpFormatter)
parser.add_argument('temp_settings',type=str,
                    help='''<template_settings>.json file which stores information 
related to the template-making part of the analysis with all relevant systematics. 
Minimally, this will have fields for:
      'template': {},
      'fiducial': {} ''')
parser.add_argument('llr_settings',type=str,
                    help='''<llr_settings>.json file which stores information 
related to the optimizer-based llr analysis.''')
parser.add_argument('ntrials',type=int,
                    help="Number of trials to run the LLR analysis.")
parser.add_argument('-s','--save_opt_steps',action='store_true',default=False,
                    help="Save all steps the optimizer takes.")
parser.add_argument('-o','--outfile',type=str,default=None,
                    help="Output filename [.json]")
parser.add_argument('-v', '--verbose', action='count', default=0,
                    help='set verbosity level')
args = parser.parse_args()

set_verbosity(args.verbose)
temp_settings = from_json(args.temp_settings)
llr_settings  = from_json(args.llr_settings)
save_opt_steps = args.save_opt_steps

full_start_time = datetime.now()

LLH_dict = {data: {'NMH': {'llh': [], 'best': [], 'opt_steps':{}},
                   'IMH': {'llh': [], 'best': [], 'opt_steps':{}}}
            for data in ['data_NMH','data_IMH'] }

temp_maker = TemplateMaker(temp_settings['binning'],temp_settings['template'])
    
for itrial in xrange(args.ntrials):
    itrial+=1
    logging.info(">>>Running trial: %d\n"%itrial)
    
    start_time = datetime.now()
    
    # This will get the template params assuming NMH (given to get_templates)
    fiducial_params = get_fiducial_params(temp_settings,use_best=True,use_nmh=True)
    fmap_nmh = get_pseudo_data_fmap(temp_maker,fiducial_params)
    
    nmh_assume_nmh = find_max_llh_opt(fmap_nmh,temp_maker,temp_settings['fiducial'],
                                      llr_settings,save_opt_steps,assume_nmh=True)    
    fill_llh_dict(LLH_dict['data_NMH']['NMH'],nmh_assume_nmh,itrial)
    stop1 = datetime.now()
    print "Total time to run optimizer: ",(stop1 - start_time)
    print "\n-------------------------------------------------------------------\n\n"

    nmh_assume_imh = find_max_llh_opt(fmap_nmh,temp_maker,temp_settings['fiducial'],
                                      llr_settings,save_opt_steps,assume_nmh=False)
    fill_llh_dict(LLH_dict['data_NMH']['IMH'],nmh_assume_imh,itrial)
    stop2 = datetime.now()
    print "Total time to run optimizer: ",(stop2 - stop1)
    print "\n-------------------------------------------------------------------\n\n"
    

    # Now for IMH pseudo data:
    fiducial_params = get_fiducial_params(temp_settings,use_best=True,use_nmh=False)
    fmap_imh = get_pseudo_data_fmap(temp_maker,fiducial_params)
    
    imh_assume_nmh = find_max_llh_opt(fmap_imh,temp_maker,temp_settings['fiducial'],
                                      llr_settings,save_opt_steps,assume_nmh=True)
    fill_llh_dict(LLH_dict['data_IMH']['NMH'],imh_assume_nmh,itrial)
    stop3 = datetime.now()
    print "Total time to run optimizer: ",(stop3 - stop2)
    print "\n-------------------------------------------------------------------\n\n"

    imh_assume_imh = find_max_llh_opt(fmap_imh,temp_maker,temp_settings['fiducial'],
                                      llr_settings,save_opt_steps,assume_nmh=False)
    fill_llh_dict(LLH_dict['data_IMH']['IMH'],imh_assume_imh,itrial)
    stop4 = datetime.now()
    print "Total time to run optimizer: ",(stop4 - stop3)
    print "\n-------------------------------------------------------------------\n\n"

LLH_dict['temp_settings'] = temp_settings
LLH_dict['llr_settings'] = llr_settings
outfile = args.outfile if args.outfile is not None else "llh_data_trials_"+str(args.ntrials)+".json"
to_json(LLH_dict,outfile)
    
print"Total time taken to run %d trials: %s"%(args.ntrials,
                                              (datetime.now() - full_start_time))

