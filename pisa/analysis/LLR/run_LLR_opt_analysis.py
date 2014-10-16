#! /usr/bin/env python
#
# run_LLR_opt_analysis.py
#
# Runs the LLR optimizer-based LLR analysis
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
from pisa.analysis.LLR.LLRAnalysis import get_pseudo_data_fmap, find_max_llh_opt
from pisa.analysis.TemplateMaker import TemplateMaker
from pisa.utils.params import get_values, select_hierarchy

def show_time(time_stop, time_start):
    '''
    This print message is repeated whenever the optimizer is finished...
    '''
    print "Total time to run optimizer: ",(time_stop - time_start)
    print "\n-------------------------------------------------------------------\n\n"

    return

parser = ArgumentParser(description='''Runs the LLR optimizer-based analysis varying a number of systematic parameters
defined in settings.json file and saves the likelihood ratios, which will be
later converted to a significance of the measurement.''',
                        formatter_class=RawTextHelpFormatter)
parser.add_argument('model_settings',type=str,metavar='JSONFILE',
                    help='''File which stores information related to the template-making part of the
analysis with all relevant systematics.''')
parser.add_argument('llr_settings',type=str,metavar='JSONFILE',
                    help='''File which stores information related to the optimizer-based llr analysis.''')
parser.add_argument('ntrials',type=int,
                    help="Number of trials to run the LLR analysis.")
parser.add_argument('-s','--save_opt_steps',action='store_true',default=False,
                    help="Save all steps the optimizer takes.")
parser.add_argument('-o','--outfile',type=str,default=None,metavar='JSONFILE',
                    help="Output filename.")
parser.add_argument('-v', '--verbose', action='count', default=0,
                    help='set verbosity level')
args = parser.parse_args()

set_verbosity(args.verbose)
model_settings = from_json(args.model_settings)
llr_settings  = from_json(args.llr_settings)

full_start_time = datetime.now()

LLH_dict = {data: {'NMH':{},'IMH':{}} for data in ['data_NMH','data_IMH']}

template_maker = TemplateMaker(get_values(model_settings['params']),**model_settings['binning'])

for itrial in xrange(1,args.ntrials+1):
    logging.info(">>>Running trial: %d\n"%itrial)

    start_time = datetime.now()

    # //////////////////////////////////////////////////////////////////////
    # 1) get a normal hierarchy pseudo data fmap from fiducial model
    #    (best fit vals of params).
    # 2) find max llh (and best fit free params) from matching pseudo data
    #    to NMH templates.
    # 3) find max llh (and best fit free params) from matching pseudo data
    #    to IMH templates (other hierarhcy).
    # //////////////////////////////////////////////////////////////////////
    fmap_nmh = get_pseudo_data_fmap(template_maker,
                                get_values(select_hierarchy(model_settings['params'],
                                                            normal_hierarchy=True)))
    llh_data = find_max_llh_opt(fmap_nmh,template_maker,model_settings['params'],
                                llr_settings,args.save_opt_steps,normal_hierarchy=True)
    LLH_dict['data_NMH']['NMH']['trial'+str(itrial)] = llh_data
    stop1 = datetime.now()
    show_time(stop1,start_time)

    llh_data = find_max_llh_opt(fmap_nmh,template_maker,model_settings['params'],
                                llr_settings,args.save_opt_steps,normal_hierarchy=False)
    LLH_dict['data_NMH']['IMH']['trial'+str(itrial)] = llh_data
    stop2 = datetime.now()
    show_time(stop2,stop1)
    # //////////////////////////////////////////////////////////////////////


    # Now repeat these steps for inverted hierarchy pseudo data.
    fmap_imh = get_pseudo_data_fmap(template_maker,
                               get_values(select_hierarchy(model_settings['params'],
                                                           normal_hierarchy=False)))
    llh_data = find_max_llh_opt(fmap_imh,template_maker,model_settings['params'],
                                llr_settings,args.save_opt_steps,normal_hierarchy=True)
    LLH_dict['data_IMH']['NMH']['trial'+str(itrial)] = llh_data
    stop3 = datetime.now()
    show_time(stop3,stop2)

    llh_data = find_max_llh_opt(fmap_imh,template_maker,model_settings['params'],
                                llr_settings,args.save_opt_steps,normal_hierarchy=False)
    LLH_dict['data_IMH']['IMH']['trial'+str(itrial)] = llh_data
    stop4 = datetime.now()
    show_time(stop4,stop3)


LLH_dict['model_settings'] = model_settings
LLH_dict['llr_settings'] = llr_settings
outfile = args.outfile if args.outfile is not None else "llh_data_trials_"+str(args.ntrials)+".json"
to_json(LLH_dict,outfile)

print"Total time taken to run %d trials: %s"%(args.ntrials,
                                              (datetime.now() - full_start_time))

