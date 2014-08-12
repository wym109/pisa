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

from pisa.utils.utils import set_verbosity
from pisa.utils.jsons import from_json,to_json
from LLRAnalysis import get_template_params, get_pseudo_data_fmap, find_max_llh_opt

from argparse import ArgumentParser,RawTextHelpFormatter

def fill_llh_dict(llh_dict,data,trial):
    '''
    Fills the LLH_dict with the data from 1 trial of data MH assume MH
    '''
    llh_dict['llh'].append(data[0])
    llh_dict['best'].append(data[1])
    llh_dict['opt_steps']['trial'+str(trial)] = dict(data[2].items())

    return


parser = ArgumentParser(description='''Runs the LLR optimizer-based analysis varying a number of systematic 
parameters defined in settings.json file and saves the likelihood ratios, 
which will be later converted to a significance of the measurement.''',
                        formatter_class=RawTextHelpFormatter)
parser.add_argument('settings',type=str,
                    help='''settings.json file which stores information related to the optimizer based analysis
with all relevant template-making systematics. Minimally, this will have fields for:
      'ebins':  [],
      'czbins': [],
      'params': {},
      'bounds': {} ''')
parser.add_argument('ntrials',type=int,
                    help="Number of trials to run the LLR analysis.")
parser.add_argument('-o','--outfile',type=str,default=None,
                    help="Output filename [.json]")
parser.add_argument('-v', '--verbose', action='count', default=0,
                    help='set verbosity level')
args = parser.parse_args()

set_verbosity(args.verbose)
settings = from_json(args.settings)

full_start_time = datetime.now()

LLH_dict = {data: {'NMH': {'llh': [], 'best': [], 'opt_steps':{}},
                   'IMH': {'llh': [], 'best': [], 'opt_steps':{}}}
            for data in ['data_NMH','data_IMH'] }

for itrial in xrange(args.ntrials):
    itrial+=1
    logging.info(">>>Running trial: %d\n"%itrial)
    
    start_time = datetime.now()
    
    # This will get the template params assuming NMH (given to get_templates)
    params = get_template_params(settings,use_best=True)
    
    fmap_nmh = get_pseudo_data_fmap(**params)
    nmh_assume_nmh = find_max_llh_opt(fmap_nmh,settings,assume_nmh=True)
    fill_llh_dict(LLH_dict['data_NMH']['NMH'],nmh_assume_nmh,itrial)
    stop1 = datetime.now()
    print "Total time to run optimizer: ",(stop1 - start_time)
    print "\n-------------------------------------------------------------------\n\n"
    
    nmh_assume_imh = find_max_llh_opt(fmap_nmh,settings,assume_nmh=False)
    fill_llh_dict(LLH_dict['data_NMH']['IMH'],nmh_assume_imh,itrial)
    stop2 = datetime.now()
    print "Total time to run optimizer: ",(stop2 - stop1)
    print "\n-------------------------------------------------------------------\n\n"
        
    # Now for IMH pseudo data:
    params['deltam31'] = -(params['deltam31'] - params['deltam21'])
    
    fmap_imh = get_pseudo_data_fmap(**params)
    imh_assume_nmh = find_max_llh_opt(fmap_imh,settings,assume_nmh=True)
    fill_llh_dict(LLH_dict['data_IMH']['NMH'],imh_assume_nmh,itrial)
    stop3 = datetime.now()
    print "Total time to run optimizer: ",(stop3 - stop2)
    print "\n-------------------------------------------------------------------\n\n"

    imh_assume_imh = find_max_llh_opt(fmap_imh,settings,assume_nmh=False)
    fill_llh_dict(LLH_dict['data_IMH']['IMH'],imh_assume_imh,itrial)
    stop4 = datetime.now()
    print "Total time to run optimizer: ",(stop4 - stop3)
    print "\n-------------------------------------------------------------------\n\n"

LLH_dict['settings'] = settings
outfile = args.outfile if args.outfile is not None else "llh_data_trials_"+str(args.ntrials)+".json"
to_json(LLH_dict,outfile)
    
print"Total time taken to run %d trials: %s"%(args.ntrials,
                                              (datetime.now() - full_start_time))

