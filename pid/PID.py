#! /usr/bin/env python
#
# PID.py
#
# Performs the particle ID step of sorting the event map templates
# of the previous stage into tracks vs. cascades. Some fraction of
# CC events is identified as tracks, all others are cascades.
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# date:   April 10, 2014
#

import logging
from argparse import ArgumentParser, RawTextHelpFormatter
from utils.utils import set_verbosity,get_binning,check_binning,get_bin_centers
from utils.jsons import from_json,to_json
from utils.proc import report_params, get_params, add_params
from PIDServicePar import PIDServicePar
import numpy as np
import scipy.stats


def get_pid_maps(reco_events,pid_service,**kwargs):
    '''
    Takes the templates of reco_events in form of:
      'nue_cc': map
      'numu_cc': map
      'nutau_cc': map
      'nuall_nc': map
    And applies PID returning a dictionary of events in form of:
      {'trk': {'ebins':ebins,'czbins':czbins,'map':map},
       'csc': {'ebins':ebins,'czbins':czbins,'map':map}}
    '''

    #Be verbose on input
    params = get_params()
    report_params(params, units = [])

    reco_events_pid = pid_service.get_pid_maps(reco_events)
    reco_events_pid['params'] = add_params(params,reco_events['params'])
    
    return reco_events_pid


if __name__ == '__main__':

    #Only show errors while parsing 
    set_verbosity(0)
    parser = ArgumentParser(description='Takes a reco event rate file '
                            'as input and produces a set of reconstructed templates '
                            'of tracks and cascades.',
                            formatter_class=RawTextHelpFormatter)
    parser.add_argument('reco_event_maps',metavar='RECOEVENTS',type=from_json,
                        help='''JSON reco event rate file with following parameters:
      {"nue_cc": {'czbins':[...], 'ebins':[...], 'map':[...]}, 
       "numu_cc": {...},
       "nutau_cc": {...},
       "nuall_nc": {...} }''')
    parser.add_argument('settings_file',metavar='SETTINGS',type=from_json,
                        help='''json file containing parameterizations of the particle ID for each event type, based on PaPA settings file.''')
    parser.add_argument('-o', '--outfile', dest='outfile', metavar='FILE', type=str,
                        action='store',default="pid.json",
                        help='''file to store the output''')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='''set verbosity level''')
    args = parser.parse_args()

    #Set verbosity level
    set_verbosity(args.verbose)

    #Check binning
    ebins, czbins = check_binning(args.reco_event_maps)

    #Initialize the PID service
    pid_service = PIDServicePar(args.settings_file,ebins,czbins)

    #Galculate event rates after PID
    event_rate_pid = get_pid_maps(args.reco_event_maps,pid_service)
    
    logging.info("Saving output to: %s"%args.outfile)
    to_json(event_rate_pid,args.outfile)
    
    
