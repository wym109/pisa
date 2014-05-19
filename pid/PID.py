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
from utils.utils import set_verbosity,is_equal_binning,get_bin_centers
from utils.jsons import from_json,to_json
from PIDService import PIDService
import numpy as np
import scipy.stats


def get_event_rate_pid(reco_events,pid_dict=None,**kwargs):
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

    # Verify consistent binning
    ebins = reco_events['nue_cc']['ebins']
    czbins = reco_events['nue_cc']['czbins']
    flavours = ['nue_cc','numu_cc','nutau_cc','nuall_nc']
    for nu in flavours:
        if not is_equal_binning(ebins,reco_events[nu]['ebins']):
            raise Exception('Event Rate maps have different energy binning!')
        if not is_equal_binning(czbins,reco_events[nu]['czbins']):
            raise Exception('Event Rate maps have different coszen binning!')

    ecen = get_bin_centers(ebins)
    czcen = get_bin_centers(czbins)
        
    pid_service = PIDService(pid_dict)
    pid_dict = pid_service.get_pid_funcs()
    
    reco_events_pid = { 'trck': {'map':np.zeros((len(ecen),len(czcen))),
                                 'czbins':czbins,
                                 'ebins':ebins},
                        'cscd': {'map':np.zeros((len(ecen),len(czcen))),
                                 'czbins':czbins,
                                 'ebins':ebins}
                        }

    for flav in flavours:
        event_map = reco_events[flav]['map']
        
        to_trck_func = pid_dict[flav]['trck']
        to_cscd_func = pid_dict[flav]['cscd']

        to_trck = to_trck_func(ecen)
        to_trck_map = np.reshape(np.repeat(to_trck, len(czcen)), 
                                 (len(ecen), len(czcen)))*event_map
        to_cscd = to_cscd_func(ecen)
        to_cscd_map = np.reshape(np.repeat(to_cscd, len(czcen)), 
                                 (len(ecen), len(czcen)))*event_map
        
        reco_events_pid['trck']['map'] += to_trck_map
        reco_events_pid['cscd']['map'] += to_cscd_map
        
        
    return reco_events_pid


if __name__ == '__main__':

    #Only show errors while parsing 
    set_verbosity(0)
    parser = ArgumentParser(description='Takes a reco event rate file '
                            'as input and produces a set of reconstructed templates '
                            'of tracks and cascades.',
                            formatter_class=RawTextHelpFormatter)
    parser.add_argument('reco_event_file',metavar='RECOEVENTS',type=from_json,
                        help='''JSON reco event rate file with following parameters:
      {"nue_cc": {'czbins':[...], 'ebins':[...], 'map':[...]}, 
       "numu_cc": {...},
       "nutau_cc": {...},
       "nuall_nc": {...} }''')
    parser.add_argument('pid_file',metavar='WEIGHTFILE',type=from_json,
                        help='''json file containing parameterizations of the particle ID for each event type.''')
    parser.add_argument('-o', '--outfile', dest='outfile', metavar='FILE', type=str,
                        action='store',default="pid.json",
                        help='''file to store the output''')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='''set verbosity level''')
    args = parser.parse_args()

    #Set verbosity level
    set_verbosity(args.verbose)

    reco_events = args.reco_event_file
    pid_dict = args.pid_file
    event_rate_pid = get_event_rate_pid(reco_events,pid_dict)

    event_rate_pid['params'] = reco_events['params']
    
    logging.info("Saving output to: %s"%args.outfile)
    to_json(event_rate_pid,args.outfile)
    
    
