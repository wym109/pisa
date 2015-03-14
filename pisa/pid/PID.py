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
# author: Sebastian Boeser
#         sboeser@physik.uni-bonn.de
#
# date:   April 10, 2014
#

import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from copy import deepcopy as copy

from pisa.utils.log import logging, set_verbosity
from pisa.utils.utils import check_binning
from pisa.utils.proc import get_params, report_params, add_params
from pisa.utils.jsons import from_json, to_json
from pisa.resources.resources import find_resource
from pisa.pid.PIDServiceParam import PIDServiceParam
from pisa.pid.PIDServiceKernelFile import PIDServiceKernelFile


def get_pid_maps(reco_events, pid_service=None, recalculate=False,
                 return_unknown=False, **kwargs):
    """
    Primary function for this service, which returns the classified
    event rate maps (sorted after tracks and cascades) from the
    reconstructed ones (sorted after nu[e,mu,tau]_cc and nuall_nc).
    """
    if recalculate:
        pid_service.recalculate_kernels(**kwargs)

    #Be verbose on input
    params = get_params()
    report_params(params, units = [])

    #Initialize return dict
    empty_map = {'map': np.zeros_like(reco_events['nue_cc']['map']),
                 'czbins': pid_service.czbins, 'ebins': pid_service.ebins}
    reco_events_pid = {'trck': copy(empty_map),
                       'cscd': copy(empty_map),
                       'params': add_params(params,reco_events['params']),
                      }
    if return_unknown:
        reco_events_pid['unkn'] = copy(empty_map)

    #Classify events
    for flav in reco_events:

        if flav=='params':
            continue
        event_map = reco_events[flav]['map']

        to_trck_map = event_map*pid_service.pid_kernels[flav]['trck']
        to_cscd_map = event_map*pid_service.pid_kernels[flav]['cscd']

        reco_events_pid['trck']['map'] += to_trck_map
        reco_events_pid['cscd']['map'] += to_cscd_map
        if return_unknown:
            reco_events_pid['unkn']['map'] += (event_map - to_trck_map - to_cscd_map)

    return reco_events_pid



if __name__ == '__main__':

    parser = ArgumentParser(description='Takes a reco event rate file '
                            'as input and produces a set of reconstructed \n'
                            'templates of tracks and cascades.',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('reco_event_maps', metavar='JSON', type=from_json,
                        help='''JSON reco event rate file with following '''
                        '''parameters:\n'''
                        '''{"nue_cc": {'czbins':[...], 'ebins':[...], 'map':[...]}, \n'''
                        ''' "numu_cc": {...}, \n'''
                        ''' "nutau_cc": {...}, \n'''
                        ''' "nuall_nc": {...} }''')
    parser.add_argument('-m', '--mode', type=str, choices=['param', 'stored'],
                        default='param', help='PID service to use')
    parser.add_argument('--param_file', metavar='JSON', type=str,
                        default='pid/V36_pid.json',
                        help='JSON file containing parameterizations '
                        'of the particle ID \nfor each event type.')
    parser.add_argument('--kernel_file', metavar='JSON', type=str, default=None,
                        help='JSON file containing pre-calculated PID kernels')
    parser.add_argument('-o', '--outfile', dest='outfile', metavar='FILE', type=str,
                        action='store',default="pid.json",
                        help='''file to store the output''')
    parser.add_argument('-v', '--verbose', action='count', default=None,
                        help='''set verbosity level''')
    args = parser.parse_args()

    #Set verbosity level
    set_verbosity(args.verbose)

    #Check binning
    ebins, czbins = check_binning(args.reco_event_maps)

    #Initialize the PID service
    if args.mode=='param':
        pid_service = PIDServiceParam(ebins, czbins, 
                            pid_paramfile=args.param_file, **vars(args))
    elif args.mode=='stored':
        pid_service = PIDServiceKernelFile(ebins, czbins, 
                            pid_kernelfile=args.kernel_file, **vars(args))

    #Calculate event rates after PID
    event_rate_pid = get_pid_maps(args.reco_event_maps, pid_service=pid_service)

    logging.info("Saving output to: %s"%args.outfile)
    to_json(event_rate_pid,args.outfile)
