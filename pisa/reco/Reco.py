#! /usr/bin/env python
#
# EventRateReco.py
#
# This module will perform the smearing of the true event rates, with
# the reconstructed parameters, using the detector response
# resolutions, in energy and coszen.
#
# The MC-based approach will take the pdf of the true_energy,
# true_coszen directly from simulations to be reconstructed at
# reco_energy, reco_coszen, and will apply these pdfs to the true
# event rates, ending with the expected reconstructed event rate
# templates for each flavor CC and an overall NC template.
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# date:   April 9, 2014
#

import logging
from argparse import ArgumentParser, RawTextHelpFormatter
from pisa.utils.utils import set_verbosity, check_binning, get_binning
from pisa.utils.jsons import from_json,to_json
from pisa.utils.proc import report_params, get_params, add_params
from pisa.reco.RecoService import RecoServiceMC
import numpy as np


def get_reco_maps(true_event_maps,simfile=None,e_reco_scale=None,
                         cz_reco_scale=None, **kwargs):
    '''
    Primary function for this module, which returns the reconstructed
    event rate maps from the true event rate maps, and from the
    smearing kernal obtained from simulations. The returned maps will
    be in the form of a dictionary with parameters:
    {'nue_cc':{'ebins':ebins,'czbins':czbins,'map':map},
     'numu_cc':{...},
     'nutau_cc':{...},
     'nuall_nc':{...}
    }
    Note that in this function, the nu<x> is now combined with nu_bar<x>.
    '''

    #Be verbose on input
    params = get_params()
    report_params(params, units = ['',''])
    
    #Initialize return dict
    reco_maps = {'params': add_params(params,true_event_maps['params'])}

    #Get kernels from reco service
    kernel_dict = reco_service.get_kernels()

    ebins, czbins = get_binning(true_event_maps)

    flavours = ['nue','numu','nutau']
    int_types = ['cc','nc']
    
    
    for int_type in int_types:
        for flavor in flavours:
            logging.info("Getting reco event rates for %s %s"%(flavor,int_type))
            reco_evt_rate = np.zeros((len(ebins)-1,len(czbins)-1),
                                     dtype=np.float32)
            for mID in ['','_bar']:
                flav = flavor+mID
                true_evt_rate = true_event_maps[flav][int_type]['map']
                
                kernels = kernel_dict[flav][int_type]
                    
                for ie,egy in enumerate(ebins[:-1]):
                    for icz,cz in enumerate(czbins[:-1]):
                        # Get kernel at these true parameters from 4D hist
                        kernel = kernels[ie,icz]
                        # normalize
                        if np.sum(kernel) > 0.0: kernel /= np.sum(kernel)
                        reco_evt_rate += true_evt_rate[ie,icz]*kernel

            reco_maps[flavor+'_'+int_type] = {'map':reco_evt_rate,
                                              'ebins':ebins,
                                              'czbins':czbins}
            logging.info("  Total counts: %.2f"%np.sum(reco_evt_rate))

    #Finally sum up all the NC contributions
    logging.info("Summing up rates for %s %s"%('all',int_type))
    reco_evt_rate = np.sum([reco_maps.pop(key)['map'] for key in reco_maps.keys()
                            if key.endswith('_nc')], axis = 0)
    reco_maps['nuall_nc'] = {'map':reco_evt_rate,
                             'ebins':ebins,
                             'czbins':czbins}
    logging.info("  Total counts: %.2f"%np.sum(reco_evt_rate))

    # Apply e_reco_scaling...
    # Apply cz_reco_scaling...
    
    return reco_maps


if __name__ == '__main__':

    #Only show errors while parsing 
    set_verbosity(0)
    parser = ArgumentParser(description='Takes a (true, triggered) event rate file '
                            'as input and produces a set of reconstructed templates '
                            'of nue CC, numu CC, nutau CC, and NC events.',
                            formatter_class=RawTextHelpFormatter)
    parser.add_argument('event_rate_maps',metavar='EVENTRATE',type=from_json,
                        help='''JSON event rate input file with following parameters:
      {"nue": {'cc':{'czbins':[], 'ebins':[], 'map':[]},'nc':...}, 
       "numu": {...},
       "nutau": {...},
       "nue_bar": {...},
       "numu_bar": {...},
       "nutau_bar": {...} }''')
    parser.add_argument('--weighted_aeff_file',metavar='WEIGHTFILE',type=str,
                        default='events/V15_weighted_aeff.hdf5',
                        help='''HDF5 File containing data from all flavours for a particular instumental geometry. 
Expects the file format to be:
      {
        'nue': {
           'cc': {
               ...
               'true_energy': np.array,
               'true_coszen': np.array,
               'reco_energy': np.array,
               'reco_coszen': np.array
            },
            'nc': {...
             }
         },
         'nue_bar' {...},...
      } ''')
    parser.add_argument('--e_reco_scale',type=float,default=1.0,
                        help='''Reconstructed energy scaling.''')
    parser.add_argument('--cz_reco_scale',type=float,default=1.0,
                        help='''Reconstructed coszen scaling.''')
    parser.add_argument('-o', '--outfile', dest='outfile', metavar='FILE', type=str,
                        action='store',default="reco.json",
                        help='''file to store the output''')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='''set verbosity level''')
    args = parser.parse_args()

    #Set verbosity level
    set_verbosity(args.verbose)

    #Check binning
    ebins, czbins = check_binning(args.event_rate_maps)

    logging.info("Defining RecoService...")
    reco_service = RecoServiceMC(ebins,czbins,simfile=args.weighted_aeff_file)

    event_rate_reco_maps = get_reco_maps(args.event_rate_maps,reco_service,args.e_reco_scale,
                                         args.cz_reco_scale)
    
    logging.info("Saving output to: %s"%args.outfile)
    to_json(event_rate_reco_maps,args.outfile)

    
