#! /usr/bin/env python
#
# EventRate.py
#
# This module is the implementation of the stage2 analysis. The main
# purpose of stage2 is to combine the "oscillated Flux maps" with the
# weighted effective areas to create oscillated event rate maps,
# using the true information.
# 
# If desired, this will create a .json output file with the results of
# the current stage of processing.
#
# author: Timothy C. Arlen
#
#         tca3@psu.edu
#
# date:   April 8, 2014
#

import os,sys
import numpy as np
import logging
from argparse import ArgumentParser, RawTextHelpFormatter
from utils.utils import set_verbosity,is_equal_binning
from utils.json import from_json, to_json
from AeffService import AeffServiceMC
from scipy.constants import Julian_year

def get_event_rates(osc_flux_maps,sim_file=None,livetime=None,nu_xsec_scale=None,
                    nu_bar_xsec_scale=None,**kwargs):
    '''
    Main function for this module, which returns the event rate maps
    for each flavor and interaction type, using true energy and zenith
    information. The content of each bin will be the weighted aeff
    multiplied by the oscillated flux, so that the returned dictionary
    will be of the form:
    {'nue': {'cc':map,'nc':map},
     'nue_bar': {'cc':map,'nc':map}, ...
     'nutau_bar': {'cc':map,'nc':map} }
    '''

    # Verify consistent binning.
    ebins = osc_flux_maps['nue']['ebins']
    czbins = osc_flux_maps['nue']['czbins']
    flavours = ['nue','numu','nutau','nue_bar','numu_bar','nutau_bar']
    if not np.alltrue([is_equal_binning(ebins,osc_flux_maps[nu]['ebins']) for nu in flavours]):
        raise Exception('Osc flux maps have different energy binning!')
    if not np.alltrue([is_equal_binning(czbins,osc_flux_maps[nu]['czbins']) for nu in flavours]):
        raise Exception('Osc flux maps have different coszen binning!')

    logging.info("Defining aeff_service...")
    aeff_service = AeffServiceMC(ebins,czbins,simfile)
    aeff_dict = aeff_service.get_aeff()
    
    # apply the scaling for nu_xsec_scale and nubar_xsec_scale...
    
    event_rate_maps = {}
    for flavour in flavours:
        osc_flux_map = osc_flux_maps[flavour]['map']
        int_type_dict = {}
        for int_type in ['cc','nc']:
            event_rate = osc_flux_map*aeff_dict[flavour][int_type]*livetime*Julian_year
            int_type_dict[int_type] = {'map':event_rate,
                                       'ebins':ebins,
                                       'czbins':czbins}
        event_rate_maps[flavour] = int_type_dict
        
    return event_rate_maps

if __name__ == '__main__':

    #Only show errors while parsing 
    set_verbosity(0)
    parser = ArgumentParser(description='Take an oscillated flux file '
                            'as input and write out a set of oscillated event counts. ',
                            formatter_class=RawTextHelpFormatter)
    parser.add_argument('osc_flux_file',metavar='FLUX',type=from_json,
                        help='''JSON osc flux input file with the following parameters:
      {"nue": {'czbins':[], 'ebins':[], 'map':[]}, 
       "numu": {...},
       "nutau": {...},
       "nue_bar": {...},
       "numu_bar": {...},
       "nutau_bar": {...} }''')
    parser.add_argument('weighted_aeff_file',metavar='WEIGHTFILE',type=str,
                        help='''HDF5 File containing data from all flavours for a particular instumental geometry. 
Expects the file format to be:
      {
        'nue': {
           'cc': {
               'weighted_aeff': np.array,
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
    parser.add_argument('--livetime',type=float,default=1.0,
                        help='''livetime in years to re-scale by.''')
    parser.add_argument('--nu_xsec_scale',type=float,default=1.0,
                        help='''Overall scale on nu xsec.''')
    parser.add_argument('--nubar_xsec_scale',type=float,default=1.0,
                        help='''Overall scale on nu_bar xsec.''')
    parser.add_argument('-o', '--outfile', dest='outfile', metavar='FILE', type=str,
                        action='store',default="event_rate.json",
                        help='''file to store the output''')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='''set verbosity level''')
    args = parser.parse_args()

    #Set verbosity level
    set_verbosity(args.verbose)

    livetime = args.livetime
    nu_xsec_scale = args.nu_xsec_scale
    nubar_xsec_scale = args.nubar_xsec_scale
    event_param_dict = {'livetime':livetime,'nu_xsec_scale':nu_xsec_scale,
                        'nubar_xsec_scale':nubar_xsec_scale}

    for name,param in zip(["livetime","nu xs scale","nubar xs scale"],
                          [livetime,nu_xsec_scale,nubar_xsec_scale]):
        logging.debug("%14s: %s "%(name,param))
        
    logging.info("Getting oscillated flux...")    
    osc_flux_maps = args.osc_flux_file
    simfile = args.weighted_aeff_file
    
    event_rate_maps = get_event_rates(osc_flux_maps,simfile,livetime,
                                      nu_xsec_scale,nubar_xsec_scale)
    
    event_rate_maps['params'] = dict(osc_flux_maps['params'].items() + 
                                     event_param_dict.items())
    logging.info("Saving output to .json file...")
    to_json(event_rate_maps,args.outfile)
    
    
