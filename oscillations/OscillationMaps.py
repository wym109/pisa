#! /usr/bin/env python
#
# OscillationMaps.py
#
# This module is the implementation of the physics oscillation step.
# The main purpose of this step is to produce oscillation probability
# maps of each neutrino flavor into the others, for a given set of
# oscillation parameters, and to multiply it by the corresponding flux
# map, producing oscillated flux maps.
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# date:   Jan. 21, 2014
#


## IMPORTS ##
import os,sys
import numpy as np
import logging
from argparse import ArgumentParser, RawTextHelpFormatter
from utils.utils import set_verbosity,is_equal_binning
from utils.json import from_json, to_json
from OscillationService import OscillationService
from datetime import datetime


def get_osc_flux(flux_maps,deltam21=None,deltam31=None,theta12=None,
                 theta13=None,theta23=None,deltacp=None,osc_code=None,**kwargs):
    '''
    Uses osc_prob_maps to calculate the oscillated flux maps.
    Inputs:
      flux_maps - dictionary of atmospheric flux ['nue','numu','nue_bar','numu_bar']
      others - oscillation parameters to compute oscillation probability maps from.
    '''

    units = ['rad','eV^2','eV^2','rad','rad','rad']
    osc_param_dict = {'deltam21':deltam21,'deltam31':deltam31,'theta12':theta12,
                      'theta13':theta13,'theta23':theta23,'deltacp':deltacp}
    for param, unit in zip(sorted(osc_param_dict),units):
        logging.debug("%10s: %.4e %s"%(param,osc_param_dict[param],unit))

    ebins = flux_maps['nue']['ebins']
    czbins = flux_maps['nue']['czbins']    
    if not np.alltrue([is_equal_binning(ebins,flux_maps[nu]['ebins']) for nu in ['nue','nue_bar','numu','numu_bar']]):
        raise Exception('Flux maps have different energy binning!')
    if not np.alltrue([is_equal_binning(czbins,flux_maps[nu]['czbins']) for nu in ['nue','nue_bar','numu','numu_bar']]):
        raise Exception('Flux maps have different coszen binning!')
    
    osc_service = OscillationService(ebins,czbins,osc_code=osc_code)
    osc_prob_maps = osc_service.get_osc_prob_maps(deltam21,deltam31,theta12,
                                                  theta13,theta23,deltacp)
    
    #test_file = "smoothed_osc_prob_maps_numu.json"
    #logging.info("Creating file %s"%test_file)
    #to_json(osc_prob_maps,test_file)
   
    osc_flux_maps = {}
    for to_flav in ['nue','numu','nutau']:
        for mID in ['','_bar']: # 'matter' ID
            nue_flux = flux_maps['nue'+mID]['map']
            numu_flux = flux_maps['numu'+mID]['map']
            oscflux = {'ebins':ebins,
                       'czbins':czbins,
                       'map':(nue_flux*osc_prob_maps['nue'+mID+'_maps'][to_flav+mID] +
                              numu_flux*osc_prob_maps['numu'+mID+'_maps'][to_flav+mID])
                       }
            osc_flux_maps[to_flav+mID] = oscflux
            
    return osc_flux_maps

        
if __name__ == '__main__':
    
    #Only show errors while parsing
    set_verbosity(0)

    # parser
    parser = ArgumentParser(description='Takes the oscillation parameters '
                            'as input and writes out a set of osc flux maps',
                            formatter_class=RawTextHelpFormatter)    
    parser.add_argument('flux_file',metavar='FLUX',type=from_json,
                        help='''JSON atm flux input file with the following parameters:
    {"nue": {'czbins':[], 'ebins':[], 'map':[]},
     "numu": {...},
     "nue_bar": {...},
     "numu_bar":{...}}''')
    parser.add_argument('--deltam21',type=float,default=7.54e-5,
                        help='''deltam21 value [eV^2]''')
    parser.add_argument('--deltam31',type=float,default=0.00246,
                        help='''deltam31 value [eV^2]''')
    parser.add_argument('--theta12',type=float,default=0.5873,
                        help='''theta12 value [rad]''')
    parser.add_argument('--theta13',type=float,default=0.1562,
                        help='''theta13 value [rad]''')
    parser.add_argument('--theta23',type=float,default=0.6745,
                        help='''theta23 value [rad]''')
    parser.add_argument('--deltacp',type=float,default=0.0,
                        help='''deltaCP value to use [rad]''')
    parser.add_argument('--osc_code',type=str,default='Prob3',
                        help='''Oscillation prob code to use ['Prob3' (default) or 'NuCraft'] ''')
    parser.add_argument('-o', '--outfile', dest='outfile', metavar='FILE', type=str,
                        action='store',default="osc_flux.json",
                        help='file to store the output')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='set verbosity level')
    args = parser.parse_args()
    
    args_dict = vars(args)
    
    #Set verbosity level
    verbose = args_dict.pop('verbose')
    set_verbosity(verbose)
    
    outfile = args_dict.pop('outfile')
    flux_maps = args_dict.pop('flux_file')
    #outfile = args.outfile
    #flux_maps = args.flux_file

    start_time = datetime.now()
    
    logging.info("Getting osc prob maps")
    osc_flux_maps = get_osc_flux(flux_maps,args.deltam21,args.deltam31,args.theta12,
                                 args.theta13,args.theta23,args.deltacp,
                                 osc_code=args.osc_code)
    
    #Merge the new parameters into the old ones
    osc_flux_maps['params'] = dict(flux_maps['params'].items() + args_dict.items())

    logging.info("Saving params: %s"%osc_flux_maps['params'].keys())

    #Write out
    logging.info("Saving osc prob maps to file: %s",outfile)
    to_json(osc_flux_maps, outfile)
    logging.info("Total time taken to run this stage: %s",(datetime.now() - start_time))
    
    
