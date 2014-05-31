#! /usr/bin/env python
#
# Oscillation.py
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
from utils.utils import set_verbosity, check_binning, get_binning
from utils.jsons import from_json, to_json
from utils.proc import report_params, get_params, add_params
from OscillationService import OscillationService
from datetime import datetime


def get_osc_flux(flux_maps,osc_service=None,deltam21=None,deltam31=None,theta12=None,
                 theta13=None,theta23=None,deltacp=None,**kwargs):
    '''
    Uses osc_prob_maps to calculate the oscillated flux maps.
    Inputs:
      flux_maps - dictionary of atmospheric flux ['nue','numu','nue_bar','numu_bar']
      osc_service - a handle to an OscillationService
      others - oscillation parameters to compute oscillation probability maps from.
    '''

    #Be verbose on input
    params = get_params()
    report_params(params, units = ['eV^2','eV^2','rad','rad','rad','rad'])

    #Initialize return dict
    osc_flux_maps = {'params': add_params(params,flux_maps['params'])}
    
    #Get oscillation probability map from service
    osc_prob_maps = osc_service.get_osc_prob_maps(deltam21,deltam31,theta12,
                                                  theta13,theta23,deltacp)

    ebins, czbins = get_binning(flux_maps)    
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
    parser.add_argument('flux_maps',metavar='FLUX',type=from_json,
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
    set_verbosity(args.verbose)

    #Get binning
    ebins, czbins = check_binning(args.flux_maps)

    #Initialize an oscillation service
    osc_service = OscillationService(ebins,czbins)

    start_time = datetime.now()
    
    logging.info("Getting osc prob maps")
    osc_flux_maps = get_osc_flux(args.flux_maps,osc_service,args.deltam21,args.deltam31,
                                 args.theta12, args.theta13,args.theta23,args.deltacp)
    
    #Write out
    logging.info("Saving output to: %s",args.outfile)
    to_json(osc_flux_maps, args.outfile)

    
