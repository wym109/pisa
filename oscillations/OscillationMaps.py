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

# Until python2.6, default json is very slow.
try: 
    import simplejson as json
except ImportError, e:
    import json


def get_osc_flux(flux_maps,deltam21=None,deltam31=None,theta12=None,
                 theta13=None,theta23=None,deltacp=None,**kwargs):
    '''
    Uses osc_prob_maps to calculate the oscillated flux maps.
    Inputs:
      flux_maps - dictionary of atmospheric flux ['nue','numu','nue_bar','numu_bar']
      others - oscillation parameters to compute oscillation probability maps from.
    '''

    ebins = flux_maps['nue']['ebins']
    czbins = flux_maps['nue']['czbins']
    
    if not np.alltrue([is_equal_binning(ebins,flux_maps[nu]['ebins']) for nu in ['nue','nue_bar','numu','numu_bar']]):
        raise Exception('Flux maps have different energy binning!')
    if not np.alltrue([is_equal_binning(czbins,flux_maps[nu]['czbins']) for nu in ['nue','nue_bar','numu','numu_bar']]):
        raise Exception('Flux maps have different coszen binning!')
    
    osc_service = OscillationService(ebins,czbins)
    osc_prob_maps = osc_service.get_osc_prob_maps(deltam21,deltam31,theta12,
                                                  theta13,theta23,deltacp)
    
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
    parser = ArgumentParser(description='Take a settings file '
                            'as input and write out a set of flux maps',
                            formatter_class=RawTextHelpFormatter)    
    parser.add_argument('flux_file',metavar='FLUX',type=from_json,
                        help='''JSON atm flux input file with the following parameters:
    {"nue": {'czbins':[], 'ebins':[], 'map':[]},
     "numu": {...},
     "nue_bar": {...},
     "numu_bar":{...}}''')
    parser.add_argument('deltam21',type=float,
                        help='''deltam21 value [eV^2]''')
    parser.add_argument('deltam31',type=float,
                        help='''deltam31 value [eV^2]''')
    parser.add_argument('theta12',type=float,
                        help='''theta12 value [rad]''')
    parser.add_argument('theta13',type=float,
                        help='''theta13 value [rad]''')
    parser.add_argument('theta23',type=float,
                        help='''theta23 value [rad]''')
    parser.add_argument('deltacp',type=float,
                        help='''deltaCP value to use [rad]''')
    parser.add_argument('-o', '--outfile', dest='outfile', metavar='FILE', type=str,
                        action='store',default="osc_flux.json",
                        help='file to store the output')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='set verbosity level')
    args = parser.parse_args()

    #Set verbosity level
    set_verbosity(args.verbose)

    outfile = args.outfile
    flux_maps = args.flux_file
    osc_param_dict = vars(args)
    osc_param_dict.pop('outfile')
    osc_param_dict.pop('flux_file')
    units = ['eV^2','eV^2','rad','rad','rad','rad']
    for param, unit in zip(osc_param_dict.keys(),units):
        logging.debug("%10s: %.4e %s"%(param,osc_param_dict[param],unit))

    logging.info("Getting osc prob maps")
    osc_flux_maps = get_osc_flux(flux_maps,args.deltam21,args.deltam31,args.theta12,
                                 args.theta13,args.theta23,args.deltacp)
    
    
    logging.info("Saving osc prob maps to file: %s",outfile)
    osc_flux_maps['params'] = osc_param_dict
    to_json(osc_flux_maps, outfile)
    
    
