#! /usr/bin/env python
#
# Oscillation.py
#
# This module is the implementation of the physics oscillation step.
# In this step, oscillation probability maps of each neutrino flavor
# into the others are produced, for a given set of oscillation
# parameters. It is then multiplied by the corresponding flux map,
# producing oscillated flux maps for each flavor.
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# date:   Jan. 21, 2014
#


import os,sys
import numpy as np
from argparse import ArgumentParser, RawTextHelpFormatter
from pisa.utils.log import logging, profile, set_verbosity
from pisa.utils.utils import check_binning, get_binning
from pisa.utils.jsons import from_json, to_json
from pisa.utils.proc import report_params, get_params, add_params
from pisa.oscillations.Prob3OscillationService import Prob3OscillationService
from pisa.oscillations.NucraftOscillationService import NucraftOscillationService


def get_osc_flux(flux_maps,osc_service=None,deltam21=None,deltam31=None,
                 energy_scale=None, theta12=None,theta13=None,theta23=None,
                 deltacp=None,**kwargs):
    '''
    Obtain a map in energy and cos(zenith) of the oscillation probabilities from
    the OscillationService and compute the oscillated flux.
    Inputs:
      flux_maps - dictionary of atmospheric flux ['nue','numu','nue_bar','numu_bar']
      osc_service - a handle to an OscillationService
      others - oscillation parameters to compute oscillation probability maps from.
    '''

    #Be verbose on input
    params = get_params()
    report_params(params, units = ['rad','eV^2','eV^2','','rad','rad','rad'])

    #Initialize return dict
    osc_flux_maps = {'params': add_params(params,flux_maps['params'])}

    #Get oscillation probability map from service
    osc_prob_maps = osc_service.get_osc_prob_maps(deltam21=deltam21,
                                                  deltam31=deltam31,
                                                  theta12=theta12,
                                                  theta13=theta13,
                                                  theta23=theta23,
                                                  deltacp=deltacp,
                                                  energy_scale=energy_scale,
                                                  **kwargs)

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
    parser.add_argument('--deltacp',type=float,default=np.pi,
                        help='''deltaCP value to use [rad]''')
    parser.add_argument('--energy_scale',type=float,default=1.0,
                        help='''Energy off scaling due to mis-calibration.''')
    parser.add_argument('--code',type=str,choices = ['prob3','table','nucraft'], 
                        default='prob3',
                        help='''Oscillation code to use, one of 
                        [table, prob3, nucraft], (default=prob3)''')
    parser.add_argument('--oversample', type=int, default=10,
                        help='''oversampling factor for *both* energy and cos(zen); 
                        i.e. every 2D bin will be oversampled by the square of the 
                        factor (default=10)''')
    parser.add_argument('-o', '--outfile', dest='outfile', metavar='FILE', type=str,
                        action='store',default="osc_flux.json",
                        help='file to store the output')
    parser.add_argument('-v', '--verbose', action='count', default=None,
                        help='set verbosity level')
    args = parser.parse_args()

    #Set verbosity level
    set_verbosity(args.verbose)

    #Get binning
    ebins, czbins = check_binning(args.flux_maps)

    #Initialize an oscillation service
    if args.code=='prob3':
      osc_service = Prob3OscillationService(ebins,czbins)
    elif args.code=='nucraft':
      osc_service = NucraftOscillationService(ebins, czbins)
    else:
      osc_service = OscillationService(ebins,czbins)

    logging.info("Getting osc prob maps")
    osc_flux_maps = get_osc_flux(args.flux_maps, osc_service, 
                                 deltam21 = args.deltam21,
                                 deltam31 = args.deltam31,
                                 deltacp = args.deltacp,
                                 theta12 = args.theta12,
                                 theta13 = args.theta13,
                                 theta23 = args.theta23,
                                 oversample = args.oversample,
                                 energy_scale = args.energy_scale)
    
    #Write out
    logging.info("Saving output to: %s",args.outfile)
    to_json(osc_flux_maps, args.outfile)
