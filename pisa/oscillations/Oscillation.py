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
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pisa.utils.log import logging, tprofile, set_verbosity
from pisa.utils.utils import check_binning, get_binning, Timer
from pisa.utils.jsons import from_json, to_json
from pisa.utils.proc import report_params, get_params, add_params
from pisa.oscillations.Prob3OscillationService import Prob3OscillationService
from pisa.oscillations.NucraftOscillationService import NucraftOscillationService
from pisa.oscillations.TableOscillationService import TableOscillationService
try:
    from pisa.oscillations.Prob3GPUOscillationService import Prob3GPUOscillationService
except:
    logging.info("NOT loading Prob3GPUOscillationService in Oscillation.py")

def get_osc_flux(flux_maps,osc_service=None,deltam21=None,deltam31=None,
                 energy_scale=None, theta12=None,theta13=None,theta23=None,
                 deltacp=None,YeI=None,YeO=None,YeM=None,**kwargs):
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

    report_params(params, units = ['','','','rad','eV^2','eV^2','','rad','rad','rad'])

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
                                                  YeI=YeI,YeO=YeO,YeM=YeM,
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
                            formatter_class=ArgumentDefaultsHelpFormatter)
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
    parser.add_argument('--earth-model',type=str,default='oscillations/PREM_12layer.dat',
                        dest='earth_model',
                        help='''Earth model data (density as function of radius)''')
    parser.add_argument('--energy-scale',type=float,default=1.0,
                        dest='energy_scale',
                        help='''Energy off scaling due to mis-calibration.''')
    parser.add_argument('--YeI',type=float,default=0.5,
                        help='''Ye (elec frac) in inner core.''')
    parser.add_argument('--YeO',type=float,default=0.5,
                        help='''Ye (elec frac) in outer core.''')
    parser.add_argument('--YeM',type=float,default=0.5,
                        help='''Ye (elec frac) in mantle.''')
    parser.add_argument('--code',type=str,choices = ['prob3','table','nucraft','gpu'],
                        default='prob3',
                        help='''Oscillation code to use''')
    parser.add_argument('--oversample_e', type=int, default=10,
                        help='''oversampling factor for energy;
                        i.e. every 2D bin will be oversampled by this factor in
                        each dimension''')
    parser.add_argument('--oversample_cz', type=int, default=10,
                        help='''oversampling factor for  cos(zen);
                        i.e. every 2D bin will be oversampled by this factor in
                        each dimension ''')
    parser.add_argument('--detector-depth', type=float, default=2.0,
                        dest='detector_depth',
                        help='''Detector depth in km''')
    parser.add_argument('--propagation-height', type=float, default=20.0,
                        dest='prop_height',
                        help='''Height in the atmosphere to begin propagation in km.
                        Prob3 default: 20.0 km
                        NuCraft default: 'sample' from a distribution''')
    parser.add_argument('--precision', type=float, default=5e-4,
                        dest='osc_precision',
                        help='''Requested precision for unitarity (NuCraft only)''')
    parser.add_argument('--tabledir', type=str, default='oscillations',
                        dest='datadir',
                        help='''Path to stored oscillation data (Tables only)''')
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
    iniargs = {'earth_model': args.earth_model,
               'detector_depth': args.detector_depth,
               'prop_height': args.prop_height,
               'osc_precision': args.osc_precision,
               'datadir': args.datadir}

    if args.code=='prob3':
      if iniargs['prop_height'] is None: iniargs['prop_height'] = 20.0
      osc_service = Prob3OscillationService(ebins, czbins, **iniargs)
    elif args.code=='nucraft':
      if iniargs['prop_height'] is None: iniargs['prop_height'] = 'sample'
      osc_service = NucraftOscillationService(ebins, czbins, **iniargs)
    elif args.code=='gpu':
        settings = vars(args)
        osc_service = Prob3GPUOscillationService(ebins, czbins, **settings)
    else:
      osc_service = TableOscillationService(ebins, czbins, **iniargs)

    logging.info("Getting osc prob maps")
    with Timer(verbose=False) as t:
        osc_flux_maps = get_osc_flux(args.flux_maps, osc_service,
                                     deltam21 = args.deltam21,
                                     deltam31 = args.deltam31,
                                     deltacp = args.deltacp,
                                     theta12 = args.theta12,
                                     theta13 = args.theta13,
                                     theta23 = args.theta23,
                                     oversample_e = args.oversample_e,
                                     oversample_cz = args.oversample_cz,
                                     energy_scale = args.energy_scale,
                                     YeI=args.YeI, YeO=args.YeO, YeM=args.YeM)
    print "       ==> elapsed time to get osc flux maps: %s sec"%t.secs

    #Write out
    logging.info("Saving output to: %s",args.outfile)
    to_json(osc_flux_maps, args.outfile)
