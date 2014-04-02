#! /usr/bin/env python
#
# OscillationMaps.py
#
# This module is the implementation of the physics oscillation step.
# The main purpose of this step is to produce oscillation probability
# maps of each neutrino flavor into the others, for a given set of
# oscillation parameters.
#
# If given a stage0 output flux file, it is read in and the
# oscillation probabilities are applied to it and an oscillated flux
# map is written out to a .json file. 
#
# If no stage0 input flux file is given, the oscillation probability
# maps themselves are written out to a .json file.
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
#from flux.HondaFlux import get_flux_maps,HondaFlux
from OscillationService import OscillationService

# Until python2.6, default json is very slow.
try: 
    import simplejson as json
except ImportError, e:
    import json
    
def get_osc_flux(flux_maps,osc_prob_maps):
    '''
    Uses osc_prob_maps to calculate the oscillated flux maps.
    Inputs:
      --flux_maps - dictionary of atmospheric flux ['nue','numu','nue_bar','numu_bar']
      --osc_prob_maps - dictionary of the osc prob maps
    '''
    
    ebins = osc_prob_maps['ebins']
    czbins = osc_prob_maps['czbins']
    
    if not np.alltrue([is_equal_binning(ebins,flux_maps[nu]['ebins']) for nu in ['nue','nue_bar','numu','numu_bar']]):
        raise Exception('Flux maps have different energy binning!')
    if not np.alltrue([is_equal_binning(czbins,flux_maps[nu]['czbins']) for nu in ['nue','nue_bar','numu','numu_bar']]):
        raise Exception('Flux maps have different coszen binning!')
    
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
    parser.add_argument('settings', metavar='SETTINGS', type=from_json,
                        help='''JSON mc_settings file with the input parameters:
    { "params": { "deltam31":[], "theta23":[],...},
      "ebins" : [1.,2.,3. ...]
      "czbins" : [-1.0,-0.9,-0.8,...]}''')
    parser.add_argument('--flux_file',metavar='FLUX',type=from_json,
                        help='''JSON atm flux input file with the following parameters:
    {"nue": {'czbins':[], 'ebins':[], 'map':[]},
     "numu": {...},
     "nue_bar": {...},
     "numu_bar":{...}}''')
    parser.add_argument('--ih',action='store_true',
                        help="Run with inverted hierarchy deltam31, otherwise NMH.")
    parser.add_argument('-o', '--outfile', dest='outfile', metavar='FILE', type=str,
                        action='store',help='file to store the output')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='set verbosity level')
    args = parser.parse_args()

    #Set verbosity level
    set_verbosity(args.verbose)


    #Check that we got all the arguments 
    try:
        osc = args.settings['osc']
        ebins = args.settings['ebins']
        czbins = args.settings['czbins']
    except KeyError, k:
        logging.error("Settings are incomplete - missing %s!"%k)
        parser.print_help()
        sys.exit(1)        

    logging.info("Getting best fit oscillation params from settings file...")
    op_dict = {}
    for key in osc.keys():
        value = osc[key]['best']
        if 'deltam31' in key:
            if args.ih:
                if '_ih' in key: key = 'deltam31'
                else: continue
            else:
                if '_nh' in key: key = 'deltam31'
                else: continue
        op_dict[key] = value

    # Check that we got all parameters that we needed...
    param_names = ['deltam21','deltam31','theta12','theta13','theta23','deltacp']
    units       = ['eV^2','eV^2','deg','deg','deg','rad']
    for param, unit in zip(param_names,units):
        logging.debug("%10s: %.4e %s"%(param,op_dict[param],unit))
        
    logging.info("Getting osc prob maps")
    osc_service = OscillationService(ebins,czbins,datadir=os.getenv('PISA')+'/resources/oscProbMaps/ebins500_czbins500/')
    
    osc_prob_maps = osc_service.get_osc_prob_maps(**op_dict)
    
    if args.flux_file is None:
        outfilename = args.outfile if args.outfile else "osc_prob_maps.json"
        logging.info("Saving osc prob maps to file: %s",outfilename)
        osc_prob_maps['params'] = op_dict
        to_json(osc_prob_maps,outfilename)
    else:
        logging.info("Loading flux maps from file.")
        flux_maps = args.flux_file
        
        osc_flux_maps = get_osc_flux(flux_maps,osc_prob_maps)

        outfilename = args.outfile if args.outfile else "osc_flux.json"
        logging.info("Saving osc prob maps to file: %s",outfilename)
        osc_flux_maps['params'] = op_dict
        to_json(osc_flux_maps, outfilename)
        
        
