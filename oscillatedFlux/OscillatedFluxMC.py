#! /usr/bin/env python
#
# stage1FullMC.py
#
# This module is the implementation of the stage1 analysis using the
# full Monte Carlo simulations. The main purpose of stage1 is to
# combine the atmospheric flux model with the oscillation probability
# map to create the "oscillated Flux maps", which becomes the input of
# stage2.
#
# If desired, this will create a .json output file with the results of
# the current stage of processing.
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
from utils.utils import set_verbosity,get_smoothed_map,get_osc_probLT_dict_hdf5
from utils.json import from_json, to_json
from flux.HondaFlux import get_flux_maps,HondaFlux

# Until python2.6, default json is very slow.
try: 
    import simplejson as json
except ImportError, e:
    import json
    
def get_osc_prob_maps(ebins, czbins, deltam21, deltam31,theta12,theta13,
                      theta23,deltacp):
    """
    This function returns an oscillation probability map dictionary calculated 
    at the input parameters:
      deltam21,deltam31,theta12,theta13,theta23,deltacp
    for flavor_from to flavor_to, with the binning of ebins,czbins.
    The dictionary is formatted as:
       'nue_maps': {'nue':map,'numu':map,'nutau':map},
       'numu_maps': {...}
       'nue_bar_maps': {...}
       'numu_bar_maps': {...}
    """
    
    ########################################################################
    ### TRUE ALGORITHM WHEN WE DECIDE ON HOW TO HANDLE OSC PROB DATA     ###
    # step 1: identify where the data is located: on disk or on server?    #
    # step 2: downsample these maps if not already done, for ebins, czbins #
    # step 3: do interpolation in oscillation parameters to arrive at the  #
    #         maps for (deltam21,deltam31,theta12,theta13,theta23,deltacp) #
    # return dictionary of smoothed, interpolated map.                     #
    ########################################################################

    ### TEMPORARY SOLUTION:
    # for now, I will grab the data from the local directory:
    import os
    maps_dir = os.getenv('PISA')+'/resources/oscProbMaps/ebins500_czbins500/'
    # for now, no interpolation
    filename = maps_dir+'oscProbLT_dm31_0.246_th23_38.645.hdf5' if deltam31 > 0.0 else maps_dir+'oscProbLT_dm31_-0.238_th23_38.645.hdf5'
    logging.info("Loading file: %s"%filename)
    osc_probLT_dict = get_osc_probLT_dict_hdf5(filename)
    ebinsLT = osc_probLT_dict['ebins']
    czbinsLT = osc_probLT_dict['czbins']

    # do smoothing
    smoothed_maps = {}
    smoothed_maps['ebins'] = ebins
    smoothed_maps['czbins'] = czbins
    for from_nu in ['nue','numu','nue_bar','numu_bar']:
        path_base = from_nu+'_maps'
        to_maps = {}
        to_nu_list = ['nue_bar','numu_bar','nutau_bar'] if 'bar' in from_nu else ['nue','numu','nutau']
        for to_nu in to_nu_list:
            to_maps[to_nu] = get_smoothed_map(osc_probLT_dict[from_nu+'_maps'][to_nu],
                                              ebinsLT,czbinsLT,ebins,czbins)
        smoothed_maps[from_nu+'_maps'] = to_maps

    return smoothed_maps
    
def get_osc_flux(flux_maps,deltam21,deltam31,theta12,theta13, theta23,
                 deltacp,**params):
    """
    Primary module function. Produces a map for probability to oscillate 
    from each of nue, nue_bar, numu, numu_bar into their respective other
    three flavours (12 maps total). 
    Inputs:
      -- osc_param_dict: dictionary for deltam31, deltam21, theta32, theta31, 
           theta21, deltacp
      -- earth_model: string corresponding to the earth model desired
      -- atm_flux_file: string filename of the atm_flux_file.json to use with
           the oscillation parameters.
     NOTE: Following two not implemented, but will be when needed...
      -- atm_flux_scale: scaling of atm_flux_file (to simulate uncertainty in
           atmospheric flux normalization)
      -- atm_flux_dIndex: spectral index variation (to simulatd uncertainty in
           atmospheric spectral index/shape)
    """
    
    nue_maps = flux_maps['nue']['map']
    numu_maps = flux_maps['numu']['map']
    nue_bar_maps = flux_maps['nue_bar']['map']
    numu_bar_maps = flux_maps['numu_bar']['map']

    ebins = flux_maps['nue']['ebins']
    czbins = flux_maps['nue']['czbins']
    
    # Get the dictionary of smoothed, interpolated osc prob maps:
    osc_prob_maps = get_osc_prob_maps(ebins,czbins,deltam21,deltam31,
                                      theta12,theta13,theta23,deltacp)
    
    osc_flux_maps = {}
    for nu in ['nue','numu','nutau']:
        osc_flux_maps[nu] = {'ebins':ebins,
                             'czbins':czbins,
                             'map':(nue_maps*osc_prob_maps['nue_maps'][nu] +
                                    numu_maps*osc_prob_maps['numu_maps'][nu])}

    for nu in ['nue_bar','numu_bar','nutau_bar']:
    #for nubar in ['nuebar','numubar','nutaubar']:
        osc_flux_maps[nu] = {'ebins':ebins,
                             'czbins':czbins,
                             'map':(nue_bar_maps*osc_prob_maps['nue_bar_maps'][nu] +
                                    numu_bar_maps*osc_prob_maps['numu_bar_maps'][nu])}
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
    parser.add_argument('-o', '--outfile', dest='outfile', metavar='FILE', type=str,
                        action='store',help='file to store the output')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='set verbosity level')
    args = parser.parse_args()

    #Set verbosity level
    set_verbosity(args.verbose)


    #Check that we got all the arguments 
    try:
        params = args.settings['params']
        ebins = args.settings['ebins']
        czbins = args.settings['czbins']
    except KeyError, k:
        logging.error("Settings are incomplete - missing %s!"%k)
        parser.print_help()
        sys.exit(1)
        
    
    logging.info("Getting oscillated flux...")
        
    if args.flux_file is None:
        logging.info("Loading flux maps from settings file.")
        flux_model = HondaFlux(**params)
        flux_maps = get_flux_maps(flux_model,ebins,czbins,**params)
    else:
        logging.info("Loading flux maps from %s"%args.flux_file)
        flux_maps = args.flux_file

    # define osc params:
    deltam21 = params['deltam21']
    deltam31 = params['deltam31_nh'][10]
    #deltam31 = params['deltam31_ih'][10]
    theta12  = params['theta12']
    theta13  = params['theta13']
    theta23  = params['theta23'][6]
    deltacp  = params['deltacp']

    logging.debug("  deltam21:    %s eV^2"%deltam21)
    logging.debug("  deltam31:    %s eV^2"%deltam31)
    logging.debug("  theta12:     %s deg"%theta12)
    logging.debug("  theta13:     %s deg"%theta13)
    logging.debug("  theta23:     %s deg"%theta23)
    logging.debug("  deltacp:     %s rad"%deltacp)

    
    osc_flux_maps = get_osc_flux(flux_maps,deltam21,deltam31,theta12,theta13,
                                 theta23,deltacp)#,**params)
    
    outfilename = args.outfile if args.outfile else "osc_flux.json"
    # Does this work?
    osc_flux_maps['params'] = params
    to_json(osc_flux_maps, outfilename)
    
