#! /usr/bin/env python
#
# Flux.py
#
# Sample the atmospheric neutrino flux from a FluxService with a given binning
# in cos(zenith) and energy.
#
# If desired, this will create a .json output file with the results of
# the current stage of processing.
#
# author: Sebastian Boeser
#         sboeser@physik.uni-bonn.de
#
# date:   2014-01-27

import os
import sys
import logging
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from utils.utils import set_verbosity
from utils.jsons import from_json, to_json, json_string
from utils.proc import report_params, get_params, add_params
from HondaFluxService import HondaFluxService, primaries

def get_flux_maps(flux_service, ebins, czbins, **params):
    '''Get a set of flux maps for the different primaries'''

    #Be verbose on input
    params = get_params()
    report_params(params, units = [])

    #Initialize return dict
    maps = {'params': params}

    for prim in primaries:

        #Get the flux for this primary
        maps[prim] = {'ebins': ebins,
                      'czbins': czbins,
                      'map': flux_service.get_flux(ebins,czbins,prim)}
        #be a bit verbose
        logging.debug("Total flux of %s is %u [s^-1 m^-2]"%
                                (prim,maps[prim]['map'].sum()))

    #return this map
    return maps


if __name__ == '__main__':

    #Only show errors while parsing
    set_verbosity(0)

    # parser
    parser = ArgumentParser(description='Take a settings file '
        'as input and write out a set of flux maps',
        formatter_class=ArgumentDefaultsHelpFormatter
        )

    parser.add_argument('--ebins', metavar='[1.0,2.0,...]', type=json_string,
        help= '''Edges of the energy bins in units of GeV, default is '''
              '''80 edges (79 bins) from 1.0 to 80 GeV in logarithmic spacing.''',
        default = np.logspace(np.log10(1.),np.log10(80),80))

    parser.add_argument('--czbins', metavar='[-1.0,-0.8.,...]', type=json_string,
        help= '''Edges of the cos(zenith) bins, default is '''
              '''21 edges (20 bins) from -1. (upward) to 0. horizontal in linear spacing.''',
        default = np.linspace(-1.,0.,21))
    
    parser.add_argument('--flux_file', metavar='FILE', type=str,
        help= '''Input flux file in Honda format. ''',
        default = os.path.expandvars('$PISA/resources/flux/frj-solmin-mountain-aa.d'))
    
    parser.add_argument('-o', '--outfile', dest='outfile', metavar='FILE', type=str, action='store',
                        help='file to store the output', default='flux.json')
    
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='set verbosity level')
    
    args = parser.parse_args()

    #Set verbosity level
    set_verbosity(args.verbose)

    logging.debug("Using %u bins in energy from %.2f to %.2f GeV"%
                                (len(args.ebins)-1,args.ebins[0],args.ebins[-1]))
    logging.debug("Using %u bins in cos(zenith) from %.2f to %.2f"%
                                (len(args.czbins)-1,args.czbins[0],args.czbins[-1]))

    #Instantiate a flux model
    flux_model = HondaFluxService(args.flux_file)
    
    #get the flux 
    flux_maps = get_flux_maps(flux_model,args.ebins,args.czbins)

    #write out to a file
    logging.info("Saving output to: %s"%args.outfile)
    to_json(flux_maps, args.outfile)
