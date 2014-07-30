#! /usr/bin/env python
#
# OscillationProbs.py
#
# This module calculates the bare neutrino oscillation probabilities 
# for all neutrino flavours and writes them to disk. 
# It is not part of the main pisa simulation chain!
#
# author: Lukas Schulte
#         lschulte@physik.uni-bon.de
#
# date:   July 28, 2014
#

import os,sys
import numpy as np
import logging
from argparse import ArgumentParser, RawTextHelpFormatter
from pisa.utils.utils import set_verbosity, check_binning, get_binning
from pisa.utils.jsons import from_json, to_json, json_string
from pisa.utils.proc import report_params, get_params, add_params
from pisa.oscillations.OscillationService import OscillationService
from pisa.oscillations.Prob3OscillationService import Prob3OscillationService

# Until python2.6, default json is very slow.
try: 
    import simplejson as json
except ImportError, e:
    import json

if __name__ == '__main__':
    
    #Only show errors while parsing
    set_verbosity(0)

    # parser
    parser = ArgumentParser(description='Takes the oscillation parameters '
                            'as input and writes out a set of osc flux maps',
                            formatter_class=RawTextHelpFormatter)    
    parser.add_argument('--ebins', metavar='[1.0,2.0,...]', type=json_string,
        help= '''Edges of the energy bins in units of GeV, default is '''
              '''80 edges (79 bins) from 1.0 to 80 GeV in logarithmic spacing.''',
        default = np.logspace(np.log10(1.),np.log10(80),80))
    parser.add_argument('--czbins', metavar='[-1.0,-0.8.,...]', type=json_string,
        help= '''Edges of the cos(zenith) bins, default is '''
              '''21 edges (20 bins) from -1. (upward) to 0. horizontal in linear spacing.''',
        default = np.linspace(-1.,0.,21))
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
    parser.add_argument('--code',type=str,choices = ['prob3','table'], default='prob3',
                        help='''Oscillation code to use, one of [table,prob3],
                        (default=prob3)''')
    parser.add_argument('--detector_depth', type=float, default=2.0,
                        help='''Depth of detector [km]''')
    parser.add_argument('--prop_height', type=float, default=20.0,
                        help='''Neutrino production height [km]''')
    parser.add_argument('-o', '--outfile', dest='outfile', metavar='FILE', type=str,
                        action='store',default="osc_probs.json",
                        help='file to store the output')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='set verbosity level')
    args = parser.parse_args()

    #Set verbosity level
    set_verbosity(args.verbose)

    #Initialize an oscillation service
    if args.code=='prob3':
      osc_service = Prob3OscillationService(**vars(args))
    else:
      osc_service = OscillationService(args.ebins,args.czbins)

    logging.info("Getting osc prob maps")
    osc_prob_maps = osc_service.get_osc_probLT_dict(**vars(args))
    
    #Write out
    logging.info("Saving output to: %s",args.outfile)
    to_json(osc_prob_maps, args.outfile)
