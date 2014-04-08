#! /usr/bin/env python
#
# EventCountsOscMC.py
#
# This module is the implementation of the stage2 analysis using the
# full Monte Carlo simulations. The main purpose of stage2 is to
# combine the atmospheric flux model with the oscillation probability 
# map to create the "oscillated Flux maps", which becomes the input of
# stage2-the reconstruction step.
# 
# If desired, this will create a .json output file with the results of
# the current stage of processing.
#
# author: Timothy C. Arlen
#
#         tca3@psu.edu
#
# date:   Jan. 21, 2014
#

import os,sys
import numpy as np
import logging
from argparse import ArgumentParser, RawTextHelpFormatter
from utils.utils import set_verbosity,is_equal_binning
from utils.json import from_json, to_json


def get_osc_counts(osc_flux_maps,sim_file=None,livetime=None,nu_xsec_scale=None,
                   nu_bar_xsec_scale=None,**kwargs):
    '''
    Main function for this module, which returns the counts maps for
    each flavor and interaction type, using true energy and zenith
    information. The content of each bin will be the simulated weight
    multiplied by the oscillation probability, so that 

    Returned dictionary will be of the form:
        {'nue':{'cc':map,'nc':map}, 
         'nue_bar':{'cc':map,'nc':map}, ...
         'nutau_bar':{'cc':map,'nc':map} 
         }
    '''

    # Verify consistent binning.
    ebins = osc_flux_maps['nue']['ebins']
    czbins = osc_flux_maps['nue']['czbins']
    flavours = ['nue','numu','nutau','nue_bar','numu_bar','nutau_bar']
    if not np.alltrue([is_equal_binning(ebins,osc_flux_maps[nu]['ebins']) for nu in flavours]):
        raise Exception('Osc flux maps have different energy binning!')
    if not np.alltrue([is_equal_binning(czbins,osc_flux_maps[nu]['czbins']) for nu in flavours]):
        raise Exception('Osc flux maps have different coszen binning!')

    osc_counts_maps = {}
    
    
    return osc_counts_maps


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
    parser.add_argument('simfile',type=str,help='''HDF5 File containing data from all flavours for a particular instumental geometry. Expects the file format to be:
      {
        'nue': {
           'cc': {
               'sim_weight': np.array,
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
    parser.add_argument('--nu_xsec',type=float,default=1.0,
                        help='''Overall uncertainty on nu xsec.''')
    parser.add_argument('--nubar_xsec',type=float,default=1.0,
                        help='''Overall uncertainty on nu_bar xsec.''')
    parser.add_argument('-o', '--outfile', dest='outfile', metavar='FILE', type=str,
                        action='store',default="evt_reco.json",
                        help='''file to store the output''')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='''set verbosity level''')
    args = parser.parse_args()

    #Set verbosity level
    set_verbosity(args.verbose)

    livetime = args.livetime
    nu_xsec_scale = args.nu_xsec
    nubar_xsec_scale = args.nubar_xsec

    for name,param in zip(["livetime","nu xs scale","nubar xs scale"],
                          [livetime,nu_xsec_scale,nubar_xsec_scale]):
        logging.debug("%14s: %s "%(name,param))

    logging.info("Getting oscillated flux...")    
    osc_flux_maps = args.osc_flux_file
    simfile = args.simfile

    evt_counts_maps = get_osc_counts(osc_flux_maps,simfile,livetime,
                                     nu_xsec_scale,nubar_xsec_scale)
    
    logging.info("Saving output to .json file...")
    to_json(evt_counts_maps,args.outfile)
    
    
