#! /usr/bin/env python
#
# stage0FullMC.py
#
#

## IMPORTS ##
import os,sys
import numpy as np
import logging
from argparse import ArgumentParser
from datetime import datetime

# Until python2.6, default json is very slow.
try: 
    import simplejson as json
except ImportError, e:
    import json
    
    
def get_atm_flux(atm_flux_file="AtmFluxModel_Honda2006.json",atm_flux_scale=1.0,
                 atm_flux_dIndex=0.0):
    
    resources_dir = os.getenv("PISA")+"/resources/"
    atm_flux_file = resources_dir+atm_flux_file
    
    logging.debug("loading file: %s"%atm_flux_file)
    atm_flux_dict = json.load(open(atm_flux_file))
    
    # Apply flux scaling to each map:
    map_names = ["maps_nue","maps_nue_bar","maps_numu","maps_numu_bar"]
    for m in map_names:
        hist = atm_flux_dict[m]['map']  # List of Lists.
        hist_scaled = []
        for row in range(len(hist[:][0])):
            new_row = [x*atm_flux_scale for x in hist[row]]
            hist_scaled.append(new_row)
        atm_flux_dict[m]['map'] = hist_scaled

    # Apply flux spectral index change to each map:
    

    
        
    return atm_flux_dict


if __name__ == '__main__':

    parser = ArgumentParser(description='''Takes a set of atmospheric flux parameters and generates atmospheric flux maps from them.''')
    parser.add_argument('--infile',type=str,default='AtmFluxModel_Honda2006.json',
                        help='atmospheric flux model file (.json), from the $PISA/resources directory.')
    parser.add_argument('--scale',type=float,default=1.0,
                        help='flux scaling to apply to the entire flux model.')
    parser.add_argument('--dindex',type=float,default=0.0,
                  help='change in spectral index from atmospheric flux model.')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='set verbosity level')
    args = parser.parse_args()

    # Set verbosity level
    levels = {0:logging.ERROR,
              1:logging.INFO,
              2:logging.DEBUG}
    logging.basicConfig(format='[%(levelname)8s] %(message)s')
    logging.root.setLevel(levels[min(2,args.verbose)])

    starttime = datetime.now()
    logging.info("Getting atm_flux_maps")
    atm_flux_maps = get_atm_flux(args.infile,args.scale,args.dindex)
    logging.info("Finished! this took %s."%(datetime.now()-starttime))
    print "Done."
    

