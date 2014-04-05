#! /usr/bin/env python
#
# HondaFlux.py
#
# Load a flux table as provided by Honda (for now only able to use
# azimuth-averaged data) and read it out in map.
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
from argparse import ArgumentParser, RawTextHelpFormatter
from scipy.interpolate import bisplrep, bisplev
from utils.utils import get_bin_centers, get_bin_sizes, set_verbosity
from utils.json import from_json, to_json

#Global definition of primaries for which there is a neutrino flux
primaries = ['numu', 'numu_bar', 'nue', 'nue_bar']

class HondaFlux():
    '''Load a neutrino flux from Honda-styles flux tables in
       units of [GeV^-1 m^-2 s^-1 sr^-1] and
       return a 2D spline interpolated function per flavour.
       For now only supports azimuth-averaged input files.
    '''
    
    def __init__(self, tables, smooth=0.05, **params):
        logging.info("Loading atmospheric flux table %s" %tables)
        
        #Load the data table
        table = np.loadtxt(os.path.expandvars(tables)).T

        #columns in Honda files are in the same order 
        cols = ['energy']+primaries 

        flux_dict = dict(zip(cols, table))
        for key in flux_dict.iterkeys():
            
            #There are 20 lines per zenith range
            flux_dict[key] = np.array(np.split(flux_dict[key], 20))
            if not key=='energy':
                flux_dict[key] = flux_dict[key].T
    
        #Set the zenith and energy range 
        flux_dict['energy'] = flux_dict['energy'][0]
        flux_dict['coszen'] = np.linspace(0.95, -0.95, 20)
    
        #Now get a spline representation of the flux table.
        logging.debug('Make spline representation of flux')
        # do this in log of energy and log of flux (more stable)
        logE, C = np.meshgrid(np.log10(flux_dict['energy']), flux_dict['coszen'])

        self.spline_dict = {}
        for nutype in primaries:
            #Get the logarithmic flux
            log_flux = np.log10(flux_dict[nutype]).T
            #Get a spline representation
            spline =  bisplrep(logE, C, log_flux, s=smooth)
            #and store
            self.spline_dict[nutype] = spline

    def get_flux(self, ebins, czbins, prim):
        '''Get the flux in units [m^-2 s^-1] for the given
           bin edges in energy and cos(zenith) and the primary.'''
        
        #Evaluate the flux at the bin centers
        evals = get_bin_centers(ebins)
        czvals = get_bin_centers(czbins)
    
        # Get the spline interpolation, which is in
        # log(flux) as function of log(E), cos(zenith)
        return_table = bisplev(np.log10(evals), czvals, self.spline_dict[prim])
        return_table = np.power(10., return_table).T
    
        #Flux is given per sr and GeV, so we need to multiply
        #by bin width in both dimensions
        #Get the bin size in both dimensions
        ebin_sizes = get_bin_sizes(ebins)
        czbin_sizes = 2.*np.pi*get_bin_sizes(czbins)
        bin_sizes = np.meshgrid(ebin_sizes, czbin_sizes)
    
        return_table *= np.abs(bin_sizes[0]*bin_sizes[1])
    
        return return_table.T

def get_flux_maps(flux, ebins, czbins, **params):
    '''Get a set of flux maps for the different primaries'''

    maps = {}
    for prim in primaries:

        #Get the flux for this primary
        maps[prim] = {'ebins': ebins,
                      'czbins': czbins,
                      'map': flux.get_flux(ebins,czbins,prim)}
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
        formatter_class=RawTextHelpFormatter)
    parser.add_argument('settings', metavar='SETTINGS', type=from_json,
        help='''JSON file with the input parameters:
         { "params": { "tables" : path/to/tablefile.d }
           "ebins" : [1.,2.,3. ...]
           "czbins" : [-1.0,-0.9,-0.8,...]}''')
    parser.add_argument('-o', '--outfile', dest='outfile', metavar='FILE', type=str, action='store',
                        help='file to store the output', default='flux.json')
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

    logging.debug("Using %u bins in energy from %.2f to %.2f GeV"%
                                (len(ebins)-1,ebins[0],ebins[-1]))
    logging.debug("Using %u bins in cos(zenith) from %.2f to %.2f"%
                                (len(czbins)-1,czbins[0],czbins[-1]))

    #Instantiate a flux model
    flux_model = HondaFlux(**params)
    
    #get the flux 
    flux_maps = get_flux_maps(flux_model,ebins,czbins,**params)

    #Store parameters along with flux_maps
    flux_maps['params'] = params

    #write out to a file
    to_json(flux_maps, args.outfile)
