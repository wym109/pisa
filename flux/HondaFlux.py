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
from utils.json import from_json, to_json, json_string

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
    parser.add_argument('--ebins', metavar='[1.0,2.0,...]', type=json_string,
        help= '''Edges of the energy bins in units of GeV, default is '''
              '''80 edges (79 bins) from 1.0 to 80 GeV in logarithmic spacing.''',
        default = np.logspace(np.log10(1.),np.log10(80),80))
    parser.add_argument('--czbins', metavar='[-1.,-0.8.,...]', type=json_string,
        help= '''Edges of the cos(zenith) bins, default is '''
              '''21 edges (20 bins) from -1. (upward) to 0. horizontal in linear spacing.''',
        default = np.linspace(-1.,0.,21))
    parser.add_argument('--flux_file', metavar='FILE', type=str,
        help= '''Input flux file in Honda format. '''
              '''Default is \'resources/flux/frj-solmin-mountain-aa.d\' ''',
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
    flux_model = HondaFlux(args.flux_file)
    
    #get the flux 
    flux_maps = get_flux_maps(flux_model,args.ebins,args.czbins)

    #Store parameters along with flux_maps (none so far)
    flux_maps['params'] = {}

    #write out to a file
    to_json(flux_maps, args.outfile)
