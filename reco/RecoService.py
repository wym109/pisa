#
# Creates the pdfs of the reconstructed energy and coszen from the
# true parameters. Provides reco event rate maps using these pdfs.
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# date:   April 9, 2014
#

import logging
import numpy as np
import h5py
import os,sys

class RecoServiceMC:
    '''
    From the simulation file, creates 4D histograms of
    [true_energy][true_coszen][reco_energy][reco_coszen] which act as
    2D pdfs for the probability that an event with (true_energy,
    true_coszen) will be reconstructed as (reco_energy,reco_coszen).

    From these histograms, and the true event rate maps, calculates
    the reconstructed even rate templates.
    '''
    def __init__(self,ebins,czbins,simfile=None):
        self.ebins = ebins
        self.czbins = czbins
        
        logging.info('Opening file: %s'%(simfile))
        simfile = os.path.expandvars(simfile)
        try:
            fh = h5py.File(simfile,'r')
        except IOError,e:
            logging.error("Unable to open simfile %s"%simfile)
            logging.error(e)
            sys.exit(1)
            
        # Create the 4D pdfs...
        
        return
    
    def get_reco_maps(self,true_event_maps=None):
        '''
        This takes the true_event_maps and applies the pdf in every
        bin to create the corresponding reco_maps...
        '''
        
        reco_maps = {}
        
        return reco_maps
        
