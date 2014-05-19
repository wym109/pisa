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
            
        # Create the 4D distribution kernels...
        self.kernel_dict = {}
        logging.info("Creating kernel dict...")
        for flavor in ['nue','nue_bar','numu','numu_bar','nutau','nutau_bar']:
            flavor_dict = {}
            logging.debug("Working on %s kernels"%flavor)
            for int_type in ['cc','nc']:
                true_energy = np.array(fh[flavor+'/'+int_type+'/true_energy'])
                true_coszen = np.array(fh[flavor+'/'+int_type+'/true_coszen'])
                reco_energy = np.array(fh[flavor+'/'+int_type+'/reco_energy'])
                reco_coszen = np.array(fh[flavor+'/'+int_type+'/reco_coszen'])

                # True binning, reco binning...
                bins = (self.ebins,self.czbins,self.ebins,self.czbins)
                data = (true_energy,true_coszen,reco_energy,reco_coszen)
                kernel,_ = np.histogramdd(data,bins=bins)
                flavor_dict[int_type] = kernel
            self.kernel_dict[flavor] = flavor_dict
            
        return

    def get_kernels(self,**kwargs):
        '''
        Returns the kernels as a dictionary
        '''

        return self.kernel_dict
    
        
