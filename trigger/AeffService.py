#
# Creates the effective areas from a simfile, then returns them
# re-scaled as desired.
#
# author: Timothy C. Arlen
# 
# date:   April 8, 2014
#

import logging
import numpy as np
from utils.utils import get_bin_centers
import h5py
import os,sys

class AeffServiceMC:
    '''
    Takes the weighted effective area files, and creates a dictionary
    of the 2D effective area in terms of energy and coszen, for each
    flavor (nue,nue_bar,numu,...) and interaction type (CC, NC)
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

        self.aeff_dict = {}
        logging.info("Creating effective area dict...")
        for flavor in ['nue','nue_bar','numu','numu_bar','nutau','nutau_bar']:
            flavor_dict = {}
            logging.debug("Working on %s effective areas"%flavor)
            for int_type in ['cc','nc']:
                weighted_aeff = np.array(fh[flavor+'/'+int_type+'/weighted_aeff'])
                true_energy = np.array(fh[flavor+'/'+int_type+'/true_energy'])
                true_coszen = np.array(fh[flavor+'/'+int_type+'/true_coszen'])
                
                bins = (self.ebins,self.czbins)
                aeff_hist = np.histogram2d(true_energy,true_coszen,weights=weighted_aeff,
                                           bins=bins)[0]
                # Divide by bin width:
                ecen = get_bin_centers(self.ebins)
                czcen = get_bin_centers(self.czbins)
                for ie,egy in enumerate(ecen):
                    ebin_width = (self.ebins[ie+1] - self.ebins[ie])
                    for icz,cz in enumerate(czcen):
                        czbin_width = (self.czbins[icz+1] - self.czbins[icz])
                        aeff_hist[ie][icz] /= (ebin_width*czbin_width*2.0*np.pi)
                
                flavor_dict[int_type] = aeff_hist
            self.aeff_dict[flavor] = flavor_dict
            
        return
    
    def get_aeff(self,nu_xsec_scale,nubar_xsec_scale):
        '''
        Returns the effective area dictionary, scaled by the
        appropriate factors.
        '''
        
        # scale each aeff by these factors...
        

        return self.aeff_dict
        
        
