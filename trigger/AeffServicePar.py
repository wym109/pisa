#
# Creates effective areas by parameterizing the detector
# response. Effective areas are always 2D in coszen and energy
#
# author: Timothy C. Arlen
# 
# date:   June 9, 2014
#

import logging
import numpy as np
from utils.utils import get_bin_centers, get_bin_sizes
from utils.jsons import from_json
from scipy.interpolate import interp1d
import os,sys

class AeffServicePar:
    '''
    Inputs a .json file to the locations of the .dat files, and
    creates a dictionary of the 2D effective area in terms of energy
    and coszen, for each flavor (nue,nue_bar,numu,...) and interaction
    type (CC, NC)
    
    The final aeff dict for each flavor is in units of [m^2] in each
    energy/coszen bin.
    '''
    def __init__(self,ebins,czbins,aeff_files=None,aeff_cz_dep=None,**kwargs):
        '''
        aeff_files and aeff_czdep are dicts of the parameterized aeff
        and cz dep per flavor
        '''
        self.ebins = ebins
        self.czbins = czbins
        
        ## Load the info from .dat files into a dict...  
        ## Parametric approach treats all NC events the same
        aeff2d_nc = self.get_aeff_flavor('NC',aeff_files,aeff_cz_dep)
        aeff2d_nc_bar = self.get_aeff_flavor('NC_bar',aeff_files,aeff_cz_dep)
        
        self.aeff_dict = {}
        logging.info("Creating effective area parametric dict...")
        for flavor in ['nue','nue_bar','numu','numu_bar','nutau','nutau_bar']:
            flavor_dict = {}
            logging.debug("Working on %s effective areas"%flavor)

            aeff2d = self.get_aeff_flavor(flavor,aeff_files,aeff_cz_dep)

            flavor_dict['cc'] = aeff2d
            flavor_dict['nc'] = aeff2d_nc_bar if 'bar' in flavor else aeff2d_nc
            
            self.aeff_dict[flavor] = flavor_dict
                    
        return

    def get_aeff_flavor(self,flavor,aeff_files,aeff_cz_dep=None):
        '''
        Creates the 2d aeff file from the parameterized aeff
        vs. energy .dat file, an input to the parametric settings file.
        '''

        #aeff_file = settings['params']['aeff_files'][flavor]
        aeff_file = aeff_files[flavor]
        aeff_arr = np.loadtxt(os.path.expandvars(aeff_file)).T
        # interpolate
        aeff_func = interp1d(aeff_arr[0], aeff_arr[1], kind='linear',
                             bounds_error=False, fill_value=0)
        
        czcen = get_bin_centers(self.czbins)
        ecen = get_bin_centers(self.ebins)
        
        # Get 1D array interpolated values at bin centers, assume no cz dep
        aeff1d = aeff_func(ecen)
        
        # Make this into a 2D array:
        aeff2d = np.reshape(np.repeat(aeff1d, len(czcen)), (len(ecen), len(czcen)))

        # If no cz_dep, return as is, otherwise add it in:
        if aeff_cz_dep is not None:
            logging.debug("  Parameterizing aeff cz dependence...")
            # Now add cz-dependence, assuming nu and nu_bar has same dependence:
            cz_dep = eval(aeff_cz_dep[flavor.strip('_bar')])(czcen)
            # Normalize:
            cz_dep *= len(cz_dep)/np.sum(cz_dep)
            
            aeff2d = aeff2d*cz_dep

        return aeff2d
    
    def get_aeff(self,*kwargs):
        '''
        Returns the effective area dictionary
        '''

        return self.aeff_dict
        
        
