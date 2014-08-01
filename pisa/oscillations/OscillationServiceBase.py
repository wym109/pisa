#
# This is the base class all other oscillation services should be derived from
#
# author: Lukas Schulte <lschulte@physik.uni-bonn.de>
#
# date:   July 31, 2014
#

import logging
import numpy as np
from utils.utils import subbinning, get_smoothed_map, integer_rebin_map

class OscillationServiceBase:
    """
    Base class for all oscillation services.
    """
    
    def __init__(self, ebins, czbins):
        """
        Parameters needed to instantiate any oscillation service:
        * ebins: Energy bin edges
        * czbins: cos(zenith) bin edges
        If further member variables are needed, override this method.
        """
        logging.debug('Instantiating %s'%self.__class__.__name__)
        self.ebins = np.array(ebins)
        self.czbins = np.array(czbins)
        for ax in [self.ebins, self.czbins]:
            if (len(np.shape(ax)) != 1):
                raise IndexError('Axes must be 1d! '+str(np.shape(ax)))
    
    
    def get_osc_prob_maps(self, deltam21=None, deltam31=None, 
                          theta12=None, theta13=None, theta23=None, 
                          deltacp=None):
        """
        Returns an oscillation probability map dictionary calculated 
        at the values of the input parameters:
          deltam21,deltam31,theta12,theta13,theta23,deltacp
        for flavor_from to flavor_to, with the binning of ebins,czbins.
        The dictionary is formatted as:
          'nue_maps': {'nue':map,'numu':map,'nutau':map},
          'numu_maps': {...}
          'nue_bar_maps': {...}
          'numu_bar_maps': {...}
        NOTES: * expects all angles in [rad]
               * this method doesn't calculate the oscillation probabi-
                 lities itself, but calls get_osc_probLT_dict internally
        """
        osc_pars = locals().copy()
        osc_pars.pop('self')
        
        #Get the finely binned maps as implemented in the derived class
        logging.info('Retrieving finely binned maps')
        fine_maps = self.get_osc_probLT_dict(**osc_pars)
        
        logging.info("Smoothing fine maps...")
        start_time = datetime.now()
        smoothed_maps = {}
        smoothed_maps['ebins'] = self.ebins
        smoothed_maps['czbins'] = self.czbins

        rebin_info = subbinning([self.ebins, self.czbins], 
                          [fine_maps['ebins'], fine_maps['czbins']])
        if rebin_info:
            #Use fast numpy magic
            logging.debug('Coarse map is true submap of fine map, '
                          'using numpy array magic for smoothing.')
            def __smoothing_func(osc_map):
                return integer_rebin_map(osc_map, rebin_info)
        else:
            def __smoothing_func(osc_map):
                return get_smoothed_map(osc_map, 
                                         fine_maps['ebins'], 
                                         fine_maps['czbins'],
                                         self.ebins, self.czbins)
        
        for from_nu, tomap_dict in fine_maps.items():
            new_tomaps = {}
            for to_nu, tomap in tomap_dict.items():
                logging.debug("Getting smoothed map %s/%s"%(from_nu,to_nu))
                new_tomaps[to_nu] = __smoothing_func(tomap)
            smoothed_maps[from_nu] = new_tomaps
        
        logging.debug("Finshed smoothing maps. This took: %s"
                        %(datetime.now()-start_time))
        
        return smoothed_maps
    
    def get_osc_probLT_dict(self, deltam21=None, deltam31=None, 
                            theta12=None, theta13=None, theta23=None, 
                            deltacp=None):
        """
        This method is called by get_osc_prob_maps and should be 
        implemented in any derived class individually
        """
        raise NotImplementedError('Method not implemented for %s'
                                    %self.__class__.__name__)
