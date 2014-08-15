#
# Base class for reconstruction services, handles the actual smearing 
# of events from the reco kernels. Kernel generation has to be implemented
# in the derived classes.
# 
#
# author: Lukas Schulte
#         schulte@physik.uni-bonn.de
#
# date:   August 15, 2014
#

import sys, os
import logging

import numpy as np
from itertools import product



class RecoServiceBase:
    """
    Base class for reconstruction services, handles the actual smearing 
    of events from the reco kernels. Kernel generation has to be implemented
    in the derived classes.
    """
    
    def __init__(self, ebins, czbins, **kwargs):
        """
        Parameters needed to instantiate any reconstruction service:
        * ebins: Energy bin edges
        * czbins: cos(zenith) bin edges
        If further member variables are needed, override this method.
        """
        logging.debug('Instantiating %s'%self.__class__.__name__)
        self.ebins = ebins
        self.czbins = czbins
        for ax in [self.ebins, self.czbins]:
            if (len(np.shape(ax)) != 1):
                raise IndexError('Axes must be 1d! '+str(np.shape(ax)))
        
        #Get kernels already now. Can be recalculated later, if needed.
        self.kernels = self.get_reco_kernels(**kwargs)
    
    
    def get_reco_kernels(self, **kwargs):
        """
        This method is called to construct the reco kernels, i.e. a 4D 
        histogram of true (1st and 2nd axis) vs. reconstructed (3rd and 
        4th axis) energy (1st and 3rd axis) and cos(zenith) (2nd and 4th 
        axis). It has to be implemented in the derived classes individually, 
        since the way the reco kernels are generated is the depends on
        the reco method. Normalization of the kernels is taken care of 
        elsewhere.
        """
        raise NotImplementedError('Method not implemented for %s'
                                    %self.__class__.__name__)
    
    
    def normalize_kernels(self):
        """
        Ensure that all reco kernels are normalized.
        """
        for channel in self.kernels:
            if channel in ['ebins', 'czbins']: continue
            k_shape = np.shape(self.kernels[channel])
            for true_bin in product(range(k_shape[0]), range(k_shape[1])):
                self.kernels[channel][true_bin] /= np.sum(self.kernels[channel][true_bin])
