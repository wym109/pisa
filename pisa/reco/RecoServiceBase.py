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

from pisa.utils.utils import is_equal_binning


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
        self.check_kernels()
        self.normalize_kernels()
    
    
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
    
    
    def check_kernels(self):
        """
        Test whether the reco kernels have the correct shape and normalize them
        """
        # check axes
        for kernel_axis, own_axis in [(self.kernels['ebins'], self.ebins),
                                       (self.kernels['czbins'], self.czbins)]:
            if not is_equal_binning(kernel_axis, own_axis):
                raise ValueError("Binning of reconstruction kernel doesn't "
                                  "match the event maps!")
            else:
                pass
        # check shape of kernels
        shape = (len(self.ebins),len(self.czbins),len(self.ebins),len(self.czbins))
        for flavour in self.kernels:
            if flavour in ['ebins', 'czbins']: continue
            for interaction in self.kernels[flavour]:
                if not np.shape(self.kernels[flavour][interaction])==shape:
                    raise IndexError('Reconstruction kernel for %s/%s has wrong shape: '
                                      '%s, %s' %(channel, interaction, str(shape)
                                      str(np.shape(self.kernels[channel]))) )
                else:
                    pass
        # normalize
        self.normalize_kernels()
        return True
    
    
    def normalize_kernels(self):
        """
        Ensure that all reco kernels are normalized.
        """
        for flavour in self.kernels:
            if flavour in ['ebins', 'czbins']: continue
            for interaction in self.kernels[flavour]:
                k_shape = np.shape(self.kernels[flavour][interaction])
                for true_bin in product(range(k_shape[0]), range(k_shape[1])):
                    self.kernels[flavour][interaction][true_bin] \
                        /= np.sum(self.kernels[flavour][interaction][true_bin])
    
    
    def apply_reconstruction(self, true_map, channel):
        """
        Apply the reconstruction kernel for the specified channel to the
        provided event map.
        """
        # check axes
        for map_axis, own_axis in [(true_map['ebins'], self.ebins),
                                    (true_map['czbins'], self.czbins)]:
            if not is_equal_binning(map_axis, own_axis):
                raise ValueError("Binning of reconstruction kernel doesn't "
                                  "match the event maps!")
            else:
                pass
        # do smearing
        data = true_map['map']
        kernel = self.kernels[channel]
        result = np.zeros_like(data)
        for i,row in enumerate(data):
            for j,col in enumerate(row):
                result += data[i,j]*kernel[i,j]
        return result
