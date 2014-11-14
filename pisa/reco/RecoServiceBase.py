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

from pisa.utils.utils import is_equal_binning, get_binning
from pisa.utils.jsons import to_json
from pisa.utils.proc import report_params, get_params, add_params


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
        Test whether the reco kernels have the correct shape.
        """
        # check axes
        logging.debug('Checking binning of reconstruction kernels')
        for kernel_axis, own_axis in [(self.kernels['ebins'], self.ebins),
                                       (self.kernels['czbins'], self.czbins)]:
            if not is_equal_binning(kernel_axis, own_axis):
                raise ValueError("Binning of reconstruction kernel doesn't "
                                  "match the event maps!")

        # check shape of kernels
        logging.debug('Checking shape of reconstruction kernels')
        shape = (len(self.ebins)-1, len(self.czbins)-1,
                 len(self.ebins)-1, len(self.czbins)-1)
        for flavour in self.kernels:
            if flavour in ['ebins', 'czbins']: continue
            for interaction in self.kernels[flavour]:
                if not np.shape(self.kernels[flavour][interaction])==shape:
                    raise IndexError('Reconstruction kernel for %s/%s has wrong shape: '
                                      '%s, %s' %(flavour, interaction, str(shape),
                                      str(np.shape(self.kernels[flavour][interaction]))) )

        logging.info('Reconstruction kernels are sane')
        return True


    def normalize_kernels(self):
        """
        Ensure that all reco kernels are normalized.
        """
        logging.debug('Normalizing reconstruction kernels')
        for flavour in self.kernels:
            if flavour in ['ebins', 'czbins']: continue
            for interaction in self.kernels[flavour]:
                k_shape = np.shape(self.kernels[flavour][interaction])
                for true_bin in product(range(k_shape[0]), range(k_shape[1])):
                    kernel_sum = np.sum(self.kernels[flavour][interaction][true_bin])
                    if kernel_sum > 0.:
                        self.kernels[flavour][interaction][true_bin] /= kernel_sum
        return

    def store_kernels(self, filename):
        """
        Store reconstruction kernels in json format
        """
        to_json(self.kernels, filename)
        return

    def recalculate_kernels(self, **kwargs):
        """
        Re-calculate reconstruction kernels and do all necessary checks.
        If new kernels are corrupted, stick with the old ones.
        """
        logging.info('Re-calculating reconstruction kernels')
        old_kernels = self.kernels.copy()
        self.get_reco_kernels(**kwargs)
        try:
            self.check_kernels()
            self.normalize_kernels()
        except:
            logging.error('Failed to recalculate reconstruction kernels, '
                          'keeping old ones: ', exc_info=True)
            self.kernels = old_kernels
        return
