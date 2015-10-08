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


import numpy as np

from pisa.utils.log import logging
from pisa.utils import utils
from pisa.utils import fileio


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
        logging.debug('Instantiating %s' % self.__class__.__name__)
        self.ebins = ebins
        self.czbins = czbins
        for ax in [self.ebins, self.czbins]:
            if len(np.shape(ax)) != 1:
                raise IndexError('Axes must be 1d! '+str(np.shape(ax)))

        # Get kernels already now. Can be recalculated later, if needed.
        self.kernels = self.get_reco_kernels(**kwargs)


    def get_reco_kernels(self, **kwargs):
        """
        Wrapper around _get_reco_kernels() that is to be used from outside,
        ensures that reco kernels are in correct shape and normalized
        """
        kernels = self._get_reco_kernels(**kwargs)
        if kernels is None:
            logging.warn("No kernels defined yet...")
            return kernels

        if self.check_kernels(kernels):
            return kernels


    def _get_reco_kernels(self, **kwargs):
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
                                  % self.__class__.__name__)


    def check_kernels(self, kernels):
        """Test whether the reco kernels have the correct shape."""
        # check axes
        logging.debug('Checking binning of reconstruction kernels')
        for kernel_axis, own_axis in [(kernels['ebins'], self.ebins),
                                      (kernels['czbins'], self.czbins)]:
            if not utils.is_equal_binning(kernel_axis, own_axis):
                raise ValueError("Binning of reconstruction kernel doesn't "
                                 "match the event maps!")

        # check shape of kernels
        logging.debug('Checking shape of reconstruction kernels')
        shape = (len(self.ebins)-1, len(self.czbins)-1,
                 len(self.ebins)-1, len(self.czbins)-1)
        for flavour in kernels:
            if flavour in ['ebins', 'czbins']:
                continue
            for interaction in kernels[flavour]:
                if not np.shape(kernels[flavour][interaction]) == shape:
                    raise IndexError(
                        'Reconstruction kernel for %s/%s has wrong shape: '
                        '%s, %s' %(flavour, interaction, str(shape),
                                   str(np.shape(kernels[flavour][interaction])))
                    )

        logging.info('Reconstruction kernels are sane')
        return True

    def store_kernels(self, filename, fmt=None):
        """Store reconstruction kernels to file"""
        fileio.to_file(self.kernels, filename, fmt=fmt)
