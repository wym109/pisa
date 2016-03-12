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


import itertools as itertools

import numpy as np

from pisa.utils.log import logging
from pisa.utils import utils
from pisa.utils import fileio
from pisa.utils.proc import get_params, add_params, report_params
from pisa.utils.utils import get_binning


class RecoServiceBase(object):
    """Base class for reconstruction services, handles the actual smearing of
    events from the reco kernels. Kernel generation has to be implemented in
    the derived classes.

    Parameters
    ----------
    ebins, czbins : array_like
        Energy and coszen bin edges
    """
    def __init__(self, ebins, czbins, **kwargs):
        logging.debug('Instantiating %s' % self.__class__.__name__)
        self.ebins = np.squeeze(ebins)
        self.czbins = np.squeeze(czbins)
        self.true_event_maps = None
        self.e_reco_scale = None
        self.cz_reco_scale = None

    def get_reco_maps(self, true_event_maps, e_reco_scale=None,
                      cz_reco_scale=None, **kwargs):
        """Primary function for this stage, which returns the reconstructed
        event rate maps from the true event rate maps. The returned maps will
        be in the form of a dictionary with parameters:
            {'nue_cc':{'ebins':ebins,'czbins':czbins,'map':map},
             'numu_cc':{...},
             'nutau_cc':{...},
             'nuall_nc':{...}}
        Note that in this function, the nu<x> is now combined with nu_bar<x>.
        """
        # Be verbose on input
        params = get_params()
        report_params(params, units = ['', ''])

        # Initialize return dict
        reco_maps = {'params': add_params(params, true_event_maps['params'])}

        # Check binning
        ebins, czbins = get_binning(true_event_maps)

        # Retrieve all reconstruction kernels
        reco_kernel_dict = self.get_reco_kernels(e_reco_scale=e_reco_scale,
                                                 cz_reco_scale=cz_reco_scale,
                                                 **kwargs)

        # DEBUG / HACK to store the computed kernels to a file
        #reco_service.store_kernels('reco_kernels.hdf5', fmt='hdf5')

        flavours = ['nue', 'numu', 'nutau']
        int_types = ['cc', 'nc']

        # Do smearing again, without loops
        flavors = ['nue', 'numu', 'nutau']
        all_int_types = ['cc', 'nc']
        n_ebins = len(ebins)-1
        n_czbins = len(czbins)-1
        for baseflavor, int_type in itertools.product(flavors, all_int_types):
            logging.info("Getting reco event rates for %s %s" % (baseflavor,
                                                                 int_type))
            reco_event_rate = np.zeros((n_ebins, n_czbins), dtype=np.float64)
            for mID in ['', '_bar']:
                flavor = baseflavor + mID
                true_event_rate = true_event_maps[flavor][int_type]['map']
                kernels = reco_kernel_dict[flavor][int_type]
                r0 = np.tensordot(true_event_rate, kernels, axes=([0,1],[0,1]))
                reco_event_rate += r0
            reco_maps[baseflavor+'_'+int_type] = {'map': reco_event_rate,
                                                  'ebins': ebins,
                                                  'czbins': czbins}
            msg = "after RECO: counts for (%s + %s) %s: %.2f" \
                % (baseflavor, baseflavor+'_bar', int_type, np.sum(reco_event_rate))
            logging.debug(msg)

        # Finally sum up all the NC contributions
        logging.info("Summing up rates for all nc events")
        reco_event_rate = np.sum(
            [reco_maps.pop(key)['map'] for key in reco_maps.keys()
             if key.endswith('_nc')], axis=0
        )
        reco_maps['nuall_nc'] = {'map':reco_event_rate,
                                 'ebins':ebins,
                                 'czbins':czbins}
        logging.debug("Total counts for nuall nc: %.2f" % np.sum(reco_event_rate))

        return reco_maps

    def get_reco_kernels(self, **kwargs):
        """
        Wrapper around _get_reco_kernels() that is to be used from outside,
        ensures that reco kernels are in correct shape and normalized
        """
        kernels = self._get_reco_kernels(**kwargs)
        if kernels is None:
            raise ValueError('No kernels defined to get')
        assert self.check_kernels(kernels)
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

