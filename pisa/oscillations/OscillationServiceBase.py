#
# This is the base class all other oscillation services should be derived from
#
# author: Lukas Schulte <lschulte@physik.uni-bonn.de>
#         Timothy C. Arlen tca3@psu.edu
#
# date:   July 31, 2014
#


import numpy as np

from pisa.utils.log import logging, tprofile
from pisa.utils.utils import get_smoothed_map, get_bin_centers, check_fine_binning, oversample_binning, Timer, get_binning
from pisa.utils.proc import report_params, get_params, add_params
from pisa.utils.utils import DictWithHash, hash_obj
from pisa.utils.cache import MemoryCache


class OscillationServiceBase(object):
    """Base class for all oscillation services.

    Parameters
    ----------
    ebins, czbins : array_like
        Energy and coszen bin edges
    """
    def __init__(self, ebins, czbins, cache_depth=2000):
        logging.trace('Instantiating %s' % self.__class__.__name__)
        self.ebins = np.array(ebins)
        self.czbins = np.array(czbins)
        for ax in [self.ebins, self.czbins]:
            if (len(np.shape(ax)) != 1):
                raise IndexError('Axes must be 1d! '+str(np.shape(ax)))

        # Instantiate caches
        self.cache_depth = cache_depth
        self.transform_cache = MemoryCache(cache_depth, is_lru=True)
        self.osc_prob_map_cache = MemoryCache(cache_depth, is_lru=True)
        self.result_cache = MemoryCache(cache_depth, is_lru=True)
        self.__smoothed_maps = DictWithHash()

    def get_osc_prob_maps(self, ebins=None, czbins=None, oversample_e=None,
                          oversample_cz=None, **kwargs):
        """
        Returns an oscillation probability map dictionary calculated
        at the values of the input parameters:
          deltam21, deltam31, theta12, theta13, theta23, deltacp
        for flavor_from to flavor_to, with the binning of ebins, czbins.
        The dictionary is formatted as:
          'nue_maps': {'nue':map, 'numu':map, 'nutau':map},
          'numu_maps': {...}
          'nue_bar_maps': {...}
          'numu_bar_maps': {...}

        Notes
        -----
          * expects all angles in [rad]
          * this method doesn't calculate the oscillation probabilities
            itself, but calls get_osc_probLT_dict internally, to get a
            high resolution map of the oscillation probs,
        """
        # Get the finely-binned maps as implemented in the derived class
        logging.info('Retrieving finely binned maps')
        with Timer(verbose=False) as t:
            # First initialize the fine binning if not explicitly given
            if not check_fine_binning(ebins, self.ebins):
                ebins = oversample_binning(self.ebins, oversample_e)
            if not check_fine_binning(czbins, self.czbins):
                czbins = oversample_binning(self.czbins, oversample_cz)
            ecen = get_bin_centers(ebins)
            czcen = get_bin_centers(czbins)

            # Call the method implemented in specialized services
            fine_maps = self.fill_osc_prob(ecen, czcen, **kwargs)

        logging.debug("       ==> elapsed time to get all fine maps: %s sec" %
                      t.secs)

        # Key on the variables that uniquely define what the output is going
        # to be
        cache_key = hash_obj((fine_maps.hash, self.ebins, self.czbins))
        try:
            self.__smoothed_maps = self.osc_prob_map_cache[cache_key]
            return self.__smoothed_maps
        except KeyError:
            pass
        #cache_key = 0

        #if not fine_maps.is_new:
        #    self.__smoothed_maps.is_new = False
        #    return self.__smoothed_maps #osc_prob_map_cache[cache_key]

        logging.info("Smoothing fine maps...")

        # Use a DictWithHash to inform the next stage of exactly what it's
        # getting, so it can more effectively implement a cache
        self.__smoothed_maps = DictWithHash()
        self.__smoothed_maps['ebins'] = self.ebins
        self.__smoothed_maps['czbins'] = self.czbins

        with Timer(verbose=False) as t:
            for from_nu, tomap_dict in fine_maps.iteritems():
                if 'vals' in from_nu:
                    continue
                new_tomaps = {}
                for to_nu, pvals in tomap_dict.iteritems():
                    logging.debug("Getting smoothed map %s/%s" % (from_nu,
                                                                  to_nu))
                    new_tomaps[to_nu] = get_smoothed_map(
                        pvals, fine_maps['evals'], fine_maps['czvals'],
                        self.ebins, self.czbins
                    )
                self.__smoothed_maps[from_nu] = new_tomaps
            self.__smoothed_maps.update_hash(cache_key)
            self.osc_prob_map_cache[cache_key] = self.__smoothed_maps
        tprofile.debug("       ==> elapsed time to smooth maps: %s sec" %
                       t.secs)

        return self.__smoothed_maps

    def fill_osc_prob(self, ecen, czcen, theta12, theta13, theta23, deltam21,
                      deltam31, deltacp, **kwargs):
        """This method is called by get_osc_probLT_dict and should be
        implemented in any derived class individually as here the actual
        oscillation code should be run.

        NOTE: Expects all angles to be in [rad], and all deltam to be in [eV^2]
        """
        raise NotImplementedError('Method not implemented for %s'
                                  % self.__class__.__name__)

    def get_osc_flux(self, flux_maps, deltam21, deltam31, energy_scale,
                     theta12, theta13, theta23, deltacp, YeI, YeO, YeM,
                     **kwargs):
        """Obtain a map in energy and cos(zenith) of the oscillation
        probabilities from the OscillationService and compute the oscillated
        flux.

        Parameters
        ----------
        flux_maps
            Dictionary of atmospheric flux with keys 'nue', 'numu', 'nue_bar',
            and 'numu_bar'
        **kwargs
            Oscillation parameters to compute oscillation probability maps
            from.
        """
        ## Be verbose on input
        #params = get_params()
        #report_params(params, units=['', '', '', 'rad', 'eV^2', 'eV^2', '',
        #                             'rad', 'rad', 'rad'])

        # Get oscillation probability map from service
        osc_prob_maps = self.get_osc_prob_maps(
            deltam21=deltam21, deltam31=deltam31,
            theta12=theta12, theta13=theta13, theta23=theta23,
            deltacp=deltacp,
            energy_scale=energy_scale,
            YeI=YeI, YeO=YeO, YeM=YeM,
            **kwargs
        )

        cache_key = hash_obj((flux_maps.hash, osc_prob_maps.hash))
        try:
            return self.result_cache[cache_key]
        except KeyError:
            pass
        #cache_key = 0
        #if not flux_maps.is_new and not osc_prob_maps.is_new:
        #    retval = self.result_cache[cache_key]
        #    retval.is_new = False
        #    return retval

        # 
        # Apply the oscillation transformation to the input flux...
        #

        # Initialize return dict
        osc_flux_maps = DictWithHash()
        osc_flux_maps['params'] = {}
        #osc_flux_maps['params'] = add_params(params, flux_maps['params'])

        ebins, czbins = get_binning(flux_maps)
        for to_flav in ['nue', 'numu', 'nutau']:
            for mID in ['', '_bar']: # 'matter' ID
                nue_flux = flux_maps['nue'+mID]['map']
                numu_flux = flux_maps['numu'+mID]['map']
                oscflux = {
                    'ebins': ebins, 'czbins': czbins,
                    'map':
                    (nue_flux*osc_prob_maps['nue'+mID+'_maps'][to_flav+mID] +
                     numu_flux*osc_prob_maps['numu'+mID+'_maps'][to_flav+mID])
                }
                osc_flux_maps[to_flav+mID] = oscflux
        osc_flux_maps.update_hash(cache_key)

        self.result_cache[cache_key] = osc_flux_maps

        return osc_flux_maps

