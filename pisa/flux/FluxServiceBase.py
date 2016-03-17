"""
Base class for all flux services
"""


import numpy as np

from pisa.utils.log import logging
from pisa.analysis.stats.Maps import apply_ratio_scale
from pisa.utils.proc import report_params, get_params
from pisa.utils.utils import get_bin_centers, get_bin_sizes, hash_obj, LRUCache, DictWithHash


class FluxServiceBase(object):
    def __init__(self, cache_depth=1000):
        self.primaries = ['numu', 'numu_bar', 'nue', 'nue_bar']
        self.cache_depth = cache_depth
        self.result_cache = LRUCache(self.cache_depth)
        self.raw_flux_cache = LRUCache(self.cache_depth*len(self.primaries))

    def get_flux_maps(self, ebins, czbins, nue_numu_ratio, nu_nubar_ratio,
                      energy_scale, atm_delta_index, **kwargs):
        """Get a set of flux maps for the different primaries.

        Parameters
        ----------
        ebins, czbins
            energy/coszenith bins within which to calculate flux

        nue_numu_ratio
            systematic to be a proxy for the realistic Flux_nue/Flux_numu and
            Flux_nuebar/Flux_numubar ratios, keeping both the total flux from
            neutrinos and antineutrinos constant. The adjusted ratios are given
            by "nue_numu_ratio * original ratio".

        nu_nubar_ratio
            systematic to be a proxy for the neutrino/anti-neutrino
            production/cross section ratio.

        energy_scale
            factor to scale energy bin centers by

        atm_delta_index
            change in spectral index from fiducial
        """
        cache_key = hash_obj((ebins, czbins, nue_numu_ratio, nu_nubar_ratio,
                              energy_scale, atm_delta_index))
        try:
            return self.result_cache.get(cache_key)
        except KeyError:
            pass

        ## Be verbose on input
        #params = get_params()
        #report_params(params, units = [''])

        # Initialize return dict
        maps = DictWithHash()
        maps['params'] = {}
        #maps['params'] = params

        for prim in self.primaries:
            # Get the flux for this primary
            maps[prim] = {
                'ebins': ebins,
                'czbins': czbins,
                'map': self.get_flux(ebins*energy_scale, czbins, prim)
            }
            # Be less verbose
            ## be a bit verbose
            #logging.trace("Total flux of %s is %u [s^-1 m^-2]"%
            #              (prim, maps[prim]['map'].sum()))

        # now scale the nue(bar) / numu(bar) flux ratios, keeping the total
        # Flux (nue + numu, nue_bar + numu_bar) constant, or return unscaled
        # maps:
        if nue_numu_ratio != 1.0:
            scaled_maps = self.apply_nue_numu_ratio(maps, nue_numu_ratio)
        else:
            scaled_maps = maps

        # now scale the nu(e/mu) / nu(e/mu)bar event count ratios, keeping the
        # total (nue + nuebar etc.) constant
        if nu_nubar_ratio != 1.0:
            scaled_maps = self.apply_nu_nubar_ratio(scaled_maps,
                                                    nu_nubar_ratio)

        median_energy = self.get_median_energy(maps['numu'])
        if atm_delta_index != 0.0:
            scaled_maps = self.apply_delta_index(scaled_maps, atm_delta_index,
                                                 median_energy)

        scaled_maps.update_hash(cache_key)
        self.result_cache.set(cache_key, scaled_maps)

        return scaled_maps

    @staticmethod
    def apply_nue_numu_ratio(flux_maps, nue_numu_ratio):
        """
        Applies the nue_numu_ratio systematic to the flux maps
        and returns the scaled maps. The actual calculation is
        done by apply_ratio_scale.
        """
        # keep both nu and nubar flux constant
        scaled_nue_flux, scaled_numu_flux = apply_ratio_scale(
            orig_maps=flux_maps,
            key1='nue',
            key2='numu',
            ratio_scale=nue_numu_ratio,
            is_flux_scale=True
        )

        scaled_nue_bar_flux, scaled_numu_bar_flux = apply_ratio_scale(
            orig_maps=flux_maps,
            key1='nue_bar',
            key2='numu_bar',
            ratio_scale=nue_numu_ratio,
            is_flux_scale=True
        )

        flux_maps['nue']['map'] = scaled_nue_flux
        flux_maps['nue_bar']['map']  =  scaled_nue_bar_flux
        flux_maps['numu']['map'] = scaled_numu_flux
        flux_maps['numu_bar']['map']  = scaled_numu_bar_flux

        return flux_maps

    @staticmethod
    def apply_nu_nubar_ratio(event_rate_maps, nu_nubar_ratio):
        """Applies the nu_nubar_ratio systematic to the event rate maps and
        returns the scaled maps. The actual calculation is done by
        apply_ratio_scale.
        """
        flavours = event_rate_maps.keys()
        if 'params' in flavours:
            flavours.remove('params')

        for flavour in flavours:
            # process nu and nubar in one go
            if not 'bar' in flavour:
                # do this for each interaction channel (cc and nc)
                scaled_nu_rates, scaled_nubar_rates = apply_ratio_scale(
                    orig_maps = event_rate_maps,
                    key1 = flavour, key2 = flavour+'_bar',
                    ratio_scale = nu_nubar_ratio,
                    is_flux_scale = True,
                )
                event_rate_maps[flavour]['map'] = scaled_nu_rates
                event_rate_maps[flavour+'_bar']['map'] = scaled_nubar_rates

        return event_rate_maps

    @staticmethod
    def apply_delta_index(flux_maps, delta_index, egy_med):
        """Applies the spectral index systematic to the flux maps by scaling
        each bin with (egy_cen/egy_med)^(-delta_index), preserving the total
        integral flux  Note that only the numu/numu_bar are scaled, because the
        nue_numu_ratio will handle the systematic on the nue flux.
        """
        for flav in ['numu', 'numu_bar']:
            ecen = get_bin_centers(flux_maps[flav]['ebins'])
            scale = np.power((ecen/egy_med), delta_index)
            flux_map = flux_maps[flav]['map']
            total_flux = flux_map.sum()

            logging.trace("flav: %s, total counts before scale: %f" %
                          (flav, total_flux))

            scaled_flux = (flux_map.T*scale).T
            scaled_flux *= (total_flux/scaled_flux.sum())
            flux_maps[flav]['map'] = scaled_flux

            logging.trace("flav: %s, total counts after scale: %f" %
                          (flav, flux_maps[flav]['map'].sum()))

        return flux_maps

    @staticmethod
    def get_median_energy(flux_map):
        """Returns the median energy of the flux_map-expected to be a dict
        with keys 'map', 'ebins', 'czbins'
        """
        ecen = get_bin_centers(flux_map['ebins'])
        energy = ecen[len(ecen)//2] # bug? previously had: [len(ecen)/2]

        return energy
