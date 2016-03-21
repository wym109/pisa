import numpy as np
from scipy.constants import Julian_year

from pisa.utils import flavInt
from pisa.utils.log import logging, set_verbosity
from pisa.utils.proc import report_params, get_params, add_params
from pisa.utils.utils import check_binning, get_binning, prefilled_map, DictWithHash, hash_obj
from pisa.utils.cache import MemoryCache


class AeffServiceBase(object):
    def __init__(self, ebins, czbins, cache_depth=1000):
        self.ebins = ebins
        self.czbins = czbins
        self.cache_depth = cache_depth
        self.transform_cache = MemoryCache(100, is_lru=True)
        self.result_cache = MemoryCache(100, is_lru=True)

    def get_event_rates(self, osc_flux_maps, livetime, aeff_scale, **kwargs):
        """Main function for this module, which returns the event rate maps
        for each flavor and interaction type, using true energy and zenith
        information. The content of each bin will be the weighted aeff
        multiplied by the oscillated flux, so that the returned dictionary
        will be of the form:
        {'nue': {'cc':map, 'nc':map},
         'nue_bar': {'cc':map, 'nc':map}, ...
         'nutau_bar': {'cc':map, 'nc':map} }
        \params:
          * osc_flux_maps - maps containing oscillated fluxes
          * livetime - detector livetime for which to calculate event counts
          * aeff_scale - systematic to be a proxy for the realistic effective
          area
        """
        # Don't get parameters, this is slow. Implications?
        ## Get parameters used here
        #params = get_params()
        #report_params(params, units=['', 'yrs', ''])
    
        # Get effective area
        aeff_fidata = self.get_aeff()

        cache_key = hash_obj((osc_flux_maps.hash, aeff_fidata.hash))
        try:
            return self.result_cache[cache_key]
        except KeyError:
            pass
    
        # Initialize return dict
        event_rate_maps = DictWithHash()
        event_rate_maps['params'] = {}
        #event_rate_maps['params'] = add_params(params, osc_flux_maps['params'])
    
        ebins, czbins = get_binning(osc_flux_maps)
    
        # apply the scaling for nu_xsec_scale and nubar_xsec_scale...
        flavours = ['nue', 'numu', 'nutau', 'nue_bar', 'numu_bar', 'nutau_bar']
        for flavour in flavours:
            osc_flux_map = osc_flux_maps[flavour]['map']
            int_type_dict = {}
            for int_type in ['cc', 'nc']:
                event_rate = osc_flux_map * aeff_fidata[flavour][int_type] \
                        * (livetime * Julian_year * aeff_scale)
    
                int_type_dict[int_type] = {'map':event_rate,
                                           'ebins':ebins,
                                           'czbins':czbins}
                logging.debug("  Event Rate before reco for %s/%s: %.2f"
                              % (flavour, int_type, np.sum(event_rate)))
            event_rate_maps[flavour] = int_type_dict
        event_rate_maps.update_hash(cache_key)
        self.result_cache[cache_key] = event_rate_maps
    
        return event_rate_maps

