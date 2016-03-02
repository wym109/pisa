#
# AeffServicePar
#
# author: J.L. Lanfranchi
#
# date:   March 1, 2016
"""Apply 1D-parameterized effective areas to 2D map"""

import os

import numpy as np
from scipy.interpolate import interp1d

from pisa.utils.log import logging
from pisa.utils.utils import get_bin_centers, get_bin_sizes
from pisa.utils import flavInt
from pisa.utils.jsons import from_json
from pisa.resources.resources import find_resource, open_resource


class AeffServicePar:
    """Takes samples from (independent) energy and coszen parameterizations,
    interpolates to a user-specified E and CZ binning, and forms a normalized
    outer product to obtain 2D effective areas

    Parameters
    ----------
    ebins, czbins : array_like
        Energy and cosine-zenity bin edges
    aeff_egy_par, aeff_coszen_par : str or dict
        Effective area vs. energy / coszen 1D parameterizations.
        If str, load parameterizations from corresponding file path.
        If dict, must be of form defined below.

    Notes
    -----
    Units of effective areas returned are [m^2].

    All neutral current effective areas are treated identically.

    The fundamental datastructures used here requires the following forms:
        * Energy-dependence parameterization:
        {
            'ebin_midpoints': <length-N_ebins array>,
            '<repr(flavintgroup0)>': {
                'histo': <length-N_ebins array>,
                'histo_err': <length-N_ebins array>,
                'smooth': <length-N_ebins array>,
            },
            '<repr(flavintgroup1)>': {
                'histo': <length-N_ebins array>,
                'histo_err': <length-N_ebins array>,
                'smooth': <length-N_ebins array>,
            },
            ...
        }

        * Coszen-dependence parameterization:
        {
            'czbin_midpoints': <length-N_ebins array>,
            '<repr(flavintgroup0)>': {
                'histo': <length-N_ebins array>,
                'histo_err': <length-N_ebins array>,
                'smooth': <length-N_ebins array>,
            },
            '<repr(flavintgroup1)>': {
                'histo': <length-N_ebins array>,
                'histo_err': <length-N_ebins array>,
                'smooth': <length-N_ebins array>,
            },
            ...
        }

    See Also
    --------
    pisa/pisa/utils/make_aeff_parameterizations.py
    """
    def __init__(self, ebins, czbins, aeff_egy_par, aeff_coszen_par, **params):
        logging.info('Initializing AeffServicePar...')
        self.__ebins = None
        self.__czbins = None
        self.__aeff_egy_par = None
        self.__aeff_coszen_par = None
        self.update(ebins=ebins, czbins=czbins, aeff_egy_par=aeff_egy_par,
                    aeff_coszen_par=aeff_coszen_par)

    def update(self, ebins, czbins, aeff_egy_par, aeff_coszen_par):
        # Return if state needn't change
        #  NOTE: this is simplistic; there might be reason to compare e.g. the
        #  data contained within a file referenced rather than just looking at
        #  string equivalency. That's a TODO if it's ever an issue...
        if ebins == self.__ebins and czbins == self.__czbins and \
                aeff_egy_par == self.__aeff_egy_par and \
                aeff_coszen_par == self.__aeff_coszen_par:
            return
        self.__ebins = ebins
        self.__czbins = czbins

        ebin_midpoints = (ebins[:-1] + ebins[1:])/2.0
        czbin_midpoints = (czbins[:-1] + czbins[1:])/2.0

        if aeff_egy_par != self.__aeff_egy_par:
            # TODO: validation on all inputs!
            if isinstance(aeff_egy_par, basestring):
                self.edep_store = fileio.from_file(find_resource(aeff_egy_par))
            elif isinstance(aeff_egy_par, collections.Mapping):
                self.edep_store = aeff_egy_par
            else:
                raise TypeError('Unhandled `aeff_egy_par` type: %s' %
                                type(aeff_egy_par))
            self.__aeff_egy_par = aeff_egy_par

        if aeff_coszen_par != self.__aeff_coszen_par:
            if isinstance(aeff_coszen_par, basestring):
                self.czdep_store = fileio.from_file(
                    find_resource(aeff_coszen_par)
                )
            elif isinstance(aeff_coszen_par, collections.Mapping):
                self.czdep_store = aeff_coszen_par
            else:
                raise TypeError('Unhandled `aeff_coszen_par` type: %s' %
                                type(aeff_coszen_par))
            self.__aeff_coszen_par = aeff_coszen_par

        grouped = [
            flavInt.NuFlavIntGroup('nuall_nc'),
            flavInt.NuFlavIntGroup('nuallbar_nc'),
        ]
        ungrouped = [flavInt.NuFlavIntGroup(fi) for fi in flavInt.ALL_NUCC]

        interpolant_ebin_midpoints = edep_store['ebin_midpoints']
        interpolant_czbin_midpoints = czdep_store['czbin_midpoints']

        aeff2d = {}
        for group in ungrouped + grouped:
            # Only need one representative flavint from this group
            rep_flavint = group[0]

            # Find where this flavint is represented in the stores
            e_keys = [k for k in edep_store.keys()
                      if rep_flavint in flavInt.NuFlavIntGroup(k)]
            assert len(e_keys) == 1, len(e_keys)
            e_key = e_keys[0]

            cz_keys = [k for k in czdep_store.keys()
                      if rep_flavint in flavInt.NuFlavIntGroup(k)]
            assert len(cz_keys) == 1, len(cz_keys)
            cz_key = cz_keys[0]

            # Grab source data
            interpolant_edep_aeff = edep_store[e_key]['smooth']
            interpolant_czdep_aeff = edep_store[cz_key]['smooth']

            # Interpolate 
            edep_interpolant = interp1d(
                x=interpolant_ebin_midpoints,
                y=interpolant_edep_aeff,
                kind='linear',
                copy=False,
                bounds_error=False,
                fill_value=np.nan,
                assume_sorted=True
            )
            czdep_interpolant = interp1d(
                x=interpolant_czbin_midpoints,
                y=interpolant_czdep_aeff,
                kind='linear',
                copy=False,
                bounds_error=False,
                fill_value=np.nan,
                assume_sorted=True
            )
            interpolated_edep_aeff = edep_interpolant(ebin_midpoints)
            interpolated_czdep_aeff = czdep_interpolant(czbin_midpoints)

            # Fill values outside interpolants' ranges with nearest neighbor
            idx = ebin_midpoints < interpolant_ebin_midpoints
            interpolated_edep_aeff[idx] = interpolant_edep_aeff[0]
            idx = ebin_midpoints > interpolant_ebin_midpoints
            interpolated_edep_aeff[idx] = interpolant_edep_aeff[-1]

            idx = czbin_midpoints < interpolant_czbin_midpoints
            interpolated_czdep_aeff[idx] = interpolant_czdep_aeff[0]
            idx = czbin_midpoints > interpolant_czbin_midpoints
            interpolated_czdep_aeff[idx] = interpolant_czdep_aeff[-1]

            # Form 2D map via outer product
            aeff2d[group] = np.outer(interpolated_edep_aeff,
                                     interpolated_czdep_aeff)

        # Store info for *all* flavints, even if they were combined
        # together for computational purposes. This is what is used by 
        # the get_event_rates function in Aeff.py file, via the get_aeff()
        # method defined below..
        self.aeff_fidata = flavInt.FlavIntData()
        for flavint in self.aeff_fidata.flavints():
            # Find where this flavint is included in aeff2d
            keys = [k for k in aeff2d.keys() if flavint in k]
            assert len(keys) == 1, flavint, keys
            self.aeff_fidata[flavint] = aeff2d[keys[0]]

    def get_aeff(self):
        """Returns the effective areas FlavIntData object"""
        return self.aeff_fidata
