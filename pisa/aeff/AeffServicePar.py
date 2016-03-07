#
# AeffServicePar
#
# author: J.L. Lanfranchi
#
# date:   March 1, 2016
"""
Generate a 2D effective area map using 1D-parameterized effective areas
(i.e., separately marginalized and then parameterized in coszen and energy).
"""

import os

import numpy as np
from scipy.interpolate import interp1d

from pisa.utils.log import logging
from pisa.utils import fileio
from pisa.utils import flavInt


class AeffServicePar(object):
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

    The fundamental datastructures used here are:
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
    pisa/pisa/utils/parameterize_aeff.py
    """
    def __init__(self, ebins, czbins, aeff_egy_par, aeff_coszen_par, **params):
        logging.info('Initializing AeffServicePar...')
        self.__ebins = None
        self.__czbins = None
        self.__aeff_egy_par = None
        self.__aeff_coszen_par = None
        self.update(ebins=ebins, czbins=czbins, aeff_egy_par=aeff_egy_par,
                    aeff_coszen_par=aeff_coszen_par)

    def update(self, ebins, czbins, aeff_egy_par, aeff_coszen_par,
               interp_kind='cubic'):
        # Return if state needn't change
        #  NOTE: this is simplistic; there might be reason to compare e.g. the
        #  data contained within a file referenced rather than just looking at
        #  string equivalency. That's a TODO if it's ever an issue...
        if np.all(ebins == self.__ebins) and np.all(czbins == self.__czbins) \
                and aeff_egy_par == self.__aeff_egy_par and \
                aeff_coszen_par == self.__aeff_coszen_par:
            return
        self.__ebins = ebins
        self.__czbins = czbins

        ebin_midpoints = (ebins[:-1] + ebins[1:])/2.0
        czbin_midpoints = (czbins[:-1] + czbins[1:])/2.0

        if aeff_egy_par != self.__aeff_egy_par:
            # TODO: validation on all inputs!
            if isinstance(aeff_egy_par, basestring):
                self.edep_store = fileio.from_file(aeff_egy_par)
            elif isinstance(aeff_egy_par, collections.Mapping):
                self.edep_store = aeff_egy_par
            else:
                raise TypeError('Unhandled `aeff_egy_par` type: %s' %
                                type(aeff_egy_par))
            self.__aeff_egy_par = aeff_egy_par

        if aeff_coszen_par != self.__aeff_coszen_par:
            if isinstance(aeff_coszen_par, basestring):
                self.czdep_store = fileio.from_file(aeff_coszen_par)
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

        aeff2d = {}
        for group in ungrouped + grouped:
            # Only need one representative flavint from this group
            rep_flavint = group[0]

            # Find where this flavint is represented in the stores
            e_keys = [k for k in self.edep_store.keys()
                      if rep_flavint in flavInt.NuFlavIntGroup(k)]
            assert len(e_keys) == 1, len(e_keys)
            e_key = e_keys[0]

            cz_keys = [k for k in self.czdep_store.keys()
                      if rep_flavint in flavInt.NuFlavIntGroup(k)]
            assert len(cz_keys) == 1, len(cz_keys)
            cz_key = cz_keys[0]

            # Grab source data
            interpolant_edep_aeff = self.edep_store[e_key]['smooth']
            interpolant_czdep_aeff = self.czdep_store[cz_key]['smooth']

            # Interpolate 
            edep_interpolant = interp1d(
                x=self.edep_store['ebin_midpoints'],
                y=interpolant_edep_aeff,
                kind=interp_kind,
                copy=False,
                bounds_error=False,
                fill_value=np.nan,
                assume_sorted=True
            )
            czdep_interpolant = interp1d(
                x=self.czdep_store['czbin_midpoints'],
                y=interpolant_czdep_aeff,
                kind=interp_kind,
                copy=False,
                bounds_error=False,
                fill_value=np.nan,
                assume_sorted=True
            )
            interpolated_edep_aeff = edep_interpolant(ebin_midpoints)
            interpolated_czdep_aeff = czdep_interpolant(czbin_midpoints)

            # Fill values outside interpolants' ranges with nearest neighbor
            idx = ebin_midpoints < self.edep_store['ebin_midpoints'][0]
            interpolated_edep_aeff[idx] = interpolant_edep_aeff[0]
            idx = ebin_midpoints > self.edep_store['ebin_midpoints'][-1]
            interpolated_edep_aeff[idx] = interpolant_edep_aeff[-1]

            idx = czbin_midpoints < self.czdep_store['czbin_midpoints'][0]
            interpolated_czdep_aeff[idx] = interpolant_czdep_aeff[0]
            idx = czbin_midpoints > self.czdep_store['czbin_midpoints'][-1]
            interpolated_czdep_aeff[idx] = interpolant_czdep_aeff[-1]

            # CZ applies shape (len(cz)-times), not absolute value;
            # also, we use values with units [m^2 / GeV / sr] (?), so
            # normalize by multiplying by # of points and dividing by sum (?)
            interpolated_czdep_aeff *= len(interpolated_czdep_aeff) \
                    / np.sum(interpolated_czdep_aeff)

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
            assert len(keys) == 1, str(flavint) + str(keys)
            self.aeff_fidata[flavint] = aeff2d[keys[0]]

    def get_aeff(self):
        """Returns the effective areas FlavIntData object"""
        return self.aeff_fidata

    @staticmethod
    def add_argparser_args(parser):
        parser.add_argument(
            '--aeff-egy-par', metavar='RESOURCE', type=str,
            default='aeff/pingu_v36/'
            'aeff_energy_dependence__pingu_v36__runs_388-390__proc_v5.json',
            help='''Resource containing energy-dependent parameterization of
            effective areas.'''
        )
        parser.add_argument(
            '--aeff-coszen-par', metavar='RESOURCE', type=str,
            default='aeff/pingu_v36/'
            'aeff_coszen_dependence__pingu_v36__runs_388-390__proc_v5.json',
            help='''Resource containing coszen-dependent parameterizations of
            effective areas.'''
        )
