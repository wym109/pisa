# author: J.L. Lanfranchi
#
# date:   March 1, 2016
"""
Generate a 2D effective area map using 1D-smoothed effective areas
(i.e., separately marginalized and then smoothed in coszen and energy).
"""

import os

import numpy as np
from scipy.interpolate import interp1d

from pisa.utils.log import logging
from pisa.utils import fileio
from pisa.utils import flavInt


class AeffServiceSmooth(object):
    """Takes smoothed samples from (independent) energy and coszen effective
    areas, interpolates to a user-specified E and CZ binning, and forms a
    normalized outer product to obtain 2D effective areas

    Parameters
    ----------
    ebins, czbins : array_like
        Energy and cosine-zenity bin edges
    aeff_energy_smooth, aeff_coszen_smooth : str or dict
        Samples from 1D-smoothed effective area vs. energy & coszen.
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
    def __init__(self, ebins, czbins, aeff_energy_smooth, aeff_coszen_smooth,
                 **kwargs):
        logging.info('Initializing AeffServicePar...')
        self.ebins = None
        self.czbins = None
        self.__aeff_energy_smooth = None
        self.__aeff_coszen_smooth = None
        self.update(ebins=ebins, czbins=czbins,
                    aeff_energy_smooth=aeff_energy_smooth,
                    aeff_coszen_smooth=aeff_coszen_smooth)

    def update(self, ebins, czbins, aeff_energy_smooth, aeff_coszen_smooth,
               interp_kind='cubic'):
        # Return if state needn't change
        #  NOTE: this is simplistic; there might be reason to compare e.g. the
        #  data contained within a file referenced rather than just looking at
        #  string equivalency. That's a TODO if it's ever an issue...
        if np.all(ebins == self.ebins) and np.all(czbins == self.czbins) \
                and aeff_energy_smooth == self.__aeff_energy_smooth and \
                aeff_coszen_smooth == self.__aeff_coszen_smooth:
            return
        self.ebins = ebins
        self.czbins = czbins

        ebin_midpoints = (ebins[:-1] + ebins[1:])/2.0
        czbin_midpoints = (czbins[:-1] + czbins[1:])/2.0

        if aeff_energy_smooth != self.__aeff_energy_smooth:
            # TODO: validation on all inputs!
            if isinstance(aeff_energy_smooth, basestring):
                self.edep_store = fileio.from_file(aeff_energy_smooth)
            elif isinstance(aeff_energy_smooth, collections.Mapping):
                self.edep_store = aeff_energy_smooth
            else:
                raise TypeError('Unhandled `aeff_energy_smooth` type: %s' %
                                type(aeff_energy_smooth))
            self.__aeff_energy_smooth = aeff_energy_smooth

        if aeff_coszen_smooth != self.__aeff_coszen_smooth:
            if isinstance(aeff_coszen_smooth, basestring):
                self.czdep_store = fileio.from_file(aeff_coszen_smooth)
            elif isinstance(aeff_coszen_smooth, collections.Mapping):
                self.czdep_store = aeff_coszen_smooth
            else:
                raise TypeError('Unhandled `aeff_coszen_smooth` type: %s' %
                                type(aeff_coszen_smooth))
            self.__aeff_coszen_smooth = aeff_coszen_smooth

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
                #assume_sorted=True
            )
            czdep_interpolant = interp1d(
                x=self.czdep_store['czbin_midpoints'],
                y=interpolant_czdep_aeff,
                kind=interp_kind,
                copy=False,
                bounds_error=False,
                fill_value=np.nan,
                #assume_sorted=True
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
            interpolated_czdep_aeff *= \
                    (np.max(self.czbins) - np.min(self.czbins)) * \
                    len(interpolated_czdep_aeff) \
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
            '--aeff-energy-smooth', metavar='RESOURCE', type=str,
            default='aeff/pingu_v36/'
            'aeff_energy_smooth__pingu_v36__runs_388-390__proc_v5.json',
            help='''Resource containing smoothed energy dependence of effective
            areas.'''
        )
        parser.add_argument(
            '--aeff-coszen-smooth', metavar='RESOURCE', type=str,
            default='aeff/pingu_v36/'
            'aeff_coszen_smooth__pingu_v36__runs_388-390__proc_v5.json',
            help='''Resource containing smmothed coszen dependence of effective
            areas.'''
        )
