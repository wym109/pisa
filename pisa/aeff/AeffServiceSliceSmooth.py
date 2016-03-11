# author: J.L. Lanfranchi
#
# date:   March 1, 2016
"""
Generate a 2D effective area map using 2D-smoothed effective areas, where the
smoothing is performed on 1D slices of the full 2D map, first in cosine-zenith
and then in energy.
"""

import os
import collections

import numpy as np
from scipy.interpolate import interp1d, interp2d, RectBivariateSpline

from pisa.utils.log import logging
from pisa.utils import hdf
from pisa.utils import flavInt


class AeffServiceSliceSmooth(object):
    """Takes smoothed samples from 2D energy / coszen plane effective
    areas, and interpolates to a user-specified E and CZ binning

    Parameters
    ----------
    ebins, czbins : array_like
        Energy and cosine-zenith bin edges
    aeff_slice_smooth : str or dict
        Samples from 2D-smoothed effective area vs. energy & coszen.
        If str, load from corresponding resource location.
        If dict, must be of form defined below.

    Notes
    -----
    Units of effective areas returned are [m^2].

    All neutral current effective areas are treated identically.

    The fundamental datastructure used here is:
        {
            'ebins': <length-N_ebins array>,
            'czbins': <length-N_czbins array>,
            '<repr(flavintgroup0)>': {
                'hist': <length-N_ebins array>,
                'hist_err': <length-N_ebins array>,
                'smooth': <N_ebins x N_czbins array>,
            },
            '<repr(flavintgroup1)>': {
                'hist': <length-N_ebins array>,
                'hist_err': <length-N_ebins array>,
                'smooth': <N_ebins x N_czbins array>,
            },
            ...
        }

    See Also
    --------
    pisa.utils.slice_smooth_aeff.py : script for performing the smoothing on
    Monte Carlo samples
    """
    def __init__(self, ebins, czbins, aeff_slice_smooth,
                 **kwargs):
        logging.info('Initializing AeffServicePar...')
        self.ebins = None
        self.czbins = None
        self.__aeff_slice_smooth = None
        self.__interp_kind = 'linear'
        self.interpolants = {}
        self.update(ebins=ebins, czbins=czbins,
                    aeff_slice_smooth=aeff_slice_smooth)

    def update(self, ebins=None, czbins=None, aeff_slice_smooth=None,
               interp_kind=None):
        if ebins is None:
            ebins = self.ebins
        if czbins is None:
            czbins = self.czbins
        if aeff_slice_smooth is None:
            aeff_slice_smooth = self.__aeff_slice_smooth
        if interp_kind is None:
            interp_kind = self.__interp_kind
        # Return if state needn't change
        #  NOTE: this is simplistic; there might be reason to compare e.g. the
        #  data contained within a file referenced rather than just looking at
        #  string equivalency. That's a TODO if it's ever an issue...
        if np.all(ebins == self.ebins) and np.all(czbins == self.czbins) \
                and aeff_slice_smooth == self.__aeff_slice_smooth \
                and interp_kind == self.__interp_kind:
            return

        if interp_kind == 'linear':
            spline_degree = 1
        elif interp_kind == 'quadratic':
            spline_degree = 2
        elif interp_kind == 'cubic':
            spline_degree = 3
        else:
            raise ValueError('Unrecognized `interp_kind`: "%s"' % interp_kind)
        self.__interp_kind = interp_kind

        self.ebins = ebins
        self.czbins = czbins
        ebin_midpoints = (ebins[:-1] + ebins[1:])/2.0
        czbin_midpoints = (czbins[:-1] + czbins[1:])/2.0

        if aeff_slice_smooth != self.__aeff_slice_smooth:
            # TODO: validation on all inputs!
            if isinstance(aeff_slice_smooth, basestring):
                self.smooth_store, self.metadata = \
                        hdf.from_hdf(aeff_slice_smooth, return_attrs=True)
            elif isinstance(aeff_slice_smooth, collections.Mapping):
                self.smooth_store = aeff_slice_smooth
            elif isinstance(aeff_slice_smooth, tuple):
                self.smooth_store, self.metadata = aeff_slice_smooth
            else:
                raise TypeError('Unhandled `aeff_slice_smooth` type: %s' %
                                type(aeff_slice_smooth))
            self.__aeff_slice_smooth = aeff_slice_smooth

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
            keys = [k for k in self.smooth_store.keys()
                    if rep_flavint in flavInt.NuFlavIntGroup(k)]
            assert len(keys) == 1, len(keys)
            key = keys[0]

            # Grab source data
            interpolant_aeff = self.smooth_store[key]['smooth']

            # Interpolate
            store_ebin_midpoints = (self.smooth_store['ebins'][:-1] +
                                    self.smooth_store['ebins'][1:])/2.0
            store_czbin_midpoints = (self.smooth_store['czbins'][:-1] +
                                     self.smooth_store['czbins'][1:])/2.0
            x = store_ebin_midpoints
            y = store_czbin_midpoints
            xmesh, ymesh = np.meshgrid(store_ebin_midpoints,
                                       store_czbin_midpoints, indexing='ij')
            interpolant = interp2d(
                x=store_czbin_midpoints, y=store_ebin_midpoints,
                z=interpolant_aeff,
                kind=interp_kind,
                copy=True,
                fill_value=None,
            )
            self.interpolants[group] = interpolant
            ebin_mp_mesh, czbin_mp_mesh = np.meshgrid(ebin_midpoints,
                                                      czbin_midpoints,
                                                      indexing='ij')
            aeff2d[group] = interpolant(czbin_midpoints, ebin_midpoints)

        # Store info for *all* flavints, even if they were combined
        # together for computational purposes. This is what is used by 
        # the get_event_rates function in Aeff.py file, which calls the
        # get_aeff() method below.
        self.aeff_fidata = flavInt.FlavIntData()
        for flavint in self.aeff_fidata.flavints():
            # Find where this flavint is included in aeff2d
            keys = [k for k in aeff2d.keys() if flavint in k]
            assert len(keys) == 1, str(flavint) + str(keys)
            self.aeff_fidata[flavint] = aeff2d[keys[0]]

    def get_aeff(self):
        """Returns the effective areas FlavIntData object"""
        return self.aeff_fidata

    def sample_aeff_surface(self, flavint, ebin_midpoints, czbin_midpoints):
        group = [g for g in self.interpolants.keys() if flavint in g][0] 
        return self.interpolants[group](ebin_midpoints, czbin_midpoints)

    @staticmethod
    def add_argparser_args(parser):
        parser.add_argument(
            '--aeff-slice-smooth', metavar='RESOURCE', type=str,
            default='aeff/pingu_v36/'
            'aeff_slice_smooth__pingu_v36__runs_388-390__proc_5.hdf5',
            help='''Resource containing sampled smoothed effective areas.'''
        )
