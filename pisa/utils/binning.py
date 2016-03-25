# author : J.L. Lanfranchi
#          jll1062+pisa@phys.psu.edu
#
# date   : March 25, 2016

"""
Class to carry information about 2D binning in energy and cosine-zenity, and to
provide basic operations with the binning.
"""

from collections import OrderedDict

import numpy
import numpy as np
# Import entire numpy namespace for `eval` called on a passed string
from numpy import *

from pisa.utils.log import logging


def verify_binning(bin_edges, is_log):
    # Bin edges must be monotonic, strictly increasing
    lin_spacing = np.diff(bin_edges)
    assert np.all(lin_spacing > 0)
    # Log binning must have equal widths in log-space
    if is_log:
        log_spacing = bin_edges[1:] / bin_edges[:-1]
        assert np.allclose(log_spacing, log_spacing[0])


class Binning(object):
    """
    Parameters
    ----------
    e_is_log
    cz_is_log
    ebins
    czbins
    n_ebins
    e_range
    n_czbins
    cz_range

    Must specify `e_is_log`; optionally specify `cz_is_log` (default=False).

    In addition, must specify either (`ebins`) or (`n_ebins` and `e_range`)
    and either (`czbins`) or (`n_czbins` and `cz_range`).

    Properties
    ----------
    ebins
    czbins
    e_is_log
    cz_is_log
    e_range
    cz_range
    n_ebins
    n_czbins
    ebin_midpoints
    czbin_midpoints
    ebin_visual_centers
    czbin_visual_centers

    Methods
    -------
    assert_array_compat
    is_binning_compat
    oversample
    __eq__
    """

    __slots = tuple()

    def __init__(self, e_is_log, cz_is_log=False, ebins=None, czbins=None,
                 n_ebins=None, e_range=None, n_czbins=None, cz_range=None):
        assert isinstance(e_is_log, bool)
        assert isinstance(cz_is_log, bool)
        if n_ebins is not None:
            assert e_range is not None
            assert ebins is None
            assert len(e_range) == 2
            if e_is_log:
                ebins = np.logspace(np.log10(e_range[0]), np.log10(e_range[1]),
                                    n_ebins)
        elif isinstance(ebins, basestring):
            ebins = eval(ebins)


        check_binning(bin_edges=ebins, is_log=e_is_log)
        object.__setattr__(self, 'ebins', ebins)

        if isinstance(czbins, basestring):
            czbins = eval(czbins)
        check_binning(bin_edges=self.czbins, is_log=cz_is_log)
        object.__setattr__(self, 'czbins', czbins)

        # Calculate meta-attributes of the specified binning
        object.__setattr__(self, 'e_is_log', e_is_log)
        object.__setattr__(self, 'cz_is_log', cz_is_log)
        object.__setattr__(self, 'e_range', (self.ebins[0], self.ebins[-1]))
        object.__setattr__(self, 'cz_range', (self.czbins[0], self.czbins[-1]))
        object.__setattr__(self, 'n_ebins', len(self.ebins))
        object.__setattr__(self, 'n_czbins', len(self.czbins))
        object.__setattr__(self, 'ebin_midpoints',
                           (self.ebins[:-1] + self.ebins[1:])/2.0)
        object.__setattr__(self, 'czbin_midpoints',
                           (self.czbins[:-1] + self.czbins[1:])/2.0)
        if self.e_is_log:
            ebin_visual_centers =  np.sqrt(self.ebins[:-1]*self.ebins[1:])
        else:
            ebin_visual_centers = self.ebin_midpoints
        object.__setattr__(self, 'ebin_visual_centers', ebin_visual_centers)
        if self.cz_is_log:
            czbin_visual_centers =  np.sqrt(self.czbins[:-1]*self.czbins[1:])
        else:
            czbin_visual_centers = self.czbin_midpoints
        object.__setattr__(self, 'czbin_visual_centers', czbin_visual_centers)

    def __setattr__(self, attr, value):
        """Only allow setting attributes defined in __slots"""
        if attr not in self.__slots:
            raise ValueError('No attribute "%s"' % attr)
        object.__setattr__(self, attr, value)

    def oversample(self, e_factor, cz_factor):
        if self.e_is_log:
            ebins = np.logspace(np.log10(self.e_range[0]),
                                np.log10(self.e_range[-1]),
                                self.n_ebins * e_factor)
        else:
            ebins = np.linspace(self.e_range[0], self.e_range[-1],
                                self.n_ebins * e_factor)
        if self.e_is_log:
            ebins = np.logspace(np.log10(self.e_range[0]),
                                np.log10(self.e_range[-1]),
                                self.n_ebins * e_factor)
        else:
            ebins = np.linspace(self.e_range[0], self.e_range[-1],
                                self.n_ebins * e_factor)
        return Binning(ebins=ebins, czbins=czbins,
                       e_is_log=self.e_is_log, cz_is_log=self.cz_is_log)

    def assert_array_compat(self, array):
        """Check if a 2D array of values fits into the defined bins"""
        required_shape = (self.n_ebins-1, self.n_czbins-1)
        if array.shape != required_shape:
            raise ValueError('Shape %s incompatible with required shape %s'
                             % (array.shape, required_shape))

    def is_binning_compat(self, binning):
        """Check if a (possibly different) binning can map onto the defined
        binning. Allows for simple re-binning schemes (but no interplation).
        """
        if binning == self:
            return True
        my_e, other_e = set(self.ebins), set(binning.ebins)
        my_cz, other_cz = set(self.czbins), set(binning.czbins)
        if not(my_e.difference(other_e)) and not(my_cz.difference(other_cz)):
            return True
        if not(other_e.difference(my_e)) and not(other_cz.difference(my_cz)):
            return True
        return False

    def __eq__(self, other):
        if not isinstance(other, Binning):
            return False
        return (self.ebins == other.ebins and self.czbins == other.czbins
                and self.e_is_log == other.e_is_log
                and self.cz_is_log == other.cz_is_log)


def test_Binning():
    pass

if __name__ == "__main__":
    test_Binning()
