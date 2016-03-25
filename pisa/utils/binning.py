# author : J.L. Lanfranchi
#          jll1062+pisa@phys.psu.edu
#
# date   : March 25, 2016

"""
Class to carry information about 2D binning in energy and cosine-zenity, and to
provide basic operations with the binning.
"""

from collections import OrderedDict
import numpy as np


def check_binning(bin_edges):
    assert np.diff(bin_edges) > 0


class Binning(object):
    __slots = tuple()

    def __init__(self, ebins, czbins, e_is_log, cz_is_log=False):
        if isinstance(ebins, basestring):
            ebins = eval(ebins)
        check_binning(ebins, e_is_log)
        object.__setattr__(self, 'ebins', ebins)

        if isinstance(czbins, basestring):
            czbins = eval(czbins)
        check_binning(self.czbins, cz_is_log)
        object.__setattr__(self, 'czbins', czbins)

        # Calculate meta-attributes
        object.__setattr__(self, 'e_is_log', e_is_log)
        object.__setattr__(self, 'cz_is_log', cz_is_log)
        object.__setattr__(self, 'e_range', (self.ebins[0], self.ebins[-1]))
        object.__setattr__(self, 'cz_range', (self.czbins[0], self.czbins[-1]))
        object.__setattr__(self, 'n_ebins', len(self.ebins))
        object.__setattr__(self, 'n_czbins', len(self.czbins))

    def __setattr__(self, attr, value):
        """Only allow setting attributes defined in __slots"""
        if attr not in self.__slots:
            raise ValueError('No attribute "%s"' % attr)
        object.__setattr__(self, attr, value)

    @property
    def ebin_midpoints(self):
        return (self.ebins[:-1] + self.ebins[1:])/2.0

    @property
    def czbin_midpoints(self):
        return (self.czbins[:-1] + self.czbins[1:])/2.0

    @property
    def ebin_visual_centers(self):
        if self.e_is_log:
            return np.sqrt(self.ebins[:-1]*self.ebins[1:])
        return self.ebin_midpoints

    @property
    def czbin_visual_centers(self):
        if self.cz_is_log:
            return np.sqrt(self.czbins[:-1]*self.czbins[1:])
        return self.czbin_midpoints

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

    def check_compatibility(self, array):
        """Check if a 2D array of values can map onto the defined binning"""
        required_shape = (self.n_ebins-1, self.n_czbins-1)
        if array.shape != required_shape:
            raise ValueError('Shape %s incompatible with required shape %s'
                             % (array.shape, required_shape))

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
