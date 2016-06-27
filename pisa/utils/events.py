#! /usr/bin/env python
#
# Events class for working with PISA events files
#
# author: Justin L. Lanfranchi
#         jll1062+pisa@phys.psu.edu
#
# date:   October 24, 2015
#
"""
Events class for working with PISA events files
"""


from collections import Iterable, Sequence

import h5py
import numpy as np
from uncertainties import unumpy as unp

from pisa.core.binning import MultiDimBinning, OneDimBinning
from pisa.resources import resources
from pisa.utils.comparisons import normQuant, recursiveEquality
from pisa.utils.flavInt import FlavIntData, NuFlavIntGroup
from pisa.utils.hash import hash_obj
from pisa.utils import hdf


# TODO: test hash function (attr)
class Events(FlavIntData):
    """Container for storing events, including metadata about the events"""
    def __init__(self, val=None):
        self.metadata = {
            'detector': '',
            'geom': '',
            'runs': [],
            'proc_ver': '',
            'cuts': [],
            'flavints_joined': [],
        }
        meta = {}
        data = FlavIntData()
        if isinstance(val, basestring) or isinstance(val, h5py.Group):
            data, meta = self.__load(val)
        elif isinstance(val, Events):
            self.metadata = val.metadata
            data = val
        elif isinstance(val, dict):
            data = val
        self.metadata.update(meta)
        self.validate(data)
        self.update(data)
        self._hash = hash_obj(normQuant(self.metadata))

    @property
    def hash(self):
        return self._hash

    def meta_eq(self, other):
        """Test whether the metadata for this object matches that of `other`"""
        return recursiveEquality(self.metadata, other.metadata)

    def data_eq(self, other):
        """Test whether the data for this object matche that of `other`"""
        return recursiveEquality(self, other)

    def __eq__(self, other):
        return self.meta_eq(other) and self.data_eq(other)

    def __load(self, fname):
        fpath = resources.find_resource(fname)
        with h5py.File(fpath, 'r') as open_file:
            meta = dict(open_file.attrs)
            data = hdf.from_hdf(open_file)
        self.validate(data)
        return data, meta

    def save(self, fname, **kwargs):
        hdf.to_hdf(self, fname, attrs=self.metadata, **kwargs)

    def histogram(self, kinds, binning, binning_cols=None, weights_col=None,
            errors=False):
        """Histogram the events of all `kinds` specified, with `binning` and
        optionally applying `weights`.

        Parameters
        ----------
        kinds : string, sequence of NuFlavInt, or NuFlavIntGroup
        binning : OneDimBinning, MultiDimBinning or sequence of arrays (one array per binning dimension)
        weights_col : string

        Returns
        -------
        hist : numpy ndarray with as many dimensions as specified by `binning` argument

        """
        if not isinstance(kinds, NuFlavIntGroup):
            kinds = NuFlavIntGroup(kinds)
        #if not isinstance(binning, (OneDimBinning, MultiDimBinning, Sequence)):
        #    binning = MultiDimBinning(binning)
        if isinstance(binning_cols, basestring):
            binning_cols = [binning_cols]
        assert weights_col is None or isinstance(weights_col, basestring)

        # TODO: units of columns, and convert bin edges if necessary
        if isinstance(binning, OneDimBinning):
            bin_edges = [binning.magnitude]
            if binning_cols is None:
                binning_cols = [binning.name]
            else:
                assert len(binning_cols) == 1 and binning_cols[0] == binning.name
        elif isinstance(binning, MultiDimBinning):
            bin_edges = [edges.magnitude for edges in binning.bin_edges]
            if binning_cols is None:
                binning_cols = binning.names
            else:
                assert set(binning_cols).issubset(set(binning.names))
        elif isinstance(binning, (Sequence, Iterable)):
            assert len(binning_cols) == len(binning)
            bin_edges = binning

        # Extract the columns' data into a list of array(s) for histogramming
        repr_flav_int = kinds[0]
        sample = [self[repr_flav_int][colname] for colname in binning_cols]
        if weights_col is not None:
            weights = self[repr_flav_int][weights_col]
        else:
            weights = None

        hist, _ = np.histogramdd(sample=sample, weights=weights, bins=bin_edges)
        if errors:
            sumw2, _ = np.histogramdd(sample=sample, weights=np.square(weights), bins=bin_edges)
            hist = unp.uarray(hist, np.sqrt(sumw2))

        return hist


def test_Events():
    events = Events()
    # TODO: add more testing here!


if __name__ == "__main__":
    test_Events()
