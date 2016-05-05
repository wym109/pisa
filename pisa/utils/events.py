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

import h5py

import pisa.resources.resources as resources
from pisa.utils.comparisons import recursiveEquality
from pisa.utils import flavInt
from pisa.utils import hdf


class Events(flavInt.FlavIntData):
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
        data = flavInt.FlavIntData()
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


def test_Events():
    events = Events()
    # TODO: add more testing here


if __name__ == "__main__":
    test_Events()
