#! /usr/bin/env python
#
# Class Events for working with PISA events files
#
# author: Justin L. Lanfranchi
#         jll1062+pisa@phys.psu.edu
#
# date:   October 24, 2015
#


import h5py

import pisa.utils.flavInt as flavInt
import pisa.utils.hdf as hdf
import pisa.resources.resources as resources
import pisa.utils.utils as utils


class Events(flavInt.FIData):
    '''Container for storing events, including metadata about the events
    '''
    def __init__(self, val=None):
        self.metadata = {
            'detector': '',
            'geom': '',
            'runs': [],
            'proc_ver': '',
            'cuts': [],
            'kinds_joined': [],
        }
        meta = {}
        d = flavInt.FIData()
        if isinstance(val, basestring) or isinstance(val, h5py.Group):
            d, meta = self.__load(val)
        elif isinstance(val, dict):
            d = val
        self.metadata.update(meta)
        self.validate(d)
        self.update(d)

    def meta_eq(self, other):
        return utils.recEq(self.metadata, other.metadata)

    def data_eq(self, other):
        return utils.recEq(self, other)

    def __eq__(self, other):
        return (self.meta_eq(other) and self.data_eq(other))

    def __load(self, fname):
        fpath = resources.find_resource(fname)
        with h5py.File(fpath, 'r') as f:
            meta = dict(f.attrs)
            d = hdf.from_hdf(f)
        self.validate(d)
        return d, meta

    def save(self, fname, overwrite=True):
        hdf.to_hdf(self, fname, attrs=self.metadata, overwrite=overwrite)


def test():
    events = Events()


if __name__ == "__main__":
    test()
