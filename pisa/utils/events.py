#! /usr/bin/env python
#
# Events.py
#
# author: Justin L. Lanfranchi
#         jll1062@phys.psu.edu
#
# date:   2015-11-07
#
# Class for working with PISA events files


import h5py

import pisa.utils.flavInt as flavInt
import pisa.utils.hdf as hdf
import pisa.resources.resources as resources


class Events(flavInt.FIData):
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
            d, meta = self.load(val)
        elif isinstance(val, dict):
            d = val
        self.metadata.update(meta)
        self.validate(d)
        self.update(d)

    @staticmethod
    def load(fname):
        fpath = resources.find_resource(fname)
        with h5py.File(fpath, 'r') as f:
            meta = dict(f.attrs)
            d = hdf.from_hdf(f)
        return d, meta

    def save(self, fname, overwrite=True):
        hdf.to_hdf(self, fname, attrs=self.metadata, overwrite=overwrite)


def test():
    events = Events()


if __name__ == "__main__":
    test()
