#
# hdf.py
#
# A set of utilities for dealing with HDF5 files.
#
# author: Sebastian Boeser
#         sboeser@uni-mainz.de
#
# date:   2015-03-05
#

import os
import numpy as np
import h5py
from pisa.utils.log import logging

def from_hdf(filename):
    """Open a file in HDF5 format, parse the content and return as dictionary
    with numpy arrays"""
    try:
        hdf5_data = h5py.File(os.path.expandvars(filename), 'r')
    except IOError, e:
        logging.error("Unable to read HDF5 file \'%s\'" % filename)
        logging.error(e)
        raise e

    data = {}

    # Iteratively parse the file to create the dictionary
    def visit_group(obj, sdict):
        name = obj.name.split('/')[-1]
        #indent = len(obj.name.split('/'))-1
        #print "  "*indent,name, obj.value if (type(obj) == h5py.Dataset) else ":"
        if type(obj) in [ h5py.Dataset ]:
            sdict[name] = obj.value
        if type(obj) in [ h5py.Group, h5py.File ]:
            sdict[name] = {}
            for sobj in obj.values():
                visit_group(sobj, sdict[name])

    # Run over the whole dataset
    for obj in hdf5_data.values():
        visit_group(obj, data)

        
    hdf5_data.close()
    return data


def to_hdf(d, filename):
    """Store a (possibly nested) dictionary to HDF5 file"""

    def store_recursively(fh, node, path=[]):
        if isinstance(node, dict):
            try:
                fh.create_group('/' + '/'.join(path))
            except ValueError:
                pass
            for key in sorted(node.iterkeys()):
                val = node[key]
                new_path = path + [key]
                store_recursively(fh=fh, node=val, path=new_path)
        else:
            fh.create_dataset(name = '/' + '/'.join(path),
                              data = node,
                              chunks = True,
                              maxshape = np.shape(node),
                              compression = None,
                              shuffle = True,
                              fletcher32 = False)

    try:
        hdf5_data = h5py.File(os.path.expandvars(filename), 'w')
    except IOError, e:
        logging.error("Unable to write to HDF5 file \'%s\'" % filename)
        logging.error(e)
        raise e

    try:
        store_recursively(fh=hdf5_data, node=d)
    finally:
        hdf5_data.close()

