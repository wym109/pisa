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
from pisa import utils


def from_hdf(filename):
    """Open a file in HDF5 format, parse the content and return as dictionary
    with numpy arrays"""
    # Function for iteratively parsing the file to create the dictionary
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

    data = {}
    try:
        h5file = h5py.File(os.path.expandvars(filename), 'r')
        # Run over the whole dataset
        for obj in h5file.values():
            visit_group(obj, data)
    except IOError, e:
        logging.error("Unable to read HDF5 file \'%s\'" % filename)
        logging.error(e)
        raise e
    finally:
        h5file.close()

    return data


def to_hdf(d, filename):
    """Store a (possibly nested) dictionary to HDF5 file, creating hardlinks
    for repeated leaf nodes (datasets).
    
    NOTE: Branch nodes are sorted before storing for consistency in the
    generated file despite Python dictionaries having no defined ordering among
    keys."""
    if not isinstance(d, dict):
        errmsg = 'Only dictionaries may be written to HDF5 files.'
        logging.error(errmsg)
        raise TypeError(errmsg)

    # Define a function for iteratively doing the work
    def store_recursively(fhandle, node, path=[], node_hashes={}):
        full_path = '/' + '/'.join(path)
        if isinstance(node, dict):
            try:
                fhandle.create_group(full_path)
            except ValueError:
                pass
            for key in sorted(node.iterkeys()):
                key_str = str(key)
                if not isinstance(key, str):
                    logging.warn('Stringifying key "' + key_str +
                                 '"for use as name in HDF5 file')
                val = node[key]
                new_path = path + [key_str]
                store_recursively(fhandle=fhandle, node=val, path=new_path,
                                  node_hashes=node_hashes)
        else:
            # Check for existing node
            node_hash = utils.utils.hash_obj(node)
            if node_hash in node_hashes:
                # Hardlink the matching existing dataset
                fhandle[full_path] = fhandle[node_hashes[node_hash]]
                return
            node_hashes[node_hash] = full_path
            # "Scalar datasets don't support chunk/filter options"; extra
            # checking that a sequence isn't a string, also. Shuffling is
            # a good idea since subsequent compression will generally benefit;
            # shuffling requires chunking. Compression is not done here
            # since it is slow.
            if hasattr(node, '__iter__') and not isinstance(node, basestring):
                shuffle = True
                chunks = True
            else:
                shuffle = False
                chunks = None
            fhandle.create_dataset(name=full_path, data=node, chunks=chunks,
                              compression=None, shuffle=shuffle,
                              fletcher32=False)
    
    # Perform the actual operation using the dict passed in by user
    try:
        h5file = h5py.File(os.path.expandvars(filename), 'w')
        store_recursively(fhandle=h5file, node=d)
    except IOError, e:
        logging.error("Unable to write to HDF5 file \'%s\'" % filename)
        logging.error(e)
        raise e
    finally:
        h5file.close()
