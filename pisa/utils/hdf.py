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
import pisa.utils.utils as utils


def from_hdf(val):
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
    myfile = False
    if isinstance(val, basestring):
        root = h5py.File(os.path.expandvars(os.path.expanduser(val)), 'r')
        myfile = True
    else:
        root = val
        logging.trace('root = %s, root.values() = %s' % (root, root.values()))
    try:
        # Run over the whole dataset
        for obj in root.values():
            visit_group(obj, data)
    finally:
        if myfile:
            root.close()

    return data


def to_hdf(data_dict, tgt, attrs=None, overwrite=True):
    """
    Store a (possibly nested) dictionary to an HDF5 file or branch node within
    an HDF5 file (an h5py Group).
    
    This creates hardlinks for duplicate non-trivial leaf nodes (h5py Datasets)
    to minimize storage space required for redundant datasets. Duplication is
    detected via object hashing.
    
    NOTE: Branch nodes are sorted before storing for consistency in the
    generated file despite Python dictionaries having no defined ordering among
    keys.
    
    Arguments
    ---------
    data_dict : dict
        Dictionary to be stored
    tgt : str or h5py.Group
        Target for storing data. If `tgt` is a str, it is interpreted as a
        filename; a file is created with that name (overwriting an existing
        file, if present). After writing, the file is closed. If `tgt` is an
        h5py.Group, the data is simply written to that Group and it is left
        open at function return.
    attrs : dict
        Attributes to apply to the root node of the file. See
        http://docs.h5py.org/en/latest/high/attr.html
    """
    if not isinstance(data_dict, dict):
        errmsg = 'to_hdf: `data_dict` only accepts top-level dict.'
        logging.error(errmsg)
        raise TypeError(errmsg)

    # Define a function for interatively doing the work
    global node_hashes
    node_hashes = {}
    def store_recursively(fhandle, node, path=None, attrs=None):
        global node_hashes
        if path is None:
            path = []
        if node_hashes is None:
            node_hashes = {}
        full_path = '/' + '/'.join(path)
        if isinstance(node, dict):
            logging.trace("  creating Group '%s'" % full_path)
            try:
                grp = fhandle.create_group(full_path)
                if attrs is not None:
                    for key in sorted(attrs.keys()):
                        dset.attrs[key] = attrs[key]
            except ValueError:
                pass
            for key in sorted(node.keys()):
                key_str = str(key)
                if not isinstance(key, str):
                    logging.warn("Stringifying key '" + key_str +
                                 "'for use as name in HDF5 file")
                val = node[key]
                new_path = path + [key_str]
                store_recursively(fhandle=fhandle, node=val, path=new_path)
        else:
            # Check for existing node
            node_hash = utils.hash_obj(node)
            if node_hash in node_hashes:
                logging.trace("  creating hardlink for Dataset: '%s' -> '%s'" %
                              (full_path, node_hashes[node_hash]))
                # Hardlink the matching existing dataset
                fhandle[full_path] = fhandle[node_hashes[node_hash]]
                return
            # For now, convert None to np.nan since h5py appears to not handle None
            if node is None:
                node = np.nan
                logging.warn("  encountered `None` at node '%s'; converting to"
                             " np.nan" % full_path)
            # "Scalar datasets don't support chunk/filter options". Shuffling
            # is a good idea otherwise since subsequent compression will
            # generally benefit; shuffling requires chunking. Compression is
            # not done here since it is slow, but can be done by
            # post-processing the generated file(s).
            if np.isscalar(node):
                shuffle = False
                chunks = None
            else:
                shuffle = True
                chunks = True
                # Store the node_hash for linking to later if this is more than
                # a scalar datatype. Assumed that "None" has 
                node_hashes[node_hash] = full_path
            # TODO: Treat strings as follows? Would this break compatibility
            # with pytables/Pandas? What are benefits? Leaving out for now.
            # if isinstance(node, basestring):
            #     dtype = h5py.special_dtype(vlen=str)
            #     fh.create_dataset(k,data=v,dtype=dtype)
            logging.trace("  creating dataset at node '%s', hash %s" %
                          (full_path, node_hash))
            try:
                dset = fhandle.create_dataset(
                    name=full_path, data=node, chunks=chunks, compression=None,
                    shuffle=shuffle, fletcher32=False
                )
            except TypeError:
                try:
                    shuffle = False
                    chunks = None
                    dset = fhandle.create_dataset(
                        name=full_path, data=node, chunks=chunks,
                        compression=None, shuffle=shuffle, fletcher32=False
                    )
                except:
                    logging.error('  full_path: ' + full_path)
                    logging.error('  chunks   : ' + str(chunks))
                    logging.error('  shuffle  : ' + str(shuffle))
                    logging.error('  node     : ' + str(node))
                    raise

            if attrs is not None:
                for key in sorted(attrs.keys()):
                    dset.attrs[key] = attrs[key]

    # Perform the actual operation using the dict passed in by user
    if isinstance(tgt, basestring):
        fpath = os.path.expandvars(os.path.expanduser(tgt))
        if os.path.exists(fpath):
            if overwrite:
                logging.warn('Overwriting file at ' + fpath)
            else:
                raise Exception('Refusing to overwrite path ' + fpath)
        try:
            h5file = h5py.File(fpath, 'w')
            if attrs is not None:
                h5file.attrs.update(attrs)
            store_recursively(fhandle=h5file, node=data_dict)
        except IOError, e:
            logging.error(e)
            logging.error("to_hdf: Unable to open `%s` for writing" % tgt)
            raise
        finally:
            h5file.close()
    elif isinstance(tgt, h5py.Group):
        store_recursively(fhandle=tgt, node=data_dict, attrs=attrs)
    else:
        errmsg = "to_hdf: Invalid `tgt` type: " + type(target_entity)
        logging.error(errmsg)
        raise TypeError(errmsg)
