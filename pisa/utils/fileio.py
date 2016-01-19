#
# fileio.py
#
# A set of utility function for generic file IO
#
# author: Justin Lanfranchi
#         jll1062@phys.psu.edu
#
# date:   2015-06-13
"""Generic file I/O, dispatching specific file readers/writers as necessary"""


import os
import pisa.utils.jsons as jsons
import pisa.utils.hdf as hdf
import pisa.resources.resources as resources
from pisa.utils.log import logging
import cPickle


JSON_EXTS = ['json']
HDF5_EXTS = ['hdf', 'h5', 'hdf5']
PKL_EXTS = ['pickle', 'pkl', 'p']


def from_pickle(fname):
    return cPickle.load(file(fname, 'rb'))


def to_pickle(obj, fname, overwrite=True):
    fpath = os.path.expandvars(os.path.expanduser(fname))
    if os.path.exists(fpath):
        if overwrite:
            logging.warn('Overwriting file at ' + fpath)
        else:
            raise Exception('Refusing to overwrite path ' + fpath)
    return cPickle.dump(obj, file(fname, 'wb'),
                        protocol=cPickle.HIGHEST_PROTOCOL)


def from_file(fname, fmt=None, **kwargs):
    """Dispatch correct file reader based on fmt (if specified) or guess
    based on file name's extension"""
    if fmt is None:
        _, ext = os.path.splitext(fname)
        ext = ext.replace('.', '').lower()
    else:
        ext = fmt.lower()
    fname = resources.find_resource(fname)
    if ext in JSON_EXTS:
        return jsons.from_json(fname, **kwargs)
    elif ext in HDF5_EXTS:
        return hdf.from_hdf(fname, **kwargs)
    elif ext in PKL_EXTS:
        return from_pickle(fname, **kwargs)
    else:
        errmsg = 'File "%s": unrecognized extension "%s"' % (fname, ext)
        logging.error(errmsg)
        raise TypeError(errmsg)


def to_file(obj, fname, fmt=None, **kwargs):
    """Dispatch correct file writer based on fmt (if specified) or guess
    based on file name's extension"""
    if fmt is None:
        _, ext = os.path.splitext(fname)
        ext = ext.replace('.', '').lower()
    else:
        ext = fmt.lower()
    if ext in JSON_EXTS:
        return jsons.to_json(obj, fname, **kwargs)
    elif ext in HDF5_EXTS:
        return hdf.to_hdf(obj, fname, **kwargs)
    elif ext in PKL_EXTS:
        return to_pickle(obj, fname, **kwargs)
    else:
        errmsg = 'Unrecognized file type/extension: ' + ext
        logging.error(errmsg)
        raise TypeError(errmsg)
