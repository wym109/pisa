#
# fileio.py
#
# A set of utility function for generic file IO
#
# author: Justin Lanfranchi
#         jll1062@phys.psu.edu
#
# date:   2015-06-13

from pisa.utils.jsons import *
from pisa.utils.hdf import *
import pisa.resources.resources as RES
from pisa.utils.log import logging, set_verbosity
import cPickle


JSON_EXTS = ['json']
HDF5_EXTS = ['hdf', 'h5', 'hdf5']
PKL_EXTS = ['pickle', 'pkl', 'p']


def from_file(fname, fmt=None):
    """Dispatch correct file reader based on fmt (if specified) or guess
    based on file name's extension"""
    if fmt is None:
        base, ext = os.path.splitext(fname)
        ext = ext.replace('.', '').lower()
    else:
        ext = fmt.lower()
    fname = RES.find_resource(fname)
    if ext in JSON_EXTS:
        return from_json(fname)
    elif ext in HDF5_EXTS:
        return from_hdf(fname)
    elif ext in PKL_EXTS:
        return cPickle.load(file(fname,'rb'))
    else:
        errmsg = 'Unrecognized file type/extension: ' + ext
        logging.error(errmsg)
        raise TypeError(errmsg)


def to_file(obj, fname, *args, **kwargs):
    """Dispatch correct file writer based on fmt (if specified) or guess
    based on file name's extension"""
    if fmt is None:
        base, ext = os.path.splitext(fname)
        ext = ext.replace('.', '').lower()
    else:
        ext = fmt.lower()
    if ext in JSON_EXTS:
        return to_json(obj, fname, *args, **kwargs)
    elif ext in HDF5_EXTS:
        return to_hdf(obj, fname, *args, **kwargs)
    elif ext in PKL_EXTS:
        return cPickle.dump(obj, file(fname, 'wb'),
                            protocol=cPickle.HIGHEST_PROTOCOL)
    else:
        errmsg = 'Unrecognized file type/extension: ' + ext
        logging.error(errmsg)
        raise TypeError(errmsg)
