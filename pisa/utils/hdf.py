#
# hdf.py
#
# A set of utilities for dealing with HDF5 files.
# 
# author: Sebastian Boeser
#         sboeser@uni-mainz.de
#
# date:   2015-03-05

import os
import sys
import numpy as np
import h5py
from pisa.utils.log import logging


def from_hdf(filename):
    '''Open a file in HDF5 format, parse the content and return as dictionary
       with numpy arrays'''
    try:
        hdf5_data = h5py.File(os.path.expandvars(filename),'r')
    except IOError, e:
        logging.error("Unable to read HDF5 file \'%s\'"%filename)
        logging.error(e)
        sys.exit(1)

    data = {}

    #Iteratively parse the file to create the dictionary
    def visit_group(obj,sdict): 
        name = obj.name.split('/')[-1]
        #indent = len(obj.name.split('/'))-1
        #print "  "*indent,name, obj.value if (type(obj) == h5py.Dataset) else ":"
        if type(obj) in [ h5py.Dataset ]:
            sdict[name] = obj.value
        if type(obj) in [ h5py.Group, h5py.File ]:
            sdict[name] = {}
            for sobj in obj.values():
                visit_group(sobj,sdict[name])


    #run over the whole dataset
    for obj in hdf5_data.values():
        visit_group(obj,data)
    return data
