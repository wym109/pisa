#! /usr/bin/env python
#
# utils.py
#
# A set of utility function to deal with maps, etc...
# 
# author: Sebastian Boeser
#         sboeser@physik.uni-bonn.de
#
# date:   2014-01-27

import logging
import numpy as np


def get_bin_centers(edges):
    '''Get the bin centers for a given set of bin edges.
       This works even if bins don't have equal width.'''
    return (np.array(edges[:-1])+np.array(edges[1:]))/2.


def get_bin_sizes(edges):
    '''Get the bin sizes for a given set of bin edges.
       This works even if bins don't have equal width.'''
    return np.array(edges[1:]) - np.array(edges[:-1]) 


def set_verbosity(verbosity):
    '''Set the verbosity level for the root logger,
       along with some better formatting.'''
    levels = {0:logging.WARN,
              1:logging.INFO,
              2:logging.DEBUG}
    logging.basicConfig(format='[%(levelname)8s] %(message)s')
    logging.root.setLevel(levels[min(2,verbosity)])
    

