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
    
def downsample_map(pmap, binsx=0, binsy=0):
    '''Downsample a map by integer factors in energy and/or cos(zen) by 
       averaging over the merged bins.'''

    #Make a copy of initial map
    rmap = n.array(pmap)
    
    #Check that the map is dividable by this number
    if len(n.nonzero(n.array(rmap.shape)%n.array([binsx,binsy]))[0]):
        raise ValueError("Can not downsample map of size %s by factors of %u,%u"%
                         (rmap.shape,binsx,binsy))
    
    #Average over each dimension
    rmap = n.average([rmap[i::binsx,:] for i in range(binsx)],axis=0)
    rmap = n.average([rmap[:,i::binsy] for i in range(binsy)],axis=0)
    return rmap


