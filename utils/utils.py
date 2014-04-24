#
# utils.py
#
# A set of utility function to deal with maps, etc...
# 
# author: Sebastian Boeser
#         sboeser@physik.uni-bonn.de
#
# author: Tim Arlen
#         tca3@psu.edu
#
# date:   2014-01-27

import logging
import numpy as np


def get_bin_centers(edges):
    '''Get the bin centers for a given set of bin edges.
       This works even if bins don't have equal width.'''
    if is_logarithmic(edges):
        return np.sqrt(np.array(edges[:-1]*np.array(edges[1:])))
    else:
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
 
def is_linear(edges, maxdev = 1e-5):
    '''Check whether the bin edges correspond to a linear axis'''
    linedges = np.linspace(edges[0],edges[-1],len(edges))
    return np.abs(edges-linedges).max() < maxdev 
    
def is_logarithmic(edges, maxdev = 1e-5):
    '''Check whether the bin edges correspond to a logarithmic axis'''
    if np.any(np.array(edges) < 0): return False
    logedges = np.logspace(np.log10(edges[0]),np.log10(edges[-1]),len(edges))
    return np.abs(edges-logedges).max() < maxdev 

def is_equal_binning(edges1,edges2,maxdev=1e-8):
    '''Check whether the bin edges are equal.'''
    if (np.shape(edges1)[0]) != (np.shape(edges2)[0]): return False
    return np.abs(edges1 - edges2).max() < maxdev

def remove_downgoing(hist,czbins):
    '''
    Takes a histogram, and the czbins and removes the downgoing
    (coszen > 0.0) bins and the corresponding part of the map. This is
    needed, since neutrinos with TRUE coszen values above the horizon
    will be reconstructed as upgoing events, and we need to keep track
    of them throughout the template making process, but when we have a
    final template for the likelihood analysis and to plot them, we
    want to show and calculate only the reconstructed upgoing events.
    '''
    czlen = len(czbins)
    czbins = czbins[np.where(czbins<=0.0)]
    bin_rm = (czlen - len(czbins))
    nebins = np.shape(hist)[0]
    nczbins = len(czbins) - 1
    hist_new = np.zeros((nebins,nczbins), dtype=np.float32)
    for ie,egy in enumerate(hist[:]):
        hist_new[ie] = hist[ie][:-bin_rm]

    return hist_new,czbins
    
# NOTE: Investigate whether we should use scipy.misc.imresize for this?
def get_smoothed_map(prob_map,ebinsLT,czbinsLT,ebinsSM,czbinsSM):
    '''
    Downsamples a map by averaging over the look up table bins whose
    bin center is within the new (coarser) binning. DOES NOT assume
    that the new (SM) binning is divisible by the old (LT)
    binning. The algorithm is that a new histogram is created from the
    entirety of the data in the Lookup Table.
    
    NOTATION: LT - "lookup table" (finely binned)
              SM - "smoothed" binning
    '''
    
    ecenLT = get_bin_centers(ebinsLT)
    czcenLT = get_bin_centers(czbinsLT)

    elist = []
    czlist = []
    weight_list = []
    for ie,egy in enumerate(ecenLT):
        for icz,cz in enumerate(czcenLT):
            czlist.append(cz)
            elist.append(egy)
            weight_list.append(prob_map[ie][icz])
            
    map_sum_wts = np.histogram2d(elist,czlist,weights=weight_list,
                                 bins=[ebinsSM,czbinsSM])[0]
    map_num = np.histogram2d(elist,czlist,bins=[ebinsSM,czbinsSM])[0]
    
    return np.divide(map_sum_wts,map_num)

