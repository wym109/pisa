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

"""
def is_contained_binning(small_bins, large_bins):
    '''Check whether small_bins lie inside of large_bins'''
    if (len(np.shape(small_bins)) != len(np.shape(large_bins))): return False
    #Make it iterable
    if len(np.shape(small_bins) == 1):
        small_bins, large_bins = [small_bins], [large_bins]
    #Check for all given axes
    for sml_ax, lrg_ax in zip(small_bins, large_bins):
        if ((sml_ax[0] < lrg_ax[0]) or (sml_ax[-1] > lrg_ax[-1])): return False
    return True
"""

def subbinning(coarse_bins, fine_bins, maxdev=1e-8):
    '''Check whether coarse_bins can be retrieved from fine_bins 
       via integer rebinning'''
    rebin_info = []
    #Make it iterable
    if (len(np.shape(coarse_bins)) == 1):
        coarse_bins, fine_bins = [coarse_bins], [fine_bins]
    
    for crs_ax, fn_ax in zip(coarse_bins, fine_bins):
        #Test all possible positions...
        for start in range(len(fn_ax)-len(crs_ax)):
            #...and rebin factors
            for rebin in range(1, (len(fn_ax)-start)/len(crs_ax)+1):
                stop = start+len(crs_ax)*rebin
                if is_equal_binning(crs_ax, 
                                    fn_ax[start:stop:rebin],
                                    maxdev=maxdev):
                    rebin_info.append((start, stop, rebin))
                    break
            else: continue # if no matching binning was found (no break)
            break # executed if 'continue' was skipped (break)
        else: break # don't search on if no binning found for first axis
    
    if (len(rebin_info) == len(coarse_bins)):
        #Matching binning was found for all axes
        return rebin_info
    else:
        return False

def get_binning(d, iterate=False, eset=[], czset=[]):
    '''Iterate over all maps in the dict, and return the ebins and czbins.
       If iterate is False, will return the first set of ebins, czbins it finds,
       otherwise will return a list of all ebins and czbins arrays'''
    #Only work on dicts
    if not type(d) == dict: return

    #Check if we are on map level
    if (sorted(d.keys()) == ['czbins','ebins','map']):
        #Immediately return if we found one
        if not iterate:
            return np.array(d['ebins']),np.array(d['czbins'])
        else:
            eset += [np.array(d['ebins'])]
            czset += [np.array(d['czbins'])]
    #Otherwise iterate through dict
    else:
        for v in d.values():
            bins = get_binning(v,iterate,eset,czset)
            if bins and not iterate: return bins

    #In iterate mode, return sets
    return eset, czset

def check_binning(data):
    '''
    Check wether all maps in data have the same binning, and return it.
    '''
    eset, czset = get_binning(data,iterate=True)
   
    for binset, label in zip([eset,czset],['energy','coszen']):
      if not np.alltrue([is_equal_binning(binset[0],bins)
                         for bins in binset[1:]]):
          raise Exception('Maps have different %s binning!'%label)

    return eset[0],czset[0]
    
    
#NOTE: Investigate whether we should use scipy.misc.imresize for this?
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


def integer_rebin_map(prob_map, rebin_info):
    '''
    Rebins a map (or a part of it) by an integer factor in every dimension.
    Merged bins will be averaged.
    '''
    #TODO: implement
    raise NotImplementedError
    
    #Make a copy of initial map
    rmap = np.array(prob_map)
    dim = len(rebin_info)
    
    for start, stop, rebin in np.array(rebin_info).T[::-1]:
        #Roll last axis to front
        rmap = np.rollaxis(rmap, dim-1)
        #Select correct part and average
        rmap = np.average([rmap[start:stop-1:rebin] for i in range(rebin)], 
                          axis=0)
    
    return rmap
