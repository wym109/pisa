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
    return np.sqrt(np.array(edges[:-1]*np.array(edges[1:]))) if is_logarithmic(edges) else  (np.array(edges[:-1])+np.array(edges[1:]))/2.

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
    logedges = np.logspace(np.log10(edges[0]),np.log10(edges[-1]),len(edges))
    return np.abs(edges-logedges).max() < maxdev 

def check_binning(edges1,edges2,maxdev=1e-8):
    '''Check whether the bin edges are equal.'''
    if (np.shape(edges1)[0]) != (np.shape(edges2)[0]): return False
    return np.abs(edges1 - edges2).max() < maxdev

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
    elist = []
    czlist = []
    weight_list = []

    ecenLT = get_bin_centers(ebinsLT)
    czcenLT = get_bin_centers(czbinsLT)
    for ie,egy in enumerate(ecenLT):
        for icz,cz in enumerate(czcenLT):
            czlist.append(cz)
            elist.append(egy)
            weight_list.append(prob_map[ie][icz])

    map_sum_wts = np.histogram2d(elist,czlist,weights=weight_list,
                                  bins=[ebinsSM,czbinsSM])[0]
    map_num = np.histogram2d(elist,czlist,bins=[ebinsSM,czbinsSM])[0]
    
    return np.divide(map_sum_wts,map_num)


def get_smoothed_map_old(prob_map, ebinsLT, czbinsLT, ebinsSM, czbinsSM):
     ''' 
     Downsamples a map by averaging over the merged bins. DOES NOT
     assume that the new binning is divisible by the old binning. FOR
     NOW, the algorithm simply asks if the old bin's center is inside
     the new bin, and if yes, includes it in the averaging. For very
     small original bin sizes, this should be sufficient.
     
     NOTATION: LT - "lookup table" (finely binned)
               SM - "smoothed" binning
     '''
 
     logging.info("Getting smoothed map...")
 
     shape = (np.shape(ebinsSM)[0]-1,np.shape(czbinsSM)[0]-1)
     smoothed_map = np.zeros(shape,dtype=np.float32)
     
     #ebinsSM = get_bin_centers(ebinsSM)
     for ie,egy in enumerate(ebinsSM[:-1]):
         emin = ebinsSM[ie]
         emax = ebinsSM[ie+1]
         for icz,cz in enumerate(czbinsSM[:-1]):
             czmin = czbinsSM[icz]
             czmax = czbinsSM[icz+1]
             smoothed_map[ie][icz] = get_smoothed_probability(prob_map,ebinsLT,czbinsLT,emin,emax,czmin,czmax)
             
     return smoothed_map
         
def get_smoothed_probability(prob_map,ebinsLT,czbinsLT,emin,emax,czmin,czmax):
    nbins = 0.0
    sum_weights = 0.0
    ebinsLT = get_bin_centers(ebinsLT)
    czbinsLT = get_bin_centers(czbinsLT)
    for ie,egy in enumerate(ebinsLT):
        if (egy < emin or egy > emax): continue
        for icz,cz in enumerate(czbinsLT):
            if ( cz < czmin or cz > czmax ): continue
            nbins+=1.0
            sum_weights += prob_map[ie][icz]
            
    return (sum_weights/nbins)
             
