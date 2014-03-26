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
import h5py

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
 
def is_linear(edges, maxdev = 1e-5):
    '''Check wether the bin edges correspond to a linear axis'''
    linedges = np.linspace(edges[0],edges[-1],len(edges))
    return np.abs(edges-linedges).max() < maxdev 
    
def is_logarithmic(edges, maxdev = 1e-5):
    '''Check wether the bin edges correspond to a logarithmic axis'''
    logedges = np.logspace(np.log10(edges[0]),np.log10(edges[-1]),len(edges))
    return np.abs(edges-logedges).max() < maxdev 

def get_map_hdf5(filename,path):
    fh = h5py.File(filename,'r')
    op_map = fh[path]
    ebins  = np.array(fh[op_map.attrs['ebins']])
    czbins = np.array(fh[op_map.attrs['czbins']])
    op_map = np.array(op_map)
    fh.close()
    return op_map,ebins,czbins

def get_osc_probLT_dict_hdf5(filename):
    '''
    Returns a dictionary of osc_prob_maps from the lookup table .hdf5 files.
    '''
    fh = h5py.File(filename,'r')
    osc_prob_maps = {}
    osc_prob_maps['ebins'] = np.array(fh['ebins'])
    osc_prob_maps['czbins'] = np.array(fh['czbins'])

    for from_nu in ['nue','numu','nue_bar','numu_bar']:
        path_base = from_nu+'_maps'
        to_maps = {}
        to_nu_list = ['nue_bar','numu_bar','nutau_bar'] if 'bar' in from_nu else ['nue','numu','nutau']
        for to_nu in to_nu_list:
            op_map = np.array(fh[path_base+'/'+to_nu])
            to_maps[to_nu] = op_map
        osc_prob_maps[from_nu+'_maps'] = to_maps
        
    fh.close()
    
    return osc_prob_maps
        
def get_smoothed_map(prob_map, ebinsLT, czbinsLT, ebinsSM, czbinsSM):
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
    
    ebinsSM = get_bin_centers(ebinsSM)
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
            

#def downsample_map(pmap, binsx=0, binsy=0):
#    '''Downsample a map by integer factors in energy and/or cos(zen) by 
#       averaging over the merged bins.'''
#
#    #Make a copy of initial map
#    rmap = n.array(pmap)
#    
#    #Check that the map is dividable by this number
#    if len(n.nonzero(n.array(rmap.shape)%n.array([binsx,binsy]))[0]):
#        raise ValueError("Can not downsample map of size %s by factors of %u,%u"%
#                         (rmap.shape,binsx,binsy))
#    
#    #Average over each dimension
#    rmap = n.average([rmap[i::binsx,:] for i in range(binsx)],axis=0)
#    rmap = n.average([rmap[:,i::binsy] for i in range(binsy)],axis=0)
#    return rmap


