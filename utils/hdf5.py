#! /usr/bin/env python
#
# hdf5.py
#
# A set of utilities for dealing with hdf5 files.
# Import hdf5 from this module everywhere 
# 
# author: Tim Arlen
#         tca3@psu.edu
#
# date:   2014-03-30
#

import logging
import h5py
import numpy as np

def get_map_hdf5(filename,path):
    '''
    description needed...
    '''
    fh = h5py.File(filename,'r')
    op_map = fh[path]
    ebins  = np.array(fh[op_map.attrs['ebins']])
    czbins = np.array(fh[op_map.attrs['czbins']])
    op_map = np.array(op_map)
    # check the map dimensions match ebins,czbins?
    fh.close()
    return op_map,ebins,czbins

###################
# NOTE: Move this next function to the OscillationService class?
# Also, better description needed...
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
