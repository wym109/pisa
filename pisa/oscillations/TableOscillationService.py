#
# This is a service which will return an oscillation probability map
# corresponding to the desired binning.
#
# author: Timothy C. Arlen
#
# date:   April 2, 2014
#

import os, sys

import numpy as np
import h5py

from pisa.utils.log import logging
from pisa.oscillations.OscillationServiceBase import OscillationServiceBase
from pisa.utils.utils import get_smoothed_map
from pisa.resources.resources import find_resource


def get_osc_probLT_dict_hdf5(filename):
    '''
    Returns a dictionary of osc_prob_maps from the lookup table .hdf5 files. 
    '''
    try:
      fh = h5py.File(find_resource(filename),'r')
    except IOError,e:
      logging.error("Unable to open oscillation map file %s"%filename)
      logging.error(e)
      sys.exit(1)

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


class TableOscillationService(OscillationServiceBase):
    """
    This class handles all tasks related to the oscillation
    probability calculations...
    """
    def __init__(self,ebins,czbins,datadir='oscillations', **kwargs):
        OscillationServiceBase.__init__(self, ebins, czbins)
        
        self.datadir = datadir
    
    
    def get_osc_probLT_dict(self,deltam21=None,deltam31=None,theta12=None, 
                            theta13=None,theta23=None,deltacp=None,**kwargs):
        """
        Returns an oscillation probability map dictionary calculated 
        at the values of the input parameters:
          deltam21,deltam31,theta12,theta13,theta23,deltacp
        for flavor_from to flavor_to, with the binning of ebins,czbins.
        The dictionary is formatted as:
          'nue_maps': {'nue':map,'numu':map,'nutau':map},
          'numu_maps': {...}
          'nue_bar_maps': {...}
          'numu_bar_maps': {...}
        """
    
        ########################################################################
        ### TRUE ALGORITHM WHEN WE DECIDE ON HOW TO HANDLE OSC PROB DATA     ###
        # step 1: identify where the data is located: on disk or on server?    #
        # step 2: downsample these maps if not already done, for ebins, czbins #
        # step 3: do interpolation in oscillation parameters to arrive at the  #
        #         maps for (deltam21,deltam31,theta12,theta13,theta23,deltacp) #
        # return dictionary of smoothed, interpolated map.                     #
        ########################################################################
        
        if deltam31 > 0.0:
          filename = os.path.join(self.datadir,'oscProbLT_dm31_0.246_th23_38.645.hdf5') 
        else:
          filename = os.path.join(self.datadir+'oscProbLT_dm31_-0.238_th23_38.645.hdf5')
        logging.info("Loading file: %s"%filename)
        osc_probLT_dict = get_osc_probLT_dict_hdf5(filename)
        
        return osc_probLT_dict
