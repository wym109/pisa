#! /usr/bin/env python
#
#

import logging
import numpy as np
from datetime import datetime
from utils.hdf5 import get_osc_probLT_dict_hdf5
from utils.utils import get_smoothed_map
import os

class OscillationService:
    """
    This class handles all tasks related to the oscillation
    probability calculations...
    """
    def __init__(self,ebins,czbins,datadir=None):
        self.ebins = ebins
        self.czbins = czbins
        self.datadir = "" if datadir==None else datadir

        return
    
    def get_osc_prob_maps(self,deltam21=7.54e-5,deltam31=None,theta12=33.647, 
                          theta13=8.931,theta23=None,deltacp=0.0):
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
        
        ### TEMPORARY SOLUTION:
        # for now, I will grab the data from the local directory:
        #import os
        #maps_dir = os.getenv('PISA')+'/resources/oscProbMaps/ebins500_czbins500/'
        # for now, no interpolation
        filename = self.datadir+'oscProbLT_dm31_0.246_th23_38.645.hdf5' if deltam31 > 0.0 else self.datadir+'oscProbLT_dm31_-0.238_th23_38.645.hdf5'
        logging.info("Loading file: %s"%filename)
        osc_probLT_dict = get_osc_probLT_dict_hdf5(filename)
        ebinsLT = osc_probLT_dict['ebins']
        czbinsLT = osc_probLT_dict['czbins']
        
        start_time = datetime.now()
        logging.info("Getting smoothed maps...")

        # do smoothing
        smoothed_maps = {}
        smoothed_maps['ebins'] = self.ebins
        smoothed_maps['czbins'] = self.czbins
        for from_nu in ['nue','numu','nue_bar','numu_bar']:
            path_base = from_nu+'_maps'
            to_maps = {}
            to_nu_list = ['nue_bar','numu_bar','nutau_bar'] if 'bar' in from_nu else ['nue','numu','nutau']
            for to_nu in to_nu_list:
                logging.info("Getting smoothed map %s"%(from_nu+'_maps/'+to_nu))
                to_maps[to_nu]=get_smoothed_map(osc_probLT_dict[from_nu+'_maps'][to_nu],
                                                ebinsLT,czbinsLT,self.ebins,self.czbins)
                
            smoothed_maps[from_nu+'_maps'] = to_maps

        total_time = (datetime.now() - start_time)
        logging.info("Finshed getting smoothed maps. This took: %s"%(datetime.now()-start_time))
        
        return smoothed_maps
    
