#! /usr/bin/env python
#
# This is a service which will return an oscillation probability map
# corresponding to the desird binning.
#
# author: Timothy C. Arlen
#
# date:   April 2, 2014
#

import logging
import numpy as np
from datetime import datetime
import h5py
import os, sys
from utils.utils import get_smoothed_map, get_bin_centers
from BargerPropagator import BargerPropagator

def get_osc_probLT_dict_hdf5(filename):
    '''
    Returns a dictionary of osc_prob_maps from the lookup table .hdf5 files. 
    '''
    try:
      fh = h5py.File(filename,'r')
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

def SaveHDF5(filename,oscprob_dict):
    # I grabbed this from OscProbMaps.py on hammer, it is just to test
    # my prob calculator for Barger, et al.
    import h5py
    fh = h5py.File(filename,'w')
    logging.info("Saving file: %s",filename)
    
    edata = fh.create_dataset('ebins',data=oscprob_dict['ebins'],dtype=np.float32)
    czdata = fh.create_dataset('czbins',data=oscprob_dict['czbins'],dtype=np.float32)
    
    for key in oscprob_dict.keys():
        if 'maps' in key:
            logging.info("  key %s",key)
            group_base = fh.create_group(key)
            for subkey in oscprob_dict[key].keys():
                logging.info("    subkey %s",subkey)
                dset = group_base.create_dataset(subkey,data=oscprob_dict[key][subkey],
                                                 dtype=np.float32)
                dset.attrs['ebins'] = edata.ref
                dset.attrs['czbins'] = czdata.ref
        
    fh.close()
    return

class OscillationService:
    """
    This class handles all tasks related to the oscillation
    probability calculations...
    """
    def __init__(self,ebins,czbins):
        self.ebins = ebins
        self.czbins = czbins
        self.barger_prop = BargerPropagator()
        self.barger_prop.UseMassEigenstates(False)

        # For lookup table binning:
        self.eminLT = 1.0
        self.emaxLT = 80.0
        self.nebinsLT = 500
        self.czminLT = -1.0
        self.czmaxLT = 1.0
        self.nczbinsLT = 500
        
        return
    
    def get_osc_prob_maps(self,deltam21=None,deltam31=None,theta12=None, 
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
        NOTE: expects all angles in [rad]
        """
            
        ### TEMPORARY SOLUTION:
        #filename = 'normal_oscelot.251cz_x_251e.hdf5' if deltam31 > 0.0 else 'invert_oscelot.251cz_x_251e.hdf5'
        #logging.info("Loading file: %s"%filename)
        #osc_probLT_dict = get_osc_probLT_dict_hdf5(filename)
        
        osc_probLT_dict = self.get_osc_probLT_dict(theta12,theta13,theta23,
                                              deltam21,deltam31,deltacp)
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

            
        logging.info("Finshed getting smoothed maps. This took: %s"%(datetime.now()-start_time))
        
        return smoothed_maps
    
    def get_osc_probLT_dict(self,theta12,theta13,theta23,deltam21,deltam31,
                            deltacp):
        '''
        This will create the oscillation probability map lookup tables
        (LT) corresponding to atmospheric neutrinos oscillation
        through the earth, and will return a dictionary of maps:
        {'nue_maps':[to_nue_map, to_numu_map, to_nutau_map],
         'numu_maps: [...],
         'nue_bar_maps': [...], 
         'numu_bar_maps': [...], 
         'czbins':czbins, 
         'ebins': ebins} 
        Uses the BargerPropagator code to calculate the individual
        probabilities on the fly.

        NOTE: Expects all angles to be in [rad], and all deltam to be in [eV^2]
        '''

        # Set to false, since we are using sin^2(2 theta) variables
        kSquared = False
        sin2th12Sq = np.sin(2.0*theta12)**2
        sin2th13Sq = np.sin(2.0*theta13)**2
        sin2th23Sq = np.sin(2.0*theta23)**2
        prop_height = 25.00  # Height in the atmosphere to begin (default= 25 km)
        
        ebins = np.logspace(np.log10(self.eminLT),np.log10(self.emaxLT),self.nebinsLT+1)
        czbins = np.linspace(self.czminLT,self.czmaxLT,self.nczbinsLT+1)
        ecen = get_bin_centers(ebins)
        czcen = get_bin_centers(czbins)

        logging.info("Defining osc_prob_dict from BargerPropagator...")
        
        osc_prob_dict = {'ebins':ebins, 'czbins':czbins}
        shape = (len(ecen),len(czcen))
        neutrinos = ['nue','numu','nutau']
        mID = ['','_bar']
        nu_barger = {'nue':1,'numu':2,'nutau':3}

        # First initialize all empty maps to use.
        for nu in ['nue_maps','numu_maps','nue_bar_maps','numu_bar_maps']:
            if 'bar' in nu:
                osc_prob_dict[nu] = {'nue_bar': np.zeros(shape,dtype=np.float32),
                                     'numu_bar': np.zeros(shape,dtype=np.float32),
                                     'nutau_bar': np.zeros(shape,dtype=np.float32)}
            else:
                osc_prob_dict[nu] = {'nue': np.zeros(shape,dtype=np.float32),
                                     'numu': np.zeros(shape,dtype=np.float32),
                                     'nutau': np.zeros(shape,dtype=np.float32)}

        for ie,energy in enumerate(ecen):
            for icz, coszen in enumerate(czcen):

                ########### FIRST FOR NEUTRINOS ##########
                kNuBar = 1 # +1 for nu -1 for nubar

                #In BargerPropagator code, it takes the "atmospheric
                #mass difference"-the nearest two mass differences, so
                #that it takes as input deltam31 for IMH and deltam32
                #for NMH
                mAtm = deltam31 if deltam31 < 0.0 else (deltam31 - deltam21)
                self.barger_prop.SetMNS(sin2th12Sq,sin2th13Sq,sin2th23Sq,deltam21,mAtm,
                                        deltacp,energy,kSquared,kNuBar)
                self.barger_prop.DefinePath(coszen, prop_height)
                self.barger_prop.propagate(1*kNuBar)

                for nu in ['nue','numu']:
                    nu_i = nu_barger[nu]
                    nu = nu+'_maps'
                    for to_nu in neutrinos:
                        nu_f = nu_barger[to_nu]
                        osc_prob_dict[nu][to_nu][ie][icz] = self.barger_prop.GetProb(nu_i,nu_f)
                        
                ########### SECOND FOR ANTINEUTRINOS ##########
                kNuBar = -1
                self.barger_prop.SetMNS(sin2th12Sq,sin2th13Sq,sin2th23Sq,deltam21,
                                        deltam31,deltacp,energy,kSquared,kNuBar)
                self.barger_prop.DefinePath(coszen, prop_height)
                self.barger_prop.propagate(1*kNuBar)
                
                for nu in ['nue','numu']:
                    nu_i = nu_barger[nu]
                    nu+=(mID[1]+'_maps')
                    for to_nu in neutrinos:
                        nu_f = nu_barger[to_nu]
                        to_nu+=mID[1]
                        osc_prob_dict[nu][to_nu][ie][icz] = self.barger_prop.GetProb(nu_i,nu_f)

        #SaveHDF5('barger_test_osc_prob.hdf5',osc_prob_dict)

        return osc_prob_dict

