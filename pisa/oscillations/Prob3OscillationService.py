#
# This is a service which will return an oscillation probability map
# corresponding to the desired binning. Code is based on the prob3 oscillation
# package.
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# author: Sebastian Boeser
#         sboeser@physik.uni-bonn.de
#

import logging
import numpy as np
from datetime import datetime 
import os, sys
from pisa.utils.utils import get_smoothed_map, get_bin_centers, is_contained binning
from pisa.oscillations.prob3.BargerPropagator import BargerPropagator
from pisa.resources.resources import find_resource


class Prob3OscillationService:
    """
    This class handles all tasks related to the oscillation
    probability calculations...
    """
    def __init__(self, ebins, czbins,
                 earth_model='oscillations/PREM_60layer.dat',
                 detector_depth=2.0, prop_height=20.0, **kwargs):
        """
        Parameters needed to instantiate a Prob3OscillationService:
        * ebins: Energy bin edges
        * czbins: cos(zenith) bin edges
        * earth_model: Earth density model used for matter oscillations.
                       Default: 60-layer PREM model shipped with pisa.
        * detector_depth: Detector depth in km. Default: 2.0
        * prop_height: Height in the atmosphere to begin in km. 
                       Default: 20.0
        """
        self.ebins = ebins
        self.czbins = czbins
        self.prop_height = prop_height

        earth_model = find_resource(earth_model)
        
        self.barger_prop = BargerPropagator(earth_model, detector_depth)
        self.barger_prop.UseMassEigenstates(False)
        
        #for key in kwargs:
        #    logging.warn('Oscillation service received unnecessary keyword argument: %s'\
        #                 %key)

    #TODO: Move this method to the base class 
    #and only implement get_osc_probLT_dict individually?
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
        
        #TODO: Check here whether the requested binning is actually 
        #smaller and coarser than the one returned by the following method
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
    
    #TODO: maybe even this should move to the base class?
    def get_osc_probLT_dict(self,theta12,theta13,theta23,
                            deltam21,deltam31,deltacp,
                            ebins=None, czbins=None, **kwargs):
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

        # First initialize all empty maps to use in osc_prob_dict
        ebins = np.logspace(np.log10(1.0), np.log10(80.0), 501) \
                if ebins is None else ebins
        czbins = np.linspace(-1.0, 1.0, 501) \
                if czbins is None else czbins
        ecen = get_bin_centers(ebins)
        czcen = get_bin_centers(czbins)
        
        osc_prob_dict = {'ebins':ebins, 'czbins':czbins}
        shape = (len(ecen),len(czcen))
        for nu in ['nue_maps','numu_maps','nue_bar_maps','numu_bar_maps']:
            if 'bar' in nu:
                osc_prob_dict[nu] = {'nue_bar': np.zeros(shape,dtype=np.float32),
                                     'numu_bar': np.zeros(shape,dtype=np.float32),
                                     'nutau_bar': np.zeros(shape,dtype=np.float32)}
            else:
                osc_prob_dict[nu] = {'nue': np.zeros(shape,dtype=np.float32),
                                     'numu': np.zeros(shape,dtype=np.float32),
                                     'nutau': np.zeros(shape,dtype=np.float32)}
        
        self.fill_osc_prob(osc_prob_dict, ecen, czcen,
                           theta12=theta12, theta13=theta13, theta23=theta23,
                           deltam21=deltam21, deltam31=deltam31, deltacp=deltacp)
        
        return osc_prob_dict
      

    #NOTE: this (and __init__) are the only one that are specific for 
    #the prob3 oscillation code!
    def fill_osc_prob(self, osc_prob_dict, ecen,czcen,
                  theta12=None, theta13=None, theta23=None,
                  deltam21=None, deltam31=None, deltacp=None):
        '''
        Loops over ecen,czcen and fills the osc_prob_dict maps, with
        probabilities calculated according to NuCraft
        '''
        
        neutrinos = ['nue','numu','nutau']
        anti_neutrinos = ['nue_bar','numu_bar','nutau_bar']
        mID = ['','_bar']

        nu_barger = {'nue':1,'numu':2,'nutau':3,
                     'nue_bar':1,'numu_bar':2,'nutau_bar':3}
        
        logging.info("Defining osc_prob_dict from BargerPropagator...")
        # Set to false, since we are using sin^2(2 theta) variables
        kSquared = False
        sin2th12Sq = np.sin(2.0*theta12)**2
        sin2th13Sq = np.sin(2.0*theta13)**2
        sin2th23Sq = np.sin(2.0*theta23)**2
        
        total_bins = int(len(ecen)*len(czcen))
        mod = total_bins/50
        ibin = 0
        for icz, coszen in enumerate(czcen):
            
            for ie,energy in enumerate(ecen):
            
                ibin+=1
                if (ibin%mod) == 0: 
                    sys.stdout.write(".")
                    sys.stdout.flush()
                
                # In BargerPropagator code, it takes the "atmospheric
                # mass difference"-the nearest two mass differences, so
                # that it takes as input deltam31 for IMH and deltam32
                # for NMH                
                mAtm = deltam31 if deltam31 < 0.0 else (deltam31 - deltam21)

                ########### FIRST FOR NEUTRINOS ##########
                kNuBar = 1 # +1 for nu -1 for nubar
                self.barger_prop.SetMNS(sin2th12Sq,sin2th13Sq,sin2th23Sq,deltam21,mAtm,
                                        deltacp,energy,kSquared,kNuBar)
                self.barger_prop.DefinePath(coszen, self.prop_height)
                self.barger_prop.propagate(kNuBar)
                
                for nu in ['nue','numu']:
                    nu_i = nu_barger[nu]
                    nu = nu+'_maps'
                    for to_nu in neutrinos:
                        nu_f = nu_barger[to_nu]
                        osc_prob_dict[nu][to_nu][ie][icz]=self.barger_prop.GetProb(nu_i,nu_f)

                ########### SECOND FOR ANTINEUTRINOS ##########
                kNuBar = -1
                self.barger_prop.SetMNS(sin2th12Sq,sin2th13Sq,sin2th23Sq,deltam21,
                                        mAtm,deltacp,energy,kSquared,kNuBar)
                self.barger_prop.DefinePath(coszen, self.prop_height)
                self.barger_prop.propagate(kNuBar)

                for nu in ['nue_bar','numu_bar']:
                    nu_i = nu_barger[nu]
                    nu+='_maps'
                    for to_nu in anti_neutrinos:
                        nu_f = nu_barger[to_nu]
                        osc_prob_dict[nu][to_nu][ie][icz] = self.barger_prop.GetProb(nu_i,nu_f)
                        
                        
        print ""        
        
        return

        
    
