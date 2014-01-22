#! /usr/bin/env python
#
# stage1FullMC.py
#
# This module will control the first stage of the PINGU
# simulation-creating the oscillated Flux maps, controlled by two
# primary physics inputs: 1) the atmospheric flux model and 2) the
# oscillation probability weights.
#
# If desired, this will create a .json output file with the results of
# the current stage of processing.
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# date:   Jan. 21, 2014
#

# Stage1Implementation

## IMPORTS ##
import os,sys
import numpy as np
import logging
#from argparse import ArgumentParser, FileType

# Until python2.6, default json is very slow.
try: 
    import simplejson as json
except ImportError, e:
    import json
    
def get_oscillated_flux(osc_param_dict, earth_model, 
     atm_flux_file="../inputFiles/AtmFluxModel_Honda2006.json",
                        atm_flux_scale=1.0,atm_flux_dIndex = 0.0):
    """
    Primary module function. Produces a map for each of
    nue, nue_bar, numu, numu_bar into their respective other three
    flavours (12 maps total). 
    Inputs:
      -- osc_param_dict: dictionary for deltam31, deltam21, theta32, theta31, 
           theta21, deltacp
      -- earth_model: string corresponding to the earth model desired
      -- atm_flux_file: string filename of the atm_flux_file.json to use with
           the oscillation parameters.
      -- atm_flux_scale: scaling of atm_flux_file (to simulate uncertainty in
           atmospheric flux normalization)
      -- atm_flux_dIndex: spectral index variation (to simulatd uncertainty in
           atmospheric spectral index/shape)
    """
    
    atm_flux_dict,czbins,ebins = load_atm_flux(atm_flux_file,atm_flux_scale,
                                               atm_flux_dIndex)
    osc_prob_dict,czbins_,ebins_ = load_osc_prob(osc_param_dict,earth_model)
    
    if not check_same_bins(czbins,ebins,czbins_,ebins_):
        exitMsg = "ERROR: atmospheric flux model and osc probability maps do NOT have the same binning!!"
        sys.exit(exitMsg)

    # Create the oscillated flux maps
    # save to .json file.
    
    
    osc_flux_map_dict = ""

    #print "\n  atm_flux_dict.keys(): ",atm_flux_dict.keys()
    #print "\n  osc_flux_map_dict.keys(): ",osc_prob_dict.keys()
    
    return osc_flux_map_dict

def load_atm_flux(atm_flux_file,atm_flux_scale,atm_flux_dIndex):
    
    atm_flux_dict = json.load(open(atm_flux_file),'r')
    czbins = atm_flux_dict['maps_nue']['czbins']
    ebins = atm_flux_dict['maps_nue']['ebins']
    
    return atm_flux_dict,czbins,ebins

def load_osc_prob(osc_param_dict,earth_model):

    # NOTE: For now, we're sticking these oscProbMaps in a directory,
    # and grabbing them when needed. We may put them into a mysql
    # database or something else later. When we decide, need to
    # rewrite this function/step

    osc_prob_dir = "../inputFiles/oscProbMaps/"
    basename = osc_prob_dir+"oscProbMapSmoothFull_dM31Sq"
    ending = "_40_20_800.json"
    dM31Sq = osc_param_dict['deltam31']
    sin2th23Sq = osc_param_dict['sin2th23Sq']
    osc_prob_filename = basename+str(dM31Sq)+"_s2th23Sq"+str(sin2th23Sq)+ending
    
    osc_prob_dict = json.load(open(osc_prob_filename),'r')
    czbins = osc_prob_dict['maps_nue'][0]['nue']['czbins']
    ebins = osc_prob_dict['maps_nue'][0]['nue']['ebins']    
    
    return osc_prob_dict,czbins,ebins

def check_same_bins(czbins1,ebins1,czbins2,ebins2):
    if(len(czbins1) != len(czbins2)):
        print "ERROR! czbins sizes not equal!"
        print "  -->num czbins1: "+str(len(czbins1))+" num czbins2: "+str(len(czbins2))
        return False
        #sys.exit()
    if(len(ebins1) != len(ebins2)):
        print "ERROR! ebins sizes not equal!"
        print "  -->num ebins1: "+str(len(ebins1))+" num ebins2: "+str(len(ebins2))
        return False
        #sys.exit()
    if( (czbins1[0] != czbins2[0]) or (czbins1[-1] != czbins2[-1]) ):
        print "ERROR! czbins not equal!"
        print "  -->czbins1: ",czbins1
        print "  -->czbins2: ",czbins2
        return False
        #sys.exit()
    if( (ebins1[0] != ebins2[0]) or (ebins1[-1] != ebins2[-1]) ):
        print "ERROR! ebins not equal!"
        print "  -->ebins1: ",ebins1
        print "  -->ebins2: ",ebins2
        return False
        #sys.exit()
        
    return True
        
if __name__ == '__main__':
    
    osc_param_dict = {'deltam31':0.246,'sin2th23Sq':0.942}
    earth_model='prem0'
    osc_flux_map_dict = get_oscillated_flux(osc_param_dict, earth_model, 
                atm_flux_file="../inputFiles/AtmFluxModel_Honda2006.json")
    
    
