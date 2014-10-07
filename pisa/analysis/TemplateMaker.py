#! /usr/bin/env python
#
# TemplateMaker.py
#
# Class to implement template-making procedure and to store as much data
# as possible to avoid re-running stages when not needed.
#
# author: Timothy C. Arlen - tca3@psu.edu
# 
# date:   7 Oct 2014
#

import logging,sys
import numpy as np
from datetime import datetime
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from scipy.constants import Julian_year

from pisa.resources.resources import find_resource
from pisa.analysis.LLR.LLRAnalysis import get_fiducial_params
from pisa.utils.jsons import from_json,to_json,json_string
from pisa.utils.utils import set_verbosity

from pisa.flux.HondaFluxService import HondaFluxService
from pisa.flux.Flux import get_flux_maps
from pisa.oscillations.Prob3OscillationService import Prob3OscillationService
from pisa.oscillations.Oscillation import get_osc_flux
from pisa.aeff.AeffServiceMC import AeffServiceMC
from pisa.aeff.AeffServicePar import AeffServicePar
from pisa.aeff.Aeff import get_event_rates
from pisa.reco.RecoService import RecoServiceMC
from pisa.reco.Reco import get_reco_maps
from pisa.pid.PIDServicePar import PIDServicePar
from pisa.pid.PID import get_pid_maps

class TemplateMaker:
    '''
    This class handles all steps needed to produce a template with a
    constant binning.
    
    The strategy employed will be to define all 'services' in the
    initialization process, make them members of the class, then use
    them later when needed.
     
    
    '''
    def __init__(self,binning,temp_settings):
        '''
        TemplateMaker class handles all of the setup and calculation of the 
        templates for a given binning.
        
        input - temp_settings, which is a dictionary directly obtained from a
        geometry_settings.json file.
        '''
        
        self.ebins = binning['ebins']
        self.czbins = binning['czbins']
        
        logging.debug("Using %u bins in energy from %.2f to %.2f GeV"%
                      (len(self.ebins)-1,self.ebins[0],self.ebins[-1]))
        logging.debug("Using %u bins in cos(zenith) from %.2f to %.2f"%
                      (len(self.czbins)-1,self.czbins[0],self.czbins[-1]))
        
        #Instantiate a flux model service
        self.flux_service = HondaFluxService(temp_settings['flux_file'])
                
        # Oscillated Flux:
        if temp_settings['osc_code']=='prob3':
            self.osc_service = Prob3OscillationService(self.ebins,self.czbins,
                               earth_model=temp_settings['earth_model'])
        else:
            raise Exception('OscillationService is only implemented for prob3! osc_code = %s'%osc_code)
        
        # Aeff/True Event Rate:
        if temp_settings['parametric']:
            logging.info("  Using effective area from PARAMETRIZATION...")
            self.aeff_service = AeffServicePar(self.ebins,self.czbins,
                              temp_settings['aeff_egy_par'],
                              temp_settings['aeff_coszen_par'])
        else:
            logging.info("  Using effective area from MC EVENT DATA...")
            self.aeff_service = AeffServiceMC(self.ebins,self.czbins,
                         simfile=temp_settings['aeff_weight_file'])    
            
        # Reco Event Rate:
        self.reco_service = RecoServiceMC(self.ebins,self.czbins,
                      simfile=temp_settings['aeff_weight_file'])
        
        # PID Service:
        self.pid_service = PIDServicePar(temp_settings['particle_ID'],
                                         self.ebins,self.czbins)
        
        return
 
    
    def get_template(self,fiducial,save_stages=False):
        '''
        Makes a template, from specific parameters found in fiducial dict.
        '''
        
        flux_maps = get_flux_maps(self.flux_service,self.ebins,self.czbins)
        
        logging.info("Getting osc prob maps...")
        osc_flux_maps = get_osc_flux(flux_maps,self.osc_service,**fiducial)
        
        logging.info("Getting event rate true maps...")
        event_rate_maps = get_event_rates(osc_flux_maps,self.aeff_service,
                                          **fiducial)
        
        logging.info("Getting event rate reco maps...")
        event_rate_reco_maps = get_reco_maps(event_rate_maps,self.reco_service,
                                             **fiducial)
        
        logging.info("Getting pid maps...")
        final_event_rate = get_pid_maps(event_rate_reco_maps,self.pid_service)
        
        if save_stages:
            final_event_rate['stages'] = {}
            stages = final_event_rate['stages']
            stages['flux_maps'] = flux_maps
            stages['osc_flux_maps'] = osc_flux_maps
            stages['event_rate_maps'] = event_rate_maps
            stages['event_rate_reco_maps'] = event_rate_reco_maps
            
        return final_event_rate
    
    
if __name__ == '__main__':

    #Only show errors while parsing
    set_verbosity(0)
    
    # parser
    parser = ArgumentParser(description='''Runs the template making process.''',
                         formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('settings',type=str,help="settings.json file to use.")
    parser.add_argument('-v','--verbose',action='count',default=0,
                        help='set verbosity level.')
    args = parser.parse_args()
    
    set_verbosity(args.verbose)

    time0 = datetime.now()  
    
    settings = from_json(args.settings)
    temp_maker = TemplateMaker(settings['binning'],settings['template'])

    time1 = datetime.now()
    print "\n  >>Finished initializing in %s sec.\n"%(time1 - time0)
    
    fiducial_params = get_fiducial_params(settings)
    temp_maker.get_template(fiducial_params)
    
    print "Finished getting template in %s sec."%(datetime.now() - time1)
    print "total time to run: %s sec."%(datetime.now() - time0)
    
