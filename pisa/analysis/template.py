#! /usr/bin/env python
#
# template.py
#
# Creates a template from user-inputted physics and detector
# parameters, utilizing the MC-based or parametric-based approach.
# 
# author: Steven Wren - steven.wren@hep.manchester.ac.uk
# author: Tim Arlen - tca3@psu.edu
#
# Date:   5 August 2014
#

import sys, logging
from datetime import datetime
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pisa.utils.utils import set_verbosity
from pisa.utils.jsons import from_json, to_json
from pisa.flux.HondaFluxService import HondaFluxService
from pisa.flux.Flux import get_flux_maps
from pisa.oscillations.OscillationService import OscillationService
from pisa.oscillations.Prob3OscillationService import Prob3OscillationService
from pisa.oscillations.Oscillation import get_osc_flux
from pisa.aeff.AeffServiceMC import AeffServiceMC
from pisa.aeff.AeffServicePar import AeffServicePar
from pisa.aeff.Aeff import get_event_rates
from scipy.constants import Julian_year
from pisa.reco.RecoService import RecoServiceMC
from pisa.reco.Reco import get_reco_maps
from pisa.pid.PIDServicePar import PIDServicePar
from pisa.pid.PID import get_pid_maps
from pisa.resources.resources import find_resource

def get_template(settings,save_stages=False):
  '''
  Run through the entire template generation chain, and at the end
  produces templates for trck/cscd channels for the given set of input
  parameters.
  
  settings - dictionary of settings relevant for template-making.
  save_stages - saves all stages inside file
  '''
  
  ebins = settings['ebins']
  czbins = settings['czbins']
  flux_file = settings['flux_file']
  
  logging.debug("Using %u bins in energy from %.2f to %.2f GeV"%
                (len(ebins)-1,ebins[0],ebins[-1]))
  logging.debug("Using %u bins in cos(zenith) from %.2f to %.2f"%
                (len(czbins)-1,czbins[0],czbins[-1]))
  
  #Instantiate a flux model
  flux_model = HondaFluxService(flux_file, **settings)
  
  #get the flux 
  flux_maps = get_flux_maps(flux_model,ebins,czbins)
  
  # Oscillated Flux:
  if settings['osc_code']=='prob3':
    osc_service = Prob3OscillationService(ebins,czbins,
                                          earth_model=settings['earth_model'])
  else:
    osc_service = OscillationService(ebins,czbins)
    
  logging.info("Getting osc prob maps")
  osc_flux_maps = get_osc_flux(flux_maps,osc_service,**settings)

  # True Event Rate:
  logging.info("Defining aeff_service...")
  if settings['parametric']:
    logging.info("Using effective area from PARAMETRIZATION...")
    aeff_service = AeffServicePar(ebins,czbins,settings=settings)
  else:
    logging.info("  Using effective area from EVENT DATA...")
    aeff_service = AeffServiceMC(ebins,czbins,simfile=settings['aeff_weight_file'])    
  event_rate_maps = get_event_rates(osc_flux_maps,aeff_service,**settings)

  # Reco Event Rate:
  logging.info("Defining RecoService...")
  reco_service = RecoServiceMC(ebins,czbins,simfile=settings['aeff_weight_file'])
  event_rate_reco_maps = get_reco_maps(event_rate_maps,reco_service,**settings)
  
  # Apply PID
  pid_service = PIDServicePar(settings['particle_ID'],ebins,czbins)
  final_event_rate = get_pid_maps(event_rate_reco_maps,pid_service)

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
    parser = ArgumentParser(description='Take a settings file '
        'as input and do all the processes currently chained '
        'together in default_chain.sh. This will only output '
        'at the end of the chain.',
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('settings', metavar='SETTNGS', type=str,
                        help = 'JSON file holding the settings for analysis')    
    
    # Oscillation Parameter Inputs
    parser.add_argument('--deltam21',type=float,default=7.54e-5,
                        help='''deltam21 value [eV^2]''')
    parser.add_argument('--deltam31',type=float,default=0.002465,
                        help='''deltam31 value [eV^2]''')
    parser.add_argument('--theta12',type=float,default=float(np.deg2rad(33.647082)),
                        help='''theta12 value [rad]''')
    parser.add_argument('--theta13',type=float,default=float(np.deg2rad(8.930817)),
                        help='''theta13 value [rad]''')
    parser.add_argument('--theta23',type=float,default=0.671,
                        help='''theta23 value [rad]''')
    parser.add_argument('--deltacp',type=float,default=0.0,
                        help='''deltaCP value to use [rad]''')    
    
    # Detector Inputs:
    parser.add_argument('--nu_xsec_scale',type=float,default=1.0,
                        help='''Overall scale on nu xsec.''')
    parser.add_argument('--nubar_xsec_scale',type=float,default=1.0,
                        help='''Overall scale on nu_bar xsec.''')
    parser.add_argument('--e_reco_scale',type=float,default=1.0,
                        help='''Reconstructed energy scaling.''')
    parser.add_argument('--cz_reco_scale',type=float,default=1.0,
                        help='''Reconstructed coszen scaling.''')

    parser.add_argument('--save_stages',action='store_true',
                        help="Stores all intermediate stages.")
    parser.add_argument('-t', '--time', action='store_true',
                        help="Displays total running time for template generation.")
    parser.add_argument('-o', '--outfile', dest='outfile', metavar='FILE', type=str, 
                        action='store', default=None, help='file to store the output')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='set verbosity level')
    
    args = parser.parse_args()

    #Set verbosity level
    set_verbosity(args.verbose)
    
    # load settings file
    settings = from_json(args.settings)

    if args.time: start_time = datetime.now()  
      
    params = vars(args)
    template_params = dict(settings['template'].items() + params.items())
    # Get rid of unwanted args:
    for key in ['verbose','outfile','settings','time']: template_params.pop(key)

    chain_output = {}
    chain_output = get_template(template_params,save_stages=args.save_stages)

    chain_output['params'] = template_params
    if args.outfile is not None: 
      logging.info("Saving output to file: %s"%args.outfile)
      to_json(chain_output, args.outfile)
    
    if args.time: 
      print "\n  Template generation process took: %s"%(datetime.now() - start_time)
      
