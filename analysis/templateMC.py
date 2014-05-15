#! /usr/bin/env python
#
# templateMC.py
#
# Creates a template with user-inputted physics and detector
# parameters, using the MC-based approach.
#
# If desired, this will create a .json output file with the template
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# date:   24 April 2014
#

import logging
from argparse import ArgumentParser, RawTextHelpFormatter
from datetime import datetime
import numpy as np
from utils.utils import set_verbosity
from utils.json import from_json,to_json

from flux.Flux import get_flux_maps
from oscillations.OscillationMaps import get_osc_flux
from trigger.EventRate import get_event_rates
from reco.EventRateReco import get_event_rates_reco
from pid.ApplyPID import get_event_rates_pid

def get_templates(save_templates=False,**params):
    '''
    This function runs though the full template simulation chain,
    generating the tracks/cascades templates with the given set of
    input parameters.

    Returns a dictionary of the templates separated into tracks "trck"
    and cascades "cscd".
    ''' 

    ebins = params.pop('ebins')
    czbins = params.pop('czbins')
    flux_file = params.pop('flux_file')

    logging.debug("Using %u bins in energy from %.2f to %.2f GeV"%
                  (len(ebins)-1,ebins[0],ebins[-1]))
    logging.debug("Using %u bins in cos(zenith) from %.2f to %.2f"%
                  (len(czbins)-1,czbins[0],czbins[-1]))

    logging.info(">>Getting flux_maps...")
    time0 = datetime.now()
    flux_maps = get_flux_maps(flux_file,ebins,czbins,**params)
    logging.info(">>This took: %s\n"%(datetime.now() - time0))
    if save_templates: to_json(flux_maps,'flux_maps.json')
    
    logging.info(">>Getting osc_flux_maps...")
    time1 = datetime.now()
    osc_flux_maps = get_osc_flux(flux_maps,**params)
    logging.info(">>This took: %s\n"%(datetime.now() - time1))
    if save_templates: to_json(osc_flux_maps,'osc_flux_maps.json')
    
    logging.info(">>Getting event rates...")
    time2 = datetime.now()
    event_rate_maps = get_event_rates(osc_flux_maps,**params)
    logging.info(">>This took: %s\n"%(datetime.now() - time2))
    if save_templates: to_json(event_rate_maps,'event_rate_maps.json')

    logging.info(">>Getting reco event rates...")
    time3 = datetime.now()
    event_rate_reco_maps = get_event_rates_reco(event_rate_maps,**params)
    logging.info(">>This took: %s\n"%(datetime.now() - time3))
    if save_templates: to_json(event_rate_reco_maps,'event_rate_reco_maps.json')

    logging.info(">>Applying pid...")
    time4 = datetime.now()
    event_rate_pid = get_event_rates_pid(event_rate_reco_maps,
                                         from_json(params['pid_file']),**params)
    logging.info(">>This took: %s\n"%(datetime.now() - time4))
    if save_templates: to_json(event_rate_pid,'event_rate_pid_maps.json')
    
    return event_rate_pid
    

if __name__ == '__main__':
    set_verbosity(0)
    
    parser = ArgumentParser(description='Takes all systematic parameters as input '
                            'and writes out the final templates at analysis level.',
                            formatter_class=RawTextHelpFormatter)

    parser.add_argument('settings',type=from_json,
                        help='''settings.json file. Minimally this will have fields for  
  'ebins': [], 'czbins': [], 'flux_file',  'weighted_aeff_file':[ ]
where the ebins,czbins are the binning of the templates, 
  'flux_file' - is the atmospheric flux file in Honda format
  'weighted_aeff_file' - is an hdf5 file containing data from all flavours to analyze. ''')

    # Oscillation Parameter Inputs
    parser.add_argument('--deltam21',type=float,default=7.54e-5,
                        help='''deltam21 value [eV^2]''')
    parser.add_argument('--deltam31',type=float,default=0.00246,
                        help='''deltam31 value [eV^2]''')
    parser.add_argument('--theta12',type=float,default=np.deg2rad(33.647082),#0.5927,
                        help='''theta12 value [rad]''')
    parser.add_argument('--theta13',type=float,default=np.deg2rad(8.930817),#0.1588,
                        help='''theta13 value [rad]''')
    parser.add_argument('--theta23',type=float,default=np.deg2rad(38.645483),#0.7051,
                        help='''theta23 value [rad]''')
    parser.add_argument('--deltacp',type=float,default=0.0,
                        help='''deltaCP value to use [rad]''')
    parser.add_argument('--osc_code',type=str,default='Prob3',
                        help='''Oscillation prob code to use ['Prob3' (default) or 'NuCraft'] ''')
    
    # Event Counts Oscillated Parameter Inputs
    parser.add_argument('--nu_xsec_scale',type=float,default=1.0,
                        help='''Overall scale on nu xsec.''')
    parser.add_argument('--nubar_xsec_scale',type=float,default=1.0,
                        help='''Overall scale on nu_bar xsec.''')

    # Reco Counts Oscillated Parameter Inputs
    parser.add_argument('--e_reco_scale',type=float,default=1.0,
                        help='''Reconstructed energy scaling.''')
    parser.add_argument('--cz_reco_scale',type=float,default=1.0,
                        help='''Reconstructed coszen scaling.''')

    parser.add_argument('--save_all_stages',action='store_true',default=False,
                        help='Save all stages of output to .json files.')
    parser.add_argument('-o', '--outfile', dest='outfile', metavar='FILE', type=str,
                        action='store',default="templates.json",
                        help='file to store the output')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='set verbosity level')
    args = parser.parse_args()

    set_verbosity(args.verbose)

    params = vars(args)
    params.pop('verbose')
    outfile = params.pop('outfile')
    settings = params.pop('settings')
    save_templates = params.pop('save_all_stages')
    ebins = settings.pop('ebins')
    czbins = settings.pop('czbins')

    params = dict(params.items() + settings['params'].items())

    start_time = datetime.now()
    templates = get_templates(save_templates,ebins=ebins,czbins=czbins,**params)
    templates['params'] = params
    to_json(templates,outfile)

    logging.info("Total time taken to generate these templates: %s"%(datetime.now() - 
                                                                     start_time))
    
