#! /usr/bin/env python
#
# TemplateMaker.py
#
# Class to implement template-making procedure and to store as much data
# as possible to avoid re-running stages when not needed.
#
# author: Timothy C. Arlen - tca3@psu.edu
#         Sebastian Boeser - sboeser@uni-mainz.de
#
# date:   7 Oct 2014
#

import sys
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from scipy.constants import Julian_year

from pisa.utils.log import logging, profile, set_verbosity
from pisa.resources.resources import find_resource
from pisa.utils.params import get_fixed_params, get_free_params, get_values, select_hierarchy
from pisa.utils.jsons import from_json,to_json,json_string

from pisa.flux.HondaFluxService import HondaFluxService
from pisa.flux.Flux import get_flux_maps
from pisa.oscillations.Prob3OscillationService import Prob3OscillationService
from pisa.oscillations.NucraftOscillationService import NucraftOscillationService
from pisa.oscillations.Oscillation import get_osc_flux

from pisa.aeff.AeffServiceMC import AeffServiceMC
from pisa.aeff.AeffServicePar import AeffServicePar
from pisa.aeff.Aeff import get_event_rates

from pisa.reco.RecoServiceMC import RecoServiceMC
from pisa.reco.RecoServiceParam import RecoServiceParam
from pisa.reco.RecoServiceKDE import RecoServiceKDE
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
    def __init__(self,template_settings,ebins=None,czbins=None,oversample_e=None,
                 oversample_cz=None,**kwargs):
        '''
        TemplateMaker class handles all of the setup and calculation of the
        templates for a given binning.

        Parameters:
        * template_settings - dictionary of all template-making settings
        * ebins - energy bin edges
        * czbins - coszen bin edges
        '''

        self.ebins = ebins
        self.czbins = czbins
        self.oversample_e = oversample_e
        self.oversample_cz = oversample_cz
        logging.debug("Using %u bins in energy from %.2f to %.2f GeV"%
                      (len(self.ebins)-1,self.ebins[0],self.ebins[-1]))
        logging.debug("Using %u bins in cos(zenith) from %.2f to %.2f"%
                      (len(self.czbins)-1,self.czbins[0],self.czbins[-1]))

        #Instantiate a flux model service
        self.flux_service = HondaFluxService(**template_settings)

        # Oscillated Flux:
        if template_settings['osc_code']=='prob3':
            self.osc_service = Prob3OscillationService(self.ebins,self.czbins,
                                                       **template_settings)
        else:
            raise NotImplementedError('OscillationService is only implemented for prob3! osc_code = %s'%template_settings['osc_code'])
	    """
	    # not working yet
            self.osc_service = NucraftOscillationService(self.ebins,self.czbins,
							 **template_settings)
            """
  
        # Aeff/True Event Rate:
        if template_settings['parametric']:
            logging.info(" Using effective area from PARAMETRIZATION...")
            self.aeff_service = AeffServicePar(self.ebins,self.czbins,
                                               **template_settings)
        else:
            logging.info(" Using effective area from MC EVENT DATA...")
            self.aeff_service = AeffServiceMC(self.ebins,self.czbins,
                                              **template_settings)

        # Reco Event Rate:
        reco_mode = template_settings['reco_mode']
        if reco_mode == 'MC':
            self.reco_service = RecoServiceMC(self.ebins,self.czbins,
                                              **template_settings)
        elif reco_mode == 'param':
            self.reco_service = RecoServiceParam(self.ebins,self.czbins,
                                               **template_settings)
        elif reco_mode == 'kde':
            self.reco_service = RecoServiceKDE(self.ebins,self.czbins,
                                               **template_settings)
        else:
            error_msg = "reco_mode: %s is not implemented! "%reco_mode
            error_msg+=" Please choose among: ['MC','kde','param']"
            raise NotImplementedError(error_msg)

        # PID Service:
        self.pid_service = PIDServicePar(self.ebins,self.czbins,
                                         **template_settings)

        return


    def get_template(self,params,return_stages=False):
        '''
        Runs entire template-making chain, using parameters found in
        'params' dict. If 'return_stages' is set to True, returns
        output from each stage as a simple tuple.
        '''

        flux_maps = get_flux_maps(self.flux_service,self.ebins,self.czbins)

        logging.info("Getting osc prob maps...")
        osc_flux_maps = get_osc_flux(flux_maps,self.osc_service,
            oversample_e=self.oversample_e,oversample_cz=self.oversample_cz,**params)

        logging.info("Getting event rate true maps...")
        event_rate_maps = get_event_rates(osc_flux_maps,self.aeff_service, **params)

        logging.info("Getting event rate reco maps...")
        event_rate_reco_maps = get_reco_maps(event_rate_maps,self.reco_service,
                                             **params)

        logging.info("Getting pid maps...")
        final_event_rate = get_pid_maps(event_rate_reco_maps,self.pid_service)

        if not return_stages:
            return final_event_rate

        # Otherwise, return all stages as a simple tuple
        return (flux_maps, osc_flux_maps, event_rate_maps, event_rate_reco_maps,
                final_event_rate)

    def get_template_no_osc(self,params):
        '''
        Runs template making chain, but without oscillations
        '''

        flux_maps = get_flux_maps(self.flux_service,self.ebins,self.czbins)

        # Create the empty nutau maps:
        test_map = flux_maps['nue']

        flavours = ['nutau','nutau_bar']
        for flav in flavours:
            flux_maps[flav] = {'map': np.zeros_like(test_map['map']),
                               'ebins': np.zeros_like(test_map['ebins']),
                               'czbins': np.zeros_like(test_map['czbins'])}

        logging.info("Getting event rate true maps...")
        event_rate_maps = get_event_rates(flux_maps,self.aeff_service, **params)

        logging.info("Getting event rate reco maps...")
        event_rate_reco_maps = get_reco_maps(event_rate_maps,self.reco_service,
                                             **params)

        logging.info("Getting pid maps...")
        final_event_rate = get_pid_maps(event_rate_reco_maps,self.pid_service)

        return final_event_rate


if __name__ == '__main__':

    # parser
    parser = ArgumentParser(description='''Runs the template making process.''',
                         formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t','--template_settings',type=str,
                        metavar='JSONFILE', required = True,
                        help='''settings for the template generation''')
    hselect = parser.add_mutually_exclusive_group(required=False)
    hselect.add_argument('--normal', dest='normal', default=True,
                        action='store_true', help="select the normal hierarchy")
    hselect.add_argument('--inverted', dest='normal', default = False,
                        action='store_false', help="select the inverted hierarchy")
    parser.add_argument('-v','--verbose',action='count',default=None,
                        help='set verbosity level.')
    parser.add_argument('-s','--save_all',action='store_true',default=False,
                        help="Save all stages.")
    parser.add_argument('-o', '--outfile', dest='outfile', metavar='FILE', type=str,
                        action='store',default="template.json",
                        help='file to store the output')
    args = parser.parse_args()

    set_verbosity(args.verbose)

    profile.info("start initializing")

    #Load all the settings
    model_settings = from_json(args.template_settings)

    #Select a hierarchy
    logging.info('Selected %s hierarchy'%
            ('normal' if args.normal else 'inverted'))
    params =  select_hierarchy(model_settings['params'],normal_hierarchy=args.normal)

    #Intialize template maker
    template_maker = TemplateMaker(get_values(params),**model_settings['binning'])

    profile.info("stop initializing")

    #Now get the actual template
    profile.info("start template calculation")
    template_maps = template_maker.get_template(get_values(params),return_stages=args.save_all)
    profile.info("stop template calculation")

    logging.info("Saving file to %s"%args.outfile)
    to_json(template_maps,args.outfile)
