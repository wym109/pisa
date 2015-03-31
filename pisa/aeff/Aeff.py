#! /usr/bin/env python
#
# Aeff.py
#
# This module is the implementation of the stage2 analysis. The main
# purpose of stage2 is to combine the "oscillated Flux maps" with the
# effective areas to create oscillated event rate maps, using the true
# information.
#
# If desired, this will create a .json output file with the results of
# the current stage of processing.
#
# author: Timothy C. Arlen
#
#         tca3@psu.edu
#
# date:   April 8, 2014
#

import os,sys
import numpy as np
from scipy.constants import Julian_year
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from pisa.utils.log import logging, set_verbosity
from pisa.utils.utils import check_binning, get_binning
from pisa.utils.jsons import from_json, to_json
from pisa.utils.proc import report_params, get_params, add_params
from pisa.resources.resources import find_resource

from pisa.aeff.AeffServiceMC import AeffServiceMC
from pisa.aeff.AeffServicePar import AeffServicePar
from pisa.analysis.stats.Maps import apply_ratio_scale


def apply_nu_nubar_ratio(event_rate_maps, nu_nubar_ratio):
    '''
    Applies the nu_nubar_ratio systematic to the event rate
    maps and returns the scaled maps. The actual calculation is
    done by apply_ratio_scale.
    '''
    flavours = event_rate_maps.keys()
    if 'params' in flavours:
        flavours.remove('params')

    for flavour in flavours:
        # process nu and nubar in one go
        if not 'bar' in flavour:
             # do this for each interaction channel (cc and nc)
             for int_type in event_rate_maps[flavour].keys():
                 scaled_nu_rates, scaled_nubar_rates = apply_ratio_scale(
                                                           orig_maps = event_rate_maps,
                                                           key1 = flavour, key2 = flavour+'_bar',
                                                           ratio_scale = nu_nubar_ratio,
                                                           is_flux_scale = False,
                                                           int_type = int_type
                                                           )

                 event_rate_maps[flavour][int_type]['map'] = scaled_nu_rates
                 event_rate_maps[flavour+'_bar'][int_type]['map'] = scaled_nubar_rates

    return event_rate_maps


def get_event_rates(osc_flux_maps,aeff_service,livetime=None,nu_nubar_ratio=None,
                    aeff_scale=None,**kwargs):
    '''
    Main function for this module, which returns the event rate maps
    for each flavor and interaction type, using true energy and zenith
    information. The content of each bin will be the weighted aeff
    multiplied by the oscillated flux, so that the returned dictionary
    will be of the form:
    {'nue': {'cc':map,'nc':map},
     'nue_bar': {'cc':map,'nc':map}, ...
     'nutau_bar': {'cc':map,'nc':map} }
    \params:
      * osc_flux_maps - maps containing oscillated fluxes
      * aeff_service - the effective area service to use
      * livetime - detector livetime for which to calculate event counts
      * nu_nubar_ratio - systematic to be a proxy for the realistic
        counts_nue(cc/nc) / counts_nuebar(cc/nc), ... ratios,
        keeping the total flavour counts constant.
        The adjusted ratios are given by "nu_nubar_ratio * original ratio".
      * aeff_scale - systematic to be a proxy for the realistic effective area
    '''

    #Get parameters used here
    params = get_params()
    report_params(params,units = ['','yrs',''])

    #Initialize return dict
    event_rate_maps = {'params': add_params(params,osc_flux_maps['params'])}

    #Get effective area
    aeff_dict = aeff_service.get_aeff()

    ebins, czbins = get_binning(osc_flux_maps)

    # apply the scaling for nu_xsec_scale and nubar_xsec_scale...
    flavours = ['nue','numu','nutau','nue_bar','numu_bar','nutau_bar']
    for flavour in flavours:
        osc_flux_map = osc_flux_maps[flavour]['map']
        int_type_dict = {}
        for int_type in ['cc','nc']:
            event_rate = osc_flux_map*aeff_dict[flavour][int_type]*aeff_scale

            event_rate *= (livetime*Julian_year)
            int_type_dict[int_type] = {'map':event_rate,
                                       'ebins':ebins,
                                       'czbins':czbins}
            logging.debug("  Event Rate before reco for %s/%s: %.2f"
                          %(flavour,int_type,np.sum(event_rate)))
        event_rate_maps[flavour] = int_type_dict

    # now scale the nu(e/mu/tau) / nu(e/mu/tau)bar event count ratios, keeping the total
    # (nue + nuebar etc.) constant
    if nu_nubar_ratio != 1.:
        return apply_nu_nubar_ratio(event_rate_maps, nu_nubar_ratio)

    # else: no scaling to be applied
    return event_rate_maps

if __name__ == '__main__':

    parser = ArgumentParser(description='Take an oscillated flux file '
                          'as input & write out a set of oscillated event counts. ',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('osc_flux_maps',metavar='FLUX',type=from_json,
                     help='''JSON osc flux input file with the following parameters:
      {"nue": {'czbins':[], 'ebins':[], 'map':[]},
       "numu": {...},
       "nutau": {...},
       "nue_bar": {...},
       "numu_bar": {...},
       "nutau_bar": {...} }''')
    parser.add_argument('--weighted_aeff_file',metavar='WEIGHTFILE',type=str,
                        default='events/V15_weighted_aeff.hdf5',
                        help='''HDF5 File containing event data for each flavours for a particular
instrumental geometry. The effective area is calculated from the event
weights in this file. Only applies in non-parametric mode.''')
    parser.add_argument('--settings_file',metavar='SETTINGS',type=str,
                        default='aeff/V36_aeff.json',
                        help='''json file containing parameterizations of the effective
area and its cos(zenith) dependence. Only applies in parametric mode.''')
    parser.add_argument('--livetime',type=float,default=1.0,
                        help='''livetime in years to re-scale by.''')
    parser.add_argument('--nu_nubar_ratio',type=float,default=1.0,
                        help='''Overall scale on nu xsec.''')
    parser.add_argument('--aeff_scale',type=float,default=1.0,
                        help='''Overall scale on aeff''')
    parser.add_argument('--mc_mode',action='store_true', default=False,
                        help='''Use MC-based effective areas instead of using the parameterized versions.''')
    parser.add_argument('-o', '--outfile', dest='outfile', metavar='FILE', type=str,
                        action='store',default="event_rate.json",
                        help='''file to store the output''')
    parser.add_argument('-v', '--verbose', action='count', default=None,
                        help='''set verbosity level''')
    args = parser.parse_args()

    #Set verbosity level
    set_verbosity(args.verbose)

    #Check binning
    ebins, czbins = check_binning(args.osc_flux_maps)

    logging.info("Defining aeff_service...")

    if args.mc_mode:
        logging.info("  Using effective area from EVENT DATA...")
        aeff_service = AeffServiceMC(ebins,czbins,aeff_weight_file=args.weighted_aeff_file)
    else:
        logging.info("  Using effective area from PARAMETRIZATION...")
        aeff_settings = from_json(find_resource(args.settings_file))
        aeff_service = AeffServicePar(ebins,czbins,**aeff_settings)


    event_rate_maps = get_event_rates(args.osc_flux_maps,aeff_service,args.livetime,
                                      args.nu_nubar_ratio,args.aeff_scale)

    logging.info("Saving output to: %s"%args.outfile)
    to_json(event_rate_maps,args.outfile)


