#! /usr/bin/env python
#
# Flux.py
#
# Sample the atmospheric neutrino flux from a FluxService with a given
# binning in cos(zenith) and energy.
#
# If desired, this will create a .json output file with the results of
# the current stage of processing.
#
# author: Sebastian Boeser
#         sboeser@physik.uni-bonn.de
#
# date:   2014-01-27

import os
import sys
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pisa.utils.log import logging, physics, set_verbosity
from pisa.utils.jsons import from_json, to_json, json_string
from pisa.utils.proc import report_params, get_params, add_params
from pisa.analysis.stats.Maps import apply_ratio_scale
from pisa.flux.HondaFluxService import HondaFluxService, primaries

def apply_nue_numu_ratio(flux_maps, nue_numu_ratio):
    '''
    Applies the nue_numu_ratio systematic to the flux maps
    and returns the scaled maps. The actual calculation is
    done by apply_ratio_scale.
    '''
    # keep both nu and nubar flux constant
    scaled_nue_flux, scaled_numu_flux = apply_ratio_scale(
        orig_maps = flux_maps,
        key1 = 'nue', key2 = 'numu',
        ratio_scale = nue_numu_ratio,
        is_flux_scale = True
    )

    scaled_nue_bar_flux, scaled_numu_bar_flux = apply_ratio_scale(
        orig_maps = flux_maps,
        key1 = 'nue_bar', key2 = 'numu_bar',
        ratio_scale = nue_numu_ratio,
        is_flux_scale = True
    )

    flux_maps['nue']['map'] = scaled_nue_flux
    flux_maps['nue_bar']['map']  =  scaled_nue_bar_flux
    flux_maps['numu']['map'] = scaled_numu_flux
    flux_maps['numu_bar']['map']  = scaled_numu_bar_flux

    return flux_maps


def get_flux_maps(flux_service, ebins, czbins, nue_numu_ratio, energy_scale, **kwargs):
    '''
    Get a set of flux maps for the different primaries.

    \params:
      * flux_service -
      * ebins/czbins - energy/coszenith bins to calculate flux
      * nue_numu_ratio - systematic to be a proxy for the realistic
        Flux_nue/Flux_numu and Flux_nuebar/Flux_numubar ratios,
        keeping both the total flux from neutrinos and antineutrinos
        constant. The adjusted ratios are given by
        "nue_numu_ratio * original ratio".
      * energy_scale - factor to scale energy bin centers by
    '''

    #Be verbose on input
    params = get_params()
    report_params(params, units = [''])

    #Initialize return dict
    maps = {'params': params}

    for prim in primaries:

        #Get the flux for this primary
        maps[prim] = {'ebins': ebins,
                      'czbins': czbins,
                      'map': flux_service.get_flux(ebins*energy_scale,czbins,prim)}

        #be a bit verbose
        logging.trace("Total flux of %s is %u [s^-1 m^-2]"%
                      (prim,maps[prim]['map'].sum()))

    # now scale the nue(bar) / numu(bar) flux ratios, keeping the total
    # flux (nue + numu, nue_bar + numu_bar) constant, or return unscaled maps:
    return apply_nue_numu_ratio(maps, nue_numu_ratio) if nue_numu_ratio != 1.0 else maps

    #if nue_numu_ratio != 1.:
    #    return apply_nue_numu_ratio(maps, nue_numu_ratio)
    # else: no scaling to be applied
    #return maps


if __name__ == '__main__':

    # parser
    parser = ArgumentParser(description='Take a settings file '
        'as input and write out a set of flux maps',
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ebins', metavar='[1.0,2.0,...]', type=json_string,
                        help= '''Edges of the energy bins in units of GeV. ''',
                        default=np.logspace(np.log10(1.0),np.log10(80.0),40) )
    parser.add_argument('--czbins', metavar='[-1.0,-0.8.,...]', type=json_string,
                        help= '''Edges of the cos(zenith) bins.''',
                        default = np.linspace(-1.,0.,21))
    parser.add_argument('--flux_file', metavar='FILE', type=str,
                        help= '''Input flux file in Honda format. ''',
                        default = 'flux/spl-solmax-aa.d')
    parser.add_argument('--nue_numu_ratio',metavar='FLOAT',type=float,
                        help='''Factor to scale nue_flux by''',default=1.0)
    parser.add_argument('--energy_scale',metavar='FLOAT',type=float,
                        help='''Factor to scale TRUE energy by''',default=1.0)
    parser.add_argument('-o', '--outfile', dest='outfile', metavar='FILE',
                        type=str, action='store', default='flux.json',
                        help='file to store the output')
    parser.add_argument('-v', '--verbose', action='count', default=None,
                        help='set verbosity level')
    args = parser.parse_args()

    #Set verbosity level
    set_verbosity(args.verbose)

    logging.debug("Using %u bins in energy from %.2f to %.2f GeV"%
                                (len(args.ebins)-1,args.ebins[0],args.ebins[-1]))
    logging.debug("Using %u bins in cos(zenith) from %.2f to %.2f"%
                                (len(args.czbins)-1,args.czbins[0],args.czbins[-1]))

    #Instantiate a flux model
    flux_model = HondaFluxService(args.flux_file)

    #get the flux
    flux_maps = get_flux_maps(flux_model,args.ebins,args.czbins,
                              args.nue_numu_ratio,args.energy_scale)


    #write out to a file
    logging.info("Saving output to: %s"%args.outfile)
    to_json(flux_maps, args.outfile)
