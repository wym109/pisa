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
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# date:   2014-01-27

import os
import sys
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from pisa.analysis.stats.Maps import apply_ratio_scale
from pisa.flux.HondaFluxService import HondaFluxService, primaries
from pisa.utils.jsons import from_json, to_json, json_string
from pisa.utils.log import logging, physics, set_verbosity
from pisa.utils.proc import report_params, get_params, add_params
from pisa.utils.utils import get_bin_centers


def apply_nue_numu_ratio(flux_maps, nue_numu_ratio):
    """
    Applies the nue_numu_ratio systematic to the flux maps
    and returns the scaled maps. The actual calculation is
    done by apply_ratio_scale.
    """
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

def apply_nu_nubar_ratio(event_rate_maps, nu_nubar_ratio):
    """
    Applies the nu_nubar_ratio systematic to the event rate
    maps and returns the scaled maps. The actual calculation is
    done by apply_ratio_scale.
    """
    flavours = event_rate_maps.keys()
    if 'params' in flavours: flavours.remove('params')

    for flavour in flavours:
        # process nu and nubar in one go
        if not 'bar' in flavour:
            # do this for each interaction channel (cc and nc)
            scaled_nu_rates, scaled_nubar_rates = apply_ratio_scale(
                orig_maps = event_rate_maps,
                key1 = flavour, key2 = flavour+'_bar',
                ratio_scale = nu_nubar_ratio,
                is_flux_scale = True,
            )
            event_rate_maps[flavour]['map'] = scaled_nu_rates
            event_rate_maps[flavour+'_bar']['map'] = scaled_nubar_rates

    return event_rate_maps


def apply_delta_index(flux_maps, delta_index, egy_med):
    """
    Applies the spectral index systematic to the flux maps by scaling
    each bin with (egy_cen/egy_med)^(-delta_index), preserving the total
    integral flux  Note that only the numu/numu_bar are scaled, because
    the nue_numu_ratio will handle the systematic on the nue flux.
    """

    for flav in ['numu','numu_bar']:
        ecen = get_bin_centers(flux_maps[flav]['ebins'])
        scale = np.power((ecen/egy_med),delta_index)
        flux_map = flux_maps[flav]['map']
        total_flux = flux_map.sum()
        logging.trace("flav: %s, total counts before scale: %f"%(flav,total_flux))
        scaled_flux = (flux_map.T*scale).T
        scaled_flux *= (total_flux/scaled_flux.sum())
        flux_maps[flav]['map'] = scaled_flux
        logging.trace("flav: %s, total counts after scale: %f"%
                      (flav,flux_maps[flav]['map'].sum()))

    return flux_maps

def get_median_energy(flux_map):
    """Returns the median energy of the flux_map-expected to be a dict
    with keys 'map', 'ebins', 'czbins'
    """

    ecen = get_bin_centers(flux_map['ebins'])
    energy = ecen[len(ecen)/2]

    return energy

def get_flux_maps(flux_service, ebins, czbins, nue_numu_ratio, nu_nubar_ratio,
                  energy_scale, atm_delta_index,**kwargs):
    """
    Get a set of flux maps for the different primaries.

    \params:
      * flux_service -
      * ebins/czbins - energy/coszenith bins to calculate flux
      * nue_numu_ratio - systematic to be a proxy for the realistic
        Flux_nue/Flux_numu and Flux_nuebar/Flux_numubar ratios,
        keeping both the total flux from neutrinos and antineutrinos
        constant. The adjusted ratios are given by
        "nue_numu_ratio * original ratio".
      * nu_nubar_ratio - systematic to be a proxy for the
        neutrino/anti-neutrino production/cross section ratio.
      * energy_scale - factor to scale energy bin centers by
      * atm_delta_index  - change in spectral index from fiducial
    """

    # Be verbose on input
    params = get_params()
    report_params(params, units = [''])

    # Initialize return dict
    maps = {'params': params}

    for prim in primaries:

        # Get the flux for this primary
        maps[prim] = {'ebins': ebins,
                      'czbins': czbins,
                      'map': flux_service.get_flux(ebins*energy_scale,czbins,prim)}

        # be a bit verbose
        logging.trace("Total flux of %s is %u [s^-1 m^-2]"%
                      (prim,maps[prim]['map'].sum()))

    # now scale the nue(bar) / numu(bar) flux ratios, keeping the total
    # Flux (nue + numu, nue_bar + numu_bar) constant, or return unscaled maps:
    scaled_maps = apply_nue_numu_ratio(maps, nue_numu_ratio) if nue_numu_ratio != 1.0 else maps

    # now scale the nu(e/mu) / nu(e/mu)bar event count ratios, keeping the total
    # (nue + nuebar etc.) constant
    if nu_nubar_ratio != 1.:
        scaled_maps = apply_nu_nubar_ratio(scaled_maps, nu_nubar_ratio)

    median_energy = get_median_energy(maps['numu'])
    if atm_delta_index != 0.0:
        scaled_maps = apply_delta_index(scaled_maps, atm_delta_index, median_energy)

    return scaled_maps


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
    parser.add_argument('--nu_nubar_ratio',metavar='FLOAT',type=float,
                        help='''Factor to scale nu_nubar_flux by''',default=1.0)
    parser.add_argument('--delta_index',metavar='FLOAT',type=float,
                        default=0.0,help='''Shift in spectral index of numu''')
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
    flux_maps = get_flux_maps(
        flux_model, args.ebins, args.czbins, args.nue_numu_ratio, args.nu_nubar_ratio,
        args.energy_scale, args.delta_index)

    #write out to a file
    logging.info("Saving output to: %s"%args.outfile)
    to_json(flux_maps, args.outfile)
