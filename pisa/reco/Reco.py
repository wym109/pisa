#! /usr/bin/env python
#
# Reco.py
#
# This module will perform the smearing of the true event rates, with
# the reconstructed parameters, using the detector response
# resolutions, in energy and coszen.
# Therefore, a RecoService is invoked that generates smearing kernels
# by some specific algorithm (see individual services for details).
# Then the true event rates are convoluted with the kernels to get the
# event rates after reconstruction.
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# author: Lukas Schulte
#         schulte@physik.uni-bonn.de
#
# date:   April 9, 2014
#

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import itertools as itertools

import numpy as np

from pisa.utils.log import logging, physics, set_verbosity
from pisa.utils.utils import check_binning
from pisa.utils.jsons import from_json,to_json


def service_factory(reco_mode, **kwargs):
    reco_mode = reco_mode.lower()
    if reco_mode == 'mc':
        from pisa.reco.RecoServiceMC import RecoServiceMC
        return RecoServiceMC(**kwargs)

    if reco_mode == 'param':
        from pisa.reco.RecoServiceParam import RecoServiceParam
        return RecoServiceParam(**kwargs)

    if reco_mode == 'stored':
        from pisa.reco.RecoServiceKernelFile import RecoServiceKernelFile
        return RecoServiceKernelFile(**kwargs)

    if reco_mode == 'vbwkde':
        from pisa.reco.RecoServiceVBWKDE import RecoServiceVBWKDE
        return RecoServiceVBWKDE(**kwargs)

    raise ValueError('Unrecognized Reco `reco_mode`: "%s"' % reco_mode)


if __name__ == '__main__':
    parser = ArgumentParser(
        description='''Takes a (true, triggered) event rate file as input and
        produces a set of reconstructed templates of nue CC, numu CC, nutau CC,
        and NC events.''',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'event_rate_maps', metavar='JSON', type=from_json,
        help='''JSON event rate input file with following parameters:
        {"nue": {'cc':{'czbins':[], 'ebins':[], 'map':[]}, 'nc':...},
         "numu": {...},
         "nutau": {...},
         "nue_bar": {...},
         "numu_bar": {...},
         "nutau_bar": {...}}'''
    )
    parser.add_argument(
        '-m', '--reco-mode', type=str,
        choices=['MC', 'param', 'vbwkde', 'stored'], default='param',
        help='Reco service to use'
    )
    parser.add_argument(
        '--mc_file', metavar='HDF5', type=str,
        default='events/V15_weighted_aeff_joined_nu_nubar.hdf5',
        help='''HDF5 File containing reconstruction data from all flavours for
        a particular instument geometry.'''
    )
    parser.add_argument(
        '--reco-vbwkde-evts-file', metavar='HDF5', type=str,
        default='events/V15_weighted_aeff_joined_nu_nubar.hdf5',
        help='''HDF5 File containing reconstruction data from all flavours for
        a particular instument geometry.'''
    )
    parser.add_argument(
        '--param-file', metavar='JSON', type=str, default='reco/V36.json',
        help='''JSON file holding the parametrization'''
    )
    parser.add_argument(
        '--kernel_file', metavar='JSON/HDF5', type=str, default=None,
        help='JSON file holding the pre-calculated kernels'
    )
    parser.add_argument(
        '--e-reco-scale', type=float, default=1.0,
        help='Reconstructed energy scaling.'
    )
    parser.add_argument(
        '--cz_reco_scale', type=float, default=1.0,
        help='Reconstructed coszen scaling.'
    )
    parser.add_argument(
        '-o', '--outfile', dest='outfile', metavar='JSON',
        type=str, action='store', default="reco.json",
        help='file to store the output'
    )
    parser.add_argument(
        '-v', '--verbose', action='count', default=None,
        help='set verbosity level'
    )
    args = parser.parse_args()

    # Set verbosity level
    set_verbosity(args.verbose)

    # Check binning
    ebins, czbins = check_binning(args.event_rate_maps)

    logging.info("Defining RecoService...")
    reco_service = service_factory(ebins=ebins, czbins=czbins, **vars(args))

    event_rate_reco_maps = reco_service.get_reco_maps(args.event_rate_maps,
                                                      **vars(args))

    to_json(event_rate_reco_maps, args.outfile)
