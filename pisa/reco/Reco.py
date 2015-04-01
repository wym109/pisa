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
import numpy as np
import itertools as itertools

from pisa.utils.log import logging, physics, set_verbosity
from pisa.utils.utils import check_binning, get_binning
from pisa.utils.jsons import from_json,to_json
from pisa.utils.proc import report_params, get_params, add_params

from pisa.reco.RecoServiceMC import RecoServiceMC
from pisa.reco.RecoServiceParam import RecoServiceParam
from pisa.reco.RecoServiceKernelFile import RecoServiceKernelFile
from pisa.reco.RecoServiceVBWKDE import RecoServiceVBWKDE


def get_reco_maps(true_event_maps, reco_service=None, e_reco_scale=None,
                  cz_reco_scale=None, **kwargs):
    """
    Primary function for this stage, which returns the reconstructed
    event rate maps from the true event rate maps. The returned maps will
    be in the form of a dictionary with parameters:
    {'nue_cc':{'ebins':ebins,'czbins':czbins,'map':map},
     'numu_cc':{...},
     'nutau_cc':{...},
     'nuall_nc':{...}}
    Note that in this function, the nu<x> is now combined with nu_bar<x>.

    """

    # Be verbose on input
    params = get_params()
    report_params(params, units = ['', ''])

    # Initialize return dict
    reco_maps = {'params': add_params(params, true_event_maps['params'])}

    # Check binning
    ebins, czbins = get_binning(true_event_maps)

    # Retrieve all reconstruction kernels
    reco_kernel_dict = reco_service.get_reco_kernels(
        e_reco_scale=e_reco_scale, cz_reco_scale=cz_reco_scale, **kwargs
    )

    # Do smearing
    flavours = ['nue', 'numu', 'nutau']
    int_types = ['cc', 'nc']

    # Do smearing again, without loops
    flavors = ['nue', 'numu', 'nutau']
    all_int_types = ['cc', 'nc']
    n_ebins = len(ebins)-1
    n_czbins = len(czbins)-1
    for baseflavor, int_type in itertools.product(flavors, all_int_types):
        logging.info("Getting reco event rates for %s %s" % (baseflavor,
                                                             int_type))
        reco_event_rate = np.zeros((n_ebins, n_czbins), dtype=np.float64)
        for mID in ['', '_bar']:
            flavor = baseflavor + mID
            true_event_rate = true_event_maps[flavor][int_type]['map']
            kernels = reco_kernel_dict[flavor][int_type]
            r0 = np.tensordot(true_event_rate, kernels, axes=([0,1],[0,1]))
            reco_event_rate += r0
        reco_maps[baseflavor+'_'+int_type] = {'map': reco_event_rate,
                                              'ebins': ebins,
                                              'czbins': czbins}
        msg = "after RECO: counts for (%s + %s) %s: %.2f" \
            % (baseflavor, baseflavor+'_bar', int_type, np.sum(reco_event_rate))
        logging.debug(msg)

    # Finally sum up all the NC contributions
    logging.info("Summing up rates for all nc events")
    reco_event_rate = np.sum(
        [reco_maps.pop(key)['map'] for key in reco_maps.keys()
         if key.endswith('_nc')], axis=0
    )
    reco_maps['nuall_nc'] = {'map':reco_event_rate,
                             'ebins':ebins,
                             'czbins':czbins}
    logging.debug("Total counts for nuall nc: %.2f" % np.sum(reco_event_rate))
    return reco_maps


if __name__ == '__main__':

    parser = ArgumentParser(
        description='Takes a (true, triggered) event rate file '
                    'as input and produces a set of reconstructed templates '
                    'of nue CC, numu CC, nutau CC, and NC events.',
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
    parser.add_argument('-m', '--mode', type=str,
                        choices=['MC', 'param', 'vbwkde', 'stored'],
                        default='param', help='Reco service to use')
    parser.add_argument('--mc_file', metavar='HDF5', type=str,
                        default='events/V15_weighted_aeff_joined_nu_nubar.hdf5',
                        help='''HDF5 File containing reconstruction data from all flavours for a particular instument geometry.''')
    parser.add_argument('--vbwkde_evts_file', metavar='HDF5', type=str,
                        default='events/V15_weighted_aeff_joined_nu_nubar.hdf5',
                        help='''HDF5 File containing reconstruction data from all flavours for a particular instument geometry.''')
    parser.add_argument('--param_file', metavar='JSON',
                        type=str, default='reco_params/V36_reco.json',
                        help='''JSON file holding the parametrization''')
    parser.add_argument('--kernel_file', metavar='JSON',
                        type=str, default=None,
                        help='''JSON file holding the pre-calculated kernels''')
    parser.add_argument('--e_reco_scale', type=float, default=1.0,
                        help='''Reconstructed energy scaling.''')
    parser.add_argument('--cz_reco_scale', type=float, default=1.0,
                        help='''Reconstructed coszen scaling.''')
    parser.add_argument('-o', '--outfile', dest='outfile', metavar='JSON',
                        type=str, action='store', default="reco.json",
                        help='''file to store the output''')
    parser.add_argument('-v', '--verbose', action='count', default=None,
                        help='''set verbosity level''')
    args = parser.parse_args()

    # Set verbosity level
    set_verbosity(args.verbose)

    # Check binning
    ebins, czbins = check_binning(args.event_rate_maps)

    logging.info("Defining RecoService...")
    if args.mode=='MC':
        reco_service = RecoServiceMC(ebins, czbins,
                                     reco_mc_wt_file=args.mc_file,
                                     **vars(args))
    elif args.mode=='param':
        reco_service = RecoServiceParam(ebins, czbins,
                                        reco_param_file=args.param_file,
                                        **vars(args))
    elif args.mode=='stored':
        reco_service = RecoServiceKernelFile(ebins, czbins,
                                             reco_kernel_file=args.kernel_file,
                                             **vars(args))
    elif args.mode=='vbwkde':
        reco_service = RecoServiceVBWKDE(
            ebins=ebins, czbins=czbins,
            reco_vbwkde_evts_file=args.reco_vbwkde_evts_file,
            **vars(args)
        )

    event_rate_reco_maps = get_reco_maps(args.event_rate_maps,
                                         reco_service, **vars(args))

    logging.info("Saving output to: %s"%args.outfile)
    to_json(event_rate_reco_maps, args.outfile)
