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

import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from pisa.utils.jsons import to_json, json_string
from pisa.utils.log import logging, set_verbosity
from pisa.utils.utils import get_bin_centers


def service_factory(flux_mode='honda', **kwargs):
    """Construct and return (Honda) FluxService class
    
    Parameters
    ----------
    flux_mode : str
        Identifier for which FluxService to instantiate
    **kwargs
        All kwargs are passed (as **kwargs) to the class being instantiated

    Returns
    -------
    Instantiated flux service
    """
    flux_mode = flux_mode.lower()
    if flux_mode == 'honda':
        from pisa.flux.FluxServiceHonda import FluxServiceHonda
        return FluxServiceHonda(**kwargs)

    raise ValueError('Unrecognized Flux `flux_mode`: "%s"' % flux_mode)


if __name__ == '__main__':
    # parser
    parser = ArgumentParser(
        description='''Take a settings file as input and write out a set of
        flux maps, formatter_class=ArgumentDefaultsHelpFormatter'''
    )
    parser.add_argument(
        '--ebins', metavar='[1.0, 2.0, ...]', type=json_string,
        default=np.logspace(np.log10(1.0), np.log10(80.0), 20),
        help='Edges of the energy bins in units of GeV.'
    )
    parser.add_argument(
        '--czbins', metavar='[-1.0, -0.8., ...]', type=json_string,
        default=np.linspace(-1., 0., 11),
        help='Edges of the cos(zenith) bins.',
    )
    parser.add_argument(
        '--flux-file', metavar='FILE', type=str, default='flux/spl-solmax-aa.d',
        help='Input flux file in Honda format'
    )
    parser.add_argument(
        '--nue-numu-ratio', metavar='FLOAT', type=float, default=1.0,
        help='Factor by which to scale nue_flux'
    )
    parser.add_argument(
        '--nu-nubar-ratio', metavar='FLOAT', type=float, default=1.0,
        help='Factor by which to scale nu_nubar_flux'
    )
    parser.add_argument(
        '--delta-index', metavar='FLOAT', type=float, default=0.0,
        help='Shift in spectral index of numu'
    )
    parser.add_argument(
        '--energy-scale', metavar='FLOAT', type=float, default=1.0,
        help='Factor by which to scale TRUE energy'
    )
    parser.add_argument(
        '-o', '--outfile', dest='outfile', metavar='FILE',
        type=str, action='store', default='flux.json',
        help='file to store the output'
    )
    parser.add_argument(
        '-v', '--verbose', action='count', default=None,
        help='set verbosity level'
    )
    args = parser.parse_args()

    # Set verbosity level
    set_verbosity(args.verbose)

    logging.debug("Using %u bins in energy from %.2f to %.2f GeV" %
                  (len(args.ebins)-1, args.ebins[0], args.ebins[-1]))
    logging.debug("Using %u bins in cos(zenith) from %.2f to %.2f" %
                  (len(args.czbins)-1, args.czbins[0], args.czbins[-1]))

    # Instantiate a flux model
    flux_service = service_factory(flux_file=args.flux_file)

    # get the flux
    flux_maps = flux_service.get_flux_maps(
        args.ebins, args.czbins, args.nue_numu_ratio,
        args.nu_nubar_ratio, args.energy_scale, args.delta_index
    )

    # write out to a file
    to_json(flux_maps, args.outfile)
