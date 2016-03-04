#! /usr/bin/env python

# author: Timothy C. Arlen
#         tca3@psu.edu
# author: Sebastian Boeser
#         sboeser@physik.uni-bonn.de
# author: J.L. Lanfranchi
#         jll1062+pisa@phys.psu.edu
#
# date:   April 10, 2014

from copy import deepcopy

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from pisa.utils.log import logging, set_verbosity
from pisa.utils.utils import check_binning
from pisa.utils.fileio import from_file, to_file


def pid_service_factory(pid_mode, **kwargs):
    """Construct and return a PIDService class based on `mode`
    
    Parameters
    ----------
    pid_mode : str
        Identifier for which PIDService class to instantiate
    **kwargs
        All subsequent kwargs are passed (as **kwargs), to the class being
        instantiated.
    """
    pid_mode = pid_mode.lower()
    if pid_mode == 'param':
        from pisa.pid.PIDServiceParam import PIDServiceParam
        return PIDServiceParam(**kwargs)

    if pid_mode == 'mc':
        from pisa.pid.PIDServiceMC import PIDServiceMC
        return PIDServiceParam(**kwargs)

    if pid_mode == 'kernel':
        from pisa.pid.PIDServiceKernelFile import PIDServiceKernelFile
        return PIDServiceKernelFile(**kwargs)

    raise ValueError('Unrecognized PID `pid_mode`: "%s"' % pid_mode)


def add_argparser_args(parser):
    from pisa.pid.PIDServiceParam import PIDServiceParam
    from pisa.pid.PIDServiceMC import PIDServiceMC
    from pisa.pid.PIDServiceKernelFile import PIDServiceKernelFile
    parser.add_argument(
        '--reco_event_maps', metavar='JSON', type=from_file,
        required=False,
        help='''JSON reco event rate file resource location; must have
        following structure:\n
        {"nue_cc": {'czbins':[...], 'ebins':[...], 'map':[...]}, \n
        "numu_cc": {...}, \n
        "nutau_cc": {...}, \n
        "nuall_nc": {...} }'''
    )

    parser.add_argument(
        '--pid-mode', type=str, choices=['param', 'mc', 'kernel'],
        default='param', help='PID service to use'
    )

    # Add args specific to the known classes
    PIDServiceParam.add_argparser_args(parser)
    PIDServiceMC.add_argparser_args(parser)
    PIDServiceKernelFile.add_argparser_args(parser)


if __name__ == "__main__":
    import numpy as np
    parser = ArgumentParser(
        description='''Takes a reco event rate file as input and produces a set
        of reconstructed templates of tracks and cascades.''',
        formatter_class=ArgumentDefaultsHelpFormatter
    )

    # Add PIDService-specific args
    add_argparser_args(parser)

    # Back to generic args
    parser.add_argument(
        '--outfile', dest='outfile', metavar='FILE', type=str,
        default="pid.json",
        help='''file to store the output'''
    )
    parser.add_argument(
        '-v', '--verbose', action='count', default=None,
        help='set verbosity level'
    )

    # Parse args & convert them to a dict
    args = vars(parser.parse_args())

    # Set verbosity level
    set_verbosity(args.pop('verbose'))

    # Output file
    outfile = args.pop('outfile')

    # Load event maps (as if they were from a reco file)
    reco_event_maps = args.pop('reco_event_maps')
    if reco_event_maps is not None:
        reco_event_maps = fileio.from_file(args.pop('reco_event_maps'))
    else:
        n_ebins = 39
        n_czbins = 20
        ebins = np.logspace(0, np.log10(80), n_ebins+1)
        czbins = np.linspace(-1, 0, n_czbins+1)
        ones = {'ebins':ebins, 'czbins':czbins,
                'map': np.ones((n_ebins, n_czbins))}
        reco_event_maps = {f:deepcopy(ones) for f in ['nue_cc', 'numu_cc',
                                                      'nutau_cc', 'nuall_nc']} 
        reco_event_maps['params'] = {}

    # Check, return binning
    args['ebins'], args['czbins'] = check_binning(reco_event_maps)

    # Initialize the PID service
    pid_service = pid_service_factory(pid_mode=args.pop('pid_mode'), **args)

    # Calculate event rates after PID
    event_rate_pid = pid_service.get_pid_maps(reco_event_maps)

    to_file(event_rate_pid, outfile)
