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
from pisa.utils.utils import check_binning, prefilled_map
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
    if pid_mode == 'mc':
        from pisa.pid.PIDServiceMC import PIDServiceMC
        return PIDServiceMC(**kwargs)

    pid_mode = pid_mode.lower()
    if pid_mode == 'param':
        from pisa.pid.PIDServiceParam import PIDServiceParam
        return PIDServiceParam(**kwargs)

    if pid_mode == 'kernel':
        from pisa.pid.PIDServiceKernelFile import PIDServiceKernelFile
        return PIDServiceKernelFile(**kwargs)

    if pid_mode == 'smooth':
        from pisa.pid.PIDServiceSmooth import PIDServiceSmooth
        return PIDServiceSmooth(**kwargs)

    raise ValueError('Unrecognized PID `pid_mode`: "%s"' % pid_mode)


def add_argparser_args(parser):
    from pisa.pid.PIDServiceParam import PIDServiceParam
    from pisa.pid.PIDServiceMC import PIDServiceMC
    from pisa.pid.PIDServiceKernelFile import PIDServiceKernelFile

    parser.add_argument(
        '--pid-mode', type=str, required=True,
        choices=['mc', 'param', 'kernel', 'smooth'], default='param',
        help='PID service to use'
    )

    # Add args specific to the known classes
    PIDServiceParam.add_argparser_args(parser)
    PIDServiceMC.add_argparser_args(parser)
    PIDServiceKernelFile.add_argparser_args(parser)

    return parser


if __name__ == "__main__":
    import numpy as np
    parser = ArgumentParser(
        description='''Takes a reco event rate file as input and produces a set
        of reconstructed templates of tracks and cascades.''',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--reco-event-maps', metavar='JSON', type=from_file, required=False,
        help='''JSON reco event rate file resource location; must have
        following structure:
        {"nue_cc": {'czbins':[...], 'ebins':[...], 'map':[...]},
        "numu_cc": {...},
        "nutau_cc": {...},
        "nuall_nc": {...} }'''
    )

    # Add PIDService-specific args
    add_argparser_args(parser)

    # Back to generic args
    parser.add_argument(
        '--outfile', dest='outfile', metavar='FILE', type=str,
        default="pid_output.json",
        help='''file to store the output'''
    )
    parser.add_argument(
        '--plot', action='store_true',
        help='plot resulting maps'
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

    reco_event_maps = args.pop('reco_event_maps')
    if reco_event_maps is not None:
        # Load event maps (expected to be something like the output from a reco
        # stage)
        reco_event_maps = fileio.from_file(args.pop('reco_event_maps'))
        flavgrps = [fg for fg in sorted(reco_event_maps)
                    if fg not in ['params', 'ebins', 'czbins']]
    else:
        # Otherwise, generate maps with all 1's to send through the PID stage
        flavgrps = ['nue_cc', 'numu_cc', 'nutau_cc', 'nuall_nc']
        n_ebins = 39
        n_czbins = 20
        ebins = np.logspace(0, np.log10(80), n_ebins+1)
        czbins = np.linspace(-1, 0, n_czbins+1)
        reco_event_maps = {f: prefilled_map(ebins, czbins, 1)
                           for f in flavgrps} 
        reco_event_maps['params'] = {}

    # Check, return binning
    args['ebins'], args['czbins'] = check_binning(reco_event_maps)

    # Initialize the PID service
    pid_service = pid_service_factory(pid_mode=args.pop('pid_mode'), **args)

    # Calculate event rates after PID
    event_rate_pid = pid_service.get_pid_maps(reco_event_maps)

    # Save the results to disk
    to_file(event_rate_pid, outfile)

    # Produce plots useful for debugging

    # TODO:
    # * Make similar plots but for counts (if user supplied a reco_events_map)
    # * Include row totals as another column at right, to show aggregate
    #   numbers per signature. (This probably is only useful if the user
    #   supplies reco_events_map; show this on the above-proposed "counts"
    #   figure.)
    if args['plot']:
        import os
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from pisa.utils import flavInt
        from pisa.utils import plot
        n_flavgrps = len(flavgrps)
        signatures = event_rate_pid.keys()
        signatures.remove('params')
        n_sigs = len(signatures)
        fig, axes = plt.subplots(n_sigs+1, n_flavgrps, figsize=(20,14),
                                 dpi=70, sharex=True, sharey=True)
        for flavgrp_num, flavgrp in enumerate(flavgrps):
            # Effect of applying PID to *just one* flavgrp
            reco_event_maps = {f: prefilled_map(ebins, czbins, 0)
                               for f in flavgrps if f != flavgrp}
            reco_event_maps[flavgrp] = prefilled_map(ebins, czbins, 1)
            reco_event_maps['params'] = {}
            fract_pid = pid_service.get_pid_maps(reco_event_maps)
            agg_map = prefilled_map(ebins, czbins, 0)

            # Actual groupings (as they stand now) include antiparticles
            # even though these do not appear in the labels given.
            # (E.g. "nue_cc" actually means "nue_cc + nuebar_cc".)
            flavintgroup = flavInt.NuFlavIntGroup(flavgrp)
            [flavintgroup.__iadd__(-f) for f in flavintgroup]
            fltex = '$' + flavintgroup.simpleTex(flavsep=r'+') + '$'

            for sig_num, sig in enumerate(signatures):
                agg_map['map'] += fract_pid[sig]['map']
                ax = axes[sig_num, flavgrp_num]
                plt.sca(ax)
                plot.show_map(fract_pid[sig], cmap=mpl.cm.GnBu_r)
                ax.get_children()[0].autoscale()
                ax.set_title('Fract. of ' + fltex + ' ID\'d as ' + sig,
                             fontsize=14)

            ax = axes[n_sigs, flavgrp_num]
            plt.sca(ax)
            plot.show_map(agg_map, cmap=mpl.cm.GnBu_r)
            ax.get_children()[0].autoscale()
            ax.set_title('Fract. of ' + fltex + ' ID\'d, total',
                         fontsize=14)

        fig.tight_layout()
        base, ext = os.path.splitext(outfile)
        fig.savefig(base + '.pdf')
        fig.savefig(base + '.png')
        plt.draw()
        plt.show()
