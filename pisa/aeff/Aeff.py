#! /usr/bin/env python
#
# Aeff.py
#
# This module is the implementation of the stage2 analysis. The main
# purpose of stage2 is to combine the "oscillated Flux maps" with the
# effective areas to create oscillated event rate maps, using the true
# information. This signifies what the "true" event rate would be for
# a detector with our effective areas, but with perfect PID and
# resolutions.
#
# If desired, this will create an output file with the results of
# the current stage of processing.
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# date:   April 8, 2014
#

import numpy as np
from scipy.constants import Julian_year
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from pisa.utils.log import logging, set_verbosity
from pisa.utils.fileio import from_file, to_file
from pisa.utils.proc import report_params, get_params, add_params
from pisa.utils.utils import check_binning, get_binning


def get_event_rates(osc_flux_maps, aeff_service, livetime=None,
                    aeff_scale=None, **kwargs):
    '''
    Main function for this module, which returns the event rate maps
    for each flavor and interaction type, using true energy and zenith
    information. The content of each bin will be the weighted aeff
    multiplied by the oscillated flux, so that the returned dictionary
    will be of the form:
    {'nue': {'cc':map, 'nc':map},
     'nue_bar': {'cc':map, 'nc':map}, ...
     'nutau_bar': {'cc':map, 'nc':map} }
    \params:
      * osc_flux_maps - maps containing oscillated fluxes
      * aeff_service - the effective area service to use
      * livetime - detector livetime for which to calculate event counts
      * aeff_scale - systematic to be a proxy for the realistic effective area
    '''

    # Get parameters used here
    params = get_params()
    report_params(params, units=['', 'yrs', ''])

    # Initialize return dict
    event_rate_maps = {'params': add_params(params, osc_flux_maps['params'])}

    # Get effective area
    aeff_dict = aeff_service.get_aeff()

    ebins, czbins = get_binning(osc_flux_maps)

    # apply the scaling for nu_xsec_scale and nubar_xsec_scale...
    flavours = ['nue', 'numu', 'nutau', 'nue_bar', 'numu_bar', 'nutau_bar']
    for flavour in flavours:
        osc_flux_map = osc_flux_maps[flavour]['map']
        int_type_dict = {}
        for int_type in ['cc', 'nc']:
            event_rate = osc_flux_map*aeff_dict[flavour][int_type]*aeff_scale

            event_rate *= (livetime*Julian_year)
            int_type_dict[int_type] = {'map':event_rate,
                                       'ebins':ebins,
                                       'czbins':czbins}
            logging.debug("  Event Rate before reco for %s/%s: %.2f"
                          % (flavour, int_type, np.sum(event_rate)))
        event_rate_maps[flavour] = int_type_dict

    # else: no scaling to be applied
    return event_rate_maps


def aeff_service_factory(aeff_mode, **kwargs):
    """Construct and return a AeffService class based on `mode`
    
    Parameters
    ----------
    aeff_mode : str
        Identifier for which AeffService class to instantiate. Currently
        understood are 'param' and 'mc'.
    **kwargs
        All subsequent kwargs are passed (as **kwargs), to the class being
        instantiated.
    """
    aeff_mode = aeff_mode.lower()
    if aeff_mode == 'param':
        from pisa.aeff.AeffServicePar import AeffServicePar
        return AeffServicePar(**kwargs)

    if aeff_mode == 'mc':
        from pisa.aeff.AeffServiceMC import AeffServiceMC
        return AeffServicePar(**kwargs)

    raise ValueError('Unrecognized Aeff `aeff_mode`: "%s"' % aeff_mode)


def add_argparser_args(parser):
    from pisa.aeff.AeffServicePar import AeffServicePar
    from pisa.aeff.AeffServiceMC import AeffServiceMC

    parser.add_argument(
        '--aeff-mode', type=str, required=True,
        choices=['param', 'mc'], default='param',
        help='Aeff service to use'
    )

    # Add args specific to the known classes
    AeffServicePar.add_argparser_args(parser)
    AeffServiceMC.add_argparser_args(parser)

    return parser


if __name__ == '__main__':
    parser = ArgumentParser(
        description='''Take an oscillated flux file as input & write out a set
        of oscillated event counts.''',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--osc-flux-maps', metavar='FLUX', type=from_file,
        help='''Osc flux input file with the following parameters:
        {"nue": {'czbins':[], 'ebins':[], 'map':[]},
         "numu": {...},
         "nutau": {...},
         "nue_bar": {...},
         "numu_bar": {...},
         "nutau_bar": {...} }'''
    )
    parser.add_argument(
        '--livetime', type=float, default=1.0,
        help='''livetime in years to re-scale by.'''
    )
    parser.add_argument(
        '--aeff_scale', type=float, default=1.0,
        help='''Overall scale on aeff'''
    )

    # Add AeffService-specific args
    add_argparser_args(parser)

    # Back to generic args
    parser.add_argument(
        '--outfile', dest='outfile', metavar='FILE', type=str,
        default="aeff_output.json",
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
    set_verbosity(args.verbose)

    # Output file
    outfile = args.pop('outfile')

    # Handy to have (TODO: move to central location)
    nil = {'ebins':ebins, 'czbins':czbins,
           'map': np.zeros((n_ebins, n_czbins))}
    unity = {'ebins':ebins, 'czbins':czbins,
             'map': np.ones((n_ebins, n_czbins))}

    osc_flux_maps = args.pop('osc_flux_maps')
    if osc_flux_maps is not None:
        # Load event maps (expected to be something like the output from a reco
        # stage)
        osc_flux_maps = fileio.from_file(args.pop('osc_flux_maps'))
        flavgrps = [fg for fg in sorted(osc_flux_maps)
                    if fg not in ['params', 'ebins', 'czbins']]
    else:
        # Otherwise, generate maps with all 1's to send through the PID stage
        flavgrps = ['nue_cc', 'numu_cc', 'nutau_cc', 'nuall_nc']
        n_ebins = 39
        n_czbins = 20
        ebins = np.logspace(0, np.log10(80), n_ebins+1)
        czbins = np.linspace(-1, 0, n_czbins+1)
        nil = {'ebins':ebins, 'czbins':czbins,
               'map': np.zeros((n_ebins, n_czbins))}
        unity = {'ebins':ebins, 'czbins':czbins,
                 'map': np.ones((n_ebins, n_czbins))}
        osc_flux_maps = {f:deepcopy(unity) for f in flavgrps} 
        osc_flux_maps['params'] = {}

    # Check, return binning
    args['ebins'], args['czbins'] = check_binning(reco_event_maps)

    # Initialize the PID service
    aeff_service = aeff_service_factory(aeff_mode=args.pop('aeff_mode'),
                                        **args)

    # Calculate event rates after Aeff
    event_rate_aeff = get_event_rates(
        osc_flux_maps=osc_flux_maps, aeff_service=aeff_service,
        livetime=args.pop('livetime'), aeff_scale=args.pop('aeff_scale')
    )

    # Save the results to disk
    to_file(event_rate_aeff, outfile)

    # Produce plots useful for debugging
    if args['plot']:
        import os
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from pisa.utils import flavInt
        from pisa.utils import plot
        n_flavgrps = len(flavgrps)
        flavgrps = event_rate_aeff.keys()
        flavgrps.remove('params')
        n_flavgrps = len(flavgrps)

        #fig, axes = plt.subplots(n_sigs+1, n_flavgrps, figsize=(20,14),
        #                         dpi=70, sharex=True, sharey=True)
        #for flavgrp_num, flavgrp in enumerate(flavgrps):
        #    # Effect of applying PID to *just one* flavgrp
        #    reco_event_maps = {f:deepcopy(nil) for f in flavgrps}
        #    reco_event_maps[flavgrp] = deepcopy(unity)
        #    reco_event_maps['params'] = {}
        #    fract_pid = pid_service.get_pid_maps(reco_event_maps)
        #    agg_map = deepcopy(nil)

        #    # Actual groupings (as they stand now) include antiparticles
        #    # even though these do not appear in the labels given.
        #    # (E.g. "nue_cc" actually means "nue_cc + nuebar_cc".)
        #    flavintgroup = flavInt.NuFlavIntGroup(flavgrp)
        #    [flavintgroup.__iadd__(-f) for f in flavintgroup]
        #    fltex = '$' + flavintgroup.simpleTex(flavsep=r'+') + '$'

        #    for sig_num, sig in enumerate(signatures):
        #        agg_map['map'] += fract_pid[sig]['map']
        #        ax = axes[sig_num, flavgrp_num]
        #        plt.sca(ax)
        #        plot.show_map(fract_pid[sig], cmap=mpl.cm.GnBu_r)
        #        ax.get_children()[0].autoscale()
        #        ax.set_title('Fract. of ' + fltex + ' ID\'d as ' + sig,
        #                     fontsize=14)

        #    ax = axes[n_sigs, flavgrp_num]
        #    plt.sca(ax)
        #    plot.show_map(agg_map, cmap=mpl.cm.GnBu_r)
        #    ax.get_children()[0].autoscale()
        #    ax.set_title('Fract. of ' + fltex + ' ID\'d, total',
        #                 fontsize=14)

        fig.tight_layout()
        base, ext = os.path.splitext(outfile)
        fig.savefig(base + '.pdf')
        fig.savefig(base + '.png')
        plt.draw()
        plt.show()
