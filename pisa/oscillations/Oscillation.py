#! /usr/bin/env python
#
# Oscillation.py
#
# This module is the implementation of the physics oscillation step.
# In this step, oscillation probability maps of each neutrino flavor
# into the others are produced, for a given set of oscillation
# parameters. It is then multiplied by the corresponding flux map,
# producing oscillated flux maps for each flavor.
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# date:   Jan. 21, 2014
#


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np

from pisa.utils.log import logging, tprofile, set_verbosity
from pisa.utils.utils import Timer
from pisa.utils.fileio import from_file, to_file
from pisa.utils.utils import DictWithHash


def newOscProbDict(ecen, czcen):
    n_ecen = len(ecen)
    n_czcen = len(czcen)
    osc_prob_dict = DictWithHash()
    for nu in ['nue_maps', 'numu_maps', 'nue_bar_maps', 'numu_bar_maps']:
        isbar = '_bar' if 'bar' in nu else ''
        osc_prob_dict[nu] = {'nue'+isbar: np.zeros(n_ecen*n_czcen),
                             'numu'+isbar: np.zeros(n_ecen*n_czcen),
                             'nutau'+isbar: np.zeros(n_ecen*n_czcen)}
    return osc_prob_dict


# TODO: so... convention is *_mode everywhere else besides here?
def service_factory(osc_code, **kwargs):
    """Construct and return an OscillationService class based on `osc_code`

    Parameters
    ----------
    osc_code : str
        Identifier for which OscillationService class to instantiate
    **kwargs
        All subsequent kwargs are passed (as **kwargs) to the class being
        instantiated.
    """
    osc_code = osc_code.lower()
    if osc_code == 'prob3':
        from pisa.oscillations.Prob3OscillationService import Prob3OscillationService
        return Prob3OscillationService(**kwargs)

    if osc_code == 'nucraft':
        from pisa.oscillations.NucraftOscillationService import NucraftOscillationService
        return NucraftOscillationService(**kwargs)

    if osc_code == 'gpu':
        from pisa.oscillations.Prob3GPUOscillationService import Prob3GPUOscillationService
        return Prob3OscillationService(**kwargs)

    if osc_code == 'table':
        from pisa.oscillations.TableOscillationService import TableOscillationService
        return TableOscillationService(**kwargs)

    raise ValueError('Unrecognized Oscillation `osc_code`: "%s"' % osc_code)


if __name__ == '__main__':
    # parser
    parser = ArgumentParser(
        description='''Takes the oscillation parameters as input and writes
        out a set of osc flux maps''',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--flux-maps', metavar='FLUX', type=from_file,
                        help='''JSON atm flux input file with the following
                        parameters:
                        {"nue": {'czbins':[], 'ebins':[], 'map':[]},
                         "numu": {...},
                         "nue_bar": {...},
                         "numu_bar":{...}}''')
    parser.add_argument('--deltam21', type=float, default=7.54e-5,
                        help='''deltam21 value [eV^2]''')
    parser.add_argument('--deltam31', type=float, default=0.00246,
                        help='''deltam31 value [eV^2]''')
    parser.add_argument('--theta12', type=float, default=0.5873,
                        help='''theta12 value [rad]''')
    parser.add_argument('--theta13', type=float, default=0.1562,
                        help='''theta13 value [rad]''')
    parser.add_argument('--theta23', type=float, default=0.6745,
                        help='''theta23 value [rad]''')
    parser.add_argument('--deltacp', type=float, default=0.0,
                        help='''deltaCP value to use [rad]''')
    parser.add_argument('--earth-model', type=str,
                        default='oscillations/PREM_12layer.dat',
                        help='''Earth model data (density as function of
                        radius)''')
    parser.add_argument('--energy-scale', type=float, default=1.0,
                        help='''Energy off scaling due to mis-calibration.''')
    parser.add_argument('--YeI', type=float, default=0.5,
                        help='''Ye (elec frac) in inner core.''')
    parser.add_argument('--YeO', type=float, default=0.5,
                        help='''Ye (elec frac) in outer core.''')
    parser.add_argument('--YeM', type=float, default=0.5,
                        help='''Ye (elec frac) in mantle.''')
    parser.add_argument('--osc-code', type=str,
                        choices = ['prob3','table','nucraft','gpu'],
                        default='prob3',
                        help='''Oscillation code to use''')
    parser.add_argument('--oversample-e', type=int, default=10,
                        help='''oversampling factor for energy;
                        i.e. every 2D bin will be oversampled by this factor
                        in each dimension''')
    parser.add_argument('--oversample-cz', type=int, default=10,
                        help='''oversampling factor for  cos(zen);
                        i.e. every 2D bin will be oversampled by this factor
                        in each dimension ''')
    parser.add_argument('--detector-depth', type=float, default=2.0,
                        help='''Detector depth in km''')
    parser.add_argument('--prop-height', type=float, default=20.0,
                        help='''Height in the atmosphere to begin propagation
                        in km. Prob3 default: 20.0 km NuCraft default:
                        'sample' from a distribution''')
    parser.add_argument('--osc-precision', type=float, default=5e-4,
                        help='''Requested precision for unitarity (NuCraft
                        only)''')
    parser.add_argument('--tabledir', type=str, default='oscillations',
                        dest='datadir',
                        help='''Path to stored oscillation data (Tables
                        only)''')
    parser.add_argument('-o', '--outfile', dest='outfile', metavar='FILE',
                        type=str, action='store', default="osc_flux.json",
                        help='file to store the output')
    parser.add_argument('-v', '--verbose', action='count', default=None,
                        help='set verbosity level')
    args = vars(parser.parse_args())

    # Set verbosity level
    set_verbosity(args.pop('verbose'))

    # Get binning
    flux_maps = args.pop('flux_maps')
    ebins, czbins = check_binning(flux_maps)

    # Initialize an oscillation service
    osc_service = service_factory(ebins=ebins, czbins=czbins, **args)

    logging.info("Getting osc prob maps")
    with Timer(verbose=False) as t:
        osc_flux_maps = get_osc_flux(args.flux_maps, osc_service,
                                     deltam21=args.deltam21,
                                     deltam31=args.deltam31,
                                     deltacp=args.deltacp,
                                     theta12=args.theta12,
                                     theta13=args.theta13,
                                     theta23=args.theta23,
                                     oversample_e=args.oversample_e,
                                     oversample_cz=args.oversample_cz,
                                     energy_scale=args.energy_scale,
                                     YeI=args.YeI, YeO=args.YeO, YeM=args.YeM)
    logging.info("       ==> elapsed time to get osc flux maps: %s sec"
                 % t.secs)

    # Write out
    logging.info("Saving output to: %s" % args.outfile)
    to_file(osc_flux_maps, args.outfile)
