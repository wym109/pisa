#! /usr/bin/env python
#
# TemplateMaker.py
#
# Class to implement template-making procedure and to store as much data
# as possible to avoid re-running stages when not needed.
#
# author: Timothy C. Arlen - tca3@psu.edu
#         Sebastian Boeser - sboeser@uni-mainz.de
#
# date:   7 Oct 2014
#

import sys

import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from pisa.utils.log import logging, set_verbosity
from pisa.utils.params import get_values, select_hierarchy
from pisa.utils.fileio import from_file, to_file
from pisa.utils.utils import Timer


from pisa.oscillations import Oscillation
from pisa.flux import Flux
from pisa.aeff import Aeff
from pisa.reco import Reco
from pisa.pid import PID


class TemplateMaker:
    """This class handles all steps needed to produce a template with a
    constant binning.

    The strategy employed will be to define all 'services' in the
    initialization process, make them members of the class, then use
    them later when needed.
    """
    def __init__(self, template_params_values, ebins, czbins,
                 oversample_e, oversample_cz, **kwargs):
        """TemplateMaker class handles all of the setup and calculation of the
        templates for a given binning.

        Parameters
        ----------
        template_params_values
        ebins, czbins
            energy and coszen bin edges
        oversample_e
        oversample_cz
        """
        self.ebins = ebins
        self.czbins = czbins

        # TODO why do these not have a proper home?
        self.oversample_e = oversample_e
        self.oversample_cz = oversample_cz

        logging.debug('Using %u bins in energy from %.2f to %.2f GeV' %
                      (len(self.ebins)-1, self.ebins[0], self.ebins[-1]))
        logging.debug('Using %u bins in cos(zenith) from %.2f to %.2f' %
                      (len(self.czbins)-1, self.czbins[0], self.czbins[-1]))

        # Instantiate the stages
        self.flux_service = Flux.service_factory(**template_params_values)
        self.osc_service = Oscillation.service_factory(
            ebins=self.ebins, czbins=self.czbins,
            oversample_e=self.oversample_e, oversample_cz=self.oversample_cz,
            **template_params_values
        )
        self.aeff_service = Aeff.service_factory(
            ebins=self.ebins, czbins=self.czbins,
            **template_params_values
        )
        self.reco_service = Reco.service_factory(
            ebins=self.ebins, czbins=self.czbins,
            **template_params_values
        )
        self.pid_service = PID.service_factory(
            ebins=self.ebins, czbins=self.czbins,
            **template_params_values
        )

    def get_template(self, template_params_values, return_stages=False):
        """Runs entire template-making chain, using parameters found in
        'template_params_values' dict. If 'return_stages' is set to True,
        returns output from each stage as a simple tuple.
        """
        logging.info("STAGE 1: Getting Atm Flux maps...")
        with Timer() as t:
            flux_maps = self.flux_service.get_flux_maps(
                ebins=self.ebins, czbins=self.czbins,
                **template_params_values
            )
        logging.debug("==> elapsed time for flux stage: %s sec" % t.secs)

        logging.info("STAGE 2: Getting osc prob maps...")
        with Timer() as t:
            osc_flux_maps = self.osc_service.get_osc_flux(
                flux_maps=flux_maps,
                oversample_e=self.oversample_e,
                oversample_cz=self.oversample_cz,
                **template_params_values
            )
        logging.debug("==> elapsed time for oscillations stage: %s sec"
                       % t.secs)

        logging.info("STAGE 3: Getting event rate true maps...")
        with Timer() as t:
            event_rate_maps = Aeff.get_event_rates(
                osc_flux_maps=osc_flux_maps, aeff_service=self.aeff_service,
                **template_params_values
            )
        logging.debug("==> elapsed time for aeff stage: %s sec" % t.secs)

        logging.info("STAGE 4: Getting event rate reco maps...")
        with Timer() as t:
            event_rate_reco_maps = self.reco_service.get_reco_maps(
                true_event_maps=event_rate_maps,
                **template_params_values
            )
        logging.debug("==> elapsed time for reco stage: %s sec" % t.secs)

        logging.info("STAGE 5: Getting pid maps...")
        with Timer(verbose=False) as t:
            final_event_rate = self.pid_service.get_pid_maps(
                event_rate_reco_maps
            )
        logging.debug("==> elapsed time for pid stage: %s sec" % t.secs)

        if not return_stages:
            return final_event_rate

        # Otherwise, return all stages as a simple tuple
        return (flux_maps, osc_flux_maps, event_rate_maps,
                event_rate_reco_maps, final_event_rate)

    def get_template_no_osc(self, template_params_values):
        """Runs template making chain, but without oscillations"""
        logging.info("STAGE 1: Getting Atm Flux maps...")
        with Timer() as t:
            flux_maps = self.flux_service.get_flux_maps(
                ebins=self.ebins, czbins=self.czbins,
                **template_params_values
            )
        logging.debug("==> elapsed time for flux stage: %s sec" % t.secs)

        # Skipping oscillation stage...
        logging.info("  >>Skipping Stage 2 in no oscillations case...")
        flavours = ['nutau', 'nutau_bar']
        # Create the empty nutau maps:
        test_map = flux_maps['nue']
        for flav in flavours:
            flux_maps[flav] = {'map': np.zeros_like(test_map['map']),
                               'ebins': np.zeros_like(test_map['ebins']),
                               'czbins': np.zeros_like(test_map['czbins'])}

        logging.info("STAGE 3: Getting event rate true maps...")
        with Timer() as t:
            event_rate_maps = Aeff.get_event_rates(
                osc_flux_maps=flux_maps, aeff_service=self.aeff_service,
                **template_params_values
            )
        logging.debug("==> elapsed time for aeff stage: %s sec" % t.secs)

        logging.info("STAGE 4: Getting event rate reco maps...")
        with Timer() as t:
            event_rate_reco_maps = self.reco_service.get_reco_maps(
                true_event_maps=event_rate_maps,
                **template_params_values
            )
        logging.debug("==> elapsed time for reco stage: %s sec" % t.secs)

        logging.info("STAGE 5: Getting pid maps...")
        with Timer(verbose=False) as t:
            final_event_rate = self.pid_service.get_pid_maps(
                event_rate_reco_maps
            )
        logging.debug("==> elapsed time for pid stage: %s sec" % t.secs)

        return final_event_rate


if __name__ == '__main__':
    # parser
    parser = ArgumentParser(
        description='''Runs the template making process.''',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-t', '--template_settings', type=str,
                        metavar='JSONFILE', required=True,
                        help='''settings for the template generation''')

    hselect = parser.add_mutually_exclusive_group(required=False)
    hselect.add_argument('--normal', dest='normal', default=True,
                         action='store_true',
                         help="select the normal hierarchy")
    hselect.add_argument('--inverted', dest='normal', default = False,
                         action='store_false',
                         help="select the inverted hierarchy")

    parser.add_argument('--plot', action='store_true',
                        help='plot resulting maps')
    parser.add_argument('-v', '--verbose', action='count', default=None,
                        help='set verbosity level.')
    parser.add_argument('-s', '--save_all', action='store_true', default=False,
                        help="Save all stages.")
    parser.add_argument('-o', '--outfile', dest='outfile', metavar='FILE',
                        type=str, action='store', default="template.json",
                        help='file to store the output')
    args = parser.parse_args()

    set_verbosity(args.verbose)

    with Timer() as t:
        # Load all the settings
        model_settings = from_file(args.template_settings)

        # Select a hierarchy
        logging.info('Selected %s hierarchy' %
                     ('normal' if args.normal else 'inverted'))
        template_params_nh = select_hierarchy(
            model_settings['params'], normal_hierarchy=True
        )
        template_params_ih = select_hierarchy(
            model_settings['params'], normal_hierarchy=False
        )

        # Intialize template maker
        template_params_values_nh = get_values(template_params_nh)
        template_params_values_ih = get_values(template_params_ih)
        ebins = model_settings['binning']['ebins']
        czbins = model_settings['binning']['czbins']
        oversample_e = model_settings['binning']['oversample_e']
        oversample_cz = model_settings['binning']['oversample_cz']
        template_maker = TemplateMaker(template_params_values_nh,
                                       **model_settings['binning'])
    logging.info('  ==> elapsed time to initialize templates: %s sec'
                  % t.secs)

    # Now get the actual template
    with Timer(verbose=False) as t:
        stage_outputs = template_maker.get_template(template_params_values_nh,
                                                    return_stages=True)
        flux_maps_nh, osc_flux_maps_nh, event_rate_maps_nh, \
        event_rate_reco_maps_nh, final_event_rate_nh = stage_outputs
    logging.info('==> elapsed time to get template: %s sec' % t.secs)

    with Timer(verbose=False) as t:
        stage_outputs_ih = template_maker.get_template(
            template_params_values_ih, return_stages=True
        )
        flux_maps_ih, osc_flux_maps_ih, event_rate_maps_ih, \
        event_rate_reco_maps_ih, final_event_rate_ih = stage_outputs_ih
    logging.info('==> elapsed time to get template: %s sec' % t.secs)

    with Timer(verbose=False) as t:
        final_event_rate_no_osc = template_maker.get_template_no_osc(
            template_params_values_nh
        )
    logging.info('==> elapsed time to get template: %s sec' % t.secs)

    logging.info('Saving file to %s' % args.outfile)
    to_file(stage_outputs, args.outfile)

    if args.plot:
        import os
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from pisa.utils import flavInt
        from pisa.utils import plot
        for k in sorted(final_event_rate_nh.keys()):
            if k == 'params':
                continue
            evtrt_nh = final_event_rate_nh[k]
            evtrt_ih = final_event_rate_ih[k]
            dist_map = plot.distinguishability_map(evtrt_ih,
                                                   evtrt_nh)
            if k == 'trck':
                clim = (-0.21, 0.21)
            else:
                clim = (-0.27, 0.27)

            f = plt.figure(figsize=(24, 5), dpi=50)
            ax = f.add_subplot(131)
            plot.show_map(evtrt_nh, title=k, log=True,
                          cmap=mpl.cm.hot)

            ax = f.add_subplot(132)
            plot.show_map(evtrt_ih, log=True,
                          cmap=mpl.cm.hot)

            ax = f.add_subplot(133)
            plot.show_map(dist_map,
                         cmap=mpl.cm.seismic)
            ax.get_children()[0].set_clim(clim)

        plt.draw()
        plt.show()
