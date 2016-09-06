#! /usr/bin/env python
# author: S.Wren
# date:   March 20, 2016
"""
Runs the pipeline in event-by-event mode to see how much it agrees with OscFit.
This is also checked against the PISA 2 legacy mode.
Test data for comparing against should be in the tests/data directory.
A set of plots will be output in your output directory for you to check.
"""

from argparse import ArgumentParser
from collections import Sequence
from copy import deepcopy
import os
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
import numpy as np

from pisa.core.map import Map, MapSet
from pisa.core.pipeline import Pipeline
from pisa.utils.fileio import mkdir, from_file
from pisa.utils.log import logging, set_verbosity
from pisa.utils.resources import find_resource
from pisa.utils.config_parser import parse_pipeline_config
from pisa.utils.tests import has_cuda, check_agreement, plot_comparisons


def compare_baseline(config, oscfitfile, outdir):
    """Compare baseline output of PISA 3 and OscFit"""

    logging.debug('>> Working on baseline comparisons between both fitters.')

    pipeline = Pipeline(config)
    outputs = pipeline['mc'].outputs
    oscfit_comparisons = from_file(oscfitfile)

    for nukey in pisa2_comparisons.keys():
        if 'nu' not in nukey:
            continue

        pisa_map_to_plot = pisa2_comparisons[nukey]

        if '_' in nukey:
            if nukey.split('_')[1] == 'bar':
                new_nukey = ""
                for substr in nukey.split('_'):
                    new_nukey += substr
                nukey = new_nukey

        cake_map = outputs[nukey]
        cake_map_to_plot = {}
        cake_map_to_plot['ebins'] = \
                cake_map.binning['true_energy'].bin_edges.magnitude
        cake_map_to_plot['czbins'] = \
                cake_map.binning['true_coszen'].bin_edges.magnitude
        cake_map_to_plot['map'] = cake_map.hist

        max_diff_ratio, max_diff = plot_comparisons(
            ref_map=pisa_map_to_plot,
            new_map=cake_map_to_plot,
            ref_abv='V2', new_abv='V3',
            outdir=outdir,
            subdir='flux',
            stagename='flux',
            servicename=servicename,
            name=nukey,
            texname=outputs[nukey].tex
        )

        check_agreement(
            testname='V3-V2 flux:%s %s %s'
                %(test_service, test_syst, nukey),
            thresh_ratio=ratio_test_threshold,
            ratio=max_diff_ratio,
            thresh_diff=diff_test_threshold,
            diff=max_diff
        )

    return pipeline


if __name__ == '__main__':
    parser = ArgumentParser(
        description='''Run a set of tests on the PISA 3 pipeline against
        the output from OscFit. If no test flags are specified, *all* tests will
        be run.'''
    )
    parser.add_argument('--baseline', action='store_true', default=False,
                        help='''Run baseline tests i.e. the output of PISA 3
                        event by event and OscFit with a set of parameters
                        agreed upon before the tests were started.''')
    parser.add_argument('-v', action='count', default=None,
                        help='set verbosity level')
    args = parser.parse_args()
    set_verbosity(args.v)

    cuda_present = has_cuda()

    if not has_cuda():
        raise RuntimeError('This system does not have a GPU so these tests '
                           'cannot be performed. Aborting')

    # Figure out which tests to do
    test_all = True
    if args.baseline:
        test_all = False

    # Perform baseline tests
    if args.baseline or test_all:
        pisa3_settings = os.path.join(
            'tests', 'settings', 'oscfit_test.ini'
        )
        pisa3_config = parse_pipeline_config(flux_settings)
        oscfitfile = os.path.join(
            'tests', 'data', 'oscfit', 'OscFit1X600Baseline.json'
        )
        oscfitfile = find_resource(pisa2file)
        pisa3_pipeline = compare_baseline(
            config=deepcopy(flux_config),
            oscfitfile=oscfitfile,
            outdir=args.outdir
        )
