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
from copy import deepcopy
import os

from pisa.core.pipeline import Pipeline
from pisa.core.map import MapSet
from pisa.utils.fileio import from_file
from pisa.utils.log import logging, set_verbosity
from pisa.utils.resources import find_resource
from pisa.utils.config_parser import parse_pipeline_config
from pisa.utils.tests import has_cuda, print_agreement, check_agreement, plot_comparisons

def consistency_test(config, testname, outdir, pisa3file, 
                     ratio_test_threshold, diff_test_threshold):
    """
    Compare baseline output of PISA 3 with an older version of itself 
    (for self-consistency checks).
    """
    logging.debug('>> Doing PISA self-consistency test for %s'%testname)
    baseline_comparisons = MapSet(from_file(pisa3file)["maps"])
    ref_abv='PISAV3Ref'

    pipeline = Pipeline(config)
    outputs = pipeline.get_outputs()

    for nukey in baseline_comparisons.names:

        pisa3ref_map = baseline_comparisons[nukey]
        pisa3ref_map_to_plot = {}
        pisa3ref_map_to_plot['ebins'] = \
                pisa3ref_map.binning['reco_energy'].bin_edges.magnitude
        pisa3ref_map_to_plot['czbins'] = \
                pisa3ref_map.binning['reco_coszen'].bin_edges.magnitude
        pisa3ref_map_to_plot['map'] = pisa3ref_map.hist
        pisa3ref_map_to_plot['map'] = pisa3ref_map_to_plot['map'].T

        cake_map = outputs[nukey]
        cake_map_to_plot = {}
        cake_map_to_plot['ebins'] = \
                cake_map.binning['reco_energy'].bin_edges.magnitude
        cake_map_to_plot['czbins'] = \
                cake_map.binning['reco_coszen'].bin_edges.magnitude
        cake_map_to_plot['map'] = cake_map.hist

        max_diff_ratio, max_diff = plot_comparisons(
            ref_map=pisa3ref_map_to_plot,
            new_map=cake_map_to_plot,
            ref_abv=ref_abv, new_abv='PISAV3',
            outdir=outdir,
            subdir='oscfit',
            stagename='baseline',
            servicename='full_mc',
            name=nukey,
            texname=outputs[nukey].tex)

        check_agreement(testname='PISAV3Ref-V3Now:%s %s'%(testname, nukey),
                        thresh_ratio=ratio_test_threshold,
                        ratio=max_diff_ratio,
                        thresh_diff=diff_test_threshold,
                        diff=max_diff)

    return pipeline
    

def compare_baseline(config, testname, outdir, oscfitfile):
    """
    Compare baseline output of PISA 3 with OscFit.
    """
    logging.debug('>> Working on baseline comparisons between both fitters.')
    logging.debug('>>> Doing %s test.'%testname)
    baseline_comparisons = from_file(oscfitfile)
    ref_abv='OscFit'

    pipeline = Pipeline(config)
    outputs = pipeline.get_outputs()

    for nukey in baseline_comparisons.keys():

        baseline_map_to_plot = baseline_comparisons[nukey]

        try:
            cake_map = outputs[nukey]
            texname = outputs[nukey].tex
        except:
            cake_map = outputs.combine_wildcard('*_%s'%nukey)
            if nukey == 'trck':
                texname = r'\rm{trck}'
            elif nukey == 'cscd':
                texname = r'\rm{cscd}'
        cake_map_to_plot = {}
        cake_map_to_plot['ebins'] = \
                cake_map.binning['reco_energy'].bin_edges.magnitude
        cake_map_to_plot['czbins'] = \
                cake_map.binning['reco_coszen'].bin_edges.magnitude
        cake_map_to_plot['map'] = cake_map.hist

        max_diff_ratio, max_diff = plot_comparisons(
            ref_map=baseline_map_to_plot,
            new_map=cake_map_to_plot,
            ref_abv=ref_abv, new_abv='PISAV3',
            outdir=outdir,
            subdir='oscfit',
            stagename=testname,
            servicename='baseline',
            name=nukey,
            texname=texname)

        print_agreement(testname='OscFit-V3:%s %s'%(testname, nukey),
                        ratio=max_diff_ratio)

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
    parser.add_argument('--outdir', metavar='DIR', type=str, required=True,
                        help='''Store all output plots to this directory. If
                        they don't exist, the script will make them, including
                        all subdirectories. If none is supplied no plots will
                        be saved.''')
    parser.add_argument('--ratio_threshold', type=float, default=1E-8,
                        help='''Sets the agreement threshold on the ratio test
                        plots. If this is not reached the tests will fail.''')
    parser.add_argument('--diff_threshold', type=float, default=2E-3,
                        help='''Sets the agreement threshold on the diff test
                        plots. If this is not reached the tests will fail. This
                        test is only important if any ratios return inf.''')
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
            'tests', 'settings', 'oscfit_fullmc_test.ini'
        )
        pisa3_config = parse_pipeline_config(pisa3_settings)
        # First ensure that we still agree with Philipp's original 
        # implementation of the event by event method. Here we can expect
        # agreement to machine precision.
        pisa3file = os.path.join(
            'tests', 'data', 'oscfit', 'PISAEventByEvent1X600Baseline.json'
        )
        pisa3file = find_resource(pisa3file)
        pisa3_pipeline = consistency_test(
            config=deepcopy(pisa3_config), 
            testname='full-mc', 
            outdir=args.outdir, 
            pisa3file=pisa3file, 
            ratio_test_threshold=args.ratio_threshold, 
            diff_test_threshold=args.diff_threshold)
        # If the above was passed we can now compare with OscFit. Agreement
        # is expected to better than 1 part in 1000.
        oscfitfile = os.path.join(
            'tests', 'data', 'oscfit', 'OscFit1X600Baseline.json'
        )
        oscfitfile = find_resource(oscfitfile)
        pisa3_pipeline = compare_baseline(
            config=deepcopy(pisa3_config),
            oscfitfile=oscfitfile,
            outdir=args.outdir,
            testname='full-mc'
        )
        pisa3_settings = os.path.join(
            'tests', 'settings', 'oscfit_standard_test.ini'
        )
        pisa3_config = parse_pipeline_config(pisa3_settings)
        oscfitfile = os.path.join(
            'tests', 'data', 'oscfit', 'OscFit1X600Baseline.json'
        )
        oscfitfile = find_resource(oscfitfile)
        pisa3_pipeline = compare_baseline(
            config=deepcopy(pisa3_config),
            oscfitfile=oscfitfile,
            outdir=args.outdir,
            testname='standard'
        )
        
