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

from pisa import ureg, Q_
from pisa.core.map import MapSet
from pisa.core.pipeline import Pipeline
from pisa.utils.config_parser import parse_pipeline_config
from pisa.utils.fileio import from_file, mkdir
from pisa.utils.log import logging, set_verbosity
from pisa.utils.resources import find_resource
from pisa.utils.tests import has_cuda, print_agreement, check_agreement, plot_comparisons, plot_cmp

FMT = 'png'

def compare_pisa(config1, config2, testname1, testname2, outdir):
    """
    Compare baseline output of PISA 3 with an older version of itself
    (for self-consistency checks).
    """
    logging.debug('>> Comparing %s with %s (both PISA)'%(testname1,testname2))

    pipeline1 = Pipeline(config1)
    outputs1 = pipeline1.get_outputs()

    cake1_trck_map = outputs1.combine_wildcard('*_trck')
    cake1_cscd_map = outputs1.combine_wildcard('*_cscd')

    pipeline2 = Pipeline(config2)
    outputs2 = pipeline2.get_outputs()

    cake2_both_map = outputs2.combine_wildcard('*')

    cake1_trck_map_to_plot = {}
    cake1_trck_map_to_plot['ebins'] = \
            cake1_trck_map.binning['reco_energy'].bin_edges.magnitude
    cake1_trck_map_to_plot['czbins'] = \
            cake1_trck_map.binning['reco_coszen'].bin_edges.magnitude
    cake1_trck_map_to_plot['map'] = cake1_trck_map.hist
    cake1_trck_map_to_plot['map'] = cake1_trck_map_to_plot['map']

    cake2_trck_map_to_plot = {}
    cake2_trck_map_to_plot['ebins'] = \
            cake2_both_map.binning['reco_energy'].bin_edges.magnitude
    cake2_trck_map_to_plot['czbins'] = \
            cake2_both_map.binning['reco_coszen'].bin_edges.magnitude
    cake2_trck_map_to_plot['map'] = cake2_both_map.hist[...,1]
    cake2_trck_map_to_plot['map'] = cake2_trck_map_to_plot['map']

    max_diff_ratio, max_diff = plot_comparisons(
        ref_map=cake1_trck_map_to_plot,
        new_map=cake2_trck_map_to_plot,
        ref_abv='5stage',
        new_abv='4stage',
        outdir=outdir,
        subdir='recopidcombinedchecks',
        stagename='baseline',
        servicename='recopid',
        name='trck',
        texname=r'\rm{trck}',
        ftype=FMT
    )

    print_agreement(testname='PISAStandard-RecoPid: %s'%('trck'),
                    ratio=max_diff_ratio)

    cake1_cscd_map_to_plot = {}
    cake1_cscd_map_to_plot['ebins'] = \
            cake1_cscd_map.binning['reco_energy'].bin_edges.magnitude
    cake1_cscd_map_to_plot['czbins'] = \
            cake1_cscd_map.binning['reco_coszen'].bin_edges.magnitude
    cake1_cscd_map_to_plot['map'] = cake1_cscd_map.hist
    cake1_cscd_map_to_plot['map'] = cake1_cscd_map_to_plot['map']

    cake2_cscd_map_to_plot = {}
    cake2_cscd_map_to_plot['ebins'] = \
            cake2_both_map.binning['reco_energy'].bin_edges.magnitude
    cake2_cscd_map_to_plot['czbins'] = \
            cake2_both_map.binning['reco_coszen'].bin_edges.magnitude
    cake2_cscd_map_to_plot['map'] = cake2_both_map.hist[...,0]
    cake2_cscd_map_to_plot['map'] = cake2_cscd_map_to_plot['map']

    max_diff_ratio, max_diff = plot_comparisons(
        ref_map=cake1_cscd_map_to_plot,
        new_map=cake2_cscd_map_to_plot,
        ref_abv='5stage',
        new_abv='4stage',
        outdir=outdir,
        subdir='recopidcombinedchecks',
        stagename='baseline',
        servicename='recopid',
        name='cscd',
        texname=r'\rm{cscd}',
        ftype=FMT
    )

    print_agreement(testname='PISAStandard-RecoPid: %s'%('cscd'),
                    ratio=max_diff_ratio)

    return pipeline2


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
            texname=texname,
            ftype=FMT
        )

        print_agreement(testname='OscFit-V3:%s %s'%(testname, nukey),
                        ratio=max_diff_ratio)

    return pipeline


def compare_systematics(baseline_oscfit, config, testname, outdir, oscfitfile):
    """
    Compare systematic variations of PISA 3 and OscFit.
    """
    logging.debug('>> Working on systematic comparisons between both fitters.')
    logging.debug('>>> Doing %s test.'%testname)
    baseline_comparisons = from_file(baseline_oscfit)
    systematic_comparisons = from_file(oscfitfile)
    ref_abv='OscFit'

    pipeline = Pipeline(config)
    outputs = pipeline.get_outputs()

    for nukey in systematic_comparisons.keys():

        systematic_map_to_plot = systematic_comparisons[nukey]
        systematic_map_to_plot['map'] = (
            systematic_map_to_plot['map'] + baseline_comparisons[nukey]['map']
        )

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
            ref_map=systematic_map_to_plot,
            new_map=cake_map_to_plot,
            ref_abv=ref_abv, new_abv='PISAV3',
            outdir=outdir,
            subdir='oscfit',
            stagename=testname,
            servicename='systematic',
            name=nukey,
            texname=texname,
            ftype=FMT
        )

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
    parser.add_argument('--continuous_systematics', action='store_true',
                        default=False,
                        help='''Run continuous systematics tests i.e. the
                        output of PISA 3 event by event and OscFit with
                        variations on the NOT discrete systematics. The
                        fiducial model was agreed upon before the tests were
                        started.''')
    parser.add_argument('--outdir', metavar='DIR', type=str, default=None,
                        required=False,
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

    if args.outdir is not None:
        mkdir(args.outdir)

    hist_cfg = 'tests/settings/vbwkde_test_reco.hist.cfg'
    vbwkde_cfg = 'tests/settings/vbwkde_test_reco.vbwkde.cfg'

    # Figure out which tests to do
    test_all = True

    # Perform internal tests
    if test_all:
        hist_pipeline = Pipeline(
            config=hist_cfg
        )
        vbwkde_pipeline = Pipeline(
            config=vbwkde_cfg
        )
        hist_maps = hist_pipeline.get_outputs()
        vbwkde_maps = vbwkde_pipeline.get_outputs()
        assert vbwkde_maps.names == hist_maps.names
        for map_name in vbwkde_maps.names:
            vmap = vbwkde_maps[map_name]
            hmap = hist_maps[map_name]
            comparisons = vmap.compare(hmap)
            for k in ['max_diff_ratio', 'max_diff', 'nanmatch', 'infmatch']:
                print '%s: %s = %s' %(map_name, k, comparisons[k])

            if args.outdir is not None:
                plot_cmp(new=vmap, ref=hmap, new_label='reco.vbwkde',
                         ref_label='reco.hist', plot_label=vmap.tex,
                         file_label=vmap.name, outdir=args.outdir,
                         ftype='png')
