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
import numpy as np

from pisa import ureg, Q_
from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.core.map import MapSet
from pisa.core.pipeline import Pipeline
from pisa.utils.config_parser import parse_pipeline_config
from pisa.utils.fileio import from_file
from pisa.utils.log import logging, set_verbosity
from pisa.utils.resources import find_resource
from pisa.utils.tests import print_event_rates, plot_comparisons


FMT = 'png'

def compare_pisa_self(config1, config2, testname1, testname2, outdir):
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
    cake1_trck_events = np.sum(cake1_trck_map_to_plot['map'])

    cake2_trck_map_to_plot = {}
    cake2_trck_map_to_plot['ebins'] = \
            cake2_both_map.binning['reco_energy'].bin_edges.magnitude
    cake2_trck_map_to_plot['czbins'] = \
            cake2_both_map.binning['reco_coszen'].bin_edges.magnitude
    cake2_trck_map_to_plot['map'] = cake2_both_map.hist[1,:,:]
    cake2_trck_map_to_plot['map'] = cake2_trck_map_to_plot['map']
    cake2_trck_events = np.sum(cake2_trck_map_to_plot['map'])

    max_diff_ratio, max_diff = plot_comparisons(
        ref_map=cake1_trck_map_to_plot,
        new_map=cake2_trck_map_to_plot,
        ref_abv=testname2,
        new_abv=testname1,
        outdir=outdir,
        subdir='recopidcombinedchecks',
        stagename='baseline',
        servicename='recopid',
        name='trck',
        texname=r'\rm{trck}',
        ftype=FMT
    )

    cake1_cscd_map_to_plot = {}
    cake1_cscd_map_to_plot['ebins'] = \
            cake1_cscd_map.binning['reco_energy'].bin_edges.magnitude
    cake1_cscd_map_to_plot['czbins'] = \
            cake1_cscd_map.binning['reco_coszen'].bin_edges.magnitude
    cake1_cscd_map_to_plot['map'] = cake1_cscd_map.hist
    cake1_cscd_map_to_plot['map'] = cake1_cscd_map_to_plot['map']
    cake1_cscd_events = np.sum(cake1_cscd_map_to_plot['map'])

    cake2_cscd_map_to_plot = {}
    cake2_cscd_map_to_plot['ebins'] = \
            cake2_both_map.binning['reco_energy'].bin_edges.magnitude
    cake2_cscd_map_to_plot['czbins'] = \
            cake2_both_map.binning['reco_coszen'].bin_edges.magnitude
    cake2_cscd_map_to_plot['map'] = cake2_both_map.hist[0,:,:]
    cake2_cscd_map_to_plot['map'] = cake2_cscd_map_to_plot['map']
    cake2_cscd_events = np.sum(cake2_cscd_map_to_plot['map'])

    max_diff_ratio, max_diff = plot_comparisons(
        ref_map=cake1_cscd_map_to_plot,
        new_map=cake2_cscd_map_to_plot,
        ref_abv=testname2,
        new_abv=testname1,
        outdir=outdir,
        subdir='recopidcombinedchecks',
        stagename='baseline',
        servicename='recopid',
        name='cscd',
        texname=r'\rm{cscd}',
        ftype=FMT
    )

    print_event_rates(
        testname1=testname1,
        testname2=testname2,
        kind='trck',
        map1_events=cake1_trck_events,
        map2_events=cake2_trck_events
    )
    print_event_rates(
        testname1=testname1,
        testname2=testname2,
        kind='cscd',
        map1_events=cake1_cscd_events,
        map2_events=cake2_cscd_events
    )

    print_event_rates(
        testname1=testname1,
        testname2=testname2,
        kind='all',
        map1_events=cake1_trck_events+cake1_cscd_events,
        map2_events=cake2_trck_events+cake2_cscd_events
    )

    return pipeline2


def compare_5stage(config, testname, outdir, oscfitfile):
    """
    Compare 5 stage output of PISA 3 with OscFit.
    """
    logging.debug('>> Working on baseline comparisons between both fitters.')
    logging.debug('>>> Doing %s test.'%testname)
    baseline_comparisons = from_file(oscfitfile)
    ref_abv='OscFit'

    pipeline = Pipeline(config)
    outputs = pipeline.get_outputs()

    total_pisa_events = 0.0
    total_oscfit_events = 0.0

    for nukey in baseline_comparisons.keys():

        baseline_map_to_plot = baseline_comparisons[nukey]
        oscfit_events = np.sum(baseline_map_to_plot['map'])

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
        pisa_events = np.sum(cake_map_to_plot['map'])

        max_diff_ratio, max_diff = plot_comparisons(
            ref_map=baseline_map_to_plot,
            new_map=cake_map_to_plot,
            ref_abv=ref_abv,
            new_abv=testname,
            outdir=outdir,
            subdir='recopidcombinedchecks',
            stagename=testname,
            servicename='baseline',
            name=nukey,
            texname=texname,
            ftype=FMT
        )

        print_event_rates(
            testname1=testname,
            testname2='OscFit',
            kind=nukey,
            map1_events=pisa_events,
            map2_events=oscfit_events
        )

        total_pisa_events += pisa_events
        total_oscfit_events += oscfit_events

    print_event_rates(
            testname1=testname,
            testname2='OscFit',
            kind='all',
            map1_events=total_pisa_events,
            map2_events=total_oscfit_events
        )

    return pipeline


def compare_4stage(config, testname, outdir, oscfitfile):
    """
    Compare 4 stage output of PISA 3 with OscFit.
    """
    logging.debug('>> Working on baseline comparisons between both fitters.')
    logging.debug('>>> Doing %s test.'%testname)
    baseline_comparisons = from_file(oscfitfile)
    ref_abv='OscFit'

    pipeline = Pipeline(config)
    outputs = pipeline.get_outputs()

    total_pisa_events = 0.0
    total_oscfit_events = 0.0

    for nukey in baseline_comparisons.keys():

        baseline_map_to_plot = baseline_comparisons[nukey]
        oscfit_events = np.sum(baseline_map_to_plot['map'])

        cake_map = outputs.combine_wildcard('*')
        cake_map_to_plot = {}
        cake_map_to_plot['ebins'] = \
                cake_map.binning['reco_energy'].bin_edges.magnitude
        cake_map_to_plot['czbins'] = \
                cake_map.binning['reco_coszen'].bin_edges.magnitude
        if nukey == 'trck':
            texname = r'\rm{trck}'
            cake_map_to_plot['map'] = cake_map.hist[1]
        elif nukey == 'cscd':
            texname = r'\rm{cscd}'
            cake_map_to_plot['map'] = cake_map.hist[0]
        pisa_events = np.sum(cake_map_to_plot['map'])

        max_diff_ratio, max_diff = plot_comparisons(
            ref_map=baseline_map_to_plot,
            new_map=cake_map_to_plot,
            ref_abv=ref_abv,
            new_abv=testname,
            outdir=outdir,
            subdir='recopidcombinedchecks',
            stagename=testname,
            servicename='baseline',
            name=nukey,
            texname=texname,
            ftype=FMT
        )

        print_event_rates(
            testname1=testname,
            testname2='OscFit',
            kind=nukey,
            map1_events=pisa_events,
            map2_events=oscfit_events
        )

        total_pisa_events += pisa_events
        total_oscfit_events += oscfit_events

    print_event_rates(
            testname1=testname,
            testname2='OscFit',
            kind='all',
            map1_events=total_pisa_events,
            map2_events=total_oscfit_events
        )

    return pipeline


def do_comparisons(config1, config2, oscfitfile,
                   testname1, testname2, outdir):
        pisa_recopid_pipeline = compare_pisa_self(
            config1=config1,
            config2=config2,
            testname1=testname1,
            testname2=testname2,
            outdir=args.outdir
        )
        pisa_standard_pipeline = compare_5stage(
            config=config1,
            testname=testname1,
            outdir=args.outdir,
            oscfitfile=oscfitfile
        )
        pisa_recopid_pipeline = compare_4stage(
            config=config2,
            testname=testname2,
            outdir=args.outdir,
            oscfitfile=oscfitfile
        )


def oversample_config(base_config, oversample):
    for stage in base_config.keys():
        for obj in base_config[stage].keys():
            if 'binning' in obj:
                if 'true' in base_config[stage][obj].names[0]:
                    base_config[stage][obj] = \
                            base_config[stage][obj].oversample(oversample)

    return base_config


if __name__ == '__main__':
    parser = ArgumentParser(
        description='''Run a set of tests on the PISA 3 pipeline to check the
        effect of combining Reco and PID in to a single stage. Output is tested
        against both the standard PISA and a full event-by-event treatment
        from OscFit in various configurations.'''
    )
    parser.add_argument('--baseline', action='store_true',
                        help='''Run baseline tests''')
    parser.add_argument('--oversampling', action='store_true',
                        help='''Run oversampling tests i.e. use a finer binning
                        through the truth stages.''')
    parser.add_argument('--outdir', metavar='DIR', type=str, required=False,
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

    # Figure out which tests to do
    test_all = True
    if args.baseline or args.oversampling:
        test_all = False

    # Want these for all tests
    pisa_standard_settings = os.path.join(
        'tests', 'settings', 'recopid_full_pipeline_5stage_test.cfg'
    )
    pisa_standard_config = parse_pipeline_config(pisa_standard_settings)
    pisa_standard_weighted_config = deepcopy(pisa_standard_config)
    reco_k = [k for k in pisa_standard_weighted_config.keys() \
              if k[0] == 'reco'][0]
    standard_reco_params = \
        pisa_standard_weighted_config[reco_k]['params'].params
    standard_reco_params.reco_weights_name.value = 'weighted_aeff'
    pid_k = [k for k in pisa_standard_weighted_config.keys() \
             if k[0] == 'pid'][0]
    standard_pid_params = \
        pisa_standard_weighted_config[pid_k]['params'].params
    standard_pid_params.pid_weights_name.value = 'weighted_aeff'
    standard_configs = [#pisa_standard_config,
                        pisa_standard_weighted_config]

    pisa_recopid_settings = os.path.join(
        'tests', 'settings', 'recopid_full_pipeline_4stage_test.cfg'
    )
    pisa_recopid_config = parse_pipeline_config(pisa_recopid_settings)
    pisa_recopid_weighted_config = deepcopy(pisa_recopid_config)
    recopid_k = [k for k in pisa_recopid_weighted_config.keys() \
                 if k[0] == 'reco'][0]
    recopid_reco_params = \
        pisa_recopid_weighted_config[recopid_k]['params'].params
    recopid_reco_params.reco_weights_name.value = 'weighted_aeff'
    recopid_configs = [#pisa_recopid_config,
                       pisa_recopid_weighted_config]

    oscfitfile = os.path.join(
        'tests', 'data', 'oscfit', 'OscFit1X600Baseline.json'
    )

    weights_names = [#'unweighted',
                     'weighted_aeff']
    short_weights_names = [#'uw',
                           'wa']

    for psc, prc, wn, swn in zip(standard_configs,
                                 recopid_configs,
                                 weights_names,
                                 short_weights_names):
        logging.info("<<<< %s reco/pid Transformations >>>>"%wn)
        # Perform baseline tests
        if args.baseline or test_all:
            logging.info("<< No oversampling >>")
            do_comparisons(
                config1=deepcopy(psc),
                config2=deepcopy(prc),
                oscfitfile=oscfitfile,
                testname1='5-stage%s'%swn,
                testname2='4-stage%s'%swn,
                outdir=args.outdir
            )

        # Perform oversampled tests
        if args.oversampling or test_all:
            oversamples = [1]
            for os in oversamples:
                psosc = oversample_config(
                    base_config=deepcopy(psc),
                    oversample=os
                )
                prosc = oversample_config(
                    base_config=deepcopy(prc),
                    oversample=os
                )
                logging.info("<< Oversampling by %i >>"%(os))
                do_comparisons(
                    config1=deepcopy(psosc),
                    config2=deepcopy(prosc),
                    oscfitfile=oscfitfile,
                    testname1='5-stage%s%i'%(swn,os),
                    testname2='4-stage%s%i'%(swn,os),
                    outdir=args.outdir
                )
