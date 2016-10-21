#! /usr/bin/env python
# author: J.L. Lanfranchi
# date:   October 8, 2016
"""
Compares reco vbwkde vs. hist.
"""


from argparse import ArgumentParser
from collections import Iterable
import os

import numpy as np
from uncertainties import unumpy as unp

from pisa.core.distribution_maker import DistributionMaker
from pisa.core.map import Map, MapSet
from pisa.utils.fileio import mkdir
from pisa.utils.log import logging, set_verbosity
from pisa.utils.tests import plot_cmp
from pisa.utils.plotter import Plotter


if __name__ == '__main__':
    parser = ArgumentParser(
        description='''Compare two entities: Maps, map sets, pipelines, or
        distribution makers. One kind can be compared against another, so long
        as the resulting map(s) have equivalent names and binning.'''
    )
    parser.add_argument(
        '--outdir', metavar='DIR', type=str, default=None, required=True,
        help='''Store output plots to this directory.'''
    )
    parser.add_argument(
        '--ref', type=str, required=True, action='append',
        help='''Pipeline settings config file that generates reference
        output, or a stored map or map set. Repeat --ref option for multiple
        pipelines, maps, or map sets'''
    )
    parser.add_argument(
        '--ref-label', type=str, required=True,
        help='''Label for reference'''
    )
    parser.add_argument(
        '--ref-param-selections', type=str, required=False, default=None,
        action='append',
        help='''Param selections to apply to --ref pipeline config(s). Not
        applicable if --ref specifies stored map or map sets'''
    )
    parser.add_argument(
        '--test', type=str, required=True, action='append',
        help='''Pipeline settings config file that generates test
        output, or a stored map or map set. Repeat --test option for multiple
        pipelines, maps, or map sets'''
    )
    parser.add_argument(
        '--test-label', type=str, required=True,
        help='''Label for test'''
    )
    parser.add_argument(
        '--test-param-selections', type=str, required=False, default=None,
        action='append',
        help='''Param selections to apply to --test pipeline config(s). Not
        applicable if --test specifies stored map or map sets'''
    )
    parser.add_argument(
        '--combine', type=str, default=None, action='append',
        help='''Combine by wildcard string, where string globbing (a la command
        line) uses asterisk for any number of wildcard characters. Use single
        quotes such that asterisks do not get expanded by the shell. Repeat the
        --combine option for multiple combine strings.'''
    )

    grp = parser.add_mutually_exclusive_group(required=False)
    grp.add_argument(
        '--pdf', action='store_true',
        help='''Save plots in PDF format.'''
    )
    grp.add_argument(
        '--png', action='store_true',
        help='''Save plots in PNG format.'''
    )

    parser.add_argument(
        '-v', action='count', default=None,
        help='Set verbosity level'
    )
    args = parser.parse_args()
    set_verbosity(args.v)

    #plot_formats = []
    #if args.pdf:
    #    plot_formats.append('pdf')
    #if args.png:
    #    plot_formats.append('png')
    plt_fmt = 'pdf' if args.pdf else 'png'

    mkdir(args.outdir)

    # Get the reference distribution(s) into the form of a test MapSet
    ref = None
    ref_source = None
    try:
        ref_dmaker = DistributionMaker(pipelines=args.ref)
    except:
        pass
    else:
        ref_source = 'DistributionMaker'
        if args.ref_param_selections is not None:
            ref_dmaker.select_params(args.ref_param_selections)
        ref = ref_dmaker.get_outputs()

    if ref is None:
        try:
            ref = [Map.from_json(f) for f in args.ref]
        except:
            pass
        else:
            ref_source = 'Map'
            ref = MapSet(ref)

    if ref is None:
        assert args.ref_param_selections is None
        assert len(args.ref) == 1, 'Can only handle one MapSet'
        try:
            ref = MapSet.from_json(args.ref[0])
        except:
            raise
        else:
            ref_source = 'MapSet'

    if ref is None:
        raise ValueError(
            'Could not instantiate the reference DistributionMaker, Map, or'
            ' MapSet from ref valu(s) %s' % args.ref
        )

    # Get the test distribution(s) into the form of a test MapSet
    test = None
    test_source = None
    try:
        test_dmaker = DistributionMaker(pipelines=args.test)
    except:
        pass
    else:
        test_source = 'DistributionMaker'
        if args.test_param_selections is not None:
            test_dmaker.select_params(args.test_param_selections)
        test = test_dmaker.get_outputs()

    if test is None:
        try:
            test = [Map.from_json(f) for f in args.test]
        except:
            pass
        else:
            test_source = 'Map'
            test = MapSet(test)

    if test is None:
        assert args.test_param_selections is None
        assert len(args.test) == 1, 'Can only handle one MapSet'
        try:
            test = MapSet.from_json(args.test[0])
        except:
            pass
        else:
            test_source = 'MapSet'

    if test is None:
        raise ValueError(
            'Could not instantiate the test DistributionMaker, Map, or MapSet'
            ' from test valu(s) %s' % args.test
        )

    if args.combine is not None:
        ref = ref.combine_wildcard(args.combine)
        test = test.combine_wildcard(args.combine)

    # Save to disk the outputs produced by any distribution makers
    if ref_source == 'DistributionMaker':
        outfile = os.path.join(args.outdir, args.ref_label + '.json.bz2')
        ref.to_json(outfile)

    if test_source == 'DistributionMaker':
        outfile = os.path.join(args.outdir, args.test_label + '.json.bz2')
        test.to_json(outfile)

    if test.names != ref.names:
        raise ValueError(
            'Test map names %s do not match ref map names %s.'
            % (test.names, ref.names)
        )

    for ref_map, test_map in zip(ref, test):
        diff = test_map - ref_map
        fract_diff = test_map / ref_map  - 1
        logging.info('Map %s ..' % ref_map.name)
        logging.info('  Totals:')
        logging.info('    Ref :' + ('%.2f' % np.nansum(ref_map.nominal_values)).rjust(8))
        logging.info('    Test:' + ('%.2f' % np.nansum(ref_map.nominal_values)).rjust(8))
        logging.info('  Means:')
        logging.info('    Ref :' + ('%.2f' % np.nanmean(ref_map.nominal_values)).rjust(8))
        logging.info('    Test:' + ('%.2f' % np.nanmean(ref_map.nominal_values)).rjust(8))
        logging.info('  Test - Ref, mean +/- std dev:')
        logging.info('    %.4e +/- %.4e' %(np.nanmean(diff.nominal_values), np.nanstd(diff.nominal_values)))
        logging.info('  Test / Ref - 1, mean +/- std dev:')
        logging.info('    %.4e +/- %.4e' %(np.nanmean(fract_diff.nominal_values), np.nanstd(fract_diff.nominal_values)))
        logging.info('')

    plotter = Plotter(stamp='', outdir=args.outdir, fmt=plt_fmt, log=False,
                      annotate=False, symmetric=False, ratio=False)
    plotter.plot_2d_array(ref, split_axis='pid', fname='%s_distr' % args.ref_label)
    plotter.plot_2d_array(test, split_axis='pid', fname='%s_distr' % args.test_label)

    plotter = Plotter(stamp='', outdir=args.outdir, fmt=plt_fmt, log=False,
                      annotate=False, symmetric=True, ratio=True)
    plotter.label = '%s/%s - 1' % (args.test_label, args.ref_label)
    plotter.plot_2d_array(test/ref-1., split_axis='pid', fname='fract_diff', cmap='seismic') #, vmin=-2,vmax=2)

    plotter = Plotter(stamp='', outdir=args.outdir, fmt=plt_fmt, log=False,
                      annotate=False, symmetric=True, ratio=False)
    plotter.label = '%s - %s' % (args.test_label, args.ref_label)
    plotter.plot_2d_array(test - ref, split_axis='pid', fname='abs_diff', cmap='seismic') #, vmin=-10, vmax=10)
