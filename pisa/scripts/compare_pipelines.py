#! /usr/bin/env python
# author: J.L. Lanfranchi
# date:   October 8, 2016
"""
Compares reco vbwkde vs. hist.
"""
import numpy as np
from uncertainties import unumpy as unp
from argparse import ArgumentParser

from pisa.core.pipeline import Pipeline
from pisa.utils.fileio import mkdir
from pisa.utils.log import logging, set_verbosity
from pisa.utils.tests import plot_cmp
from pisa.utils.plotter import Plotter


if __name__ == '__main__':
    parser = ArgumentParser(
        description='''Compare reco.vbwkde against reco.hist.'''
    )
    parser.add_argument(
        '--outdir', metavar='DIR', type=str, default=None, required=False,
        help='''Store output plots to this directory.'''
    )
    parser.add_argument(
        '--ref', type=str, required=True, metavar='CONFIGFILE',
        help='''Pipeline settings config file that generates reference
        output.'''
    )
    parser.add_argument(
        '--ref-label', type=str, required=True,
        help='''Label for reference'''
    )
    parser.add_argument(
        '--test', type=str, required=True, metavar='CONFIGFILE',
        help='''Pipeline settings config file that generates test output.'''
    )
    parser.add_argument(
        '--test-label', type=str, required=True,
        help='''Label for test'''
    )
    parser.add_argument(
        '--combine', type=str, required=False, default=None,
        help='''Combine by wildcard string. Use single quotes such that
        asterisk does not get expanded by shell.'''
    )
    parser.add_argument(
        '--pdf', action='store_true',
        help='''Save plots in PDF format.'''
    )
    parser.add_argument(
        '--png', action='store_true',
        help='''Save plots in PNG format.'''
    )
    parser.add_argument(
        '-v', action='count', default=None,
        help='Set verbosity level'
    )
    args = parser.parse_args()
    set_verbosity(args.v)

    plot_formats = []
    if args.pdf:
        plot_formats.append('pdf')
    if args.png:
        plot_formats.append('png')

    ref_pipeline = Pipeline(config=args.ref)
    test_pipeline = Pipeline(config=args.test)

    if args.outdir is not None:
        mkdir(args.outdir)

    ref_maps = ref_pipeline.get_outputs()
    if args.combine is not None:
        ref_maps = ref_maps.combine_wildcard(args.combine)
    test_maps = test_pipeline.get_outputs()
    if args.combine is not None:
        test_maps = test_maps.combine_wildcard(args.combine)

    if test_maps.names != ref_maps.names:
        raise ValueError(
            'Test map names %s do not match ref map names %s.'
            %(test_maps.names, ref_maps.names)
        )

    my_plotter = Plotter(stamp='', outdir=args.outdir, fmt='pdf', log=False, annotate=False, symmetric=False, ratio=True)
    for map in ref_maps:
        print '%s:\t%.2f'%(map.name, np.sum(unp.nominal_values(map.hist)))
    my_plotter.plot_2d_array(ref_maps, split_axis='pid', fname='%s_nominal'%args.ref_label)
    my_plotter.plot_2d_array(test_maps, split_axis='pid', fname='%s_nominal'%args.test_label)
    my_plotter.label = '%s/%s - 1'%(args.test_label, args.ref_label)
    my_plotter.plot_2d_array(test_maps/ref_maps-1., split_axis='pid', fname='ratio', cmap='RdBu', vmin=-2,vmax=2)
    my_plotter.label = '%s - %s'%(args.ref_label, args.test_label)
    my_plotter.plot_2d_array(ref_maps - test_maps, split_axis='pid', fname='abs_diff', cmap='RdBu', vmin=-10, vmax=10)
