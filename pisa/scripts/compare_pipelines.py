#! /usr/bin/env python
# author: J.L. Lanfranchi
# date:   October 8, 2016
"""
Compares reco vbwkde vs. hist.
"""

from argparse import ArgumentParser
import os

from pisa.core.pipeline import Pipeline
from pisa.utils.fileio import mkdir, to_file
from pisa.utils.log import logging, set_verbosity
from pisa.utils.tests import plot_cmp


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
    test_maps = test_pipeline.get_outputs()

    if test_maps.names != ref_maps.names:
        raise ValueError(
            'Test map names %s do not match ref map names %s.'
            %(test_maps.names, ref_maps.names)
        )

    for map_name in test_maps.names:
        print 'Map %s:' %map_name
        test_map = test_maps[map_name]
        ref_map = ref_maps[map_name]
        comparisons = test_map.compare(ref_map)
        for k in ['max_diff_ratio', 'max_diff', 'nanmatch', 'infmatch']:
            print '%s = %s' %(k, comparisons[k])
        if args.outdir is not None:
            to_file(test_map, os.path.expandvars(os.path.expanduser(os.path.join(args.outdir, map_name + '__' + args.test_label + '.json.bz2'))))
            to_file(ref_map, os.path.expandvars(os.path.expanduser(os.path.join(args.outdir, map_name + '__' + args.ref_label + '.json.bz2'))))
            for fmt in plot_formats:
                plot_cmp(new=test_map, ref=ref_map, new_label=args.test_label,
                         ref_label=args.ref_label, plot_label=test_map.tex,
                         file_label=test_map.name, outdir=args.outdir,
                         ftype=fmt)
        print ''
