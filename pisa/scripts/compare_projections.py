#! /usr/bin/env python
# author: J.L. Lanfranchi
# date:   October 8, 2016
"""
Compare two entities: Maps, map sets, pipelines, or distribution makers. One
kind can be compared against another, so long as the resulting map(s) have
equivalent names and binning. The result each entity specification is formatted
into a MapSet and stored to disk, so that e.g. re-running a DistributionMaker
is unnecessary to reproduce the results.

"""

# TODO: make use of `MapSet.compare()` method (and/or expand that until it is
# equally useful here)

from argparse import ArgumentParser
from collections import OrderedDict
import os
from copy import deepcopy

import numpy as np

from pisa.core.distribution_maker import DistributionMaker
from pisa.core.map import Map, MapSet
from pisa.core.pipeline import Pipeline
from pisa.utils.fileio import mkdir, to_file
from pisa.utils.log import logging, set_verbosity
from pisa.utils.plotter import Plotter


__all__ = ['DISTRIBUTIONMAKER_SOURCE_STR', 'PIPELINE_SOURCE_STR',
           'MAP_SOURCE_STR', 'MAPSET_SOURCE_STR',
           'parse_args', 'main']


DISTRIBUTIONMAKER_SOURCE_STR = (
    'DistributionMaker instantiated from multiple pipeline config files'
)
PIPELINE_SOURCE_STR = 'Pipeline instantiated from a pipelinen config file'
MAP_SOURCE_STR = 'Map stored on disk'
MAPSET_SOURCE_STR = 'MapSet stored on disk'

def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        '--outdir', metavar='DIR', type=str, required=True,
        help='''Store output plots to this directory.'''
    )
    parser.add_argument(
        '--ref', type=str, required=True, action='append',
        help='''Pipeline settings config file that generates reference
        output, or a stored map or map set. Repeat --ref option for multiple
        pipelines, maps, or map sets'''
    )
    parser.add_argument(
        '--ref-abs', action='store_true',
        help='''Use the absolute value of the reference plot for
        comparisons.'''
    )
    parser.add_argument(
        '--ref-label', type=str, default='ref',
        help='''Label for reference'''
    )
    parser.add_argument(
        '--ref-param-selections', type=str, required=False,
        action='append',
        help='''Param selections to apply to --ref pipeline config(s). Not
        applicable if --ref specifies stored map or map sets'''
    )
    for t in range(10):
        parser.add_argument(
            '--test%i'%t, default=None, action='append',
            help='''Pipeline settings config file that generates test
            output, or a stored map or map set. Repeat --test option for multiple
            pipelines, maps, or map sets'''
        )
        parser.add_argument(
            '--test%i-abs'%t, action='store_true',
            help='''Use the absolute value of the test plot for
            comparisons.'''
        )
        parser.add_argument(
            '--test%i-label'%t, default='test%i'%t,
            help='''Label for test'''
        )
        parser.add_argument(
            '--test%i-param-selections'%t, type=str, required=False,
            action='append',
            help='''Param selections to apply to --test pipeline config(s). Not
            applicable if --test specifies stored map or map sets'''
        )
    parser.add_argument(
        '--combine', type=str, action='append',
        help='''Combine by wildcard string, where string globbing (a la command
        line) uses asterisk for any number of wildcard characters. Use single
        quotes such that asterisks do not get expanded by the shell. Repeat the
        --combine option for multiple combine strings.'''
    )
    parser.add_argument(
        '--json', action='store_true',
        help='''Save output maps in compressed json (json.bz2) format.'''
    )
    parser.add_argument(
        '--pdf', action='store_true',
        help='''Save plots in PDF format. If neither this nor --png is
        specified, no plots are produced.'''
    )
    parser.add_argument(
        '--png', action='store_true',
        help='''Save plots in PNG format. If neither this nor --pdf is
        specfied, no plots are produced.'''
    )
    parser.add_argument(
        '--diff-min', type=float, required=False,
        help='''Difference plot vmin; if you specify only one of --diff-min or
        --diff-max, symmetric limits are automatically used (min = -max).'''
    )
    parser.add_argument(
        '--diff-max', type=float, required=False,
        help='''Difference plot max; if you specify only one of --diff-min or
        --diff-max, symmetric limits are automatically used (min = -max).'''
    )
    parser.add_argument(
        '--fract-diff-min', type=float, required=False,
        help='''Fractional difference plot vmin; if you specify only one of
        --fract-diff-min or --fract-diff-max, symmetric limits are
        automatically used (min = -max).'''
    )
    parser.add_argument(
        '--fract-diff-max', type=float, required=False,
        help='''Fractional difference plot max; if you specify only one of
        --fract-diff-min or --fract-diff-max, symmetric limits are
        automatically used (min = -max).'''
    )
    parser.add_argument(
        '--asymm-min', type=float, required=False,
        help='''Asymmetry plot vmin; if you specify only one of --asymm-min or
        --asymm-max, symmetric limits are automatically used (min = -max).'''
    )
    parser.add_argument(
        '--asymm-max', type=float, required=False,
        help='''Fractional difference plot max; if you specify only one of
        --asymm-min or --asymm-max, symmetric limits are automatically used
        (min = -max).'''
    )
    parser.add_argument(
        '-v', action='count',
        help='Set verbosity level; repeat -v for higher level.'
    )
    parser.add_argument(
        '--only-stage', metavar='STAGE', type=str,
        help='''Test stage: Instantiate a single stage in the pipeline
        specification and run it in isolation (as the sole stage in a
        pipeline). If it is a stage that requires inputs, these can be
        specified with the --infile argument, or else dummy stage input maps
        (numpy.ones(...), matching the input binning specification) are
        generated for testing purposes. See also --infile and --transformfile
        arguments.'''
    )
    args = parser.parse_args()
    return args

def get_outputs(f_list, label, combine, only_stage):
    # Get the distribution(s) into the form of a test MapSet
    if f_list is None:
        return None
    elif f_list[0].endswith('.cfg'):
        if len(f_list) == 1:
            out_pipeline = Pipeline(f_list[0])
            if only_stage is None:
                out = out_pipeline.get_outputs()
            else:
                idx = out_pipeline.index(only_stage)
                stage = out_pipeline[idx]
                indices = slice(idx, idx+1)
                # Create dummy inputs
                if hasattr(stage, 'input_binning'):
                    logging.warn(
                        'Stage requires input, so building dummy'
                        ' inputs of random numbers, with random state set to the input'
                        ' index according to alphabetical ordering of input names and'
                        ' filled in alphabetical ordering of dimension names.'
                    )
                    input_maps = []
                    tmp = deepcopy(stage.input_binning)
                    alphabetical_binning = tmp.reorder_dimensions(sorted(tmp.names))
                    for input_num, input_name in enumerate(sorted(stage.input_names)):
                        # Create a new map with all 1's; name according to the input
                        hist = np.full(shape=alphabetical_binning.shape,
                                       fill_value=1.0)
                        input_map = Map(name=input_name, binning=alphabetical_binning,
                                        hist=hist)

                        # Apply Poisson fluctuations to randomize the values in the map
                        #input_map.fluctuate(method='poisson', random_state=input_num)

                        # Reorder dimensions according to user's original binning spec
                        input_map.reorder_dimensions(stage.input_binning)
                        input_maps.append(input_map)
                    inputs = MapSet(maps=input_maps, name='ones', hash=1)
                else:
                    inputs = None
                out = stage.get_outputs(inputs=inputs)
        else:
            out_pipeline = DistributionMaker(f_list)
            outputs = out_pipeline.get_outputs()
            out = sum(outputs)
    elif f_list[0].endswith('.json') or f_list[0].endswith('.json.bz2'):
        out = sum([Map.from_json(f) for f in f_list])
    else:
        raise ValueError('no method for input %s implemented'%out)

    if combine is not None:
        out = out.combine_wildcard(combine)
        if isinstance(out, Map):
            out = MapSet([out])
    out.name = label
    out.tex = label
    return out

def main():
    args = parse_args()
    set_verbosity(args.v)

    plot_formats = []
    if args.pdf:
        plot_formats.append('pdf')
    if args.png:
        plot_formats.append('png')


    args.outdir = os.path.expanduser(os.path.expandvars(args.outdir))
    mkdir(args.outdir)


    # get the maps
    ref = get_outputs(args.ref, args.ref_label, args.combine, args.only_stage)

    tests = []
    for t in range(10):
        test = get_outputs(getattr(args,'test%i'%t),getattr(args,'test%i_label'%t), args.combine, args.only_stage)
        if test is not None:
            assert(set(test.names) == set(ref.names),
                'Test%i map names %s do not match ref map names %s.'%(t, ' '.join(sorted(test.names)), ' '.join(sorted(ref.names)))
            )
            reordered_test = []
            for ref_map in ref:
                test_map = test[ref_map.name].reorder_dimensions(ref_map.binning)
                reordered_test.append(test_map)
            reordered_test =  MapSet(reordered_test)
            reordered_test.name = test.name
            reordered_test.tex = test.tex
            tests.append(reordered_test)

    


    # Save to disk the maps being plotted (excluding optional aboslute value
    # operations)
    if args.json:
        for maps in [ref] + tests:
            path = os.path.join(
                args.outdir, 'maps_%s.json.bz2' %maps.name
            )
            to_file(maps, path)


    # Plot the raw distributions
    plotter = Plotter(stamp='', outdir=args.outdir, fmt=plot_formats,
                      log=False, annotate=False,
                      symmetric=False,
                      ratio=False)


    # individual 2d maps
    for maps in [ref] + tests:
        plotter = Plotter(stamp='', outdir=args.outdir, fmt=plot_formats,
                      log=False, annotate=False,
                      symmetric=False,
                      ratio=False)
        plotter.plot_2d_array(maps, fname='2d_distr-%s'
                          %maps.name)

    # 1d arrays
    #for axis in ref[0].binning.names:
    #    plotter = Plotter(stamp='', outdir=args.outdir, fmt=plot_formats,
    #                  log=False, annotate=False,
    #                  symmetric=False,
    #                  ratio=False)
    #    plotter.ratio = False
    #    for maps in [ref] + tests:
    #        plotter.plot_1d_array(
    #            maps, axis,
    #            fname='%s-%s'%(axis,maps.name),
    #        )

    # 1d comparisons
    for axis in ref[0].binning.names:
        if 'energy' in axis:
            log=True
        else:
            log=False
        plotter = Plotter(stamp='', outdir=args.outdir, fmt=plot_formats,
                      log=log, annotate=False,
                      symmetric=False,
                      ratio=False)
        plotter.ratio = True
        plotter.plot_1d_cmp(
            [ref] + tests, axis,
            fname='%s-%s-%s'%(axis, ref.name, '-'.join([t.name for t in tests]))
        )


main.__doc__ = __doc__


if __name__ == '__main__':
    main()
