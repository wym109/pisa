#!/usr/bin/env python


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from copy import deepcopy
import os

from pisa.core.analysis import LLRAnalysis
from pisa.core.distribution_maker import DistributionMaker
from pisa.utils.config_parser import parse_pipeline_config
from pisa.utils.fileio import to_file, expandPath
from pisa.utils.log import logging, set_verbosity
from pisa.utils.resources import find_resource


def parse_args():
    parser = ArgumentParser(
        description='''Perform the LLR analysis for calculating the NMO
        sensitivity of the distribution made from data-settings compared with
        hypotheses generated from template-settings.

        Currently the output should be a json file containing the dictionary
        of best fit and likelihood values.'''
    )
    parser.add_argument(
        '--alt-hypo-pipeline', required=True,
        type=str, action='append', metavar='PIPELINE_CFG',
        help='''Settings for the generation of alternate hypothesis
        distributions; repeat this argument to specify multiple pipelines.'''
    )
    parser.add_argument(
        '--alt-hypo-param-selections',
        type=str, default=None, metavar='PARAM_SELECTOR_LIST',
        help='''Comma-separated (no spaces) list of param selectors to apply to
        the alt hypothesis distribution maker's pipelines.'''
    )
    parser.add_argument(
        '--null-hypo-pipeline',
        type=str, action='append', default=None, metavar='PIPELINE_CFG',
        help='''Settings for the generation of null hypothesis distributions;
        repeat this argument to specify multiple pipelines. If omitted, the
        same settings as specified for --alt-hypo-pipeline are used to generate
        the null hypothesis distributions (and so you have to use the
        --null-hypo-param-selections argument to generate a hypotheses distinct
        from the alt hypothesis while using alt hypo's distribution maker).'''
    )
    parser.add_argument(
        '--null-hypo-param-selections',
        type=str, default=None, metavar='PARAM_SELECTOR_LIST',
        help='''Comma-separated (no spaces) list of param selectors to apply to
        the null hypothesis distribution maker's pipelines.'''
    )
    parser.add_argument(
        '--data-pipeline',
        type=str, action='append', default=None, metavar='PIPELINE_CFG',
        help='''Settings for the generation of "data" distributions; repeat
        this argument to specify multiple pipelines. If omitted, the same
        settings as specified for --alt-hypo-pipeline are used to generate data
        distributions (i.e., data is assumed to come from the alternate
        hypothesis.'''
    )
    parser.add_argument(
        '--data-param-selections',
        type=str, default=None, metavar='PARAM_SELECTOR_LIST',
        help='''Comma-separated list of param selectors to apply to the data
        distribution maker's pipelines. If neither --data-pipeline nor
        --data-param-selections are specified, *both* are copied from
        --alt-hypo-pipeline and --alt-param-selections, respectively. However,
        if --data-pipeline is specified while --data-param-selections is not,
        then the param selections in the pipeline config file(s) specified are
        used to produce data distributions.'''
    )
    parser.add_argument(
        '-m', '--minimizer-settings',
        type=str, metavar='MINIMIZER_CFG', required=True,
        help='''Settings related to the optimizer used in the LLR analysis.'''
    )
    parser.add_argument(
        '--fluctuate-data',
        type=bool, action='store_true'
        help='''Apply fluctuations to the data distribution. This should *not*
        be set for analyzing "real" (measured) data, and it is common to not
        use this feature even for Monte Carlo analysis. If this is not set,
        --num-data-trials is forced to 1.'''
    )
    parser.add_argument(
        '--fluctuate-fid-data',
        type=bool, action='store_true'
        help='''Apply fluctuations to the fiducaial data distributions. If this
        is not set, --num-fid-data-trials is forced to 1.'''
    )
    parser.add_argument(
        '--num-data-trials',
        type=int, default=1,
        help='''When performing Monte Carlo analysis, set to > 1 to produce
        multiple pseudodata distributions from the data distribution maker's
        Asimov data distribution. This is overridden if --fluctuate-data is not
        set (since each data distribution will be identical if it is not
        fluctuated). This is typically left at 1 (i.e., the Asimov distribution
        is assumed to be representative.'''
    )
    parser.add_argument(
        '-n', '--num-fid-data-trials',
        type=int, default=1,
        help='''Number of fiducial pseudodata trials to run. In our experience,
        it takes ~10^3-10^5 fiducial psuedodata trials to achieve low
        uncertainties on the resulting significance, though that exact number
        will vary based upon the details of an analysis.'''
    )
    parser.add_argument(
        '-d', '--logdir', required=True,
        metavar='DIR', type=str,
        help='Directory into which to store results and metadata.'
    )
    parser.add_argument(
        '--alt-hypo-name',
        type=str, metavar='NAME', default='alt hypo',
        help='''Name for the alternate hypothesis. E.g., "NO" for normal
        ordering in the neutrino mass ordering analysis. Note that the name
        here has no bearing on the actual process, so it's important that you
        be careful to use a name that appropriately identifies the
        hypothesis.'''
    )
    parser.add_argument(
        '--null-hypo-name',
        type=str, metavar='NAME', default='null hypo',
        help='''Name for the null hypothesis. E.g., "IO" for inverted
        ordering in the neutrino mass ordering analysis. Note that the name
        here has no bearing on the actual process, so it's important that you
        be careful to use a name that appropriately identifies the
        hypothesis.'''
    )
    parser.add_argument(
        '--data-name',
        type=str, metavar='NAME', default='data',
        help='''Name for the data. E.g., "NO" for normal ordering in the
        neutrino mass ordering analysis. Note that the name here has no bearing
        on the actual process, so it's important that you be careful to use a
        name that appropriately identifies the hypothesis.'''
    )
    parser.add_argument(
        '--no-post-processing',
        type=bool, action='store_true'
        help='''Do not run post-processing for the trials run. This is useful
        if the analysis is divided and run in separate processes, whereby only
        after all processes are run should post-processing be performed
        (once).'''
    )
    parser.add_argument(
        '-v', action='count', default=None,
        help='set verbosity level'
    )
    return parser.parse_args()


# TODO: make this work with Python package resources, not merely absolute
# paths! ... e.g. hash on the file or somesuch?
def normpath(path, checkdir=False):
    normpath = os.path.abspath(os.path.expandvars(os.path.expanduser(path)))
    if checkdir:
        kind = 'dir'
        check = os.path.isdir
    else:
        kind = 'file'
        check = os.path.isfile

    if not check(path):
        raise IOError('Path "%s" which resolves to "%s" is not a %s.'
                      %(path, newpath, %kind))

    return normpath


if __name__ == '__main__':
    args_d = vars(parse_args())

    set_verbosity(args_d.pop('v'))

    # Normalize and convert `*_pipeline` filenames; store to `*_maker` in
    # order which is argument that LLRAnalysis init takes.
    for maker in ['alt_hypo', 'null_hypo', 'data']:
        filenames = args_d.pop(maker + '_pipeline')
        if filenames is not None:
            filenames = sorted(
                [normpath(fname) for fname in pipeline_config_filenames]
            )
        args_d[maker + '_maker'] = filenames

    # Instantiate the analysis object
    llr_analysis = LLRAnalysis(**args_d)


    # Run the analysis
    llr_analysis.run_analysis()

    # Run postprocessing if called to do so
