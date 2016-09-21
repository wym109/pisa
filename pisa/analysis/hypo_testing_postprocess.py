#!/usr/bin/env python

# authors: J.L. Lanfranchi, P.Eller, and S. Wren
# email:   jll1062+pisa@phys.psu.edu
# date:    March 20, 2016
"""
Hypothesis testing: How do two hypotheses compare for describing MC or data?

This script/module computes significances, etc. from the logfiles recorded by
the `hypo_testing.py` script.

"""


from __future__ import division

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import Mapping, OrderedDict, Sequence
from copy import copy, deepcopy
import getpass
import os
import random
import socket
import string
import sys
import time
from traceback import format_exc

import numpy as np
import pint

from pisa import ureg, _version, __version__
from pisa.analysis.hypo_testing import Labels
#from pisa.core.map import VALID_METRICS
#from pisa.utils.comparisons import normQuant
#from pisa.utils.fileio import from_file, get_valid_filename, mkdir, to_file
#from pisa.utils.hash import hash_obj
#from pisa.utils.log import logging, set_verbosity
#from pisa.utils.random_numbers import get_random_state
#from pisa.utils.resources import find_resource
#from pisa.utils.timing import timediffstamp, timestamp


def get_config(logdir):
    config_summary_fpath = os.path.join(logdir, 'config_summary.json')
    summary = from_file(config_summary_fpath, sort_keys=False)
    return summary

    d['metric_optimized'] = self.metric
    summary['minimizer_info'] = d

    summary['data_name'] = self.data_name
    summary['data_is_data'] = self.data_is_data
    summary['data_hash'] = self.data_hash
    summary['data_param_selections'] = ','.join(self.data_param_selections)
    summary['data_params_state_hash'] = self.data_maker.params.state_hash
    summary['data_params'] = [str(p) for p in self.data_maker.params]
    summary['data_pipelines'] = self.summarize_dist_maker(self.data_maker)

    self.h0_maker.select_params(self.h0_param_selections)
    self.h0_maker.reset_free()
    summary['h0_name'] = self.h0_name
    summary['h0_hash'] = self.h0_hash
    summary['h0_param_selections'] = ','.join(self.h0_param_selections)
    summary['h0_params_state_hash'] = self.h0_maker.params.state_hash
    summary['h0_params'] = [str(p) for p in self.h0_maker.params]
    summary['h0_pipelines'] = self.summarize_dist_maker(self.h0_maker)

    self.h1_maker.select_params(self.h1_param_selections)
    self.h1_maker.reset_free()
    summary['h1_name'] = self.h1_name
    summary['h1_hash'] = self.h1_hash
    summary['h1_param_selections'] = ','.join(self.h1_param_selections)
    summary['h1_params_state_hash'] = self.h1_maker.params.state_hash
    summary['h1_params'] = [str(p) for p in self.h1_maker.params]
    summary['h1_pipelines'] = self.summarize_dist_maker(self.h1_maker)

    # Reverse the order so it serializes to a file as intended
    # (want top-to-bottom file convention vs. fifo streaming data
    # convention)
    od = OrderedDict()
    for ok, ov in (summary.items()):
        if isinstance(ov, OrderedDict):
            od1 = OrderedDict()
            for ik, iv in (ov.items()):
                od1[ik] = iv
            ov = od1
        od[ok] = ov

    to_file(od, self.config_summary_fpath, sort_keys=False)


def extract_trials(logdir, fluctuate_fid, fluctuate_data=False):
    """Extract and aggregate analysis results.

    Parameters
    ----------
    logdir : string
        Path to logging directory where files are stored. This should contain
        e.g. the "config_summary.json" file.

    fluctuate_fid : bool
        Whether the trials you're interested in applied fluctuations to the
        fiducial-fit Asimov distributions. `fluctuate_fid` False is equivalent
        to specifying an Asimov analysis (so long as the metric used was
        chi-squared).

    fluctuate_data : bool
        Whether the trials you're interested in applied fluctuations to the
        (toy) data. This is invalid if actual data was processed.

    Note that a single `logdir` can have different kinds of analyses run and
    results be logged within, so `fluctuate_fid` and `fluctuate_data` allows
    these to be separated from one another.

    """
    config_summary_fpath = os.path.join(logdir, 'config_summary.json')
    cfg = from_file(config_summary_fpath, sort_keys=False)

    data_is_data = cfg['data_is_data']
    if data_is_data and fluctuate_data:
        raise ValueError('Analysis was performed on data, so `fluctuate_data`'
                         ' is not supported.')

    # Get naming scheme 
    labels = Labels(
        h0_name=cfg['h0_name'], h1_name=cfg['h1_name'],
        data_name=cfg['data_name'], data_is_data=self.data_is_data,
        fluctuate_data=self.fluctuate_data, fluctuate_fid=self.fluctuate_fid
    )



def extract_data_asimov_trials(logdirs):
    pass
def extract_data_llr_trials(logdirs):
    pass
def extract_mc_asimov_trials(logdirs):
    pass
def extract_mc_llr_trials(logdirs):
    pass

def parse_args():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description='''Postprocess logfiles produced by hypo_testing.py
        script.'''
    )
    parser.add_argument(
        '-d', '--logdir', required=True,
        metavar='DIR', type=str,
        help='Directory into which to store results and metadata.'
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--asimov-analysis', action='store_true',
        help='''Analyze the Asimov trials in the specified directories.'''
    )
    group.add_argument(
        '--llr-analysis', action='store_true',
        help='''Analyze the LLR trials in the specified directories.'''
    )

    parser.add_argument(
        '--allow-dirty',
        action='store_true',
        help='''Warning: Use with caution. (Allow for run despite dirty
        repository.)'''
    )
    parser.add_argument(
        '--allow-no-git-info',
        action='store_true',
        help='''*** DANGER! Use with extreme caution! (Allow for run despite
        complete inability to track provenance of code.)'''
    )
    parser.add_argument(
        '-v', action='count', default=None,
        help='set verbosity level'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    init_args_d = vars(args)

    # NOTE: Removing extraneous args that won't get passed to instantiate the
    # HypoTesting object via dictionary's `pop()` method.

    set_verbosity(init_args_d.pop('v'))
    init_args_d['check_octant'] = not init_args_d.pop('no_octant_check')

    init_args_d['data_is_data'] = not init_args_d.pop('data_is_mc')

    init_args_d['store_minimizer_history'] = (
        not init_args_d.pop('no_minimizer_history')
    )

    other_metrics = init_args_d.pop('other_metric')
    if other_metrics is not None:
        other_metrics = [s.strip().lower() for s in other_metrics]
        if 'all' in other_metrics:
            other_metrics = sorted(VALID_METRICS)
        if init_args_d['metric'] in other_metrics:
            other_metrics.remove(init_args_d['metric'])
        if len(other_metrics) == 0:
            other_metrics = None
        else:
            logging.info('Will evaluate other metrics %s' %other_metrics)
        init_args_d['other_metrics'] = other_metrics

    # Normalize and convert `*_pipeline` filenames; store to `*_maker`
    # (which is argument naming convention that HypoTesting init accepts).
    for maker in ['h0', 'h1', 'data']:
        filenames = init_args_d.pop(maker + '_pipeline')
        if filenames is not None:
            filenames = sorted(
                [normcheckpath(fname) for fname in filenames]
            )
        init_args_d[maker + '_maker'] = filenames

        ps_name = maker + '_param_selections'
        ps_str = init_args_d[ps_name]
        if ps_str is None:
            ps_list = None
        else:
            ps_list = [x.strip().lower() for x in ps_str.split(',')]
        init_args_d[ps_name] = ps_list

    # Instantiate the analysis object
    hypo_testing = HypoTesting(**init_args_d)

    # Run the analysis
    hypo_testing.run_analysis()
