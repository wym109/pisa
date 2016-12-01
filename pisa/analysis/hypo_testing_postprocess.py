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

from argparse import ArgumentParser
from collections import OrderedDict
import os
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
plt.rcParams['text.usetex'] = True
import numpy as np

from pisa import ureg
from pisa.analysis.hypo_testing import Labels
from pisa.utils.fileio import from_file, nsort
from pisa.utils.log import set_verbosity


__all__ = ['extract_trials', 'extract_fit', 'parse_args', 'main']


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
    logdir = os.path.expanduser(os.path.expandvars(logdir))
    config_summary_fpath = os.path.join(logdir, 'config_summary.json')
    cfg = from_file(config_summary_fpath)

    data_is_data = cfg['data_is_data']
    if data_is_data and fluctuate_data:
        raise ValueError('Analysis was performed on data, so `fluctuate_data`'
                         ' is not supported.')

    # Get naming scheme
    labels = Labels(
        h0_name=cfg['h0_name'], h1_name=cfg['h1_name'],
        data_name=cfg['data_name'], data_is_data=data_is_data,
        fluctuate_data=fluctuate_data, fluctuate_fid=fluctuate_fid
    )

    #for key in labels.dict.keys():
    #    print key

    # Find all relevant data dirs, and from each extract the fiducial fit(s)
    # information contained
    data_sets = OrderedDict()
    for basename in nsort(os.listdir(logdir)):
        m = labels.subdir_re.match(basename)
        if m is None:
            continue

        if fluctuate_data:
            data_ind = int(m.groupdict()['data_ind'])
            dset_label = data_ind
        else:
            dset_label = labels.data_prefix
            if not labels.data_name in [None, '']:
                dset_label += '_' + labels.data_name
            if not labels.data_suffix in [None, '']:
                dset_label += '_' + labels.data_suffix

        lvl2_fits = OrderedDict()
        lvl2_fits['h0_fit_to_data'] = None
        lvl2_fits['h1_fit_to_data'] = None

        # Account for failed jobs. Get the set of file numbers that exist
        # for all h0 an h1 combinations
        file_nums = OrderedDict()
        subdir = os.path.join(logdir, basename)
        for fnum, fname in enumerate(nsort(os.listdir(subdir))):
            fpath = os.path.join(subdir, fname)
            for x in ['0', '1']:
                for y in ['0','1']:
                    k = 'h{x}_fit_to_h{y}_fid'.format(x=x, y=y)
                    r = labels.dict[k + '_re']
                    m = r.match(fname)
                    if m is None:
                        continue
                    if fluctuate_fid:
                        fid_label = int(m.groupdict()['fid_ind'])
                    else:
                        fid_label = labels.fid
                    if k not in file_nums:
                        file_nums[k] = []
                    file_nums[k].append(fid_label)
                    break

        set_file_nums = []
        for hypokey in file_nums.keys():
            if len(set_file_nums) == 0:
                set_file_nums = set(file_nums[hypokey])
            else:
                set_file_nums = set_file_nums.intersection(file_nums[hypokey])

        for fnum, fname in enumerate(nsort(os.listdir(subdir))):
            fpath = os.path.join(subdir, fname)
            for x in ['0', '1']:
                k = 'h{x}_fit_to_data'.format(x=x)
                if fname == labels.dict[k]:
                    lvl2_fits[k] = extract_fit(fpath, 'metric_val')
                    break
                for y in ['0','1']:
                    k = 'h{x}_fit_to_h{y}_fid'.format(x=x, y=y)
                    r = labels.dict[k + '_re']
                    m = r.match(fname)
                    if m is None:
                        continue
                    if fluctuate_fid:
                        fid_label = int(m.groupdict()['fid_ind'])
                    else:
                        fid_label = labels.fid
                    if k not in lvl2_fits:
                        lvl2_fits[k] = OrderedDict()
                    if fid_label in set_file_nums:
                        lvl2_fits[k][fid_label] = \
                            extract_fit(fpath, ['metric_val','fit_history'])
                        fit_history = lvl2_fits[k][fid_label].pop('fit_history')
                        lvl2_fits[k][fid_label]['best_fit_params'] = \
                            fit_history[-1]
                    break
        data_sets[dset_label] = lvl2_fits
        data_sets[dset_label]['params'] = \
            extract_fit(fpath, ['params'])['params']
    return data_sets, labels


def extract_fit(fpath, keys=None):
    """Extract fit info from a file.

    Parameters
    ----------
    fpath : string
        Path to the file

    keys : None, string, or iterable of strings
        Keys to extract. If None, all keys are extracted.

    """
    info = from_file(fpath)
    if keys is None:
        return info
    if isinstance(keys, basestring):
        keys = [keys]
    for key in info.keys():
        if key not in keys:
            info.pop(key)
    return info


def make_llr_plots(data, labels, detector, selection):

    h0_fit_to_h0_fid_metrics = np.array(data['h0_fit_to_h0_fid']['metric_val'])
    h1_fit_to_h0_fid_metrics = np.array(data['h1_fit_to_h0_fid']['metric_val'])
    h0_fit_to_h1_fid_metrics = np.array(data['h0_fit_to_h1_fid']['metric_val'])
    h1_fit_to_h1_fid_metrics = np.array(data['h1_fit_to_h1_fid']['metric_val'])

    num_trials = len(h0_fit_to_h0_fid_metrics)
    
    LLRh0 = h0_fit_to_h0_fid_metrics - h1_fit_to_h0_fid_metrics
    LLRh1 = h0_fit_to_h1_fid_metrics - h1_fit_to_h1_fid_metrics

    minLLR = min(min(LLRh0), min(LLRh1))
    maxLLR = max(max(LLRh0), max(LLRh1))
    rangeLLR = maxLLR - minLLR
    binning = np.linspace(minLLR - 0.1*rangeLLR,
                          maxLLR + 0.1*rangeLLR,
                          num_trials/3)
    binwidth = binning[1]-binning[0]

    LLRh0hist, LLRh0binedges = np.histogram(LLRh0,bins=binning)
    LLRh1hist, LLRh1binedges = np.histogram(LLRh1,bins=binning)

    LLRhistmax = max(max(LLRh0hist),max(LLRh1hist))

    inj_name = labels['data_name']
    h0_name = labels['h0_name']
    h1_name = labels['h1_name']

    plot_labels = []
    plot_labels.append(
        (r"%s best fit - $\log\left[\mathcal{L}\left(\mathcal{H}_{%s}\right)/"
         r"\mathcal{L}\left(\mathcal{H}_{%s}\right)\right]$"
         %(h0_name, h0_name, h1_name))
    )
    plot_labels.append(
        (r"%s best fit - $\log\left[\mathcal{L}\left(\mathcal{H}_{%s}\right)/"
         r"\mathcal{L}\left(\mathcal{H}_{%s}\right)\right]$"
         %(h1_name, h0_name, h1_name))
    )
    plot_title = ('%s %s Event Selection LLR Distributions for true '
                  '%s (%i trials)'
                  %(detector,selection,inj_name,num_trials))

    # Factor with which to make everything visible
    plot_scaling_factor = 1.55

    plt.hist(LLRh0,bins=binning,color='r',histtype='step')
    plt.hist(LLRh1,bins=binning,color='b',histtype='step')
    plt.xlabel(r'Log-Likelihood Ratio')
    plt.ylabel(r'Number of Trials (per %.2f)'%binwidth)
    plt.ylim(0,plot_scaling_factor*LLRhistmax)
    plt.legend(plot_labels, loc='upper left')
    plt.title(plot_title)
    plt.savefig('llr.png')
    plt.close()

    
def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        '-d', '--dir', required=True,
        metavar='DIR', type=str,
        help='Directory into which to store results and metadata.'
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--asimov', action='store_true',
        help='''Analyze the Asimov trials in the specified directories.'''
    )
    group.add_argument(
        '--llr', action='store_true',
        help='''Analyze the LLR trials in the specified directories.'''
    )
    parser.add_argument('--detector',type=str,default='',
                        help="Name of detector to put in histogram titles")
    parser.add_argument('--selection',type=str,default='',
                        help="Name of selection to put in histogram titles")
    parser.add_argument(
        '-v', action='count', default=None,
        help='set verbosity level'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    init_args_d = vars(args)

    # NOTE: Removing extraneous args that won't get passed to instantiate the
    # HypoTesting object via dictionary's `pop()` method.

    set_verbosity(init_args_d.pop('v'))

    detector = init_args_d.pop('detector')
    selection = init_args_d.pop('selection')

    if args.asimov:
        data_sets, labels = extract_trials(logdir=args.dir, fluctuate_fid=False,
                                           fluctuate_data=False)
        od = data_sets.values()[0]
        #if od['h1_fit_to_h0_fid']['fid_asimov']['metric_val'] > od['h0_fit_to_h1_fid']['fid_asimov']['metric_val']:
        print np.sqrt(np.abs(od['h1_fit_to_h0_fid']['fid_asimov']['metric_val'] - od['h0_fit_to_h1_fid']['fid_asimov']['metric_val']))

    else:
        data_sets, labels = extract_trials(logdir=args.dir,
                                           fluctuate_fid=True,
                                           fluctuate_data=False)

        datakeys = ['h0_fit_to_data', 'h1_fit_to_data', 'h1_fit_to_h1_fid', 'h1_fit_to_h0_fid', 'h0_fit_to_h1_fid', 'h0_fit_to_h0_fid']

        values = {}

        for injkey in data_sets.keys():
            values[injkey] = {}
            alldata = data_sets[injkey]
            paramkeys = alldata['params'].keys()
            for datakey in alldata.keys():
                if datakey is not 'params':
                    if alldata[datakey] is not None:
                        values[injkey][datakey] = {}
                        values[injkey][datakey]['metric_val'] = []
                        for paramkey in paramkeys:
                            values[injkey][datakey][paramkey] = []
                        trials = alldata[datakey]
                        for trial_num in trials.keys():
                            trial = trials[trial_num]
                            values[injkey][datakey]['metric_val'].append(
                                trial['metric_val']
                            )
                            for param_name, param_val in zip(
                                    paramkeys,
                                    trial['best_fit_params']):
                                values[injkey][datakey][param_name].append(
                                    param_val
                                )

        for injkey in values.keys():

            make_llr_plots(
                data = values[injkey],
                labels = labels.dict,
                detector = detector,
                selection = selection
            )
            
        
if __name__ == '__main__':
    main()
