#!/usr/bin/env python

# authors: T. Ehrhardt
# email:   tehrhardt@icecube.wisc.edu
# date:    October 21, 2016
"""
Theta23 Octant/Maximal Mixing

This script/module calculates and plots octant of theta23/maximal mixing
sensitivity etc. from the files recorded by the `theta23_octant.py` script.

"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from copy import deepcopy
import os

import numpy as np

from pisa.core.param import ParamSet
from pisa.utils.fileio import from_file, get_valid_filename, mkdir, to_file, nsort
from pisa.utils.log import logging, set_verbosity


def extract_octant_data(datadir):
    """Extract and aggregate analysis results.

    Parameters
    ----------
    datadir : string
        Path to directory where files are stored.
    """
    datadir = os.path.expanduser(os.path.expandvars(datadir))
    data_dicts_per_file = []
    truth_points_per_file = []
    for basename in nsort(os.listdir(datadir)):
        fpath = os.path.join(datadir, basename)
        octant_data = from_file(fpath)
        injected_points = octant_data['truth_sampled']
        fits_d = octant_data['fits']
        hypos = fits_d.keys()
        hypo_d = {hypo: {'gof': [], 'best fits': []} for hypo in hypos}
        # record the goodness-of-fit values
        data_d = {'wrong octant': deepcopy(hypo_d),
                  'maximal mixing': deepcopy(hypo_d),
                  'combined': deepcopy(hypo_d)}
        metric_str = None
        for hypo in fits_d:
            hypo_data = fits_d[hypo]
            for fit_data in hypo_data:
                assert ('wrong octant' in fit_data and
                        'maximal mixing' in fit_data)
                # very explicit for now
                wo_metric_val = fit_data['wrong octant']['metric_val']
                data_d['wrong octant'][hypo]['gof'].append(wo_metric_val)
                wo_bf_vals = get_bf_vals(fit_data['wrong octant']['params'])
                data_d['wrong octant'][hypo]['best fits'].append(wo_bf_vals)
                max_mix_metric_val = fit_data['maximal mixing']['metric_val']
                data_d['maximal mixing'][hypo]['gof'].append(max_mix_metric_val)
                max_mix_bf_vals = get_bf_vals(fit_data['maximal mixing']['params'])
                data_d['maximal mixing'][hypo]['best fits'].append(max_mix_bf_vals)
                if metric_str is None:
                    metric_str = fit_data['wrong octant']['metric']
                    assert fit_data['maximal mixing']['metric'] == metric_str
                """If the solution at maximal mixing happens to be better than
                the best fit resulting from injecting the wrong octant mirror
                point of the truth, the maximal mixing fit needs to be
                considered as the best wrong octant solution. So combine the two
                for the same hypothesis."""
                data_d['combined'][hypo]['gof'].append(min(wo_metric_val,
                                                       max_mix_metric_val))
        # minimize over the hypo selections, also minimize the combined
        # wrong octant/maximal mixing fit over the various hypotheses
        # (no "combined" best fit parameter values for now)
        for fit_key in ('wrong octant', 'maximal mixing', 'combined'):
            hypo_gofs = [data_d[fit_key][h]['gof'] for h in fits_d]
            data_d[fit_key]['best hypo'] = \
                [np.min(metrics) for metrics in zip(*hypo_gofs)]
        # get one list entry per file found in datadir
        # TODO: Keys ('wrong octant' needed at all? -> probably not)
        data_dicts_per_file.append({'wrong octant': data_d['wrong octant'],
                                    'maximal mixing': data_d['maximal mixing'],
                                    'combined': data_d['combined'],
                                    'metric': metric_str})
        truth_points_per_file.append(injected_points)
    return data_dicts_per_file, truth_points_per_file

def get_bf_vals(params):
    free_vals = {}
    for p in params:
        if not params[p]['is_fixed']:
            free_vals[p] = params[p]['value']
    return free_vals

def plot_gof(oct_dat_d, running_groups, all_fixed_vals, fixed_dim,
             metric_str, ax, xlab):
    for m, r_vals in enumerate(running_groups):
        f_val = all_fixed_vals[m]
        running_inds = [int(i) for i in np.array(r_vals)[:,0]]
        y1 = np.array(oct_dat_d['combined']['best hypo'])[running_inds]
        y2 = np.array(oct_dat_d['maximal mixing']['best hypo'])[running_inds]
        ax[0].plot(np.array(r_vals)[:,1], y1, label="%s %s"%
                                    (f_val, fixed_dim.replace("_", " ")), lw=3)
        ax[0].set_ylabel(r"Octant Sensitivity (%s)"%metric_str, fontsize=18,
                                                                   labelpad=20)
        ax[1].plot(np.array(r_vals)[:,1], y2, lw=3)
        ax[1].set_ylabel(r"Maximal Mixing Sensitivity (%s)"%metric_str,
                                                      fontsize=18, labelpad=20)
        for axi in ax:
            plt.setp(axi.spines.values(), linewidth=2)
            axi.set_xlabel(xlab, fontsize=18, labelpad=20)
            axi.tick_params(axis='both', which='major', pad=10, width=3)
            #axi.set_ylim(min(min(y1),min(y2)), max(max(y1), max(y2)))
            axi.set_ylim(0, max(max(y1), max(y2)))
            for lab in axi.get_xticklabels() + axi.get_yticklabels():
                lab.set_fontsize(18)
        ax[0].legend(loc='best', frameon=False, fontsize=14)
    return ax

def parse_args():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description='''Postprocess output files produced by theta23_octant.py
        script.'''
    )
    parser.add_argument(
        '-d', '--dir', required=True, metavar='DIR', type=str,
        help='''Directory where octant data files are stored. Will try to
             combine data as much as possible.'''
    )
    # TODO: implement
    parser.add_argument(
        '--plot-best-fits', default=False, action='store_true',
        help='''*Not implemented yet*. Also plot parameters' best fit values as
        function of injected points (in addition to goodness of fit).'''
    )
    parser.add_argument(
        '-v', action='count', default=None,
        help='set verbosity level'
    )
    return parser.parse_args()


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    try:
        plt.rcParams['text.usetex'] = True
    except:
        print "Could not use tex"
    args = parse_args()
    init_args_d = vars(args)
    data_dicts_per_file, truth_points_per_file = \
                                           extract_octant_data(datadir=args.dir)
    for (k, (oct_dat_d, tps)) in enumerate(zip(data_dicts_per_file,
                                                        truth_points_per_file)):
        metric_used = oct_dat_d['metric']
        # all livetimes in sample
        all_lt_vals = list(set([tp['livetime'][0] for tp in tps]))
        lt_dim = tps[0]['livetime'][1][0][0]
        t23_dim = tps[0]['theta23'][1][0][0]
        # all theta23 values
        all_t23_vals = list(set([tp['theta23'][0] for tp in tps]))
        fixed_lt_running_t23 = []
        fixed_lts = []
        fixed_t23_running_lt = []
        fixed_t23s = []
        for lt in all_lt_vals:
            lt_t23_vals = [(i, tp['theta23'][0]) for i,tp in enumerate(tps)
                                                    if tp['livetime'][0] == lt]
            fixed_lt_running_t23.append(lt_t23_vals)

        for t23 in all_t23_vals:
            t23_lt_vals = [(i, tp['livetime'][0]) for i,tp in enumerate(tps)
                                                    if tp['theta23'][0] == t23]
            fixed_t23_running_lt.append(t23_lt_vals)
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))
        ax1, ax2 = plot_gof(oct_dat_d, fixed_lt_running_t23, all_lt_vals,
                            lt_dim, metric_used, (ax1, ax2),
                            xlab=r"$\theta_{23}$ [%s]"%
                                                      t23_dim.replace("_", " "))
        f.tight_layout()
        plt.savefig("./fixed_lt_running_t23_%d.png"%k)

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))
        ax1, ax2 = plot_gof(oct_dat_d, fixed_t23_running_lt, all_t23_vals,
                            t23_dim, metric_used, (ax1, ax2),
                            xlab="livetime [%s]"%lt_dim.replace("_", " "))
        f.tight_layout()
        plt.savefig("./fixed_t23_running_lt_%d.png"%k)

        if args.plot_best_fits:
            pass
