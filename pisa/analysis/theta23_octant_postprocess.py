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
        injected_points = octant_data["truth_sampled"]
        fits_d = octant_data["fits"]
        hypos = fits_d.keys()
        wo_metric = {hypo: [] for hypo in hypos}
        #wo_metric["best_hypo"] = []
        max_mix_metric = deepcopy(wo_metric)
        wo_max_comb_metric = deepcopy(wo_metric)
        for hypo in fits_d:
            hypo_data = fits_d[hypo]
            for fit_data in hypo_data:
                for fit_key in fit_data:
                    # TODO: replace the two separate dicts with two keys within
                    # the same dict?
                    # Make two variables so we can access them later when
                    # combining the two fits.
                    # TODO: Can one of the two keys not exist for some reason?
                    # If so, need to make at least one of the two variables invalid
                    if fit_key == "wrong octant":
                        wo_metric_val = fit_data[fit_key]['metric_val']
                        wo_metric[hypo].append(wo_metric_val)
                    elif fit_key == "maximal mixing":
                        max_mix_metric_val = fit_data[fit_key]['metric_val']
                        max_mix_metric[hypo].append(max_mix_metric_val)
                """If the solution at maximal mixing happens to be better than
                the best fit resulting from injecting the wrong octant mirror
                point of the truth, the maximal mixing fit needs to be
                considered as the best wrong octant solution. So combine the two
                for the same hypothesis."""
                wo_max_comb_metric[hypo].append(min(
                                            wo_metric_val, max_mix_metric_val))
        # minimize over the hypo selections
        wo_metric["best_hypo"] = [np.min(metrics) for metrics in
                                             zip(*wo_metric.values())]
        max_mix_metric["best_hypo"] = [np.min(metrics) for metrics in
                                    zip(*max_mix_metric.values())]
        # also minimize the combined fit wrong octant/maximal mixing fit over the
        # various hypotheses
        wo_max_comb_metric["best_hypo"] = [np.min(metrics) for metrics in
                                    zip(*wo_max_comb_metric.values())]
        # get one list entry per file found in datadir
        # TODO: Keys!
        data_dicts_per_file.append({"WO": wo_metric, "MM": max_mix_metric,
                                    "Comb": wo_max_comb_metric})
        truth_points_per_file.append(injected_points)
    return data_dicts_per_file, truth_points_per_file

def plot_octant_data(wo_metric, max_mix_metric, truth_sampled):
    pass


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
    data_dicts_per_file, truth_points_per_file = extract_octant_data(datadir=args.dir)
    for (i, (oct_dat_d, tp)) in enumerate(zip(data_dicts_per_file,
                                                        truth_points_per_file)):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))
        # TODO: don't simply assume chi2 metric
        # TODO: determine which true variables are running
        y1 = np.sqrt(oct_dat_d["Comb"]["best_hypo"])
        y2 = np.sqrt(oct_dat_d["MM"]["best_hypo"])
        # TODO: x-values
        ax1.plot(np.arange(38, 54, 2), y1)
        ax1.set_ylabel(r"Octant Sensitivity", fontsize=18)
        # TODO: x-values
        ax2.plot(np.arange(38, 54, 2), y2)
        ax2.set_ylabel(r"Maximal Mixing Sensitivity", fontsize=18)
        for ax in (ax1, ax2):
            ax.set_xlabel(r"$\theta_{23}$", fontsize=18)
            ax.tick_params(axis='both', which='major', pad=10)
            ax.grid()
            ax.set_ylim(min(min(y1),min(y2)), max(max(y1), max(y2)))
            for lab in ax.get_xticklabels() + ax.get_yticklabels():
                lab.set_fontsize(16)
        f.tight_layout()
        plt.savefig("./test_%d.png"%i)
