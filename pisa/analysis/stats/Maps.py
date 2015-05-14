#
# Maps.py
#
# Utilities for dealing with event rate maps in analysis
#
# author: Tim Arlen   <tca3@psu.edu>
#

import os
import numpy as np
from pisa.analysis.stats.LLHStatistics import get_random_map
from pisa.utils.log import logging


def apply_ratio_scale(orig_maps, key1, key2, ratio_scale, is_flux_scale, int_type = None):
    """
    Scales the ratio of the entries of two maps, conserving the total.
    """

    if is_flux_scale: log_str = 'flux'
    else: log_str = 'event rate (%s)'%int_type

    if not is_flux_scale:
	# we have maps of event counts of a certain interaction type
        orig_sum = orig_maps[key1][int_type]['map'] + orig_maps[key2][int_type]['map']
        orig_total1 = orig_maps[key1][int_type]['map'].sum()
        orig_total2 = orig_maps[key2][int_type]['map'].sum()
        orig_ratio = orig_maps[key1][int_type]['map'] / orig_maps[key2][int_type]['map']
    else:
        # we have flux_maps
        orig_sum = orig_maps[key1]['map'] + orig_maps[key2]['map']
        orig_total1 = orig_maps[key1]['map'].sum()
        orig_total2 = orig_maps[key2]['map'].sum()
        orig_ratio = orig_maps[key1]['map'] / orig_maps[key2]['map']

    # conserved total:
    scaled_map2 = orig_sum / (1 + ratio_scale*orig_ratio)
    scaled_map1 = ratio_scale*orig_ratio*scaled_map2

    logging.trace(' %s / %s %s ratio before scaling: %.3f'%(key1, key2, log_str,
                    orig_total1/orig_total2))
    logging.trace(' %s / %s %s ratio after scaling with %.2f: %.3f'%(key1, key2, log_str,
                    ratio_scale, scaled_map1.sum()/scaled_map2.sum()))

    return scaled_map1, scaled_map2


def get_pseudo_data_fmap(template_maker,fiducial_params,seed=None,chan=None):
    """
    Creates a true template from fiducial_params, then uses Poisson statistics
    to vary the expected counts per bin to create a pseudo data set.
    If seed is provided, the random state is seeded with seed before the map is
    created.

    IMPORTANT: returns a SINGLE flattened map of trck/cscd combined
    \params:
      * chan = channel of flattened fmap to use.
        if 'all': returns a single flattened map of trck/cscd combined.
        if 'cscd' or 'trck' only returns the channel requested.
    """

    true_template = template_maker.get_template(fiducial_params)
    true_fmap = flatten_map(true_template,chan=chan)
    fmap = get_random_map(true_fmap, seed=seed)

    return fmap

def get_asimov_fmap(template_maker,fiducial_params,chan=None):
    """Creates a true template from fiducial_params"""

    true_template = template_maker.get_template(fiducial_params)
    return flatten_map(true_template,chan=chan)

def flatten_map(template,chan='all'):
    """
    Takes a final level true (expected) template of trck/cscd, and returns a
    single flattened map of trck appended to cscd, with all zero bins
    removed.
    """

    logging.trace("Getting flattened map of chan: %s"%chan)
    if chan == 'all':
        cscd = template['cscd']['map'].flatten()
        trck = template['trck']['map'].flatten()
        fmap = np.append(cscd,trck)
    elif chan == 'trck':
        trck = template[chan]['map'].flatten()
        fmap = np.array(trck)
        #fmap = np.array(fmap)[np.nonzero(fmap)]
    elif chan == 'cscd':
        cscd = template[chan]['map'].flatten()
        fmap = np.array(cscd)
        #fmap = np.array(fmap)[np.nonzero(fmap)]
    elif chan == 'no_pid':
        cscd = template['cscd']['map'].flatten()
        trck = template['trck']['map'].flatten()
        fmap = cscd + trck
        #fmap = np.array(fmap)[np.nonzero(fmap)]
    else:
        raise ValueError(
            "chan: '%s' not implemented! Allowed: ['all', 'trck', 'cscd','no_pid']"
            %chan)

    fmap = np.array(fmap)[np.nonzero(fmap)]
    return fmap

def get_seed():
    """
    Returns a random seed from /dev/urandom that can be used to seed the random
    state, e.g. for the poisson random variates.
    """

    return int(os.urandom(4).encode('hex'),16)

