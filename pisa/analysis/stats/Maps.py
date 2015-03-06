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

def get_pseudo_data_fmap(template_maker,fiducial_params,seed=None,chan='all'):
    '''
    Creates a true template from fiducial_params, then uses Poisson statistics
    to vary the expected counts per bin to create a pseudo data set.
    If seed is provided, the random state is seeded with seed before the map is
    created.

    IMPORTANT: returns a SINGLE flattened map of trck/cscd combined
    \params:
      * chan = channel of flattened fmap to use.
        if 'all': returns a single flattened map of trck/cscd combined.
        if 'cscd' or 'trck' only returns the channel requested.
    '''

    true_template = template_maker.get_template(fiducial_params)
    true_fmap = flatten_map(true_template,chan=chan)
    fmap = get_random_map(true_fmap, seed=seed)

    return fmap

def flatten_map(template,chan='all'):
    '''
    Takes a final level true (expected) template of trck/cscd, and returns a
    single flattened map of trck appended to cscd, with all zero bins
    removed.
    '''

    logging.trace("Getting flattened map of chan: %s"%chan)
    if chan == 'all':
        cscd = template['cscd']['map'].flatten()
        trck = template['trck']['map'].flatten()
        fmap = np.append(cscd,trck)
        fmap = np.array(fmap)[np.nonzero(fmap)]
    elif chan == 'trck':
        trck = template[chan]['map'].flatten()
        fmap = np.array(trck)
        fmap = np.array(fmap)[np.nonzero(fmap)]
    elif chan == 'cscd':
        cscd = template[chan]['map'].flatten()
        fmap = np.array(cscd)
        fmap = np.array(fmap)[np.nonzero(fmap)]
    elif chan == 'combined':
        cscd = template['cscd']['map'].flatten()
        trck = template['trck']['map'].flatten()
        fmap = cscd + trck
        fmap = np.array(fmap)[np.nonzero(fmap)]
    else:
        raise ValueError("chan: '%s' not implemented! Only 'all', 'trck', or 'cscd' is allowed")

    return fmap

def get_seed():
    '''
    Returns a random seed from /dev/urandom that can be used to seed the random
    state, e.g. for the poisson random variates.
    '''
    return int(os.urandom(4).encode('hex'),16)

