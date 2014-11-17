#
# Maps.py
#
# Utilities for dealing with event rate maps in analysis
#
# author: Tim Arlen   <tca3@psu.edu>
#

import numpy as np
from pisa.analysis.stats.LLHStatistics import get_random_map

def get_pseudo_data_fmap(template_maker,fiducial_params):
    '''
    Creates a true template from fiducial_params, then uses Poisson statistics
    to vary the expected counts per bin to create a pseudo data set.

    IMPORTANT: returns a SINGLE flattened map of trck/cscd combined
    '''

    true_template = template_maker.get_template(fiducial_params)
    true_fmap = flatten_map(true_template)
    fmap = get_random_map(true_fmap)

    return fmap

def flatten_map(template):
    '''
    Takes a final level true (expected) template of trck/cscd, and returns a
    single flattened map of trck appended to cscd, with all zero bins
    removed.
    '''
    cscd = template['cscd']['map'].flatten()
    trck = template['trck']['map'].flatten()
    fmap = np.append(cscd,trck)
    fmap = np.array(fmap)[np.nonzero(fmap)]

    return fmap



