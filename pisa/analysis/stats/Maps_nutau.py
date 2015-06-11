#
# Maps_nutau.py
#
# Utilities augmenting and/or replacing those in Maps.py for dealing with event
# rate maps in analysis
#
# author: Feifei Huang <fxh140@psu.edu>
# date:   2015-06-11
#

import os
import numpy as np
from pisa.analysis.stats.LLHStatistics import get_random_map
from pisa.utils.log import logging
import pisa.analysis.stats.Maps as Maps


def get_asimov_data_fmap_up_down(template_maker, fiducial_params, channel=None):
    if fiducial_params['residual_up_down'] or fiducial_params['simp_up_down'] or fiducial_params['ratio_up_down']:
        template_maker_up = template_maker[0]
        template_maker_down = template_maker[1]
        template_up = template_maker_up.get_template(fiducial_params)  
        template_down = template_maker_down.get_template(fiducial_params)  
        reflected_template_down = get_flipped_map(template_down, channel=fiducial_params['channel'])
        true_fmap_up = Maps.flatten_map(template_up, channel=fiducial_params['channel'])
        true_fmap_down = Maps.flatten_map(reflected_template_down, channel=fiducial_params['channel'])
        fmap_up = np.int32(true_fmap_up+0.5)
        fmap_down = np.int32(true_fmap_down+0.5)
        if fiducial_params['residual_up_down']:
            fmap = fmap_up-fmap_down
        elif fiducial_params['ratio_up_down']:
            fmap = np.array([fmap_up, fmap_down])
        else:
            fmap = np.append(fmap_up, fmap_down)
    else:
        true_template = template_maker.get_template(fiducial_params)  
        true_fmap = Maps.flatten_map(true_template, channel=channel)
        fmap = get_random_map(true_fmap, seed=seed)
    return fmap

def get_pseudo_data_fmap(template_maker, fiducial_params, channel, seed=None):
    """
    Creates a true template from fiducial_params, then uses Poisson statistics
    to vary the expected counts per bin to create a pseudo data set.
    If seed is provided, the random state is seeded with seed before the map is
    created.

    IMPORTANT: returns a SINGLE flattened map of trck/cscd combined
    \params:
      * channel = channel of flattened fmap to use.
        if 'all': returns a single flattened map of trck/cscd combined.
        if 'cscd' or 'trck' only returns the channel requested.
    """

    if fiducial_params['residual_up_down'] or fiducial_params['simp_up_down'] or fiducial_params['ratio_up_down']:
        template_maker_up = template_maker[0]
        template_maker_down = template_maker[1]
        template_up = template_maker_up.get_template(fiducial_params)  
        template_down = template_maker_down.get_template(fiducial_params)  
        reflected_template_down = get_flipped_map(template_down, channel=fiducial_params['channel'])
        true_fmap_up = Maps.flatten_map(template_up, channel=fiducial_params['channel'])
        true_fmap_down = Maps.flatten_map(reflected_template_down, channel=fiducial_params['channel'])
        fmap_up = get_random_map(true_fmap_up, seed=Maps.get_seed())
        fmap_down = get_random_map(true_fmap_down, seed=Maps.get_seed())
        if fiducial_params['residual_up_down']:
            fmap = fmap_up-fmap_down
        elif fiducial_params['ratio_up_down']:
            fmap = np.array([fmap_up, fmap_down])
        else:
            fmap = np.append(fmap_up, fmap_down)
    else:
        true_template = template_maker.get_template(fiducial_params)
        true_fmap = Maps.flatten_map(true_template, channel=channel)
        fmap = get_random_map(true_fmap, seed=seed)
    return fmap

def get_true_template(template_params, template_maker):
    if template_params['theta23'] == 0.0:
        logging.info("Zero theta23, so generating no oscillations template...")
        true_template = template_maker.get_template_no_osc(template_params)
        true_fmap = Maps.flatten_map(true_template, channel=template_params['channel'])
    elif type(template_maker)==list and len(template_maker)==2:
        template_maker_up = template_maker[0]
        template_maker_down = template_maker[1]
        template_up = template_maker_up.get_template(template_params)  
        template_down = template_maker_down.get_template(template_params)  
        reflected_template_down = get_flipped_map(template_down, channel=template_params['channel'])
        true_fmap_up = Maps.flatten_map(template_up, channel=template_params['channel'])
        true_fmap_down = Maps.flatten_map(reflected_template_down, channel=template_params['channel'])
        if template_params['residual_up_down'] or template_params['ratio_up_down']:
            true_fmap = np.array([true_fmap_up, true_fmap_down])
        else:
            true_fmap = np.append(true_fmap_up, true_fmap_down)
    else:
        true_template = template_maker.get_template(template_params)  
        true_fmap = Maps.flatten_map(true_template, channel=template_params['channel'])

    return true_fmap

def get_pseudo_tau_fmap(template_maker, fiducial_params, channel=None, seed=None):
    '''
    Creates a true template from fiducial_params, then uses Poisson statistics
    to vary the expected counts per bin to create a pseudo data set.
    If seed is provided, the random state is seeded with seed before the map is
    created.

    IMPORTANT: returns a SINGLE flattened map of trck/cscd combined
    \params:
      * channel = channel of flattened fmap to use.
        if 'all': returns a single flattened map of trck/cscd combined.
        if 'cscd' or 'trck' only returns the channel requested.
    ''' 
    if fiducial_params['residual_up_down'] or fiducial_params['simp_up_down'] or fiducial_params['ratio_up_down']:
        template_maker_up = template_maker[0]
        template_maker_down = template_maker[1]
        template_up = template_maker_up.get_tau_template(fiducial_params)  
        template_down = template_maker_down.get_tau_template(fiducial_params)  
        reflected_template_down = get_flipped_map(template_down, channel=fiducial_params['channel'])
        true_fmap_up = Maps.flatten_map(template_up, channel=fiducial_params['channel'])
        true_fmap_down = Maps.flatten_map(reflected_template_down, channel=fiducial_params['channel'])
        fmap_up = get_random_map(true_fmap_up, seed=Maps.get_seed())
        fmap_down = get_random_map(true_fmap_down, seed=Maps.get_seed())
        if fiducial_params['residual_up_down']:
            fmap = fmap_up-fmap_down
        elif fiducial_params['ratio_up_down']:
            fmap = np.array([fmap_up, fmap_down])
        else:
            fmap = np.append(fmap_up, fmap_down)
    else:
        true_template = template_maker.get_tau_template(fiducial_params)  
        true_fmap = Maps.flatten_map(true_template, channel=channel)
        fmap = get_random_map(true_fmap, seed=seed)
    return fmap

def get_up_map(map, channel):
    ''' Gets the upgoing map from a full sky map and flip it.'''
    if channel =='all':
        flavs=['trck', 'cscd']
    elif channel =='trck':
        flavs=['trck']
    elif channel =='cscd':
        flavs=['cscd']
    elif channel == 'no_pid':
        return {'no_pid':{
            'map': map['trck']['map'][:,0:czbin_mid_idx]+map['cscd']['map'][:,0:czbin_mid_idx],
            'ebins':map['trck']['ebins'],
            'czbins': map['trck']['czbins'][0:czbin_mid_idx+1] }}
    else:
        raise ValueError("channel: '%s' not implemented! Allowed: ['all', 'trck', 'cscd', 'no_pid']"%channel)
    czbin_edges = len(map['cscd']['czbins'])
    czbin_mid_idx = (czbin_edges-1)/2
    return {flav:{
        'map': map[flav]['map'][:,0:czbin_mid_idx],
        'ebins':map[flav]['ebins'],
        'czbins': map[flav]['czbins'][0:czbin_mid_idx+1] }
            for flav in flavs}

def get_flipped_down_map(map, channel):
    ''' Gets the downgoing map from a full sky map and flip it.'''
    czbin_edges = len(map['cscd']['czbins'])
    czbin_mid_idx = (czbin_edges-1)/2
    if  channel=='all':
        flavs=['trck', 'cscd']
    elif  channel=='trck':
        flavs=['trck']
    elif  channel=='cscd':
        flavs=['cscd']
    elif  channel == 'no_pid':
        return {'no_pid':{
            'map': np.fliplr(map['trck']['map'][:,czbin_mid_idx:]+map['cscd']['map'][:,czbin_mid_idx:]),
            'ebins':map['trck']['ebins'],
            'czbins': np.sort(-map['trck']['czbins'][czbin_mid_idx:]) }}
    else:
        raise ValueError(" channel: '%s' not implemented! Allowed: ['all', 'trck', 'cscd', 'no_pid']"%channel)
    return {flav:{
        'map': np.fliplr(map[flav]['map'][:,czbin_mid_idx:]),
        'ebins':map[flav]['ebins'],
        'czbins': np.sort(-map['trck']['czbins'][czbin_mid_idx:]) }
            for flav in flavs}

def get_flipped_map(map, channel):
    ''' Flip a map.'''
    if not np.alltrue(map['cscd']['czbins']>=0):
        raise ValueError("This map has to be down-going neutrinos!")
    if channel=='all':
        flavs=['trck', 'cscd']
    elif channel=='trck':
        flavs=['trck']
    elif channel=='cscd':
        flavs=['cscd']
    elif channel == 'no_pid':
        return {'no_pid':{
            'map': np.fliplr(map['trck']['map']+map['cscd']['map']),
            'ebins':map['trck']['ebins'],
            'czbins': np.sort(-map['cscd']['czbins']) }}
    else:
        raise ValueError("channel: '%s' not implemented! Allowed: ['all', 'trck', 'cscd', 'no_pid']"%channel)
    return {flav:{
        'map': np.fliplr(map[flav]['map']),
        'ebins':map[flav]['ebins'],
        'czbins': np.sort(-map[flav]['czbins']) }
            for flav in flavs}
