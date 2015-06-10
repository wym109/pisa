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

def get_asimov_data_fmap_up_down(template_maker, fiducial_params, channel=None):
    if fiducial_params['residual_up_down'] or fiducial_params['simp_up_down'] or fiducial_params['ratio_up_down']:
        template_maker_up = template_maker[0]
        template_maker_down = template_maker[1]
        template_up = template_maker_up.get_template(fiducial_params)  
        template_down = template_maker_down.get_template(fiducial_params)  
        reflected_template_down = get_flipped_map(template_down, channel=fiducial_params['channel'])
        true_fmap_up = flatten_map(template_up, channel=fiducial_params['channel'])
        true_fmap_down = flatten_map(reflected_template_down, channel=fiducial_params['channel'])
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
        true_fmap = flatten_map(true_template, channel=channel)
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
        reflected_template_down = get_flipped_map(template_down,channel=fiducial_params['channel'])
        true_fmap_up = flatten_map(template_up,channel=fiducial_params['channel'])
        true_fmap_down = flatten_map(reflected_template_down,channel=fiducial_params['channel'])
        fmap_up = get_random_map(true_fmap_up, seed=get_seed())
        fmap_down = get_random_map(true_fmap_down, seed=get_seed())
        if fiducial_params['residual_up_down']:
            fmap = fmap_up-fmap_down
        elif fiducial_params['ratio_up_down']:
            fmap = np.array([fmap_up,fmap_down])
        else:
            fmap = np.append(fmap_up,fmap_down)
    else:
        true_template = template_maker.get_template(fiducial_params)
        true_fmap = flatten_map(true_template, channel=channel)
        fmap = get_random_map(true_fmap, seed=seed)
    return fmap

def get_true_template(template_params,template_maker):
    if template_params['theta23'] == 0.0:
        logging.info("Zero theta23, so generating no oscillations template...")
        true_template = template_maker.get_template_no_osc(template_params)
        true_fmap = flatten_map(true_template,channel=template_params['channel'])
    elif type(template_maker)==list and len(template_maker)==2:
        template_maker_up = template_maker[0]
        template_maker_down = template_maker[1]
        template_up = template_maker_up.get_template(template_params)  
        template_down = template_maker_down.get_template(template_params)  
        reflected_template_down = get_flipped_map(template_down,chan=template_params['channel'])
        true_fmap_up = flatten_map(template_up,chan=template_params['channel'])
        true_fmap_down = flatten_map(reflected_template_down,chan=template_params['channel'])
        if template_params['residual_up_down'] or template_params['ratio_up_down']:
            true_fmap = np.array([true_fmap_up,true_fmap_down])
        else:
            true_fmap = np.append(true_fmap_up,true_fmap_down)
    else:
        true_template = template_maker.get_template(template_params)  
        true_fmap = flatten_map(true_template,chan=template_params['channel'])

    return true_fmap

def get_pseudo_tau_fmap(template_maker,fiducial_params,seed=None,chan=None):
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
    if fiducial_params['residual_up_down'] or fiducial_params['simp_up_down'] or fiducial_params['ratio_up_down']:
        template_maker_up = template_maker[0]
        template_maker_down = template_maker[1]
        template_up = template_maker_up.get_tau_template(fiducial_params)  
        template_down = template_maker_down.get_tau_template(fiducial_params)  
        reflected_template_down = get_flipped_map(template_down,chan=fiducial_params['channel'])
        true_fmap_up = flatten_map(template_up,chan=fiducial_params['channel'])
        true_fmap_down = flatten_map(reflected_template_down,chan=fiducial_params['channel'])
        fmap_up = get_random_map(true_fmap_up, seed=get_seed())
        fmap_down = get_random_map(true_fmap_down, seed=get_seed())
        if fiducial_params['residual_up_down']:
            fmap = fmap_up-fmap_down
        elif fiducial_params['ratio_up_down']:
            fmap = np.array([fmap_up,fmap_down])
        else:
            fmap = np.append(fmap_up,fmap_down)
    else:
        true_template = template_maker.get_tau_template(fiducial_params)  
        true_fmap = flatten_map(true_template,chan=chan)
        fmap = get_random_map(true_fmap, seed=seed)
    return fmap

def get_asimov_fmap(template_maker, fiducial_params, channel=None):
    """Creates a true template from fiducial_params"""

    true_template = template_maker.get_template(fiducial_params)
    return flatten_map(true_template,channel=channel)

def flatten_map(template, channel='all'):
    """
    Takes a final level true (expected) template of trck/cscd, and returns a
    single flattened map of trck appended to cscd, with all zero bins
    removed.
    """

    logging.trace("Getting flattened map of channel: %s"%channel)

    if channel == 'all':
        cscd = template['cscd']['map'].flatten()
        trck = template['trck']['map'].flatten()
        fmap = np.append(cscd,trck)
    elif channel == 'trck':
        trck = template[channel]['map'].flatten()
        fmap = np.array(trck)
        #fmap = np.array(fmap)[np.nonzero(fmap)]
    elif channel == 'cscd':
        cscd = template[channel]['map'].flatten()
        fmap = np.array(cscd)
        #fmap = np.array(fmap)[np.nonzero(fmap)]
    elif channel == 'no_pid':
        cscd = template['cscd']['map'].flatten()
        trck = template['trck']['map'].flatten()
        fmap = cscd + trck
        #fmap = np.array(fmap)[np.nonzero(fmap)]
    else:
        raise ValueError(
            "channel: '%s' not implemented! Allowed: ['all', 'trck', 'cscd','no_pid']"%channel)

    fmap = np.array(fmap)[np.nonzero(fmap)]
    return fmap

def get_seed():
    """
    Returns a random seed from /dev/urandom that can be used to seed the random
    state, e.g. for the poisson random variates.
    """

    return int(os.urandom(4).encode('hex'),16)

def get_up_map(map,channel):
    ''' Gets the upgoing map from a full sky map and flip it.'''
    if channel =='all':
        flavs=['trck','cscd']
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
        raise ValueError("channel: '%s' not implemented! Allowed: ['all', 'trck', 'cscd','no_pid']"%channel)
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
        flavs=['trck','cscd']
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
        raise ValueError(" channel: '%s' not implemented! Allowed: ['all', 'trck', 'cscd','no_pid']"%channel)
    return {flav:{
        'map': np.fliplr(map[flav]['map'][:,czbin_mid_idx:]),
        'ebins':map[flav]['ebins'],
        'czbins': np.sort(-map['trck']['czbins'][czbin_mid_idx:]) }
            for flav in flavs}

def get_flipped_map(map,channel):
    ''' Flip a map.'''
    if not np.alltrue(map['cscd']['czbins']>=0):
        raise ValueError("This map has to be down-going neutrinos!")
    if channel=='all':
        flavs=['trck','cscd']
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
        raise ValueError("channel: '%s' not implemented! Allowed: ['all', 'trck', 'cscd','no_pid']"%channel)
    return {flav:{
        'map': np.fliplr(map[flav]['map']),
        'ebins':map[flav]['ebins'],
        'czbins': np.sort(-map[flav]['czbins']) }
            for flav in flavs}
