#! /usr/bin/env python
#
# Gradients.py
#
# Tools for calculating the gradients.
#
# author: Lukas Schulte - schulte@physik.uni-bonn.de
#         Sebastian Boeser - sboeser@uni-mainz.de
#	  Thomas Ehrhardt - tehrhard@uni-mainz.de

import numpy as np
import os
import tempfile
import copy

from pisa.utils.jsons import to_json
from pisa.utils.log import logging, profile
from pisa.utils.params import get_values
from pisa.utils.utils import Timer


def derivative_from_polycoefficients(coeff, loc):
    """
    Return derivative of a polynomial of the form

        f(x) = coeff[0] + coeff[1]*x + coeff[2]*x**2 + ...

    at x = loc
    """

    result = 0.

    for n in range(len(coeff))[1:]: # runs from 1 to len(coeff)

        result += n*coeff[n]*loc**(n-1)

    return result


def get_derivative_map(data, fiducial=None , degree=2):
  """
  Get the approximate derivative of data w.r.t parameter par
  at location loc with polynomic degree of approximation, default: 2.

  Data is a dictionary of the form
  {
  'test_point1': {'params': {},
		  'trck': {'map': [[],[],...],
			    'ebins': [],
			    'czbins': []
			  },
		  'cscd': {'map': [[],[],...],
			    'ebins': [],
			    'czbins': []
			  }
		  }

  'test_point2': ...
  }
  """
  derivative_map = {'trck':{},'cscd':{}}
  test_points = sorted(data.keys())

  # TODO: linearity check?
  for channel in ['trck','cscd']:
    # Flatten data map for use with polyfit
    channel_data = [ np.array(data[pvalue][channel]['map']).flatten() for pvalue in test_points ]
    # Polynomial fit of bin counts
    channel_fit_params = np.polyfit(test_points, channel_data, deg=degree)
    # Get partial derivatives at fiducial values
    derivative_map[channel]['map'] = derivative_from_polycoefficients(channel_fit_params[::-1], fiducial['value'])

  return derivative_map



def get_steps(param, grid_settings, fiducial_params):
  """
  Prepare the linear sequence of test points: use a globally valid
  number of test points if grid_settings makes no specifications
  for the parameter.
  """
  try:
    n_points = grid_settings['npoints'][param]
  except:
    n_points = grid_settings['npoints']['default']

  return np.linspace(fiducial_params[param]['range'][0],fiducial_params[param]['range'][1],n_points)



def get_hierarchy_gradients(data_tag, fiducial_maps, fiducial_params,
                            grid_settings, store_dir):
  """
  Use the hierarchy interpolation between the two fiducial maps to obtain the
  gradients.
  """
  logging.info("Working on parameter hierarchy.")

  steps = get_steps('hierarchy', grid_settings, fiducial_params)

  hmap = {step:{'trck':{},'cscd':{}} for step in steps}

  for h in steps:
    for channel in ['trck','cscd']:
   	# Superpose bin counts
    	hmap[h][channel]['map'] = fiducial_maps['NMH'][channel]['map']*h + fiducial_maps['IMH'][channel]['map']*(1.-h)
	# Obtain binning from one of the maps, since identical by construction (cf. FisherAnalysis)
	hmap[h][channel]['ebins'] = fiducial_maps['NMH'][channel]['ebins']
	hmap[h][channel]['czbins'] = fiducial_maps['NMH'][channel]['czbins']

  # TODO: give hmap the same structure as pmaps?
  # Get_derivative_map works even if 'params' and 'ebins','czbins' not in 'data'

  # Store the maps used to calculate partial derivatives
  if store_dir != tempfile.gettempdir():
  	logging.info("Writing maps for parameter 'hierarchy' to %s"%store_dir)
  to_json(hmap,os.path.join(store_dir,"hierarchy_"+data_tag+".json"))

  gradient_map = get_derivative_map(hmap, fiducial_params['hierarchy'],degree=2)

  return gradient_map



def get_gradients(data_tag, param, template_maker, fiducial_params,
                  grid_settings, store_dir):
  """
  Use the template maker to create all the templates needed to obtain the gradients.
  """
  logging.info("Working on parameter %s."%param)

  steps = get_steps(param, grid_settings, fiducial_params)

  pmaps = {}

  # Generate one template for each value of the parameter in question and store in pmaps
  for param_value in steps:

      # Make the template corresponding to the current value of the parameter
      with Timer() as t:
          maps = template_maker.get_template(
              get_values(dict(fiducial_params,**{param:dict(fiducial_params[param],
                                                            **{'value': param_value})})))
      profile.info("==> elapsed time for template: %s sec"%t.secs)

      pmaps[param_value] = maps

  # Store the maps used to calculate partial derivatives
  if store_dir != tempfile.gettempdir():
  	logging.info("Writing maps for parameter %s to %s"%(param,store_dir))

  to_json(pmaps, os.path.join(store_dir,param+"_"+data_tag+".json"))

  gradient_map = get_derivative_map(pmaps,fiducial_params[param],degree=2)

  return gradient_map

