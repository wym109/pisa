#! /usr/bin/env python
#
# Gradients.py
#
# Tools for calculating the gradients.
#
# author: Lukas Schulte - schulte@physik.uni-bonn.de
#         Sebastian Boeser - sboeser@uni-mainz.de
#

from pisa.utils.jsons import to_json
from pisa.utils.log import logging
import numpy as np
import os


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


# copied from PaPA, modify treatment of data
def getDerivativeMap(data, test_points=None, fiducial=None , degree=2):
  """
  Get the approximate derivative of data w.r.t parameter par
  at location loc with polynomic degree of approximation, default: 2
  Data is a 3D-array, i.e. a list of coszen-enery maps
  """

  # TODO: chi2 test

  # All bins are treated equally, so we just flatten out the maps to linear arrays
  fdata = [ n.array(pmap['map']).flatten() for pmap in data ]

  # Now we have one row per map, and one column per bin, so data is
  # in the right format for making the polyfit
  fitparams = n.polyfit(test_points,fdata, deg=degree)

  #Get the actual derivatives at the best fit point
  #(no need to reshape result as map)
  return derivative_from_polycoefficients(fitparams[::-1], fiducial['value'])




def get_steps(param, grid_settings, fiducial_params):
  
  try:
    n_points = grid_settings['npoints'][param]
  except:
    n_points = grid_settings['npoints']['default']

  return np.linspace(*fiducial_params[param]['range'],n_points)

  

def get_gradients(param,template_maker,fiducial_params,grid_settings,store_directory):

  logging.info("Working on parameter %s."%param)

  # store_subdir = os.path.join(store_directory,param)

  # os.mkdir(store_subdir)

  steps = get_steps(param, grid_settings, fiducial_params)
  
  pmaps = {}  

  for param_value in steps:
      maps = template_maker.get_template(dict(fiducial_params,
                                              **{param: param_value}))
      
      pmaps[param_value] = maps

  # gradient_maps = get_derivative_map(pmaps,steps,fiducial_params[param])    

  to_json(pmaps, os.path.join(store_directory,param+".json"))            

  return gradient_maps

 # if param=='hierarchy':
 #    get the special hiearchy interpolation to get the maps
 # else:
 #    use templateMaker to get the maps
     
 
