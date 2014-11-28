#! /usr/bin/env python
#
# Gradients.py
#
# Tools for calculating the gradients.
#
# author: Lukas Schulte - schulte@physik.uni-bonn.de
#         Sebastian Boeser - sboeser@uni-mainz.de
#

from pisa.utils.log import logging

def get_gradients(param,template_maker,fiducial_params,grid_settings):

  logging.info("Working on parameter %s."%param)

  steps = get_steps(param, grid_settings)

  for param_value in steps:
      maps = template_maker.get_template(dict(fiducial_params,
                                              **{param: param_value}))
  return

 # if param=='hierarchy':
 #    get the special hiearchy interpolation to get the maps
 # else:
 #    use templateMaker to get the maps
     
