#! /usr/bin/env python
#
# FisherAnalysis.py
#
# Runs the Fisher Analysis method
#
# author: Lukas Schulte - schulte@physik.uni-bonn.de
#         Sebastian Boeser - sboeser@uni-mainz.de
#	  Thomas Ehrhardt - tehrhard@uni-mainz.de

import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import tempfile
import os

from pisa.utils.log import logging, profile, physics, set_verbosity
from pisa.utils.jsons import from_json,to_json
from pisa.analysis.TemplateMaker import TemplateMaker
from pisa.utils.params import select_hierarchy, get_free_params, get_values
from pisa.utils.utils import Timer

from pisa.analysis.fisher.gradients import get_gradients, get_hierarchy_gradients
from pisa.analysis.fisher.BuildFisherMatrix import build_fisher_matrix
from pisa.analysis.fisher.Fisher import FisherMatrix


def get_fisher_matrices(template_settings, grid_settings, IMH=True, NMH=False, dump_all_stages=False,
		        save_templates=False, outdir=None):
  '''
  Main function that runs the Fisher analysis for the chosen hierarchy(ies) (inverted by default).

  Returns a dictionary of Fisher matrices, in the format:
  {'IMH': {'cscd': [...],
          'trck': [...],
          'comb': [...],
          },
  'NMH': {'cscd': [...],
          'trck': [...],
          'comb': [...],
         }
  }

  If save_templates=True and no hierarchy is given, only fiducial templates will be written out;
  if one is given, then the templates used to obtain the gradients will be written out in
  addition.
  '''
  if outdir is None and (save_templates or dump_all_stages):
    logging.info("No output directory specified. Will save templates to current working directory.")
    outdir = os.getcwd()

  profile.info("start initializing")

  # Get the parameters
  params = template_settings['params']
  bins = template_settings['binning']

  # Artifically add the hierarchy parameter to the list of parameters
  # The method get_hierarchy_gradients below will know how to deal with it
  params['hierarchy_nh'] = { "value": 1., "range": [0.,1.],
                           "fixed": False, "prior": None}
  params['hierarchy_ih'] = { "value": 0., "range": [0.,1.],
                           "fixed": False, "prior": None}

  chosen_data = []
  if IMH:
    chosen_data.append(('IMH',False))
    logging.info("Fisher matrix will be built for IMH.")
  if NMH:
    chosen_data.append(('NMH',True))
    logging.info("Fisher matrix will be built for NMH.")
  if chosen_data == []:
    # In this case, only the fiducial maps (for both hierarchies) will be written
    logging.info("No Fisher matrices will be built.")

  # There is no sense in performing any of the following steps if no Fisher matrices are to be built
  # and no templates are to be saved.
  if chosen_data!=[] or dump_all_stages or save_templates:

    # Initialise return dict to hold Fisher matrices
    fisher = { data_tag:{'cscd':[],'trck':[],'comb':[]} for data_tag, data_normal in chosen_data }

    # Get a template maker with the settings used to initialize
    template_maker = TemplateMaker(get_values(params),**bins)

    profile.info("stop initializing\n")

    # Generate fiducial templates for both hierarchies (needed for partial derivatives
    # w.r.t. hierarchy parameter)
    fiducial_maps = {}
    for hierarchy in ['NMH','IMH']:

      logging.info("Generating fiducial templates for %s."%hierarchy)

      # Get the fiducial parameter values corresponding to this hierarchy
      fiducial_params = select_hierarchy(params,normal_hierarchy=(hierarchy=='NMH'))

      # Generate fiducial maps, either all of them or only the ultimate one
      profile.info("start template calculation")
      with Timer() as t:
        fid_maps = template_maker.get_template(get_values(fiducial_params),
                                               return_stages=dump_all_stages)
      profile.info("==> elapsed time for template: %s sec"%t.secs)

      fiducial_maps[hierarchy] = fid_maps[4] if dump_all_stages else fid_maps

      # save fiducial map(s)
      # all stages
      if dump_all_stages:
        stage_names = ("0_unoscillated_flux","1_oscillated_flux","2_oscillated_counts","3_reco","4_pid")
        stage_maps = {}
        for stage in xrange(0,len(fid_maps)):
          stage_maps[stage_names[stage]] = fid_maps[stage]
        logging.info("Writing fiducial maps (all stages) for %s to %s."%(hierarchy,outdir))
        to_json(stage_maps,os.path.join(outdir,"fid_map_"+hierarchy+".json"))
      # only the final stage
      elif save_templates:
        logging.info("Writing fiducial map (final stage) for %s to %s."%(hierarchy,outdir))
        to_json(fiducial_maps[hierarchy],os.path.join(outdir,"fid_map_"+hierarchy+".json"))

    # Get_gradients and get_hierarchy_gradients will both (temporarily)
    # store the templates used to generate the gradient maps
    store_dir = outdir if save_templates else tempfile.gettempdir()

    # Calculate Fisher matrices for the user-defined cases (NHM true and/or IMH true)
    for data_tag, data_normal in chosen_data:

      logging.info("Running Fisher analysis for %s."%(data_tag))

      # The fiducial params are selected from the hierarchy case that does NOT match
      # the data, as we are varying from this model to find the 'best fit'
      fiducial_params = select_hierarchy(params,not data_normal)

      # Get the free parameters (i.e. those for which the gradients should be calculated)
      free_params = select_hierarchy(get_free_params(params),not data_normal)
      gradient_maps = {}
      for param in free_params.keys():
        # Special treatment for the hierarchy parameter
        if param=='hierarchy':
          gradient_maps[param] = get_hierarchy_gradients(data_tag,
						     fiducial_maps,
						     fiducial_params,
						     grid_settings,
						     store_dir,
						     )
        else:
          gradient_maps[param] = get_gradients(data_tag,
					   param,
                                           template_maker,
                                           fiducial_params,
                                           grid_settings,
                                           store_dir
                                           )

      logging.info("Building Fisher matrix for %s."%(data_tag))

      # Build Fisher matrices for the given hierarchy
      fisher[data_tag] = build_fisher_matrix(gradient_maps,fiducial_maps['IMH'] if data_normal else fiducial_maps['NMH'],fiducial_params)

      # If Fisher matrices exist for both channels, add the matrices to obtain the combined one.
      if len(fisher[data_tag].keys()) > 1:
        fisher[data_tag]['comb'] = FisherMatrix(matrix=np.array([f.matrix for f in fisher[data_tag].itervalues()]).sum(axis=0),
                                              parameters=gradient_maps.keys(),  #order is important here!
                                              best_fits=[fiducial_params[par]['value'] for par in gradient_maps.keys()],
                                              priors=[fiducial_params[par]['prior'] for par in gradient_maps.keys()],
                                              )
    return fisher

  else:
    logging.info("Nothing to be done.")
    return {}


if __name__ == '__main__':
  parser = ArgumentParser(description='''Runs the Fisher analysis method by varying a number of systematic parameters
			defined in a settings.json file, taking the number of test points from a grid_settings.json file, and saves the
			Fisher matrices for the desired hierarchy.''',
                        formatter_class=ArgumentDefaultsHelpFormatter)

  parser.add_argument('-t','--template_settings',type=str,
                    metavar='JSONFILE', required = True,
                    help='Settings related to template generation and systematics.')

  parser.add_argument('-g','--grid_settings',type=str,
                    metavar='JSONFILE', required = True,
                    help='''Settings for the grid on which the gradients are
                    calculated (number of test points for each parameter).''')

  parser.add_argument('-nh','--normal-truth',action='store_true',
		    default=False, dest='normal_truth',
		    help='Compute Fisher matrix for true normal hierarchy.')

  parser.add_argument('-ih','--inverted-truth',action='store_true',
		    default=False, dest='inverted_truth',
		    help='Compute Fisher matrix for true inverted hierarchy.')

  parser.add_argument('-d','--dump-all-stages', action='store_true',dest='dump_all_stages', default=False,
                    help='''Store histograms at all simulation stages for fiducial model in 
                    normal and inverted hierarchy.''')

  sselect = parser.add_mutually_exclusive_group(required=False)

  sselect.add_argument('-s','--save-templates',action='store_true',
                    default=True, dest='save_templates',
                    help="Save all the templates used to obtain the gradients.")

  sselect.add_argument('-n','--no-save-templates', action='store_false',
                    default=False, dest='save_templates',
                    help="Do not save the templates for the different test points.")

  parser.add_argument('-o','--outdir',type=str,default=os.getcwd(),metavar='DIR',
                    help="Output directory")

  parser.add_argument('-v', '--verbose', action='count', default=None,
                    help='set verbosity level')

  args = parser.parse_args()

  # Set verbosity level
  set_verbosity(args.verbose)

  # Read the template settings
  template_settings = from_json(args.template_settings)

  # This file only contains the number of test points for each parameter (and perhaps eventually a non-linearity criterion)
  grid_settings  = from_json(args.grid_settings)

  # Get the Fisher matrices for the desired hierarchy and fiducial settings
  fisher_matrices = get_fisher_matrices(template_settings,grid_settings,args.inverted_truth,args.normal_truth,
                                    args.dump_all_stages,args.save_templates,args.outdir)

  
  # Fisher matrices are saved in any case
  for data_tag in fisher_matrices:
    fisher_basename = 'fisher_data_%s'%data_tag
    for chan in fisher_matrices[data_tag]:
      if chan == 'comb':
        outfile = os.path.join(args.outdir,fisher_basename+'.json')
        logging.info("%s: writing combined Fisher matrix to %s"%(data_tag,outfile))
      else:
        outfile = os.path.join(args.outdir,fisher_basename+'_%s.json'%chan)
        logging.info("%s: writing Fisher matrix for channel %s to %s"%(data_tag,chan,outfile))
      fisher_matrices[data_tag][chan].saveFile(outfile)
