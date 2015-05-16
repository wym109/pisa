#! /usr/bin/env python
#
# AsimovOptimizerAnalysis.py
#
# Runs the Asimov optimizer-based analysis. Based on
# LLROptimizerAnalysis, but the primary difference is that it only
# uses the one fiducial model template of the "pseudo data set" and
# fits to the templates finding the best fit template by maximizing
# the LLH / or minimizing the chisquare using the optimizer.
#
# author: Tim Arlen - tca3@psu.edu
#         Sebatian Boeser - sboeser@uni-mainz.de
#
# date:   02-July-2014
#

import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from pisa.utils.log import logging, profile, physics, set_verbosity
from pisa.utils.jsons import from_json,to_json
from pisa.analysis.llr.LLHAnalysis import find_max_llh_bfgs, find_min_chisquare_bfgs
from pisa.analysis.stats.Maps import get_asimov_fmap
from pisa.analysis.TemplateMaker import TemplateMaker
from pisa.utils.params import get_values, select_hierarchy

parser = ArgumentParser(description='''Runs the Asimov optimizer-based analysis varying a number of systematic parameters
defined in settings.json file and saves the likelihood (or chisquare) values for all
combinations of hierarchies.''',
                        formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-t','--template_settings',type=str,
                    metavar='JSONFILE', required = True,
                    help='''Settings related to the template generation and systematics.''')
parser.add_argument('-m','--minimizer_settings',type=str,
                    metavar='JSONFILE', required = True,
                    help='''Settings related to the optimizer used in the LLR analysis.''')
parser.add_argument('-c','--chisquare',action='store_true', default=False,
                    dest='use_chisquare', help='''Use chisquare metric instead of log-likelihood.''')
parser.add_argument('-pd','--pseudo_data_settings',type=str,
                    metavar='JSONFILE',default=None,
                    help='''Settings for pseudo data templates, if desired to be different from template_settings.''')
parser.add_argument('-s','--save-steps',action='store_true',default=False,
                    dest='save_steps',
                    help="Save all steps the optimizer takes.")
parser.add_argument('-o','--outfile',type=str,default='llh_data.json',metavar='JSONFILE',
                    help="Output filename.")
parser.add_argument('-v', '--verbose', action='count', default=None,
                    help='set verbosity level')
args = parser.parse_args()

set_verbosity(args.verbose)

#Read in the settings
template_settings = from_json(args.template_settings)
minimizer_settings  = from_json(args.minimizer_settings)
pseudo_data_settings = from_json(args.pseudo_data_settings) if args.pseudo_data_settings is not None else template_settings

#Workaround for old scipy versions
import scipy
if scipy.__version__ < '0.12.0':
    logging.warn('Detected scipy version %s < 0.12.0'%scipy.__version__)
    if 'maxiter' in minimizer_settings:
      logging.warn('Optimizer settings for \"maxiter\" will be ignored')
      minimizer_settings.pop('maxiter')


# make sure that both pseudo data and template are using the same
# channel. Raise Exception and quit otherwise
channel = template_settings['params']['channel']['value']
if channel != pseudo_data_settings['params']['channel']['value']:
    error_msg = "Both template and pseudo data must have same channel!\n"
    error_msg += " pseudo_data_settings chan: '%s', template chan: '%s' "%(pseudo_data_settings['params']['channel']['value'],channel)
    raise ValueError(error_msg)


template_maker = TemplateMaker(get_values(template_settings['params']),
                               **template_settings['binning'])
if args.pseudo_data_settings:
    pseudo_data_template_maker = TemplateMaker(get_values(pseudo_data_settings['params']),
                                               **pseudo_data_settings['binning'])
else:
    pseudo_data_template_maker = template_maker


# //////////////////////////////////////////////////////////////////////
# Generate two pseudo-data experiments (one for each hierarchy),
# and for each experiment, find the best matching template in each
# of the hierarchy hypotheses.
# //////////////////////////////////////////////////////////////////////
results = {}
for data_tag, data_normal in [('data_NMH',True),('data_IMH',False)]:

    results[data_tag] = {}
    # 1) get "Asimov" average fidicual template:
    asimov_fmap = get_asimov_fmap(pseudo_data_template_maker,
                                  get_values(select_hierarchy(pseudo_data_settings['params'],
                                                              normal_hierarchy=data_normal)),
                                  chan=channel)

    # 2) find max llh (and best fit free params) from matching pseudo data
    #    to templates.
    for hypo_tag, hypo_normal in [('hypo_NMH',True),('hypo_IMH',False)]:
        physics.info("Finding best fit for %s under %s assumption"%(data_tag,hypo_tag))
        profile.info("start optimizer")
	if not args.use_chisquare:
	    profile.info("Using llh")
	    opt_data = find_max_llh_bfgs(asimov_fmap,template_maker,template_settings['params'],
                                         minimizer_settings,args.save_steps,
                                         normal_hierarchy=hypo_normal)
        else:
	    profile.info("Using chisquare")
	    if data_normal==hypo_normal:
		# skip the case where hypothesis corresponds to truth, since chi2 zero by construction
		profile.info("Skipping %s"%hypo_tag)
		continue
	    opt_data = find_min_chisquare_bfgs(asimov_fmap,template_maker,template_settings['params'],
                                               minimizer_settings,args.save_steps,
                                               normal_hierarchy=hypo_normal)
	profile.info("stop optimizer")

        #Store the LLH data
        results[data_tag][hypo_tag] = opt_data

#Assemble output dict
output = {'results' : results,
          'template_settings' : template_settings,
          'minimizer_settings' : minimizer_settings}

if args.pseudo_data_settings is not None:
    output['pseudo_data_settings'] = pseudo_data_settings

if not args.use_chisquare:
    llr_nmh = -(np.min(results['data_NMH']['hypo_IMH']['llh']) - np.min(results['data_NMH']['hypo_NMH']['llh']))
    llr_imh = -(np.min(results['data_IMH']['hypo_IMH']['llh']) - np.min(results['data_IMH']['hypo_NMH']['llh']))
    logging.info('(hypo NMH is numerator): llr_nmh: %.4f, llr_imh: %.4f'%(llr_nmh,llr_imh))
else:
    chi2_nmh = np.min(results['data_NMH']['hypo_IMH']['chisquare'])
    chi2_imh = np.min(results['data_IMH']['hypo_NMH']['chisquare'])
    logging.info('chi2_nmh: %.4f, chi2_imh: %.4f'%(chi2_nmh, chi2_imh))

#And write to file
to_json(output,args.outfile)
