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

from pisa.utils.log import logging, tprofile, physics, set_verbosity
from pisa.utils.jsons import from_json,to_json
from pisa.analysis.llr.LLHAnalysis import find_opt_bfgs
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
parser.add_argument('--check-octant', action='store_true', default=False,
		    dest='check_octant', help='''After first optimisation, check whether seeding with theta23
		    mirrored into the alternative octant leads to a better solution (requires 2x the amount of time).''')
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

# Read in the settings
template_settings = from_json(args.template_settings)
minimizer_settings  = from_json(args.minimizer_settings)
pseudo_data_settings = from_json(args.pseudo_data_settings) if args.pseudo_data_settings is not None else template_settings

# Parse the metric to be used
metric_name = 'chisquare' if args.use_chisquare else 'llh'

# Workaround for old scipy versions
import scipy
if scipy.__version__ < '0.12.0':
    logging.warn('Detected scipy version %s < 0.12.0'%scipy.__version__)
    if 'maxiter' in minimizer_settings:
      logging.warn('Optimizer settings for \"maxiter\" will be ignored')
      minimizer_settings.pop('maxiter')


# Make sure that both pseudo data and template are using the same
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
    # 1) get "Asimov" average fiducial template:
    asimov_fmap = get_asimov_fmap(pseudo_data_template_maker,
                                  get_values(select_hierarchy(pseudo_data_settings['params'],
                                                              normal_hierarchy=data_normal)),
                                  channel=channel)

    # 2) find max llh or min chisquare (and best fit free params) from matching pseudo data
    #    to templates.
    for hypo_tag, hypo_normal in [('hypo_NMH',True),('hypo_IMH',False)]:
        physics.info("Finding best fit for %s under %s assumption"%(data_tag,hypo_tag))
	tprofile.info("start optimizer")
	tprofile.info("Using %s"%metric_name)

	opt_data = find_opt_bfgs(asimov_fmap,template_maker,template_settings['params'],
				 minimizer_settings,args.save_steps,
				 normal_hierarchy=hypo_normal, check_octant=args.check_octant,
				 metric_name=metric_name)

	tprofile.info("stop optimizer")

	# Store the optimum data
	results[data_tag][hypo_tag] = opt_data

# Assemble output dict
output = {'results' : results,
          'template_settings' : template_settings,
          'minimizer_settings' : minimizer_settings}

if args.pseudo_data_settings is not None:
    output['pseudo_data_settings'] = pseudo_data_settings

# And write to file
to_json(output,args.outfile)

# Finally, report on what we have
if not args.use_chisquare:
    llr_nmh = -(np.min(results['data_NMH']['hypo_IMH'][metric_name]) - np.min(results['data_NMH']['hypo_NMH'][metric_name]))
    llr_imh = -(np.min(results['data_IMH']['hypo_IMH'][metric_name]) - np.min(results['data_IMH']['hypo_NMH'][metric_name]))
    logging.info('(hypo NMH is numerator): llr_nmh: %.4f, llr_imh: %.4f'%(llr_nmh,llr_imh))
else:
    chi2_nmh = np.min(results['data_NMH']['hypo_IMH'][metric_name])
    chi2_imh = np.min(results['data_IMH']['hypo_NMH'][metric_name])
    logging.info('chi2_nmh: %.4f, chi2_imh: %.4f'%(chi2_nmh, chi2_imh))
