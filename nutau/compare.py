from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np

from pisa import ureg, Q_
from pisa.core.distribution_maker import DistributionMaker
from pisa.utils.log import set_verbosity
from pisa.utils.plotter import Plotter
from pisa.utils.fileio import from_file

parser = ArgumentParser()
parser.add_argument('-t', '--template-settings',
		    metavar='configfile', required=True,
		    action='append',
		    help='''settings for the template generation''')
parser.add_argument('-t1', '--template-settings1',
		    metavar='configfile', required=True,
		    action='append',
		    help='''settings for the template generation''')
parser.add_argument('-o', '--outdir', type=str,
		    default='plots',
		    help='outdir')
parser.add_argument('-v', action='count', default=None,
                    help='set verbosity level')
parser.add_argument('--livetime', type=float, default=None)
args = parser.parse_args()
set_verbosity(args.v)



my_plotter = Plotter(stamp='test', outdir=args.outdir, fmt='pdf', log=False, annotate=True, symmetric=False, ratio=True)

template_maker = DistributionMaker(args.template_settings)
template_maker1 = DistributionMaker(args.template_settings1)

template_nominal = template_maker.get_total_outputs()
for map in template_nominal: map.tex = map.name
new_maps = []

combined = template_nominal.combine_wildcard(['*nc', 'nue*cc', 'numu*cc', 'nutau*cc'])
combined[0].name = 'nuall_nc+nuallbar_nc'
combined[1].name = 'nue_cc+nuebar_cc'
combined[2].name = 'numu_cc+numubar_cc'
combined[3].name = 'nutau_cc+nutaubar_cc'

template_nominal = combined

template_nominal1 = template_maker1.get_total_outputs()
for map in template_nominal1: map.tex = map.name

#my_plotter.plot_2d_array(template_nominal/template_nominal1, split_axis='pid', fname='ratio', cmap='summer')
my_plotter.symmetric = True
my_plotter.label = 'hist - MC'
my_plotter.plot_2d_array(template_nominal-template_nominal1, split_axis='pid', fname='abs_diff', cmap='BrBG')
my_plotter.label = '(hist - MC)/MC'
my_plotter.plot_2d_array((template_nominal-template_nominal1)/template_nominal1,split_axis='pid', fname='rel_diff', cmap='BrBG')
