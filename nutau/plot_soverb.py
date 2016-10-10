from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np

from pisa import ureg, Q_
from pisa.core.distribution_maker import DistributionMaker
from pisa.core.map import Map, MapSet
from pisa.utils.log import set_verbosity
from pisa.utils.plotter import Plotter

import pandas as pd

parser = ArgumentParser()
parser.add_argument('-t', '--template-settings',
		    metavar='configfile', required=True,
		    action='append',
		    help='''settings for the template generation''')
parser.add_argument('-v', action='count', default=None,
                    help='set verbosity level')
parser.add_argument('-o', '--outdir', type=str,
                    default='plots',
                    help='outdir')
args = parser.parse_args()
set_verbosity(args.v)

my_plotter = Plotter(stamp='nutau', outdir=args.outdir, fmt='pdf', log=False, annotate=False, symmetric=False)
template_maker = DistributionMaker(args.template_settings)

template_nominal = template_maker.get_total_outputs()
p = template_maker.params['nutau_norm']
p.value = 0. * ureg.dimensionless
template_maker.update_params(p)
template_bkgd = template_maker.get_total_outputs()

template_nutau = template_nominal - template_bkgd

#for n in template_nominal: n.tex = n.name
#for n in template_nutau: n.tex = n.name
#for n in template_bkgd: n.tex = n.name

my_plotter.label = r'$S/\sqrt{B}$'
my_plotter.plot_2d_array(template_nutau/template_bkgd.sqrt(), fname='soverb', split_axis='pid',cmap='YlOrRd')
