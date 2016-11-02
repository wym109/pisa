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
parser.add_argument('-o', '--outdir', type=str,
                    default='plots',
                    help='outdir')
parser.add_argument('-v', action='count', default=None,
                    help='set verbosity level')
args = parser.parse_args()
set_verbosity(args.v)

my_plotter = Plotter(stamp='nutau', outdir=args.outdir, fmt='pdf', log=False, annotate=True, symmetric=False)
template_maker = DistributionMaker(args.template_settings)

sys_dict = {'hole_ice_fwd':-1.0 * ureg.dimensionless, 'hole_ice':15 * ureg.dimensionless}


template_nominal = template_maker.get_total_outputs()

for sys, val in sys_dict.items():
    template_maker.reset_all()
    p = template_maker.params[sys]
    p.value = val
    template_maker.update_params(p)
    template_sys = template_maker.get_total_outputs()

    my_plotter.plot_2d_array((template_sys - template_nominal)/template_nominal, fname=sys, split_axis='pid',cmap='RdBu')
