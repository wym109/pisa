from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np

from pisa import ureg, Q_
from pisa.core.distribution_maker import DistributionMaker
from pisa.utils.log import set_verbosity
from pisa.utils.plotter import plotter

parser = ArgumentParser()
parser.add_argument('-t', '--template-settings',
		    metavar='configfile', required=True,
		    action='append',
		    help='''settings for the template generation''')
parser.add_argument('-v', action='count', default=None,
                    help='set verbosity level')
args = parser.parse_args()
set_verbosity(args.v)

my_plotter = plotter(stamp='nutau sys test', outdir='.', fmt='pdf', log=False, annotate=True, symmetric=True)
template_maker = DistributionMaker(args.template_settings)

template_nominal = template_maker.get_outputs()
my_plotter.plot_2d_array(template_nominal, fname='nominal',cmap='RdBu')

variation = {'deltam31': 0.2e-3*ureg.eV**2,
            'theta23': 0.1 * ureg.rad,
            'theta13': 0.008 * ureg.rad,
            'aeff_scale': 0.12,
            'nutau_cc_norm': 0.5,
            'nu_nc_norm': 0.2,
            'nue_numu_ratio': 0.05,
            'nu_nubar_ratio': 0.1,
            'delta_index': 0.1,
            'dom_eff': 0.1,
            'hole_ice': 10.,
            'hole_ice_fwd': -1.,
            'atm_muon_scale': 0.1
            }

for sys, var in variation.items():
    template_maker.params.reset_free()
    p = template_maker.params[sys]
    p.value += var
    template_maker.update_params(p)
    template_sys = template_maker.get_outputs()
    my_plotter.plot_2d_array(template_sys - template_nominal, fname='%s_variation'%sys,cmap='RdBu')
