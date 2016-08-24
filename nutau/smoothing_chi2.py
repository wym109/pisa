from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
from uncertainties import unumpy as unp

from pisa import ureg, Q_
from pisa.core.distribution_maker import DistributionMaker
from pisa.core.map import Map, MapSet
from pisa.utils.log import set_verbosity
from pisa.utils.plotter import plotter

import pandas as pd

parser = ArgumentParser()
parser.add_argument('-t', '--template-settings',
		    metavar='configfile', required=True,
		    action='append',
		    help='''settings for the template generation''')
parser.add_argument('-v', action='count', default=None,
                    help='set verbosity level')
args = parser.parse_args()
set_verbosity(args.v)

my_plotter = plotter(stamp='nutau sys test', outdir='chi2_plots/', fmt='pdf', log=False, annotate=True, symmetric=True)

variation = {
    'hole_ice':30*ureg.dimensionless,
    'hole_ice_fwd':1.0*ureg.dimensionless,
    'dom_eff':0.9*ureg.dimensionless,
}

for sys, var in variation.items():
    template_maker = DistributionMaker(args.template_settings)
    p = template_maker.params[sys]
    p.value = var
    template_maker.update_params(p)

    template_nominal = template_maker.get_outputs()


    p = template_maker.params[sys+'_file']
    new = p.value.replace('raw','gauss')
    p.value = new
    p.set_nominal_to_current_value()
    template_maker.update_params(p)

    template_sys = template_maker.get_outputs()
    my_plotter.label = r'$\Delta(sys-nominal)$'
    my_plotter.plot_2d_array(template_sys - template_nominal, fname='%s_variation'%sys,cmap='RdBu')


    print sys
    print 'cscd chi2 = %.4f'%np.sum((unp.nominal_values(template_nominal['cscd'].hist) - unp.nominal_values(template_sys['cscd'].hist))**2/unp.std_devs(template_nominal['cscd'].hist)**2)
    print 'cscd ndf = ',reduce(lambda x, y: x*y, template_nominal['cscd'].hist.shape)
    print 'trck chi2 = %.4f'%np.sum((unp.nominal_values(template_nominal['trck'].hist) - unp.nominal_values(template_sys['trck'].hist))**2/unp.std_devs(template_nominal['trck'].hist)**2)
    print 'trck ndf = ',reduce(lambda x, y: x*y, template_nominal['trck'].hist.shape)
