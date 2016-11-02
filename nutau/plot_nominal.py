from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
from uncertainties import unumpy as unp

from pisa import ureg, Q_
from pisa.core.distribution_maker import DistributionMaker
from pisa.utils.log import set_verbosity
from pisa.utils.plotter import Plotter
from pisa.utils.fileio import from_file, to_file

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
parser.add_argument('--livetime', type=float, default=None)
args = parser.parse_args()
set_verbosity(args.v)



my_plotter = Plotter(stamp='', outdir=args.outdir, fmt='pdf', log=False, annotate=False, symmetric=False, ratio=True)

template_maker = DistributionMaker(args.template_settings)
template_nominal = template_maker.get_outputs(sum=True)
template_nominal = template_maker.get_outputs(sum=True)
to_file(template_nominal, args.outdir+'/maps.json')
for map in template_nominal:
    print '%s:\t%.2f'%(map.name, np.sum(unp.nominal_values(map.hist)))
my_plotter.plot_2d_array(template_nominal, split_axis='pid', fname='nominal', cmap='YlOrRd')
#my_plotter.plot_2d_array(template_nominal, fname='nominal', cmap='YlOrRd')
