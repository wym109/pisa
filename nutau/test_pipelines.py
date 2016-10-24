from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
from uncertainties import unumpy as unp

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
parser.add_argument('-o', '--outdir', type=str,
		    default='plots',
		    help='outdir')
parser.add_argument('-v', action='count', default=None,
                    help='set verbosity level')
parser.add_argument('--livetime', type=float, default=None)
args = parser.parse_args()
set_verbosity(args.v)



#my_plotter = Plotter(stamp='', outdir=args.outdir, fmt='pdf', log=False, annotate=True, symmetric=False, ratio=True)

template_maker = DistributionMaker(args.template_settings)
template_nominal0 = template_maker.get_total_outputs()
#for n in range(10):
#    template_nominal1 = template_maker.get_total_outputs()
template_nominal2 = template_maker.get_total_outputs()

delta = template_nominal2 -  template_nominal0
#print 'map0'
#print unp.nominal_values(template_nominal0[0].hist)
#print 'map1'
#print unp.nominal_values(template_nominal2[0].hist)
#print sum([np.sum(map.hist) for map in delta])
print [sum(unp.nominal_values(map.hist)) for map in delta]
print sum([np.sum(unp.nominal_values(map.hist)) for map in delta])

#for map in template_nominal:
#    print '%s:\t%.2f'%(map.name, np.sum(unp.nominal_values(map.hist)))
#my_plotter.plot_2d_array(template_nominal, split_axis='pid', fname='nominal', cmap='summer')
