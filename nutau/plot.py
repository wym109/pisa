from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np

from pisa import ureg, Q_
from pisa.core.distribution_maker import DistributionMaker
from pisa.utils.log import set_verbosity
from pisa.utils.plotter import plotter
from pisa.utils.jsons import from_json, to_json

parser = ArgumentParser()
parser.add_argument('-t', '--template-settings',
		    metavar='configfile', required=True,
		    action='append',
		    help='''settings for the template generation''')
parser.add_argument('-d', '--data-settings', type=str,
		    metavar='configfile', default=None,
		    help='settings for the generation of "data"')
parser.add_argument('-sp', '--set-param', type=str, default='',
                    help='Set a param to a certain value.')
parser.add_argument('-v', action='count', default=None,
                    help='set verbosity level')
parser.add_argument('--livetime', type=float, default=None)
args = parser.parse_args()
set_verbosity(args.v)


if args.data_settings is not None:
    data_maker = DistributionMaker(args.data_settings)
    data = data_maker.get_outputs()
    data.set_poisson_errors()
    stamp=r'$\nu_\tau$ appearance'+'\nBurnsample comparison'+'\nPrefit Distributions'
else:
    data = None
    stamp=r'$\nu_\tau$ appearance'+'\nMC'

if not args.set_param == '':
    p_name,value = args.set_param.split("=")
    print "p_name,value= ", p_name, " ", value
    value = parse_quantity(value)
    value = value.n * value.units
    test = template_maker.params[p_name]
    test.value = value
    template_maker.update_params(test)
    if args.data_settings is not None and p_name in data_maker.params.names:
        test = data_maker.params[p_name]
        test.value = value
        data_maker.update_params(test)

my_plotter = plotter(stamp=stamp, outdir='.', fmt='pdf', log=False, annotate=True, symmetric=False, ratio=True)

template_maker = DistributionMaker(args.template_settings)
if args.livetime is not None:
    livetime = template_maker.params['livetime']
    livetime.value = args.livetime * ureg.common_year
    template_maker.update_params(livetime) 

template_nominal = template_maker.get_outputs()
for map in template_nominal: map.tex = 'MC'

print "template_nominal = ", template_nominal['cscd'].hist

my_plotter.plot_1d_array(template_nominal, 'reco_coszen', fname='p_coszen')
my_plotter.plot_1d_array(template_nominal, 'reco_energy', fname='p_energy')
my_plotter.plot_2d_array(template_nominal, fname='nominal',cmap='PiYG')

if args.data_settings is not None:

    nutau_cc_norm = template_maker.params['nutau_cc_norm']
    nutau_cc_norm.value = 0.0 * ureg.dimensionless
    template_maker.update_params(nutau_cc_norm) 
    template_notau = template_maker.get_outputs()
    for map in template_notau: map.tex = r'MC\ (\nu_\tau^{CC}=0)'

    my_plotter.plot_1d_cmp(mapsets=[template_nominal,template_notau, data], plot_axis='reco_energy', fname='p_energy')
    my_plotter.plot_1d_cmp(mapsets=[template_nominal,template_notau, data], plot_axis='reco_coszen', fname='p_coszen')
