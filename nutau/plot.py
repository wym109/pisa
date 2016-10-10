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
parser.add_argument('-d', '--data-settings', type=str,
		    metavar='configfile', default=None,
		    help='settings for the generation of "data"')
parser.add_argument('-sp', '--set-param', type=str, default='',
                    help='Set a param to a certain value.')
parser.add_argument('-f', '--fit', type=str,
		    metavar='fit file', default=None,
		    help='settings for the generation of "data"')
parser.add_argument('-f0', '--fit0', type=str,
		    metavar='fit file for null hypo', default=None,
		    help='settings for the generation of "data"')
parser.add_argument('-o', '--outdir', type=str,
		    default='plots',
		    help='outdir')
parser.add_argument('-v', action='count', default=None,
                    help='set verbosity level')
parser.add_argument('--livetime', type=float, default=None)
args = parser.parse_args()
set_verbosity(args.v)


if args.data_settings is not None:
    data_maker = DistributionMaker(args.data_settings)
    data = data_maker.get_total_outputs()
    data.set_poisson_errors()
    stamp=r'$\nu_\tau$ appearance'+'\nBurnsample comparison'
    if args.fit is None:
        stamp+='\nPrefit Distributions'
        out_name = 'prefit'
    else:
        stamp+='\nPostfit Distributions'
        out_name = 'postfit'
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

my_plotter = Plotter(stamp=stamp, outdir=args.outdir, fmt='pdf', log=False, annotate=True, symmetric=False, ratio=True)

template_maker = DistributionMaker(args.template_settings)
template_maker_H0 = DistributionMaker(args.template_settings)

nutau_cc_norm = template_maker_H0.params['nutau_cc_norm']
nutau_cc_norm.value = 0.0 * ureg.dimensionless
template_maker_H0.update_params(nutau_cc_norm) 

def update(template_maker, fitfile):
    params = from_file(fitfile)
    for key,val in params[0].items():
        if key in template_maker.params.names:
            p = template_maker.params[key]
            if len(val[1]) == 0:
                p.value = Q_(val[0])
            else:
                p.value = Q_(val[0],'%s^%s'%(val[1][0][0],val[1][0][1]))
            template_maker.update_params(p)


if args.fit is not None:
    update(template_maker, args.fit)

if args.fit0 is not None:
    update(template_maker_H0, args.fit0)

if args.livetime is not None:
    for tm in [template_maker, template_maker_H0]:
        livetime = tm.params['livetime']
        livetime.value = args.livetime * ureg.common_year
        tm.update_params(livetime) 

template_nominal = template_maker.get_total_outputs()
for map in template_nominal: map.tex = map.name
#for map in template_nominal: map.tex = 'MC'
template_notau = template_maker_H0.get_total_outputs()
for map in template_notau: map.tex = map.name
#for map in template_notau: map.tex = r'MC\ (\nu_\tau^{CC}=0)'

#print "template_nominal = ", template_nominal['cscd'].hist

#my_plotter.plot_1d_array(template_nominal, 'reco_coszen', fname='p_coszen')
#my_plotter.plot_1d_array(template_nominal, 'reco_energy', fname='p_energy')
my_plotter.plot_2d_array(template_nominal, fname='nominal', cmap='summer')

if args.data_settings is not None:


    my_plotter.plot_1d_slices_array(mapsets=[template_nominal,template_notau, data], plot_axis='reco_coszen', fname=out_name+'_s_coszen')
    my_plotter.plot_1d_slices_array(mapsets=[template_nominal,template_notau, data], plot_axis='reco_energy', fname=out_name+'_s_energy')
    my_plotter.plot_1d_cmp(mapsets=[template_nominal,template_notau, data], plot_axis='reco_energy', fname=out_name+'_p_energy')
    my_plotter.plot_1d_cmp(mapsets=[template_nominal,template_notau, data], plot_axis='reco_coszen', fname=out_name+'_p_coszen')
