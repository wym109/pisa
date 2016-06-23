from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
from numpy.polynomial.polynomial import polyval
from scipy.optimize import curve_fit
from scipy import interpolate

from pisa import ureg, Q_
from pisa.core.distribution_maker import DistributionMaker
from pisa.utils.fileio import from_file, to_file
from pisa.utils.log import set_verbosity
from pisa.utils.parse_config import parse_config
from pisa.utils.plotter import plotter
from pisa.core.map import MapSet

parser = ArgumentParser()
parser.add_argument('-t', '--template-settings', type=str,
                    metavar='configfile', required=True,
                    help='settings for the generation of templates')
parser.add_argument('-p', '--plot', action='store_true',
                    help='plot')
parser.add_argument('-v', action='count', default=None,
                    help='set verbosity level')
args = parser.parse_args()
set_verbosity(args.v)

# 1: linear, 2: quadratic, etc...
degree = 2
# force fit to go through nominal point
force_through_nominal = True

# define fit function
if force_through_nominal:
    function = "fit_fun = lambda x, *p: polyval(x, [1.] + list(p))"
else:
    function = "fit_fun = lambda x, *p: polyval(x, list(p))"
    # add free parameter for constant term
    degree += 1

exec(function)

if args.plot: 
    import matplotlib.pyplot as plt

template_maker_settings = from_file(args.template_settings)
template_maker_configurator = parse_config(template_maker_settings)
template_maker = DistributionMaker(template_maker_configurator)

path = '/Users/peller/PSU/cake/nutau/event_files/'
fname_s = 'events__deepcore__IC86__runs_12%s1-12%s3,14%s1-14%s3,16%s1-16%s3__proc_v5digit__'
fname_unjoined = 'unjoined.hdf5'
fname_joined = 'joined_G_nue_cc+nuebar_cc_G_numu_cc+numubar_cc_G_nutau_cc+nutaubar_cc_G_nuall_nc+nuallbar_nc.hdf5'

pname = 'dom_eff'
nominal = 1.0
#runs = [('601', 0.88), ('603', 0.94), ('604', 0.97), ('605', 1.03), ('606', 1.06), ('608', 1.12)]
runs = [('601', 0.88), ('603', 0.94), ('606', 1.06), ('608', 1.12)]

x_values = np.array(sorted([r[1] for r in runs] + [nominal]))

categories = ['cscd', 'trck']
plt_colors = {'cscd':'b', 'trck':'r'}

inputs = {}
for cat in categories:
     inputs[cat] = {}

# get templates
template = template_maker.get_outputs()
for cat in categories:
    inputs[cat][nominal] = sum([map.hist for map in template if
            map.name.endswith(cat)])

for run, value in runs:
    # adjust params
    param = template_maker.params['aeff_weight_file']
    param.value = path + fname_s%(tuple([run]*6)) + fname_unjoined
    param.set_nominal_to_current_value()
    template_maker.update_params(param)
    param = template_maker.params['reco_weight_file']
    param.value = path + fname_s%(tuple([run]*6)) + fname_joined
    param.set_nominal_to_current_value()
    template_maker.update_params(param)
    param = template_maker.params['pid_events']
    param.value = path + fname_s%(tuple([run]*6)) + fname_joined
    param.set_nominal_to_current_value()
    template_maker.update_params(param)
    # retreive maps
    template = template_maker.get_outputs()
    for cat in categories:
        inputs[cat][value] = sum([map.hist for map in template if
            map.name.endswith(cat)])

arrays = {}
for cat in categories:
    arrays[cat] = []

for x in x_values:
    for cat in categories:
        arrays[cat].append(inputs[cat][x]/inputs[cat][nominal])

for cat in categories:
    arrays[cat] = np.array(arrays[cat]).transpose(1,2,0)

nx, ny = inputs[categories[0]][nominal].shape
bins_x = np.arange(nx)
bins_y = np.arange(ny)

grid_x, grid_y = np.meshgrid(bins_x, bins_y)

grid_x = np.ravel(grid_x)
grid_y = np.ravel(grid_y)

# shift to get deltas
x_values -= nominal

# array to store params
outputs = {}
for cat in categories:
    outputs[cat] = np.ones((nx, ny, degree))

    for i, j in np.ndindex((nx,ny)):
    #for i, j in np.ndindex(inputs[cat][nominal].shape):
        y_values = arrays[cat][i,j,:]
        popt, pcov = curve_fit(fit_fun, x_values,
                y_values, p0=np.ones(degree))
        for k, p in enumerate(popt):
            outputs[cat][i,j,k] = p
        # plot
        if args.plot:
            fig_num = i + nx * j
            if fig_num == 0:
                fig = plt.figure(num=1, figsize=( 4*nx, 4*ny))
            subplot_idx = nx*(ny-1-j)+ i + 1
            plt.subplot(ny, nx, subplot_idx)
            plt.scatter(x_values, y_values, color=plt_colors[cat])
            # nominal point
            plt.scatter([0.0], [1.0], color='k')
            f_values = fit_fun(x_values, *popt)
            fun_plot, = plt.plot(x_values, f_values,
                    color=plt_colors[cat])
            plt.ylim(np.min(arrays[cat])*0.9, np.max(arrays[cat])*1.1)
            if i > 0:
                plt.setp(plt.gca().get_yticklabels(), visible=False)
            if j > 0:
                plt.setp(plt.gca().get_xticklabels(), visible=False)

if args.plot:
    fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(wspace=0)
    plt.show()
    plt.savefig('%s_sysfits.pdf'%pname)
    plt.clf()

outputs['pname'] = pname
outputs['nominal'] = nominal
outputs['function'] = function
to_file(outputs, './%s_sysfits.json'%pname)

#spline = interpolate.SmoothBivariateSpline(grid_x, grid_y,
#        np.ravel(cscd_slopes), kx=2, ky=2)
#smooth = spline(bins_x, bins_y)
#plt.imshow(smooth,interpolation='nearest')
