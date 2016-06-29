from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage.filters import gaussian_filter
from scipy import interpolate
import copy
import itertools

from pisa import ureg, Q_
from pisa.core.pipeline import Pipeline
from pisa.utils.fileio import from_file, to_file
from pisa.utils.log import set_verbosity
from pisa.utils.parse_config import parse_config
from pisa.utils.plotter import plotter
from pisa.core.map import Map, MapSet

parser = ArgumentParser()
parser.add_argument('-t', '--template-settings', type=str,
                    metavar='configfile', required=True,
                    help='settings for the generation of templates')
parser.add_argument('-f', '--fit-settings', type=str,
                    metavar='configfile', required=True,
                    help='settings for the generation of templates')
parser.add_argument('-p', '--plot', action='store_true',
                    help='plot')
parser.add_argument('-v', action='count', default=None,
                    help='set verbosity level')
args = parser.parse_args()
set_verbosity(args.v)

if args.plot: 
    import matplotlib.pyplot as plt
    from pisa.utils.plotter import plotter

cfg = from_file(args.fit_settings)
sys_list = cfg.get('general','sys_list').replace(' ','').split(',')
categories = cfg.get('general','categories').replace(' ','').split(',')
idx = cfg.getint('general','stop_after_stage')

# setup plotting colors
colors = itertools.cycle(["r", "b", "g"])
plt_colors = {}
for cat in categories:
    plt_colors[cat] = next(colors)

for sys in sys_list:
    # parse info for given systematic
    nominal = cfg.getfloat(sys,'nominal')
    degree = cfg.getint(sys, 'degree')
    force_through_nominal = cfg.getboolean(sys, 'force_through_nominal')
    runs = eval(cfg.get(sys,'runs'))
    smooth = cfg.get(sys, 'smooth')

    x_values = np.array(sorted(runs))

    # build fit function
    if force_through_nominal:
        function = "fit_fun = lambda x, *p: np.polynomial.polynomial.polyval(x, [1.] + list(p))"
    else:
        function = "fit_fun = lambda x, *p: np.polynomial.polynomial.polyval(x, list(p))"
        # add free parameter for constant term
        degree += 1
    exec(function)

    # instantiate template maker
    template_maker_settings = from_file(args.template_settings)
    template_maker_configurator = parse_config(template_maker_settings)
    template_maker = Pipeline(template_maker_configurator)

    inputs = {}
    for cat in categories:
        inputs[cat] = {}

    # get sys templates
    for run in runs:
        for key, val in cfg.items('%s:%s'%(sys, run)):
            if key.startswith('param.'):
                _,pname = key.split('.')
                param = template_maker.params[pname]
                param.value = val
                param.set_nominal_to_current_value()
                template_maker.update_params(param)
        # retreive maps
        template = template_maker.get_outputs(idx=idx)
        for cat in categories:
            inputs[cat][run] = sum([map.hist for map in template if
                map.name.endswith(cat)])

    # binning object
    binning = template.maps[0].binning

    # numpy acrobatics:
    arrays = {}
    for cat in categories:
        arrays[cat] = []
        for x in x_values:
            arrays[cat].append(inputs[cat][x]/inputs[cat][nominal])
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

    # now actualy perform some fits
    for cat in categories:
        outputs[cat] = np.ones((nx, ny, degree))

        for i, j in np.ndindex((nx,ny)):
        #for i, j in np.ndindex(inputs[cat][nominal].shape):
            y_values = arrays[cat][i,j,:]
            popt, pcov = curve_fit(fit_fun, x_values,
                    y_values, p0=np.ones(degree))
            for k, p in enumerate(popt):
                outputs[cat][i,j,k] = p

            # maybe plot
            if args.plot:
                fig_num = i + nx * j
                if fig_num == 0:
                    fig = plt.figure(num=1, figsize=( 4*nx, 4*ny))
                subplot_idx = nx*(ny-1-j)+ i + 1
                plt.subplot(ny, nx, subplot_idx)
                plt.scatter(x_values, y_values, color=plt_colors[cat])
                # plot nominal point again in black
                plt.scatter([0.0], [1.0], color='k')
                f_values = fit_fun(x_values, *popt)
                fun_plot, = plt.plot(x_values, f_values,
                        color=plt_colors[cat])
                plt.ylim(np.min(arrays[cat])*0.9, np.max(arrays[cat])*1.1)
                if i > 0:
                    plt.setp(plt.gca().get_yticklabels(), visible=False)
                if j > 0:
                    plt.setp(plt.gca().get_xticklabels(), visible=False)

    # smoothing
    if not smooth == 'raw':
        raw_outputs = copy.deepcopy(outputs)
        errors = {}
        for cat in categories:
            for d in range(degree):
                if smooth == 'spline':
                    spline = interpolate.SmoothBivariateSpline(grid_x, grid_y,
                            np.ravel(outputs[cat][:,:,d]), kx=2, ky=2)
                    outputs[cat][:,:,d] = spline(bins_x, bins_y)
                elif smooth == 'gauss':
                    outputs[cat][:,:,d] = gaussian_filter(outputs[cat][:,:,d],
                            sigma=1)

            if args.plot:
                for i, j in np.ndindex((nx,ny)):
                    fig_num = i + nx * j
                    if fig_num == 0:
                        fig = plt.figure(num=1, figsize=( 4*nx, 4*ny))
                    subplot_idx = nx*(ny-1-j)+ i + 1
                    plt.subplot(ny, nx, subplot_idx)
                    p_smooth = outputs[cat][i,j,:]
                    f_values = fit_fun(x_values, *p_smooth)
                    fun_plot, = plt.plot(x_values, f_values,
                            color=plt_colors[cat], linestyle='--')

    if args.plot:
        fig.subplots_adjust(hspace=0)
        fig.subplots_adjust(wspace=0)
        plt.show()
        plt.savefig('%s_sysfits_%s.pdf'%(sys,smooth))
        plt.clf()

        if not smooth == 'raw':
            for d in range(degree):
                maps = []
                for cat in categories:
                    maps.append(Map(name='%s_raw'%cat, hist=raw_outputs[cat][:,:,d],
                                binning=binning))
                    maps.append(Map(name='%s_smooth'%cat, hist=outputs[cat][:,:,d],
                                binning=binning))
                    maps.append(Map(name='%s_diff'%cat,
                                hist=raw_outputs[cat][:,:,0] - outputs[cat][:,:,d],
                                binning=binning))
                maps = MapSet(maps)
                my_plotter = plotter(stamp='PISA cake test', outdir='.',
                    fmt='pdf',log=False, label='')
                my_plotter.plot_2d_array(maps, fname='%s_smooth_%s_p%s'%(sys,smooth,d))


    outputs['pname'] = sys
    outputs['nominal'] = nominal
    outputs['function'] = function
    outputs['categories'] = categories
    #outputs['binning'] = binning
    to_file(outputs, 'pisa/resources/sys/%s_sysfits_%s.json'%(sys,smooth))
