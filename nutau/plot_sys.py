from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np

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

my_plotter = plotter(stamp='nutau sys test', outdir='sys_plots/', fmt='pdf', log=False, annotate=True, symmetric=True)
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
            'atm_muon_scale': 0.0749*1.112,
            'Barr_uphor_ratio': 1.0,
            'Barr_nu_nubar_ratio': 1.0, 
            'Genie_Ma_RES': 1.0,
            'Genie_Ma_QE' : 1.0,

            }

sys_jp_name = {'theta23': 'theta23',
	     'theta13': 'theta13',
	     'deltam31': 'dm31',
	     'aeff_scale': 'norm_nu',
	     'nutau_cc_norm': 'norm_tau',
	     'nu_nc_norm': 'norm_nc',
	     'atm_muon_scale': 'atmmu_f',
	     'nue_numu_ratio': 'norm_e',
	     'delta_index': 'gamma',
	     'dom_eff': 'domeff',
	     'hole_ice': 'hole_ice',
	     'hole_ice_fwd': 'hi_fwd',
	     'Genie_Ma_QE': 'axm_qe',
	     'Genie_Ma_RES': 'axm_res',
	     'Barr_nu_nubar_ratio': 'nubar_ratio',
	     'Barr_uphor_ratio': 'uphor_ratio'
	     }

path = 'JP/'

for sys, var in variation.items():
    print sys
    template_maker.params.reset_free()
    p = template_maker.params[sys]
    p.value += var
    template_maker.update_params(p)
    template_sys = template_maker.get_outputs()
    my_plotter.label = r'$\Delta(sys-nominal)$'
    my_plotter.plot_2d_array(template_sys - template_nominal, fname='%s_variation'%sys,cmap='RdBu')


    # JP comp:
    maps = []
    for channel,channel_jp in zip(['cscd','trck'],['cascade','track']):
        if sys in sys_jp_name.keys():
            if sys in ['dom_eff', 'hole_ice', 'hole_ice_fwd']:
                file_name = '1X600_newmc_forceBaselineCrossing_detailedSystematics/diff_%s_baseline_%s.csv'% (sys_jp_name[sys], channel_jp)
            elif sys == 'atmmu_f':
                file_name = '1X600_newDataForIC/diff_%s_baseline_%s.csv'% (sys_jp_name[sys], channel_jp)
            elif sys == 'atm_delta_index':
                file_name = '1X600_SI_pivotE27.2602103972/diff_%s_baseline_%s.csv'% (sys_jp_name[sys], channel_jp)
            else:
                file_name = '1X600_diff_csv/diff_%s_baseline_%s.csv'% (sys_jp_name[sys], channel_jp)
            oscFit_data = pd.read_csv(path+file_name, sep=',',header=None)
            oscFit_data_x = oscFit_data[0].values
            oscFit_data_y = oscFit_data[1].values
            oscFit_data_z = oscFit_data[2].values
            oscFit_hist, x_edges, y_edges = np.histogram2d(oscFit_data_x, oscFit_data_y, weights = oscFit_data_z)
            maps.append(Map(name=channel, hist=oscFit_hist, binning=template_nominal[0].binning))
    if len(maps) > 0:
        oscfit = MapSet(maps, name='oscfit')
        my_plotter.plot_2d_array(oscfit, fname='%s_variation_oscfit'%sys,cmap='BrBG')
        my_plotter.label = r'$(\Delta_{OscFit} - \Delta_{PISA})/PISA$'
        my_plotter.plot_2d_array((oscfit - template_sys + template_nominal)/template_nominal, fname='%s_variation_diff_percent'%sys,cmap='PiYG')
