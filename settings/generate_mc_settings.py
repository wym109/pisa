#! /usr/bin/env python
#
# This script generates a settings file for the Monte Carlo-based analysis.
# 
# 
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# date:   2014-03-13

from utils.json import to_json
import numpy as np
try:
    import simplejson as json
except ImportError:
    import json


from argparse import ArgumentParser
parser = ArgumentParser(description="This script generates the settings file for the Monte Carlo-based NMH analysis. The user can many inputs to the simulation here.")
parser.add_argument('-o', '--outfile', dest='outfile', metavar='FILE', type=str, action='store',
                    default='mc_settings.json',
                    help='file to store the output settings file [default: mc_settings.json]')
### Map Binning Params ###
parser.add_argument('--emin',type=float,default=1.0,action='store',
                    help='Minimum energy bin edge to use in maps [GeV].')
parser.add_argument('--emax',type=float,default=80.0,action='store',
                    help='Maximum energy bin edge to use in maps [GeV].')
parser.add_argument('--num_ebins',type=int,default=40,action='store',
                    help='Number of energy bins to use in maps.')
parser.add_argument('--logE',action='store_true',default=False,
                    help='Use log10 bins in energy.')
parser.add_argument('--czmin',type=float,default=-1.0,action='store',
                    help='Minimum coszen bin edge in maps.')
parser.add_argument('--czmax',type=float,default=1.0,action='store',
                    help='Minimum coszen bin edge in all TRUE maps.')
parser.add_argument('--num_czbins',type=int,default=40,action='store',
                    help='Number of cz bins to use in maps.')

### Oscillation Params ###
# NOTE: Best fit values and range, based loosely on 
# Fogli, et al. (2012) DOI: 10.1103/PhysRevD.86.013012
parser.add_argument('--deltam21_best',type=float,default=7.54e-05,action='store',
                    help='deltam21 value to use [eV^2].')
parser.add_argument('--deltam21_min',type=float,default=7.32e-05,action='store',
                    help='deltam21 minimum bound to use [eV^2].')
parser.add_argument('--deltam21_max',type=float,default=7.80e-05,action='store',
                    help='deltam21 maximum bound to use [eV^2].')

parser.add_argument('--deltam31_nh_best',type=float,default=0.00246,action='store',
                    help='best fit value of deltam31 to use for NH [eV^2].')
parser.add_argument('--deltam31_nh_min',type=float,default=0.00230,action='store',
                    help='min value of deltam31 value to use for NH [eV^2].')
parser.add_argument('--deltam31_nh_max',type=float,default=0.00260,action='store',
                    help='max value of deltam31 value to use for NH [eV^2].')
parser.add_argument('--deltam31_ih_best',type=float,default=-0.00238,action='store',
                    help='best fit value of deltam31 value to use for IH [eV^2].')
parser.add_argument('--deltam31_ih_min',type=float,default=-0.00250,action='store',
                    help='min value of deltam31 value to use for IH [eV^2].')
parser.add_argument('--deltam31_ih_max',type=float,default=-0.00220,action='store',
                    help='max value of deltam31 value to use for IH [eV^2].')

parser.add_argument('--theta12_best',type=float,default=33.6471,action='store',
                    help='theta12 best fit value to use [deg].')
parser.add_argument('--theta12_min',type=float,default=32.65,action='store',
                    help='theta12 min value to use [deg].')
parser.add_argument('--theta12_max',type=float,default=34.76,action='store',
                    help='theta12 max value to use [deg].')

parser.add_argument('--theta13_best',type=float,default=8.931,action='store',
                    help='theta13 best fit value to use [deg].')
parser.add_argument('--theta13_min',type=float,default=8.4,action='store',
                    help='theta13 min value to use [deg].')
parser.add_argument('--theta13_max',type=float,default=9.4,action='store',
                    help='theta13 max value to use [deg].')

parser.add_argument('--theta23_best',type=float,default=38.6455,action='store',
                    help='min value of theta23 to use.')
parser.add_argument('--theta23_min',type=float,default=36.0,action='store',
                    help='min value of theta23 to use.')
parser.add_argument('--theta23_max',type=float,default=45.0,action='store',
                    help='max value of theta23 to use.')

parser.add_argument('--deltacp_best',type=float,default=np.pi,action='store',
                    help='delta CP value to use [rad].')
parser.add_argument('--deltacp_min',type=float,default=0.0,action='store',
                    help='min delta CP value to use.')
parser.add_argument('--deltacp_max',type=float,default=2.0*np.pi,action='store',
                    help='max delta CP value to use.')

### Other Parameters ###
parser.add_argument('--e_reco_scale',type=float,default=1.0,help="energy reco scale.")
parser.add_argument('--runtime',type=float,default=1.0,help="Livetime in years.")
parser.add_argument('--nu_xsec_scale',type=float,default=0.15,help="Cross section uncertainty.")
parser.add_argument('--nu_bar_xsec_scale',type=float,default=0.15,help="Cross section uncertainty.")
parser.add_argument('--nue_data',type=str,
                    default='$PISA/resources/data_files/V15_nue_sim_wt_array.hdf5',
                    help='nue_data file [.hdf5 file format.]')
parser.add_argument('--nue_bar_data',type=str,
                    default='$PISA/resources/data_files/V15_nue_bar_sim_wt_array.hdf5',
                    help='nue_bar_data file [.hdf5 file format.]')
parser.add_argument('--numu_data',type=str,
                    default='$PISA/resources/data_files/V15_numu_sim_wt_array.hdf5',
                    help='numu_data file [.hdf5 file format.]')
parser.add_argument('--numu_bar_data',type=str,
                    default='$PISA/resources/data_files/V15_numu_bar_sim_wt_array.hdf5',
                    help='numu_bar_data file [.hdf5 file format.]')
parser.add_argument('--nutau_data',type=str,
                    default='$PISA/resources/data_files/V15_nutau_sim_wt_array.hdf5',
                    help='nutau_data file [.hdf5 file format.]')
parser.add_argument('--nutau_bar_data',type=str,
                    default='$PISA/resources/data_files/V15_numu_bar_sim_wt_array.hdf5',
                    help='nutau_bar_data file [.hdf5 file format.]')
args = parser.parse_args()


ebins = np.logspace(np.log10(args.emin),np.log10(args.emax),args.num_ebins+1) if args.logE else np.linspace(args.emin,args.emax,args.num_ebins+1)
czbins = np.linspace(args.czmin, args.czmax, args.num_czbins+1)

sim_wt_dict = {'nue':args.nue_data,'nue_bar':args.nue_bar_data,
               'numu':args.numu_data,'numu_bar':args.numu_bar_data,
               'nutau':args.nutau_data,'nutau_bar':args.nutau_bar_data}

arg_dict = vars(args)
osc_param_names = ['deltam21','deltam31_nh','deltam31_ih',
                   'theta12','theta13','theta23','deltacp']
endings = ['best','min','max']
osc_dict = {name:{val:arg_dict[name+'_'+val] for val in endings} for name in osc_param_names}

params = {"tables":"$PISA/resources/flux/frj-solmin-mountain-aa.d",
          "e_reco_scale":args.e_reco_scale,
          "runtime":args.runtime,
          "nu_xsec_scale":args.nu_xsec_scale,
          "nu_bar_xsec_scale":args.nu_bar_xsec_scale}

json_content = {"ebins":ebins,
                "czbins":czbins,
                "params":params,
                "osc": osc_dict,
                "sim_wt_arrays":sim_wt_dict}

to_json(json_content,args.outfile)

