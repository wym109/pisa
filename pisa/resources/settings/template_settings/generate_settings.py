#! /usr/bin/env python
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# date:   9 April 2015
#
# Generate a settings file in .json using command line arguments only.
# Most of the time, and for most analyses, we will vary one or several
# input parameters that do not relate to the systematics (or nuisance
# parameters), leaving the systematics constant. This script generates
# new settings, configureable from the command line, but queries the
# database of systematic parameters and adds their values.
#


from argparse import ArgumentParser,ArgumentDefaultsHelpFormatter
import sqlite3
import numpy as np

from pisa.utils.jsons import to_json
from pisa.resources.resources import find_resource

try:
    from tabulate import tabulate
except:
    pass

def processDatabase(dbfile,free_params):

    con = sqlite3.connect(dbfile)
    cur = con.cursor()

    cur.execute('SELECT ParamName,Value,GaussianPrior,Scale,Max,Min,Fixed from SystematicParams ORDER BY ParamName')
    data = cur.fetchall()

    try:
         print tabulate(data,headers=["Name","Value","GaussianPrior","Scale","Max","Min","Fixed"],tablefmt="grid")
    except:
        print "PARAMETER TABLE: \n"
        col_names = [col[0] for col in cur.description]
        print "  %s %s %s %s %s %s %s" % tuple(col_names)
        for row in data: print "  %s %s %s %s %s %s %s" % row

    # Systematic params dict:
    params = {row[0]: {'value': row[1], 'range': [row[5],row[4]],
                       'fixed': bool(row[6]),'scale': row[3] ,'prior': row[2]}
              for row in data}

    # Convert deg to rad:
    for key in params.keys():
        if 'theta' in key:
            for subkey in ['value','prior','range']:
                if params[key][subkey] is not None:
                    params[key][subkey] = np.deg2rad(params[key][subkey])

    # now make fixed/free:
    if free_params is not None:
        # modify the free params to include the '_ih'/'_nh' tags:
        mod_free_params = []
        for p in free_params:
            if ('theta23' in p) or ('deltam31' in p):
                mod_free_params.append(p+'_ih')
                mod_free_params.append(p+'_nh')
            else:
                mod_free_params.append(p)

        print "\nmod free params: ",mod_free_params
        #Loop over the free params and set to fixed/free
        for key in params.keys():
            if key in mod_free_params: params[key]['fixed'] = False
            else: params[key]['fixed'] = True

            if not params[key]['fixed']:
                print "  Leaving parameter free: ",key
        print "  ...all others fixed!"

    return params

parser = ArgumentParser('''Creates a .json file for the template_settings used in a PISA analysis.''',
                        formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('outfile',metavar='STR',type=str,default='settings.json',
                    help='''Settings output file name.''')
parser.add_argument('dbfile',metavar='STR',type=str,
                    help='''Database file containing latest default settings on
                    nuisance parameters in analysis.''')
# Binning parameters:
parser.add_argument('--ebins',metavar='FUNC',type=str,
                    default='np.logspace(np.log10(1.0),np.log10(80.0),40)',
                    help='''Python text to convert to lambda function for energy
                    bin edges. i.e. np.linspace(1,40,40)''')
parser.add_argument('--czbins',metavar='FUNC',type=str,
                    default='np.linspace(-1.0,0.0,21)',
                    help='''Python text to convert to lambda function for coszen
                    bin edges. i.e. np.linspace(-1,0,21)''')
parser.add_argument('--oversample_e',metavar='INT',type=int,default=10,
                    help='''Oversampling factor for energy bins''')
parser.add_argument('--oversample_cz',metavar='INT',type=int,default=10,
                    help='''Oversampling factor for coszen bins''')
# Parameters for template making stages:
parser.add_argument('--flux_file',metavar='STR',type=str,
                    default="flux/frj-solmin-mountain-aa.d",
                    help='''Atmospheric flux in Honda Format''')
#     Oscillations Stage:
parser.add_argument('--osc_code',metavar='STR',type=str,default="prob3",
                    choices = ['prob3','table','nucraft','gpu'],
                    help='''Oscillation Code to use for transition probabilities.''')
parser.add_argument('--earth_model',type=str,default='oscillations/PREM_12layer.dat',
                    help='''Earth model data. For Prob3, it is shells of density within
                    the layer, whereas for NuCraft, it is used to make a spline fit for
                    density vs. radius.''')
parser.add_argument('--detector_depth', type=float, default=2.0,
                    help='''Detector depth in km
                    [Should not change for IceCube/DeepCore/PINGU]''')
parser.add_argument('--prop_height', type=float, default=20.0,
                    help='''Height in the atmosphere to begin propagation in km.
                    Prob3 default: 20.0 km
                    NuCraft default: 'sample' from a distribution''')
#     Aeff Stage
parser.add_argument('--aeff_mode',metavar='STR',type=str,default="param",
                    choices = ['prob3','MC'],
                    help='''Use parameterized or Monte Carlo-based Aeff stage.''')
parser.add_argument('--aeff_egy_par_dir',metavar='STR',type=str,default='aeff/V36/cuts_V5/',
                    help='''Directory (using find_resource) to aeff files, which are expected
                    to be in format [<dir>/a_aeff_<flav>.dat]''')
parser.add_argument('--aeff_coszen_par',metavar='STR',type=str,default='aeff/V36/V36_aeff_cz.json',
                    help='''Directory (using find_resource) to aeff coszen dependence''')
parser.add_argument('--aeff_weight_file',metavar='STR',type=str,default=None,
                    help='''HDF5 file of weighted aeff for MC mode''')
#     Reco Stage
parser.add_argument('--reco_mode',metavar='STR', type=str,
                    choices=['MC', 'param', 'vbwkde', 'stored'],
                    default='vbwkde', help='''Reco service to use''')
parser.add_argument('--reco_mc_wt_file',metavar='STR', type=str,
                    default="events/V36_weighted_aeff_joined_nu_nubar.hdf5",
                    help='''File from which to get the MC-based reconstruction''')
parser.add_argument('--reco_vbwkde_evts_file',metavar='STR', type=str,
                    default="events/V36_weighted_aeff_joined_nu_nubar.hdf5",
                    help='''File from which to define the VBW KDE reconstruction''')
parser.add_argument('--reco_kernel_file',metavar='STR', type=str,
                    default=None,
                    help='''File from which to define the reconstruction from a kernel file''')
parser.add_argument('--reco_param_file',metavar='STR', type=str,default="reco/V36.json",
                    help='''File containing the double gauss parameterized reconstruction''')
#     PID Stage
parser.add_argument('--pid_mode',metavar='STR', type=str,
                    choices=['param', 'stored'],default='param',
                    help='''Mode for PID service to use''')
parser.add_argument('--pid_paramfile',metavar='STR', type=str,default="pid/V36_pid_recoEgy.json",
                    help='''File of parameterized PID vs. Energy''')
parser.add_argument('--pid_kernelfile',metavar='STR', type=str,default=None,
                    help='''Kernelfile containing PID vs. Energy''')

parser.add_argument('--livetime',metavar='FLOAT',type=float,default=1.0,
                    help='''Livetime for detector''')
parser.add_argument('--channel',metavar='STR', type=str,
                    choices=['trck', 'cscd','all','no_pid'],default='all',
                    help='''Channel to use in analysis''')
parser.add_argument('--free_params',metavar='LIST',default=None,nargs='*',
                    help='''List of parameters to leave as free from systematics database.
                    If not set, will use fixed as set in database.''')


args = parser.parse_args()

template_settings = {"binning": {}, "params": {}}

ebins = eval(args.ebins)
czbins = eval(args.czbins)

# Form dict of tempalte_settings, then write to output .json file:
# Binning field first:
template_settings['binning']['ebins'] = ebins
template_settings['binning']['czbins'] = czbins
template_settings['binning']['oversample_e'] = args.oversample_e
template_settings['binning']['oversample_cz'] = args.oversample_cz

# Now process the (possibly not fixed) systematic parameters of the database:
dbparams = processDatabase(args.dbfile,args.free_params)
outfile = args.outfile

# Get all other parameters as cmd line arguments:
arg_dict = vars(args)
for key in (template_settings['binning'].keys()+['free_params','dbfile','outfile']):
    arg_dict.pop(key)

# Now params:
template_settings['params'] = {key: {"value": val, "fixed": True}
                               for key,val in vars(args).items()
                               if key not in template_settings['binning'].keys()}

# Add aeff part:
stem = 'a_eff_'
aeff_dict = {'NC': stem+'nuall_nc.dat',
             'NC_bar': stem+'nuallbar_nc.dat',
             'nue': stem+'nue.dat',
             'nue_bar': stem+'nuebar.dat',
             'numu': stem+'numu.dat',
             'numu_bar': stem+'numubar.dat',
             'nutau': stem+'nutau.dat',
             'nutau_bar': stem+'nutaubar.dat'}
aeff_dir = template_settings['params']['aeff_egy_par_dir']['value']
template_settings['params'].pop('aeff_egy_par_dir')
template_settings['params']['aeff_egy_par'] = {'value':{},'fixed':True}

for key,val in aeff_dict.items():
    template_settings['params']['aeff_egy_par']['value'][key] = aeff_dir+val

for key,val in dbparams.items(): template_settings['params'][key] = val

print "Writing all settings to: ",outfile
to_json(template_settings,outfile)
