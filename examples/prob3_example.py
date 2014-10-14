#! /usr/bin/env python
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# date:   14 October 2014
#
# Demonstrates the usage of the prob3 oscillation code to compute
# oscillation probabilities in a stand-alone form. The prob3
# oscillation code, written by members of the Super K collaboration
# (http://www.phy.duke.edu/~raw22/public/Prob3++/) is a fast
# oscillation probability calculator for and is based on the analytic
# solution for the propagation of three flavor neutrino probabilities
# of: Barger et al. Phys. Rev. D22 (1980) 2718.
# 
# Output of all oscillation maps:
#   nue -> nue, numu, nutau
#   numu -> nue, numu, nutau
#   nue_bar -> nue_bar, numu_bar, nutau_bar
#   numu_bar -> nue_bar, numu_bar, nutau_bar
# Is saved to a file: args.outfile.
#
# To load the output file into a python dictionary:
#   > from pisa.utils.jsons import from_json
#   > data = from_json('osc_prob_dict.json')
#   > data.keys()
#

from argparse import ArgumentParser
import logging
from datetime import datetime
import sys
import numpy as np

from pisa.oscillations.prob3.BargerPropagator import BargerPropagator
from pisa.resources.resources import find_resource
from pisa.utils.jsons import to_json
from pisa.utils.utils import set_verbosity, get_bin_centers
from pisa.utils.proc import report_params
from pisa.resources.resources import find_resource

parser = ArgumentParser('Example usage of the prob3 oscillation code (through BargerPropagator module).')
parser.add_argument('--earth_model',type=str,default='oscillations/PREM_60layer.dat',
                    help='Earth model to use.')
parser.add_argument('--detector_depth',type=float,default=2.0,
        help='detector depth [km]. (Default: 2.0 km) NOTE: IceCube should be 2.0 km')
parser.add_argument('--prod_height',type=float,default=20.0,
       help='Atmospheric production height at which neutrinos are created. [km]')
parser.add_argument('--deltam21',type=float,default=7.54e-5,
                    help='\Delta m_{21}^2 parameter-solar mass splitting [eV^2]')
parser.add_argument('--deltam31',type=float,default=2.46e-3,
                    help='\Delta m_{31}^2 parameter  [eV^2]')
parser.add_argument('--theta12',type=float,default=0.5873,
                    help='''theta12 value [rad]''')
parser.add_argument('--theta13',type=float,default=0.1562,
                    help='''theta13 value [rad]''')
parser.add_argument('--theta23',type=float,default=0.6745,
                    help='''theta23 value [rad]''')
parser.add_argument('--deltacp',type=float,default=0.0,
                    help='''deltaCP value to use [rad]''')
parser.add_argument('--outfile',type=str,default='osc_prob_dict.json',
                    help="output file for osc probs [.json]")
parser.add_argument('--plot',action='store_true',default=False,
                    help="Plot the numu survival probability when finished.")
parser.add_argument('-v', '--verbose', action='count', default=0,
                    help='set verbosity level')
args = parser.parse_args()

set_verbosity(args.verbose)

start_time = datetime.now()

osc_params = {'deltacp':args.deltacp, 'deltam21':args.deltam21,
              'deltam31':args.deltam31,'theta12':args.theta12,
              'theta13':args.theta13,'theta23':args.theta23}
report_params(osc_params, units = ['rad','eV^2','eV^2','rad','rad','rad'])

# Initialize binning for prob maps:
ebins = np.linspace(1,80,150)
czbins = np.linspace(-1,0,150)

# Initialize barger propagator which contains the methods for
# extracting the oscillation probabilities through the earth.
earth_model = find_resource(args.earth_model)
barger_prop = BargerPropagator(earth_model, args.detector_depth)
barger_prop.UseMassEigenstates(False)

mAtm = args.deltam31 if args.deltam31 < 0.0 else (args.deltam31 - args.deltam21)
# Set to false, since we are using sin^2(2 theta) variables                   
kSquared = False
sin2th12Sq = np.sin(2.0*args.theta12)**2
sin2th13Sq = np.sin(2.0*args.theta13)**2
sin2th23Sq = np.sin(2.0*args.theta23)**2

neutrinos = ['nue','numu','nutau']
anti_neutrinos = ['nue_bar','numu_bar','nutau_bar']


nu_barger = {'nue':1,'numu':2,'nutau':3,
             'nue_bar':1,'numu_bar':2,'nutau_bar':3}

# Initialize dictionary to hold the osc prob maps
osc_prob_dict = {'ebins':ebins, 'czbins':czbins}
ecen = get_bin_centers(ebins)
czcen = get_bin_centers(czbins)
shape = (len(ebins),len(czbins))
for nu in ['nue_maps','numu_maps','nue_bar_maps','numu_bar_maps']:
    isbar = '_bar' if 'bar' in nu else ''
    osc_prob_dict[nu] = {'nue'+isbar: np.zeros(shape,dtype=np.float32),
                         'numu'+isbar: np.zeros(shape,dtype=np.float32),
                         'nutau'+isbar: np.zeros(shape,dtype=np.float32)}
    
    
print "Getting oscillation probability maps..."
total_bins = int(len(ebins)*len(czbins))
mod = total_bins/50
ibin = 0
for icz, coszen in enumerate(czcen):
    for ie,energy in enumerate(ecen):
        ibin+=1
        if (ibin%mod) == 0:
            sys.stdout.write(".")
            sys.stdout.flush()
            
        # First for neutrinos:
        kNuBar = 1
        barger_prop.SetMNS(sin2th12Sq,sin2th13Sq,sin2th23Sq,args.deltam21,mAtm,
                           args.deltacp,energy,kSquared,kNuBar)
        barger_prop.DefinePath(coszen, args.prod_height)
        barger_prop.propagate(kNuBar)
        
        for nu in ['nue','numu']:
            nu_i = nu_barger[nu]
            for to_nu in neutrinos:
                nu_f = nu_barger[to_nu]
                osc_prob_dict[nu+'_maps'][to_nu][ie][icz]=barger_prop.GetProb(nu_i, nu_f)
                
        # Second for anti-neutrinos:
        kNuBar = -1
        barger_prop.SetMNS(sin2th12Sq,sin2th13Sq,sin2th23Sq,args.deltam21,mAtm,
                           args.deltacp,energy,kSquared,kNuBar)
        barger_prop.DefinePath(coszen, args.prod_height)
        barger_prop.propagate(kNuBar)
        
        for nu in ['nue_bar','numu_bar']:
            nu_i = nu_barger[nu]
            for to_nu in anti_neutrinos:
                nu_f = nu_barger[to_nu]
                osc_prob_dict[nu+'_maps'][to_nu][ie][icz]=barger_prop.GetProb(nu_i, nu_f)


print "\nSaving to file: ",args.outfile
to_json(osc_prob_dict,args.outfile)

print "\nFinished in %s seconds!"%(datetime.now() - start_time)

if args.plot:
    from matplotlib import pyplot as plt
    from pisa.utils.plot import show_map
    pmap = {'map':osc_prob_dict['numu_maps']['numu'],
            'ebins':osc_prob_dict['ebins'],
            'czbins': osc_prob_dict['czbins']}
    show_map(pmap)
    plt.show()
    
    
