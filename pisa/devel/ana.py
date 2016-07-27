import sys, os
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pisa.utils.resources import find_resource
from pisa.utils.log import logging
from pisa.devel.prob3gpu import Prob3GPU
from pisa.devel.hist import GPUhist
from pisa.devel.weight import GPUweight
from pisa.devel.const import FTYPE
from pisa.utils.events import Events
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import time

def copy_dict_to_d(events):
    d_events = {}
    for key, val in events.items():
        d_events[key] = cuda.mem_alloc(val.nbytes)
        cuda.memcpy_htod(d_events[key], val)
    return d_events

# --- PARAMS (fixed for now) ---
# layer params
detector_depth = 2.0
earth_model = find_resource('osc/PREM_12layer.dat')
prop_height = 20.0
YeI = 0.4656
YeO = 0.4656
YeM = 0.4957
# events file
fname = '/fastio/peller/events__deepcore__IC86__runs_126001-126003,146001-146003,166001-166003__proc_v5digit__unjoined_with_fluxes_GENIE_Barr.hdf5'
# histo bins
bin_edges_e = np.logspace(0.75,2,11).astype(FTYPE)
bin_edges_cz = np.linspace(-1,1,17).astype(FTYPE)
# nuisance params:
# flux
nu_nubar_ratio = 1.0
nue_numu_ratio = 1.0
# osc
theta12 = 0.5839958715755919
theta13 = 0.14819001778459273
theta23 = 0.7373241279447564
deltam21 = 7.5e-05
deltam31 = 0.002457
deltacp = 5.340707511102648
# aeff
livetime = 4.0
aeff_scale = 1.0
# pid
pid_bound = 2.0
pid_remove = -3.0
# ------

# initialize classes
osc = Prob3GPU(detector_depth, earth_model, prop_height,  YeI, YeO, YeM)
weight = GPUweight()
#histogram from GPU
histogrammer = GPUhist(bin_edges_cz, bin_edges_e)

# --- Load events
# open Events file
evts = Events(fname)

# Load and copy events
variables = ['true_energy', 'true_coszen', 'reco_energy', 'reco_coszen', 'neutrino_nue_flux', 'neutrino_numu_flux', 'neutrino_oppo_nue_flux', 'neutrino_oppo_numu_flux', 'weighted_aeff', 'pid']
empty = ['prob_e', 'prob_mu', 'weight_trck', 'weight_cscd']
flavs = ['nue_cc', 'numu_cc', 'nutau_cc', 'nue_nc', 'numu_nc', 'nutau_nc', 'nuebar_cc', 'numubar_cc', 'nutaubar_cc', 'nuebar_nc', 'numubar_nc', 'nutaubar_nc']
kFlavs = [0, 1, 2] * 4
kNuBars = [1] *6 + [-1] * 6

print 'read in events and copy to GPU'
start_t = time.time()
events_dict = {}
for flav, kFlav, kNuBar in zip(flavs, kFlavs, kNuBars):
    events_dict[flav] = {}
    # neutrinos: 1, anti-neutrinos: -1 
    events_dict[flav]['kNuBar'] = kNuBar
    # electron: 0, muon: 1, tau: 2
    events_dict[flav]['kFlav'] = kFlav
    # host arrays
    events_dict[flav]['host'] = {}
    for var in variables:
        events_dict[flav]['host'][var] = evts[flav][var].astype(FTYPE)
    events_dict[flav]['n_evts'] = np.uint32(len(events_dict[flav]['host'][variables[0]]))
    for var in empty:
        events_dict[flav]['host'][var] = np.zeros(events_dict[flav]['n_evts'], dtype=FTYPE)
    # calulate layers
    events_dict[flav]['host']['numLayers'], events_dict[flav]['host']['densityInLayer'], events_dict[flav]['host']['distanceInLayer'] = osc.calc_Layers(events_dict[flav]['host']['true_coszen'])
    # copy to device arrays
    events_dict[flav]['device'] = copy_dict_to_d(events_dict[flav]['host'])
end_t = time.time()
print 'copy done in %.4f ms'%((end_t - start_t) * 1000)
# ------

for i in range(1):
# --- do the calculation ---
    print 'retreive weighted histo'
    start_t = time.time()
    osc.update_MNS(theta12, theta13, theta23, deltam21, deltam31, deltacp)
    tot = 0
    for flav in flavs:
        osc.calc_probs(events_dict[flav]['kNuBar'], events_dict[flav]['kFlav'], events_dict[flav]['n_evts'], **events_dict[flav]['device'])
        weight.calc_weight(events_dict[flav]['n_evts'], livetime=livetime, pid_bound=pid_bound, pid_remove=pid_remove, aeff_scale=aeff_scale, nue_numu_ratio=nue_numu_ratio, nu_nubar_ratio=nu_nubar_ratio, **events_dict[flav]['device'])
        events_dict[flav]['hist_cscd'] = histogrammer.get_hist(events_dict[flav]['n_evts'], events_dict[flav]['device']['reco_coszen'], events_dict[flav]['device']['reco_energy'], events_dict[flav]['device']['weight_cscd'])
        events_dict[flav]['hist_trck'] = histogrammer.get_hist(events_dict[flav]['n_evts'], events_dict[flav]['device']['reco_coszen'], events_dict[flav]['device']['reco_energy'], events_dict[flav]['device']['weight_trck'])
        tot += events_dict[flav]['n_evts']
    end_t = time.time()
    print 'GPU done in %.4f ms for %s events'%(((end_t - start_t) * 1000),tot)
# ------


# --- CAKE stuff ---
from pisa import ureg, Q_
from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.core.map import Map, MapSet
from pisa.utils.plotter import plotter


e_binning = OneDimBinning(name='energy', tex=r'$E_\nu$', 
                               bin_edges=bin_edges_e*ureg.GeV, is_log=True)
cz_binning = OneDimBinning(name='coszen', tex=r'$\cos\,\theta$', 
                               bin_edges=bin_edges_cz, is_lin=True)
binning = MultiDimBinning([cz_binning, e_binning])

maps = []
for flav in flavs:
    maps.append(Map(name='%s_cscd'%flav, hist=events_dict[flav]['hist_cscd'], binning=binning))
    maps.append(Map(name='%s_trck'%flav, hist=events_dict[flav]['hist_trck'], binning=binning))

template = MapSet(maps,name='test')

nutau_cc_cscd = template.pop('nutau_cc_cscd') + template.pop('nutaubar_cc_cscd')
nutau_cc_trck = template.pop('nutau_cc_trck') + template.pop('nutaubar_cc_trck')
nutau_cc_all = nutau_cc_trck + nutau_cc_cscd

cscd = sum([map for map in template if map.name.endswith('cscd')])
trck = sum([map for map in template if map.name.endswith('trck')])
all = cscd + trck

m = MapSet((nutau_cc_cscd/cscd.sqrt(), nutau_cc_trck/trck.sqrt(),
            nutau_cc_all/all.sqrt()))
m[0].tex = 'cascades'
m[1].tex = 'tracks'
m[2].tex = 'all'

my_plotter = plotter(stamp='GPU MC test', outdir='.', fmt='pdf', log=False,
        label=r'$s/\sqrt{b}$')
my_plotter.plot_2d_array(m, fname='nutau_test',cmap='OrRd')

my_plotter = plotter(stamp='GPU MC test', outdir='.',fmt='pdf', log=False)
my_plotter.plot_2d_array(template, fname='GPU_test',cmap='OrRd')
