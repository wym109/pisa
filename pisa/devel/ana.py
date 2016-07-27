import sys, os
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pisa.utils.resources import find_resource
from pisa.utils.log import logging
from pisa.devel.prob3gpu import Prob3GPU
from pisa.devel.hist import GPUhist
from pisa.devel.weight import GPUweight
from pisa.utils.events import Events
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import time

FTYPE = np.float64

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
# osc
theta12 = 0.5839958715755919
theta13 = 0.14819001778459273
theta23 = 0.7373241279447564
deltam21 = 7.5e-05
deltam31 = 0.002457
deltacp = 5.340707511102648
# histo bins
bin_edges_e = np.logspace(0,2,500).astype(FTYPE)
bin_edges_cz = np.linspace(-1,1,500).astype(FTYPE)
# ------

# initialize classes
osc = Prob3GPU(detector_depth, earth_model, prop_height,  YeI, YeO, YeM)
weight = GPUweight()
#histogram from GPU
histogrammer = GPUhist(bin_edges_e, bin_edges_cz)

# --- Load events
# open Events file
evts = Events(fname)

# Load and copy events
events = {}
events['true_energy'] = evts['nue_cc']['true_energy'].astype(FTYPE)
events['true_coszen'] = evts['nue_cc']['true_coszen'].astype(FTYPE)
events['reco_energy'] = evts['nue_cc']['reco_energy'].astype(FTYPE)
events['reco_coszen'] = evts['nue_cc']['reco_coszen'].astype(FTYPE)
events['neutrino_nue_flux'] = evts['nue_cc']['neutrino_nue_flux'].astype(FTYPE)
events['neutrino_numu_flux'] = evts['nue_cc']['neutrino_numu_flux'].astype(FTYPE)
events['weighted_aeff'] = evts['nue_cc']['weighted_aeff'].astype(FTYPE)
n_evts = np.uint32(len(events['true_coszen']))
events['prob_e'] = np.zeros(n_evts, dtype=FTYPE)
events['prob_mu'] = np.zeros(n_evts, dtype=FTYPE)
events['weight'] = np.zeros(n_evts, dtype=FTYPE)
# neutrinos: 1, anti-neutrinos: -1 
kNuBar = np.int32(1)
# electron: 0, muon: 1, tau: 2
kFlav = np.int32(0)
# calulate layers
events['numLayers'], events['densityInLayer'], events['distanceInLayer'] = osc.calc_Layers(events['true_coszen'])
# copy to GPU
print 'copy to GPU'
d_events = copy_dict_to_d(events)
print 'retreive weighted histo'
# ------

# --- do the calculation ---
start_t = time.time()
osc.update_MNS(theta12, theta13, theta23, deltam21, deltam31, deltacp)
osc.calc_probs(kNuBar, kFlav, n_evts, **d_events)
weight.calc_weight(n_evts, **d_events)
hist2d = histogrammer.get_hist(n_evts, d_events['reco_energy'], d_events['reco_coszen'], d_events['weight'])
end_t = time.time()
print 'GPU done in %.4f ms for %s events'%(((end_t - start_t) * 1000),n_evts)
# ------

plt.imshow(hist2d,origin='lower',interpolation='nearest')
plt.show()
plt.savefig('test.pdf')
