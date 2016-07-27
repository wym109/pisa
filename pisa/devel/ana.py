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
bin_edges_e = np.logspace(0,2,50).astype(FTYPE)
bin_edges_cz = np.linspace(-1,1,50).astype(FTYPE)
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
variables = ['true_energy', 'true_coszen', 'reco_energy', 'reco_coszen', 'neutrino_nue_flux', 'neutrino_numu_flux', 'weighted_aeff']
empty = ['prob_e', 'prob_mu', 'weight']
flavs = ['nue_cc', 'numu_cc', 'nutau_cc', 'nue_nc', 'numu_nc', 'nutau_nc', 'nuebar_cc', 'numubar_cc', 'nutaubar_cc', 'nuebar_nc', 'numubar_nc', 'nutaubar_nc']
kFlavs = [0, 1, 2] * 4
kNuBars = [1] *6 + [-1] * 6

print 'read in events and copy to GPU'
events_dict = {}
for flav, kFlav, kNuBar in zip(flavs, kFlavs, kNuBars):
    events_dict[flav] = {}
    # neutrinos: 1, anti-neutrinos: -1 
    events_dict[flav]['kNuBar'] = np.int32(kNuBar)
    # electron: 0, muon: 1, tau: 2
    events_dict[flav]['kFlav'] = np.int32(kFlav)
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
# ------


# --- do the calculation ---
print 'retreive weighted histo'
start_t = time.time()
osc.update_MNS(theta12, theta13, theta23, deltam21, deltam31, deltacp)
for flav in flavs:
    osc.calc_probs(events_dict[flav]['kNuBar'], events_dict[flav]['kFlav'], events_dict[flav]['n_evts'], **events_dict[flav]['device'])
    weight.calc_weight(events_dict[flav]['n_evts'], **events_dict[flav]['device'])
    events_dict[flav]['hist'] = histogrammer.get_hist(events_dict[flav]['n_evts'], events_dict[flav]['device']['reco_energy'], events_dict[flav]['device']['reco_coszen'], events_dict[flav]['device']['weight'])
end_t = time.time()
print 'GPU done in %.4f ms'%((end_t - start_t) * 1000)
# ------


for flav in flavs:
    plt.imshow(events_dict[flav]['hist'],origin='lower',interpolation='nearest')
    plt.show()
    plt.savefig('%s.pdf'%flav)
