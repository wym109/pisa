import sys, os
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

from pisa.utils.resources import find_resource
from pisa.utils.log import logging
from pisa.devel.prob3gpu import Prob3GPU
from pisa.devel.hist import GPUhist
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

FTYPE = np.float64

def copy_dict_to_d(events):
    d_events = {}
    for key, val in events.items():
        d_events['d_%s'%key] = cuda.mem_alloc(val.nbytes)
        cuda.memcpy_htod(d_events['d_%s'%key], val)
    return d_events

events = {}
e = np.linspace(1,100,1000).astype(FTYPE)[:-1]
cz = np.linspace(-1,1,1000).astype(FTYPE)[:-1]
ee, czcz = np.meshgrid(e,cz)
events['energy'] = ee.ravel()
events['coszen'] = czcz.ravel()
n_evts = np.uint32(len(events['coszen']))
events['prob_e'] = np.zeros(n_evts, dtype=FTYPE)
events['prob_mu'] = np.zeros(n_evts, dtype=FTYPE)
# neutrinos: 1, anti-neutrinos: -1 
kNuBar = np.int32(1)
# electron: 0, muon: 1, tau: 2
kFlav = np.int32(2)

# layer params
detector_depth = 2.0
earth_model = find_resource('osc/PREM_12layer.dat')
prop_height = 20.0
YeI = 0.4656
YeO = 0.4656
YeM = 0.4957

osc = Prob3GPU(detector_depth, earth_model, prop_height,  YeI, YeO, YeM)

# SETUP ARRAYS
# calulate layers
events['numLayers'], events['densityInLayer'], events['distanceInLayer'] = osc.calc_Layers(events['coszen'])

d_events = copy_dict_to_d(events)

# SETUP MNS
theta12 = 0.5839958715755919
theta13 = 0.14819001778459273
theta23 = 0.7373241279447564
deltam21 = 7.5e-05
deltam31 = 0.002457
deltacp = 5.340707511102648
osc.update_MNS(theta12, theta13, theta23, deltam21, deltam31, deltacp)

osc.calc_probs(kNuBar, kFlav, n_evts, **d_events)

cuda.memcpy_dtoh(events['prob_e'],d_events['d_prob_e'])
#cuda.memcpy_dtoh(events['prob_mu'],d_events['d_prob_mu'])
#print events['prob_e'], events['prob_mu']

#histogram from GPU

bin_edges_e = np.linspace(1,100,1000).astype(FTYPE)
bin_edges_cz = np.linspace(-1,1,1000).astype(FTYPE)

histogrammer = GPUhist(bin_edges_e, bin_edges_cz)
hist2d = histogrammer.get_hist(n_evts, d_events['d_energy'], d_events['d_coszen'], d_events['d_prob_mu'])

#cuda.memcpy_dtoh(events['energy'],d_events['d_energy'])
#cuda.memcpy_dtoh(events['coszen'],d_events['d_coszen'])
#print events['coszen']
#print events['energy']

#np_hist2d,_,_ = np.histogram2d(events['energy'], events['coszen'],bins=(bin_edges_e, bin_edges_cz), weights=events['prob_mu'])
#print hist2d
#print np_hist2d
#print hist2d - np_hist2d
#assert (np.sum(hist2d - np_hist2d) == 0.)
plt.imshow(hist2d,origin='lower',interpolation='nearest')
plt.show()
plt.savefig('test.pdf')
