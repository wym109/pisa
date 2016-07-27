import os
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
from pisa.utils.events import Events

class GPUweight(object):
    
    def __init__(self):
        kernel_template = """//CUDA//
          #include "constants.h"
          #include "utils.h"
          
          __global__ void weights(const int n_evts, fType *weighted_aeff, fType *neutrino_nue_flux, fType *neutrino_numu_flux, fType *prob_e, fType *prob_mu, fType *weight)
          {
            int idx = threadIdx.x + blockDim.x * blockIdx.x;
            if (idx < n_evts) {
               weight[idx] = 4 * 31557600 * weighted_aeff[idx] * ((neutrino_nue_flux[idx] * prob_e[idx]) + (neutrino_numu_flux[idx] * prob_mu[idx]));
            }
          }
          """
        include_path = os.path.expandvars('$PISA/pisa/stages/osc/grid_propagator/')
        module = SourceModule(kernel_template, include_dirs=[include_path], keep=True)
        self.weights_fun = module.get_function("weights")


    def calc_weight(self, n_evts, weighted_aeff, neutrino_nue_flux, neutrino_numu_flux, prob_e, prob_mu, weight, **kwargs):
        # block and grid dimensions
        bdim = (256,1,1)
        dx, mx = divmod(n_evts, bdim[0])
        gdim = ((dx + (mx>0)) * bdim[0], 1)
        self.weights_fun(n_evts, weighted_aeff, neutrino_nue_flux, neutrino_numu_flux, prob_e, prob_mu, weight, block=bdim, grid=gdim)
