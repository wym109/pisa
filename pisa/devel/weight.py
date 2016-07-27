import os
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
from pisa.devel.const import FTYPE
from pisa.utils.events import Events

class GPUweight(object):
    
    def __init__(self):
        kernel_template = """//CUDA//
          #include "constants.h"
          #include "utils.h"

	  __device__ void apply_ratio_scale(fType flux1, fType flux2, fType ratio_scale, bool sum_const, fType &scaled_flux1, fType &scaled_flux2){
		if (sum_const){
		    // keep sum of flux1, flux2 constant
		    fType orig_ratio = flux1/flux2;
		    fType orig_sum = flux1 + flux2;
		    scaled_flux2 = orig_sum / (1 + ratio_scale*orig_ratio);
		    scaled_flux1 = ratio_scale*orig_ratio*scaled_flux2;
		    }
		else {
		    // don't keep sum of flux1, flux2 constant
		    scaled_flux1 = ratio_scale*flux1;
                    scaled_flux2 = flux2;
		    }
		}
 
          
          __global__ void weights(const int n_evts, fType *weighted_aeff, fType *neutrino_nue_flux, fType *neutrino_numu_flux,
                                    fType *prob_e, fType *prob_mu, fType *pid, fType *weight_cscd, fType *weight_trck,
                                    fType livetime, fType pid_bound, fType pid_remove, fType aeff_scale, fType nue_numu_ratio, fType nu_nubar_ratio)
                {
                    int idx = threadIdx.x + blockDim.x * blockIdx.x;
                    if (idx < n_evts) {

                        //apply flux systematics
                        fType scaled_nue_flux, scaled_numu_flux;
                        apply_ratio_scale(neutrino_nue_flux[idx], neutrino_numu_flux[idx], nue_numu_ratio, true, scaled_nue_flux, scaled_numu_flux);

                        fType w = aeff_scale * livetime * 31557600 * weighted_aeff[idx] * ((scaled_nue_flux * prob_e[idx]) + (scaled_numu_flux * prob_mu[idx]));
                        weight_cscd[idx] = ((pid[idx] < pid_bound) && (pid[idx] >= pid_remove)) * w;
                        weight_trck[idx] = (pid[idx] >= pid_bound) * w;
                    }
                }
          """
        include_path = os.path.expandvars('$PISA/pisa/stages/osc/grid_propagator/')
        module = SourceModule(kernel_template, include_dirs=[include_path], keep=True)
        self.weights_fun = module.get_function("weights")


    def calc_weight(self, n_evts, weighted_aeff, neutrino_nue_flux, neutrino_numu_flux,
                    prob_e, prob_mu, pid, weight_cscd, weight_trck,
                    livetime, pid_bound, pid_remove, aeff_scale, nue_numu_ratio, nu_nubar_ratio, **kwargs):
        # block and grid dimensions
        bdim = (256,1,1)
        dx, mx = divmod(n_evts, bdim[0])
        gdim = ((dx + (mx>0)) * bdim[0], 1)
        self.weights_fun(n_evts, weighted_aeff, neutrino_nue_flux, neutrino_numu_flux,
                            prob_e, prob_mu, pid, weight_cscd, weight_trck,
                            FTYPE(livetime), FTYPE(pid_bound), FTYPE(pid_remove), FTYPE(aeff_scale),
                            FTYPE(nue_numu_ratio), FTYPE(nu_nubar_ratio), block=bdim, grid=gdim)
