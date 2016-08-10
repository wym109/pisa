import os
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
from pisa.utils.const import FTYPE
from pisa.utils.events import Events

class GPUweight(object):
    
    def __init__(self):
        kernel_template = """//CUDA//
          #include "constants.h"
          #include "utils.h"
          #include "math.h"

	  __device__ void apply_ratio_scale(fType flux1, fType flux2,
                                            fType ratio_scale, bool sum_const,
                                            fType &scaled_flux1, fType &scaled_flux2){
              // apply a ratio for two fluxes, taken from pisa v2
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

            __device__ fType spectral_index_scale(fType true_energy, fType egy_pivot, fType delta_index){
                    fType scale = pow((true_energy/egy_pivot),delta_index);
                    return scale;
                }
 
            __global__ void weights(const int n_evts, fType *weighted_aeff, fType *true_energy,
                                    fType *neutrino_nue_flux, fType *neutrino_numu_flux,
                                    fType *neutrino_oppo_nue_flux, fType *neutrino_oppo_numu_flux,
                                    fType *prob_e, fType *prob_mu, fType *pid, fType *weight_cscd, fType *weight_trck,
                                    fType livetime, fType pid_bound, fType pid_remove, fType aeff_scale,
                                    fType nue_numu_ratio, fType nu_nubar_ratio, const int kNuBar, fType delta_index)
                {
                    int idx = threadIdx.x + blockDim.x * blockIdx.x;
                    if (idx < n_evts) {

                        //apply flux systematics
                        // nue/numu ratio
                        // for neutrinos
                        fType idx_scale = spectral_index_scale(true_energy[idx], 24.0900951261, delta_index);
                        
                        fType scaled_nue_flux, scaled_numu_flux;
                        apply_ratio_scale(neutrino_nue_flux[idx], neutrino_numu_flux[idx], nue_numu_ratio, true,
                                            scaled_nue_flux, scaled_numu_flux);
                        // and the opposite (bar) type
                        fType scaled_nue_oppo_flux, scaled_numu_oppo_flux;
                        apply_ratio_scale(neutrino_oppo_nue_flux[idx], neutrino_oppo_numu_flux[idx], nue_numu_ratio, true,
                                            scaled_nue_oppo_flux, scaled_numu_oppo_flux);
                        // nu/nubar ratio
                        fType scaled_nue_flux2, scaled_nue_oppo_flux2;
                        fType scaled_numu_flux2, scaled_numu_oppo_flux2;
                        if (kNuBar < 0){
                            apply_ratio_scale(scaled_nue_oppo_flux, scaled_nue_flux, nu_nubar_ratio, true,
                                                scaled_nue_oppo_flux2, scaled_nue_flux2);
                            apply_ratio_scale(scaled_numu_oppo_flux, scaled_numu_flux, nu_nubar_ratio, true,
                                                scaled_numu_oppo_flux2, scaled_numu_flux2);
                        }
                        else {
                            apply_ratio_scale(scaled_nue_flux, scaled_nue_oppo_flux, nu_nubar_ratio, true,
                                                scaled_nue_flux2, scaled_nue_oppo_flux2);
                            apply_ratio_scale(scaled_numu_flux, scaled_numu_oppo_flux, nu_nubar_ratio, true,
                                                scaled_numu_flux2, scaled_numu_oppo_flux2);
                        }
                        
                        // calc weight
                        fType w = idx_scale * aeff_scale * livetime * weighted_aeff[idx] *
                                 ((scaled_nue_flux2 * prob_e[idx]) + (scaled_numu_flux2 * prob_mu[idx]));
                        // distinguish between PID classes
                        weight_cscd[idx] = ((pid[idx] < pid_bound) && (pid[idx] >= pid_remove)) * w;
                        weight_trck[idx] = (pid[idx] >= pid_bound) * w;
                    }
                }

            __global__ void sumw2(const int n_evts, fType *weight_cscd, fType *weight_trck,
                                    fType *sumw2_cscd, fType *sumw2_trck) {
                    int idx = threadIdx.x + blockDim.x * blockIdx.x;
                    if (idx < n_evts) {
                        sumw2_cscd[idx] = weight_cscd[idx] * weight_cscd[idx];
                        sumw2_trck[idx] = weight_trck[idx] * weight_trck[idx];
                    }
                }
          """
        include_path = os.path.expandvars('$PISA/pisa/stages/osc/grid_propagator/')
        module = SourceModule(kernel_template, include_dirs=[include_path], keep=True)
        self.weights_fun = module.get_function("weights")
        self.sumw2_fun = module.get_function("sumw2")


    def calc_weight(self, n_evts, weighted_aeff, true_energy,
                    neutrino_nue_flux, neutrino_numu_flux,
                    neutrino_oppo_nue_flux, neutrino_oppo_numu_flux,
                    prob_e, prob_mu, pid, weight_cscd, weight_trck,
                    livetime, pid_bound, pid_remove, aeff_scale,
                    nue_numu_ratio, nu_nubar_ratio, kNuBar, delta_index, **kwargs):
        # block and grid dimensions
        bdim = (256,1,1)
        dx, mx = divmod(n_evts, bdim[0])
        gdim = ((dx + (mx>0)) * bdim[0], 1)
        self.weights_fun(n_evts, weighted_aeff, true_energy, 
                            neutrino_nue_flux, neutrino_numu_flux,
                            neutrino_oppo_nue_flux, neutrino_oppo_numu_flux,
                            prob_e, prob_mu, pid, weight_cscd, weight_trck,
                            FTYPE(livetime), FTYPE(pid_bound), FTYPE(pid_remove), FTYPE(aeff_scale),
                            FTYPE(nue_numu_ratio), FTYPE(nu_nubar_ratio), np.int32(kNuBar), FTYPE(delta_index),  block=bdim, grid=gdim)

    def calc_sumw2(self, n_evts, weight_cscd, weight_trck, sumw2_cscd, sumw2_trck, **kwargs):
        bdim = (256,1,1)
        dx, mx = divmod(n_evts, bdim[0])
        gdim = ((dx + (mx>0)) * bdim[0], 1)
        self.sumw2_fun(n_evts, weight_cscd, weight_trck, sumw2_cscd, sumw2_trck, block=bdim, grid=gdim)

