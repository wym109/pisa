import os
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
from pisa.utils.const import FTYPE
from pisa.core.events import Events

class GPUweight(object):
    # This is as ugly as it gets, from copying python code from oscfit and c++-izing it
    
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

            __device__ fType sign(fType x) { 
		// why tf is there no signum function in math.h ??
		int sgn; 
		if (x > 0.0) sgn= 1.; 
		if (x < 0.0) sgn= -1.; 
		if (x == 0.0) sgn= 0.; 
		return sgn; 
	    } 

            __device__ fType spectral_index_scale(fType true_energy, fType egy_pivot, fType delta_index){
                    fType scale = pow((true_energy/egy_pivot),delta_index);
                    return scale;
                }

	    // These parameters are obtained from fits to the paper of Barr
	    // E dependent ratios, max differences per flavor (Fig.7)
	    __device__ fType e1max_mu = 3.;
	    __device__ fType e2max_mu = 43;
	    __device__ fType e1max_e  = 2.5;
	    __device__ fType e2max_e  = 10;
	    __device__ fType e1max_mu_e = 0.62;
	    __device__ fType e2max_mu_e = 11.45;
	    // Evaluated at
	    __device__ fType x1e = 0.5;
	    __device__ fType x2e = 3.;

	    // Zenith dependent amplitude, max differences per flavor (Fig. 9)
	    __device__ fType z1max_mu = 0.6;
	    __device__ fType z2max_mu = 5.;
	    __device__ fType z1max_e  = 0.3;
	    __device__ fType z2max_e  = 5.;
	    __device__ fType nue_cutoff  = 650.;
	    __device__ fType numu_cutoff = 1000.;
	    // Evaluated at
	    __device__ fType x1z = 0.5;
	    __device__ fType x2z = 2.;

            __device__ fType LogLogParam(fType energy, fType y1, fType y2, fType x1, fType x2, bool use_cutoff, fType cutoff_value){
                fType nu_nubar = sign(y2);
                if (nu_nubar == 0.0) nu_nubar = 1.;
                y1 = sign(y1)*log10(abs(y1)+0.0001);
                y2 = log10(abs(y2+0.0001));
                fType modification = nu_nubar*pow(10.,(((y2-y1)/(x2-x1))*(log10(energy)-x1)+y1-2.));
                if (use_cutoff) modification *= exp(-1.*energy/cutoff_value);
                return modification;
            }

            __device__ fType norm_fcn(fType x, fType A, fType sigma){
                return A/sqrt(2*M_PI*pow(sigma,2)) * exp(-pow(x,2)/(2*pow(sigma,2)));
            }

            __device__ fType ModNuMuFlux(fType energy, fType czenith, fType e1, fType e2, fType z1, fType z2){
		fType A_ave = LogLogParam(energy, e1max_mu*e1, e2max_mu*e2, x1e, x2e, false, 0);
		fType A_shape = 2.5*LogLogParam(energy, z1max_mu*z1, z2max_mu*z2, x1z, x2z, true, numu_cutoff);
		return A_ave - (norm_fcn(czenith, A_shape, 0.32) - 0.75*A_shape);
            }

            __device__ fType ModNuEFlux(fType energy, fType czenith, fType e1mu, fType e2mu, fType z1mu, fType z2mu, fType e1e, fType e2e, fType z1e, fType z2e){
		fType A_ave = LogLogParam(energy, e1max_mu*e1mu + e1max_e*e1e, e2max_mu*e2mu + e2max_e*e2e, x1e, x2e, false, 0);
		fType A_shape = 1.*LogLogParam(energy, z1max_mu*z1mu + z1max_e*z1e, z2max_mu*z2mu + z2max_e*z2e, x1z, x2z, true, nue_cutoff);
		return A_ave - (1.5*norm_fcn(czenith, A_shape, 0.4) - 0.7*A_shape);
            }

            __device__ fType modRatioUpHor(const int kFlav, fType true_energy, fType true_coszen, fType uphor) {
                fType A_shape;
                if (kFlav == 0){
                    A_shape = 1.*abs(uphor)*LogLogParam(true_energy, (z1max_e+z1max_mu),(z2max_e+z2max_mu),x1z, x2z, true, nue_cutoff);
                }
                if (kFlav == 1){
                    A_shape = 1.*abs(uphor)*LogLogParam(true_energy, z1max_mu, z2max_mu, x1z, x2z, true, numu_cutoff);
                }
                return 1-3.5*sign(uphor)*norm_fcn(true_coszen, A_shape, 0.35);
            }

            __device__ fType modRatioNuBar(const int kNuBar, const int kFlav, fType true_e, fType true_cz, fType nu_nubar, fType nubar_sys){
		//not sure what nu_nubar is, only found this line in the documentation:
		// +1 applies the change to neutrinos, 0 to antineutrinos. Anything in between is shared
                fType modfactor;
		if (kFlav == 0){
		    modfactor = nubar_sys * ModNuEFlux(true_e, true_cz, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
                }
		if (kFlav == 1){
		    modfactor = nubar_sys * ModNuMuFlux(true_e, true_cz, 1.0, 1.0,1.0,1.0);
                }
		if (kNuBar < 0){
		    return 1./(1+(1-nu_nubar)*modfactor);
                }
		if (kNuBar > 0){
		    return 1. + modfactor*nu_nubar;
                }
            }

 
            __global__ void weights(const int n_evts, fType *weighted_aeff, fType *true_energy, fType *true_coszen,
                                    fType *neutrino_nue_flux, fType *neutrino_numu_flux,
                                    fType *neutrino_oppo_nue_flux, fType *neutrino_oppo_numu_flux,
                                    fType *linear_fit_MaCCQE, fType *quad_fit_MaCCQE,
                                    fType *linear_fit_MaCCRES, fType *quad_fit_MaCCRES,
                                    fType *prob_e, fType *prob_mu, fType *pid, fType *weight_cscd, fType *weight_trck,
                                    fType livetime, fType pid_bound, fType pid_remove, fType aeff_scale,
                                    fType nue_numu_ratio, fType nu_nubar_ratio, const int kNuBar, fType delta_index,
                                    fType Barr_uphor_ratio, fType Barr_nu_nubar_ratio,
                                    fType Genie_Ma_QE, fType Genie_Ma_RES
                                    )
                {
                    int idx = threadIdx.x + blockDim.x * blockIdx.x;
                    if (idx < n_evts) {

                        //apply flux systematics
                        // nue/numu ratio
                        // for neutrinos
                        fType idx_scale = spectral_index_scale(true_energy[idx], 24.0900951261, delta_index);
                        
                        fType scaled_nue_flux, scaled_numu_flux;
                        apply_ratio_scale(neutrino_nue_flux[idx], neutrino_numu_flux[idx], nue_numu_ratio, false,
                                            scaled_nue_flux, scaled_numu_flux);
                        // and the opposite (bar) type
                        fType scaled_nue_oppo_flux, scaled_numu_oppo_flux;
                        apply_ratio_scale(neutrino_oppo_nue_flux[idx], neutrino_oppo_numu_flux[idx], nue_numu_ratio, false,
                                            scaled_nue_oppo_flux, scaled_numu_oppo_flux);
                        // nu/nubar ratio
                        fType scaled_nue_flux2, scaled_nue_oppo_flux2;
                        fType scaled_numu_flux2, scaled_numu_oppo_flux2;
                        if (kNuBar < 0){
                            apply_ratio_scale(scaled_nue_oppo_flux, scaled_nue_flux, nu_nubar_ratio, false,
                                                scaled_nue_oppo_flux2, scaled_nue_flux2);
                            apply_ratio_scale(scaled_numu_oppo_flux, scaled_numu_flux, nu_nubar_ratio, false,
                                                scaled_numu_oppo_flux2, scaled_numu_flux2);
                        }
                        else {
                            apply_ratio_scale(scaled_nue_flux, scaled_nue_oppo_flux, nu_nubar_ratio, false,
                                                scaled_nue_flux2, scaled_nue_oppo_flux2);
                            apply_ratio_scale(scaled_numu_flux, scaled_numu_oppo_flux, nu_nubar_ratio, false,
                                                scaled_numu_flux2, scaled_numu_oppo_flux2);
                        }
                        // Barr flux
                        scaled_nue_flux2 *= modRatioNuBar(kNuBar, 0, true_energy[idx], true_coszen[idx], 1.0, Barr_nu_nubar_ratio);
                        scaled_nue_flux2 *= modRatioUpHor(0, true_energy[idx], true_coszen[idx], Barr_uphor_ratio);
                        scaled_numu_flux2 *= modRatioNuBar(kNuBar, 1, true_energy[idx], true_coszen[idx], 1.0, Barr_nu_nubar_ratio);
                        scaled_numu_flux2 *= modRatioUpHor(1, true_energy[idx], true_coszen[idx], Barr_uphor_ratio);

                        // GENIE axial mass sys
                        fType aeff_QE =  1. + quad_fit_MaCCQE[idx]*pow(Genie_Ma_QE,2) + linear_fit_MaCCQE[idx]*Genie_Ma_QE;
                        fType aeff_RES =  1. + quad_fit_MaCCRES[idx]*pow(Genie_Ma_RES,2) + linear_fit_MaCCRES[idx]*Genie_Ma_RES;
                        
                        // calc weight
                        fType w = idx_scale * aeff_scale * livetime * weighted_aeff[idx] * aeff_QE * aeff_RES *
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


    def calc_weight(self, n_evts, weighted_aeff, true_energy, true_coszen,
                    neutrino_nue_flux, neutrino_numu_flux,
                    neutrino_oppo_nue_flux, neutrino_oppo_numu_flux,
                    linear_fit_MaCCQE, quad_fit_MaCCQE,
                    linear_fit_MaCCRES, quad_fit_MaCCRES,
                    prob_e, prob_mu, pid, weight_cscd, weight_trck,
                    livetime, pid_bound, pid_remove, aeff_scale,
                    nue_numu_ratio, nu_nubar_ratio, kNuBar, delta_index,
                    Barr_uphor_ratio, Barr_nu_nubar_ratio,
                    Genie_Ma_QE, Genie_Ma_RES,
                    **kwargs):
        # block and grid dimensions
        bdim = (256,1,1)
        dx, mx = divmod(n_evts, bdim[0])
        gdim = ((dx + (mx>0)) * bdim[0], 1)
        self.weights_fun(n_evts, weighted_aeff, true_energy, true_coszen,
                            neutrino_nue_flux, neutrino_numu_flux,
                            neutrino_oppo_nue_flux, neutrino_oppo_numu_flux,
                            linear_fit_MaCCQE, quad_fit_MaCCQE,
                            linear_fit_MaCCRES, quad_fit_MaCCRES,
                            prob_e, prob_mu, pid, weight_cscd, weight_trck,
                            FTYPE(livetime), FTYPE(pid_bound), FTYPE(pid_remove), FTYPE(aeff_scale),
                            FTYPE(nue_numu_ratio), FTYPE(nu_nubar_ratio), np.int32(kNuBar), FTYPE(delta_index),  
                            FTYPE(Barr_uphor_ratio), FTYPE(Barr_nu_nubar_ratio),
                            FTYPE(Genie_Ma_QE), FTYPE(Genie_Ma_RES),
                            block=bdim, grid=gdim)

    def calc_sumw2(self, n_evts, weight_cscd, weight_trck, sumw2_cscd, sumw2_trck, **kwargs):
        bdim = (256,1,1)
        dx, mx = divmod(n_evts, bdim[0])
        gdim = ((dx + (mx>0)) * bdim[0], 1)
        self.sumw2_fun(n_evts, weight_cscd, weight_trck, sumw2_cscd, sumw2_trck, block=bdim, grid=gdim)

