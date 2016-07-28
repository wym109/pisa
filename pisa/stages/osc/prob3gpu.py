# calculate the osc probabilities for an array of events
#
# authors: Philipp Eller
#          pde3@psu.edu
#          Timothy C. Arlen
#          tca3@psu.edu
#
# NOTE: 2015-05-21 (TCA) attempted to use single precision, and at
# least on my system, I got all junk in the output of my osc prob
# maps. Unfortunately, I don't want to spend the time right now to
# figure out WHY this is the case, but until someone figures this out,
# keep fType to double and np.float64.

import sys, os
import numpy as np

from pisa.stages.osc.grid_propagator.GridPropagator import GridPropagator
from pisa.utils.resources import find_resource
from pisa.utils.log import logging

# Put CUDA imports in the constructor
import pycuda.driver as cuda
import pycuda.compiler
from pycuda.compiler import SourceModule

class Prob3GPU(object):
    """
    This class handles all tasks related to the oscillation
    probability calculations using the prob3_GPU oscillation code,
    which is identical to the prob3 code, but adapted for use on
    the GPU.
    """
    def __init__(self, detector_depth=None, earth_model=None, prop_height=None,  YeI=0., YeO=0., YeM=0.):
        self.FTYPE = np.float64
        self.initialize_kernel()
        self.detector_depth = detector_depth
        self.earth_model = earth_model
        self.prop_height = prop_height
        self.YeI = YeI
        self.YeO = YeO
        self.YeM = YeM
        self.grid_prop = None

    def calc_Layers(self, coszen):
        """
        \params:
          * energy: array of energies in GeV
          * coszen: array of coszen values
          * kNuBar: +1 for neutrinos, -1 for anti neutrinos
          * earth_model: Earth density model used for matter oscillations.
          * detector_depth: Detector depth in km.
          * prop_height: Height in the atmosphere to begin in km.
        """
        self.grid_prop  = GridPropagator(self.earth_model, self.FTYPE(coszen), self.detector_depth)

        self.grid_prop.SetEarthDensityParams(self.prop_height,self.YeI,self.YeO,self.YeM)

        n_evts = np.uint32(len(coszen))

        self.maxLayers  = self.grid_prop.GetMaxLayers()
        numLayers       = np.zeros(n_evts,dtype=np.int32)
        densityInLayer  = np.zeros((n_evts*self.maxLayers),dtype=self.FTYPE)
        distanceInLayer = np.zeros((n_evts*self.maxLayers),dtype=self.FTYPE)

        self.grid_prop.GetNumberOfLayers(numLayers)
        self.grid_prop.GetDensityInLayer(densityInLayer)
        self.grid_prop.GetDistanceInLayer(distanceInLayer)

        return numLayers, densityInLayer, distanceInLayer


    def update_MNS(self, theta12, theta13, theta23, deltam21, deltam31,
                          deltacp):
        """
        Returns an oscillation probability map dictionary calculated
        at the values of the input parameters:
          deltam21,deltam31,theta12,theta13,theta23,deltacp
          * theta12,theta13,theta23 - in [rad]
          * deltam21, deltam31 - in [eV^2]
        """

        sin2th12Sq = np.sin(theta12)**2
        sin2th13Sq = np.sin(theta13)**2
        sin2th23Sq = np.sin(theta23)**2

        mAtm = deltam31 if deltam31 < 0.0 else (deltam31 - deltam21)

        # Comment BargerPropagator.cc::SetMNS()
        # "For the inverted Hierarchy, adjust the input
        # by the solar mixing (should be positive)
        # to feed the core libraries the correct value of m32."
        #if mAtm < 0.0: mAtm -= deltam21;

        self.grid_prop.SetMNS(deltam21,mAtm,sin2th12Sq,sin2th13Sq,sin2th23Sq,deltacp)

        dm_mat = np.zeros((3,3),dtype=self.FTYPE)
        self.grid_prop.Get_dm_mat(dm_mat)
        mix_mat = np.zeros((3,3,2),dtype=self.FTYPE)
        self.grid_prop.Get_mix_mat(mix_mat)

        logging.debug("dm_mat: \n %s"%str(dm_mat))
        logging.debug("mix[re]: \n %s"%str(mix_mat[:,:,0]))

        self.d_dm_mat = cuda.mem_alloc(dm_mat.nbytes)
        self.d_mix_mat = cuda.mem_alloc(mix_mat.nbytes)
        cuda.memcpy_htod(self.d_dm_mat,dm_mat)
        cuda.memcpy_htod(self.d_mix_mat,mix_mat)


    def initialize_kernel(self):
        ###############################################
        ###### DEFINE KERNEL
        ###############################################
        kernel_template = """//CUDA//
          #include "mosc.cu"
          #include "mosc3.cu"
          //#include "utils.h"
          #include "constants.h"
          #include <stdio.h>


          __global__ void propagateArray(
                                        fType* d_prob_e,
                                        fType* d_prob_mu,
                                        fType d_dm[3][3],
                                        fType d_mix[3][3][2],
                                        const int n_evts,
                                        const int kNuBar,
                                        const int kFlav,
                                        const int maxLayers,
                                        const fType* const d_energy,
                                        const int* const d_numberOfLayers,
                                        const fType* const d_densityInLayer,
                                        const fType* const d_distanceInLayer)
          {

            const int idx = blockIdx.x*blockDim.x + threadIdx.x;

            // ensure we don't access memory outside of bounds!
            if(idx >= n_evts) return;

            bool kUseMassEstates = false;

            fType TransitionMatrix[3][3][2];
            fType TransitionProduct[3][3][2];
            fType TransitionTemp[3][3][2];
            fType RawInputPsi[3][2];
            fType OutputPsi[3][2];
            fType Probability[3][3];

            clear_complex_matrix( TransitionMatrix );
            clear_complex_matrix( TransitionProduct );
            clear_complex_matrix( TransitionTemp );
            clear_probabilities( Probability );

            int layers = *(d_numberOfLayers + idx);

            fType energy = d_energy[idx];
            for( int i=0; i<layers; i++) {
              fType density = *(d_densityInLayer + idx*maxLayers + i);
              fType distance = *(d_distanceInLayer + idx*maxLayers + i);

              get_transition_matrix( kNuBar,
                                     energy,
                                     density,
                                     distance,
                                     TransitionMatrix,
                                     0.0,
                                     d_mix,
                                     d_dm);

              if(i==0) { copy_complex_matrix(TransitionMatrix, TransitionProduct);
              } else {
                clear_complex_matrix( TransitionTemp );
                multiply_complex_matrix( TransitionMatrix, TransitionProduct, TransitionTemp );
                copy_complex_matrix( TransitionTemp, TransitionProduct );
              }
            } // end layer loop

            // loop on neutrino types, and compute probability for neutrino i:
            // We actually don't care about nutau -> anything since the flux there is zero!
            for( unsigned i=0; i<2; i++) {
              for ( unsigned j = 0; j < 3; j++ ) {
                RawInputPsi[j][0] = 0.0;
                RawInputPsi[j][1] = 0.0;
              }

              if( kUseMassEstates ) convert_from_mass_eigenstate(i+1,kNuBar,RawInputPsi,d_mix);
              else RawInputPsi[i][0] = 1.0;

              // calculate 'em all here, from legacy code...
              multiply_complex_matvec( TransitionProduct, RawInputPsi, OutputPsi );
              Probability[i][0] +=OutputPsi[0][0]*OutputPsi[0][0]+OutputPsi[0][1]*OutputPsi[0][1];
              Probability[i][1] +=OutputPsi[1][0]*OutputPsi[1][0]+OutputPsi[1][1]*OutputPsi[1][1];
              Probability[i][2] +=OutputPsi[2][0]*OutputPsi[2][0]+OutputPsi[2][1]*OutputPsi[2][1];

			}

            d_prob_e[idx] = Probability[0][kFlav];
            d_prob_mu[idx] = Probability[1][kFlav];

          }
        """

        include_path = os.path.expandvars('$PISA/pisa/stages/osc/grid_propagator/')
        logging.info("  pycuda INC PATH: %s"%include_path)
        logging.info("  pycuda FLAGS: %s"%pycuda.compiler.DEFAULT_NVCC_FLAGS)
        self.module = SourceModule(kernel_template,
                                   include_dirs=[include_path],
                                   keep=True)
        self.propagateArray = self.module.get_function("propagateArray")


    def calc_probs(self, kNuBar, kFlav, n_evts, true_energy, numLayers, densityInLayer, distanceInLayer, prob_e, prob_mu, **kwargs):

        bdim = (32,1,1)
        dx, mx = divmod(n_evts, bdim[0])
        gdim = ((dx + (mx>0)) * bdim[0], 1)

        self.propagateArray(prob_e,
                      prob_mu,
                      self.d_dm_mat,
                      self.d_mix_mat,
                      n_evts,
                      np.int32(kNuBar),
                      np.int32(kFlav),
                      np.uint32(self.maxLayers),
                      true_energy,
                      numLayers,
                      densityInLayer,
                      distanceInLayer,
                      block=bdim, grid=gdim)

if __name__ == '__main__':
    import pycuda.autoinit

    def copy_dict_to_d(events):
        d_events = {}
        for key, val in events.items():
            d_events['d_%s'%key] = cuda.mem_alloc(val.nbytes)
            cuda.memcpy_htod(d_events['d_%s'%key], val)
        return d_events


    events = {}
    events['energy'] = np.linspace(1,100,100)
    events['coszen'] = np.linspace(-1,1,100)
    n_evts = np.uint32(len(events['coszen']))
    events['prob_e'] = np.zeros(n_evts, dtype=np.float64)
    events['prob_mu'] = np.zeros(n_evts, dtype=np.float64)
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
    cuda.memcpy_dtoh(events['prob_mu'],d_events['d_prob_mu'])
    print events['prob_e'], events['prob_mu']
