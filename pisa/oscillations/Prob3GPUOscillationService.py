#
# This service will perform the oscillation probability calculations on a grid
# of high resolution fine energy/coszen maps using a GPU-based implementation of
# the oscillation probability calculation.
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# date:   26 Feb 2015
#


import sys,os
import numpy as np

from pisa.oscillations.grid_propagator.GridPropagator import GridPropagator
from pisa.resources.resources import find_resource
from pisa.utils.log import logging, tprofile
from pisa.utils.proc import get_params, report_params
from pisa.utils.utils import get_bin_centers, is_coarser_binning, is_linear
from pisa.utils.utils import is_logarithmic, check_fine_binning, oversample_binning, Timer

# Put CUDA imports in the constructor
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

#
# NOTE: Unlike the other oscillation services, this GPU version does NOT inherit from
# OscillationServiceBase, because get_osc_prob_maps() is different.
#
class Prob3GPUOscillationService():
    """
    This class handles all tasks related to the oscillation
    probability calculations using the prob3_GPU oscillation code,
    which is identical to the prob3 code, but adapted for use on
    the GPU.
    """
    def __init__(self, ebins, czbins, detector_depth=None, earth_model=None,
                 prop_height=None, oversample_e=None,oversample_cz=None,gpu_id=None,
                 **kwargs):
        """
        \params:
          * ebins: Energy bin edges
          * czbins: cos(zenith) bin edges
          * earth_model: Earth density model used for matter oscillations.
          * detector_depth: Detector depth in km.
          * prop_height: Height in the atmosphere to begin in km.
          * gpu_id: If running on a system with multiple GPUs, it will choose
            the one with gpu_id. Otherwise, defaults to default context
        """

        self.gpu_id = gpu_id
        try:
            import pycuda.autoinit
            self.context = cuda.Device(self.gpu_id).make_context()
            print "Initializing PyCUDA using gpu id: %d"%self.gpu_id
        except:
            import pycuda.autoinit
            print "Auto initializing PyCUDA..."


        logging.info('Instantiating %s'%self.__class__.__name__)
        self.ebins = np.array(ebins)
        self.czbins = np.array(czbins)
        self.prop_height = prop_height
        for ax in [self.ebins, self.czbins]:
            if (len(np.shape(ax)) != 1):
                raise IndexError('Axes must be 1d! '+str(np.shape(ax)))

        report_params(get_params(),['km','','','','km'])

        earth_model = find_resource(earth_model)
        self.earth_model = earth_model
        self.FTYPE = np.float64

        self.ebins_fine = oversample_binning(self.ebins, oversample_e)
        self.czbins_fine = oversample_binning(self.czbins, oversample_cz)
        self.ecen_fine = get_bin_centers(self.ebins_fine)
        self.czcen_fine = get_bin_centers(self.czbins_fine)

        self.initialize_kernel(detector_depth,**kwargs)

        return

    def initialize_kernel(self,detector_depth,**kwargs):
        '''
        Initializes: 1) the grid_propagator class, 2) the device arrays
        that will be passed to the propagateGrid() kernel and 3) the
        kernel module.
        '''

        self.grid_prop  = GridPropagator(self.earth_model,self.czcen_fine,detector_depth)

        ###############################################
        ###### DEFINE KERNEL
        ###############################################
        kernel_template = """
          #include "mosc.cu"
          #include "mosc3.cu"
          #include "utils.h"
          #include <stdio.h>

          __global__ void propagateGrid(double* d_smooth_maps,
                                        double d_dm[3][3], double d_mix[3][3][2],
                                        const double* const d_ecen_fine,
                                        const double* const d_czcen_fine,
                                        const int nebins_fine, const int nczbins_fine,
                                        const int nebins, const int nczbins, const int maxLayers,
                                        const int* const d_numberOfLayers,
                                        const double* const d_densityInLayer,
                                        const double* const d_distanceInLayer)
          {

            const int2 thread_2D_pos = make_int2(blockIdx.x*blockDim.x + threadIdx.x,
                                                 blockIdx.y*blockDim.y + threadIdx.y);

            // ensure we don't access memory outside of bounds!
            if(thread_2D_pos.x >= nczbins_fine || thread_2D_pos.y >= nebins_fine) return;
            const int thread_1D_pos = thread_2D_pos.y*nczbins_fine + thread_2D_pos.x;

            int eidx = thread_2D_pos.y;
            int czidx = thread_2D_pos.x;

            int kNuBar;
            if(blockIdx.z == 0) kNuBar = 1;
            else kNuBar=-1;

            bool kUseMassEstates = false;

            double TransitionMatrix[3][3][2];
            double TransitionProduct[3][3][2];
            double TransitionTemp[3][3][2];
            double RawInputPsi[3][2];
            double OutputPsi[3][2];
            double Probability[3][3];

            clear_complex_matrix( TransitionMatrix );
            clear_complex_matrix( TransitionProduct );
            clear_complex_matrix( TransitionTemp );
            clear_probabilities( Probability );

            int layers = *(d_numberOfLayers + czidx);

            double energy = d_ecen_fine[eidx];
            //double coszen = d_czcen_fine[czidx];
            for( int i=0; i<layers; i++) {
              double density = *(d_densityInLayer + czidx*maxLayers + i);
              double distance = *(d_distanceInLayer + czidx*maxLayers + i);

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

              multiply_complex_matvec( TransitionProduct, RawInputPsi, OutputPsi );
              Probability[i][0] +=OutputPsi[0][0]*OutputPsi[0][0]+OutputPsi[0][1]*OutputPsi[0][1];
              Probability[i][1] +=OutputPsi[1][0]*OutputPsi[1][0]+OutputPsi[1][1]*OutputPsi[1][1];
              Probability[i][2] +=OutputPsi[2][0]*OutputPsi[2][0]+OutputPsi[2][1]*OutputPsi[2][1];

            }//end of neutrino loop

            int efctr = nebins_fine/nebins;
            int czfctr = nczbins_fine/nczbins;
            int eidx_smooth = eidx/efctr;
            int czidx_smooth = czidx/czfctr;
            double scale = double(efctr*czfctr);
            for (int i=0;i<2;i++) {
              int iMap = 0;
              if (kNuBar == 1) iMap = i*3;
              else iMap = 6 + i*3;

              for (unsigned to_nu=0; to_nu<3; to_nu++) {
                int k = (iMap+to_nu);
                double prob = Probability[i][to_nu];
                atomicAdd((d_smooth_maps + k*nczbins*nebins + eidx_smooth*nczbins +
                           czidx_smooth),prob/scale);
              }
            }

          }
        """

        include_path = os.path.expandvars('$PISA/pisa/oscillations/grid_propagator/')
        #cache_dir=os.path.expandvars('$PISA/pisa/oscillations/'+'.cache_dir')
        logging.trace("  pycuda INC PATH: %s"%include_path)
        #logging.trace("  pycuda cache_dir: %s"%cache_dir)
        #logging.trace("  pycuda FLAGS: %s"%pycuda.compiler.DEFAULT_NVCC_FLAGS)
        self.module = SourceModule(kernel_template,
                                   include_dirs=[include_path],
                                   #cache_dir=cache_dir,
                                   keep=True)
        self.propGrid = self.module.get_function("propagateGrid")

        return

    def prepare_device_arrays(self):

        self.maxLayers  = self.grid_prop.GetMaxLayers()
        nczbins_fine    = len(self.czcen_fine)
        numLayers       = np.zeros(nczbins_fine,dtype=np.int32)
        densityInLayer  = np.zeros((nczbins_fine*self.maxLayers),dtype=self.FTYPE)
        distanceInLayer = np.zeros((nczbins_fine*self.maxLayers),dtype=self.FTYPE)

        self.grid_prop.GetNumberOfLayers(numLayers)
        self.grid_prop.GetDensityInLayer(densityInLayer)
        self.grid_prop.GetDistanceInLayer(distanceInLayer)

        # Copy all these earth info arrays to device:
        self.d_numLayers       = cuda.mem_alloc(numLayers.nbytes)
        self.d_densityInLayer  = cuda.mem_alloc(densityInLayer.nbytes)
        self.d_distanceInLayer = cuda.mem_alloc(distanceInLayer.nbytes)
        cuda.memcpy_htod(self.d_numLayers,numLayers)
        cuda.memcpy_htod(self.d_densityInLayer,densityInLayer)
        cuda.memcpy_htod(self.d_distanceInLayer,distanceInLayer)

        self.d_ecen_fine = cuda.mem_alloc(self.ecen_fine.nbytes)
        self.d_czcen_fine = cuda.mem_alloc(self.czcen_fine.nbytes)
        cuda.memcpy_htod(self.d_ecen_fine,self.ecen_fine)
        cuda.memcpy_htod(self.d_czcen_fine,self.czcen_fine)

        return

    def get_osc_prob_maps(self, theta12, theta13, theta23, deltam21, deltam31,
                          deltacp, energy_scale, YeI, YeO, YeM, **kwargs):
        """
        Returns an oscillation probability map dictionary calculated
        at the values of the input parameters:
          deltam21,deltam31,theta12,theta13,theta23,deltacp
        for flavor_from to flavor_to, with the binning defined in the constructor.
        The dictionary is formatted as:
          'nue_maps': {'nue':map,'numu':map,'nutau':map},
          'numu_maps': {...}
          'nue_bar_maps': {...}
          'numu_bar_maps': {...}

        \params:
          * theta12,theta13,theta23 - in [rad]
          * deltam21, deltam31 - in [eV^2]
          * energy_scale - factor to scale energy bin centers
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
        self.grid_prop.SetEarthDensityParams(self.prop_height,YeI,YeO,YeM)
        self.prepare_device_arrays()

        dm_mat = np.zeros((3,3),dtype=self.FTYPE)
        self.grid_prop.Get_dm_mat(dm_mat)
        mix_mat = np.zeros((3,3,2),dtype=self.FTYPE)
        self.grid_prop.Get_mix_mat(mix_mat)

        logging.debug("dm_mat: \n %s"%str(dm_mat))
        logging.debug("mix[re]: \n %s"%str(mix_mat[:,:,0]))

        d_dm_mat = cuda.mem_alloc(dm_mat.nbytes)
        d_mix_mat = cuda.mem_alloc(mix_mat.nbytes)
        cuda.memcpy_htod(d_dm_mat,dm_mat)
        cuda.memcpy_htod(d_mix_mat,mix_mat)


        # NEXT: set up smooth maps to give to kernel, and then use PyCUDA to launch kernel...
        logging.info("Initialize smooth maps...")
        smoothed_maps = {}
        smoothed_maps['ebins'] = self.ebins
        smoothed_maps['czbins'] = self.czbins

        nebins_fine = np.uint32(len(self.ecen_fine))
        nczbins_fine = np.uint32(len(self.czcen_fine))
        nebins = np.uint32(len(self.ebins)-1)
        nczbins = np.uint32(len(self.czbins)-1)

        # This goes here, so it can use the energy_scale systematic:
        cuda.memcpy_htod(self.d_ecen_fine,self.ecen_fine*energy_scale)

        smooth_maps = np.zeros((nczbins*nebins*12),dtype=self.FTYPE)
        d_smooth_maps = cuda.mem_alloc(smooth_maps.nbytes)
        cuda.memcpy_htod(d_smooth_maps,smooth_maps)
        block_size = (10,10,1)
        grid_size = (nczbins_fine/block_size[0] + 1, nebins_fine/block_size[1] + 1,2)
        self.propGrid(d_smooth_maps,
                 d_dm_mat,d_mix_mat,
                 self.d_ecen_fine,self.d_czcen_fine,
                 nebins_fine,nczbins_fine,
                 nebins,nczbins,
                 np.uint32(self.maxLayers),
                 self.d_numLayers,self.d_densityInLayer,
                 self.d_distanceInLayer,
                 block=block_size,grid=grid_size)

        cuda.memcpy_dtoh(smooth_maps,d_smooth_maps)

        # Now put these into smoothed_maps in the correct format as
        # the other oscillation services, to interface properly with
        # the rest of the code:
        smooth_maps = np.reshape(smooth_maps,(12,nebins,nczbins))
        flavs = ['nue','numu','nutau']
        iMap = 0
        for from_nu in ['nue','numu','nue_bar','numu_bar']:
            from_nu += '_maps'
            smoothed_maps[from_nu] = {}
            for to_nu in flavs:
                if '_bar' in from_nu: to_nu+='_bar'
                smoothed_maps[from_nu][to_nu] = smooth_maps[iMap]
                iMap+=1

        return smoothed_maps

