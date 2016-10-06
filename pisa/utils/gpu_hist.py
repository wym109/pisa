# authors: P.Eller (pde3@psu.edu)
# date:   September 2016
import os
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
from pisa.utils.profiler import profile
from pisa.utils.log import set_verbosity
#set_verbosity(3)

class GPUhist(object):
    ''' histogramming class for GPUs
        
    PARAMETERS
    ---------

    bin_edges_x : array
    bin_edges_y : array

    METHODS
    -------

    get_hist
        retreive weighted histogram of given events
        * n_evts : number of events
        * d_x : CUDA device array of length n_evts with x-values
        * d_y : CUDA device array of length n_evts with y-values
        * d_w : CUDA device array of length n_evts with weights
    clear
        clear buffer

    '''
    
    def __init__(self, bin_edges_x, bin_edges_y):
        ''' initialize 2d histogram, with given bin_edges '''
        self.FTYPE = np.float64
        # events to be histogrammed per thread
        self.n_thread = 20
        self.n_bins_x = np.int32(len(bin_edges_x)-1)
        self.n_bins_y = np.int32(len(bin_edges_y)-1)
        self.hist2d = np.ravel(np.zeros((self.n_bins_x, self.n_bins_y))).astype(self.FTYPE)

        # allocate
        self.d_hist2d = cuda.mem_alloc(self.hist2d.nbytes)
        self.d_bin_edges_x = cuda.mem_alloc(bin_edges_x.nbytes)
        self.d_bin_edges_y = cuda.mem_alloc(bin_edges_y.nbytes)

        # copy
        cuda.memcpy_htod(self.d_hist2d, self.hist2d)
        cuda.memcpy_htod(self.d_bin_edges_x, bin_edges_x)
        cuda.memcpy_htod(self.d_bin_edges_y, bin_edges_y)

        kernel_template = """//CUDA//
          // total number of bins (must be known at comiple time)
          #define N_BINS %i
          // number of events to be histogrammed per thread
          #define N_THREAD %i
            
          #include "constants.h"
          #include "utils.h"
          
          __device__ int GetBin(fType x, const int n_bins, fType *bin_edges){
            // search what bin an event belongs in, given the event values x, the number of bins n_bins and the bin_edges array
            int first = 0;
            int last = n_bins -1;
            int bin;
            // binary search to speed things up and allow for arbitrary binning
            while (first <= last) {
                bin = (first + last)/2;
                if (x >= bin_edges[bin]){
                    if ((x < bin_edges[bin+1]) || ((x <= bin_edges[n_bins])) && (bin == n_bins - 1)){
                        break;
                    }
                    else {
                        first = bin + 1;
                    }
                }
                else {
                    last = bin - 1;
                }
            }
            return bin; 
          }

          __global__ void Hist2D(fType *X, fType *Y, fType *W, const int n_evts, fType *hist, const int n_bins_x, const int n_bins_y, fType *bin_edges_x, fType *bin_edges_y)
          {
            __shared__ fType temp_hist[N_BINS];
            // zero out (reset) shared histogram buffer
            int iterations = (N_BINS / blockDim.x) + 1;
            int bin;
            for (int i = 0; i < iterations; i++){
                bin = (i * blockDim.x) + threadIdx.x;
                if (bin < N_BINS) temp_hist[bin] = 0;
            }
            __syncthreads();

            int idx = N_THREAD * (threadIdx.x + blockDim.x * blockIdx.x);
                for (int i = 0; i < N_THREAD; i++){

                if (idx < n_evts) {
                    fType x = X[idx];
                    fType y = Y[idx];
                    // check if event is even in range
                    if ((x >= bin_edges_x[0]) && (x <= bin_edges_x[n_bins_x]) && (y >= bin_edges_y[0]) && (y <= bin_edges_y[n_bins_y])){
                        int bin_x = GetBin(x, n_bins_x, bin_edges_x);
                        int bin_y = GetBin(y, n_bins_y, bin_edges_y);
                        atomicAdd_custom(&temp_hist[bin_y + bin_x * n_bins_y], W[idx]);
                    }
                }
                idx++;
            }
            __syncthreads();
            // write shared buffer into global memory
            for (int i = 0; i < iterations; i++){
                bin = (i * blockDim.x) + threadIdx.x;
                if (bin < N_BINS) atomicAdd_custom( &(hist[bin]), temp_hist[bin] );
            }

          }
          """%(self.n_bins_x*self.n_bins_y, self.n_thread)
        include_path = os.path.expandvars('$PISA/pisa/stages/osc/grid_propagator/')
        module = SourceModule(kernel_template, include_dirs=[include_path], keep=True)
        self.hist2d_fun = module.get_function("Hist2D")

    def clear(self):
        # very dumb way to reset to zero...
        self.hist2d = np.ravel(np.zeros((self.n_bins_x, self.n_bins_y))).astype(self.FTYPE)
        cuda.memcpy_htod(self.d_hist2d, self.hist2d)

    def get_hist(self, n_evts, d_x, d_y, d_w):
        ''' retrive 2d histogram, given device arrays for x&y values as well as weights w '''
        # block and grid dimensions
        bdim = (256,1,1)
        dx, mx = divmod(n_evts/self.n_thread+1, bdim[0])
        gdim = ((dx + (mx>0)) * bdim[0], 1)
        self.clear()
        # calculate hist
        self.hist2d_fun(d_x, d_y, d_w, n_evts, self.d_hist2d, self.n_bins_x, self.n_bins_y, self.d_bin_edges_x, self.d_bin_edges_y, block=bdim, grid=gdim)
        # copy bask
        cuda.memcpy_dtoh(self.hist2d, self.d_hist2d)
        hist2d = self.hist2d.reshape(self.n_bins_x, self.n_bins_y)
        return hist2d

if __name__ == '__main__':
    import pycuda.autoinit
    FTYPE = np.float64

    e = np.linspace(1,100,1000).astype(FTYPE)
    cz = np.linspace(-1,1,1000).astype(FTYPE)
    n_evts = np.int32(len(e))
    w = np.ones(n_evts).astype(FTYPE)
    d_e = cuda.mem_alloc(e.nbytes)
    d_cz = cuda.mem_alloc(cz.nbytes)
    d_w = cuda.mem_alloc(w.nbytes)
    cuda.memcpy_htod(d_e, e)
    cuda.memcpy_htod(d_cz, cz)
    cuda.memcpy_htod(d_w, w)
    print '%s events'%n_evts

    bin_edges_e = np.linspace(1,100,10).astype(FTYPE)
    bin_edges_cz = np.linspace(-1,1,10).astype(FTYPE)

    histogrammer = GPUhist(bin_edges_e, bin_edges_cz)
    hist2d = histogrammer.get_hist(n_evts, d_e, d_cz, d_w)

    np_hist2d,_,_ = np.histogram2d(e, cz,bins=(bin_edges_e, bin_edges_cz), weights=w)
    print hist2d
    print np_hist2d
    assert (np.sum(hist2d - np_hist2d) == 0.)
