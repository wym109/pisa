import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from pisa.utils.events import Events

fname = '/home/peller/cake/pisa/resources/events/pingu_v39/events__pingu__v39__runs_620-622__proc_v5.1__unjoined.hdf5'
events = Events(fname)

# CUDA
e = events['nue_cc']['reco_energy'].astype(np.float32)
n = np.int32(len(e))
#cz = events['nue_cc']['reco_coszen'].astype(np.float32)

bin_edges = np.logspace(0,2,100).astype(np.float32)

n_bins = np.int32(len(bin_edges)-1)
hist = np.zeros(n_bins).astype(np.float32)

print '%s events'%n

# block and grid dimensions
bdim = (256,1,1)
dx, mx = divmod(n, bdim[0])
gdim = ((dx + (mx>0)) * bdim[0], 1)

# allocate
e_gpu = cuda.mem_alloc(e.nbytes)
#cz_gpu = cuda.mem_alloc(cz.nbytes)
hist_gpu = cuda.mem_alloc(hist.nbytes)
bin_edges_gpu = cuda.mem_alloc(bin_edges.nbytes)

# copy
cuda.memcpy_htod(e_gpu, e)
#cuda.memcpy_htod(cz_gpu, cz)
cuda.memcpy_htod(hist_gpu, hist)
cuda.memcpy_htod(bin_edges_gpu, bin_edges)

#mod = SourceModule("""
#  __global__ void histogram(float *e, const int len, float *hist, const int n_bins, float *bin_edges)
#  {
#    int idx = threadIdx.x + blockDim.x * blockIdx.x;
#    if (idx < len) {
#        if ((e[idx] > bin_edges[0]) && (e[idx] <= bin_edges[n_bins])){
#            int bin = 0;
#            while ((e[idx] > bin_edges[bin]) && (bin <= n_bins)){
#                bin++;
#            }
#            if (bin <= n_bins){
#                atomicAdd(&hist[bin-1], 1);
#            }
#        }
#    }
#  }
#  """)

mod = SourceModule("""
  __global__ void histogram(float *e, const int len, float *hist, const int n_bins, float *bin_edges)
  {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < len) {
        float E = e[idx];
        // check if event is even in range
        if ((E > bin_edges[0]) && (E <= bin_edges[n_bins])){
            int first = 0;
            int last = n_bins -1;
            int bin;
            // binary search
            while (first <= last) {
                bin = (first + last)/2;
                if (E > bin_edges[bin]){
                    if (E <= bin_edges[bin+1]){
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
            atomicAdd(&hist[bin], 1);
        }
    }
  }
  """)

func = mod.get_function("histogram")
func(e_gpu, n, hist_gpu, n_bins, bin_edges_gpu, block=bdim, grid=gdim)

cuda.memcpy_dtoh(hist, hist_gpu)
np_hist = np.histogram(e,bins=bin_edges)
print hist
print hist - np_hist[0]
