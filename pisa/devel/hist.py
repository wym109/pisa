import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray
import numpy as np
from pisa.utils.events import Events

fname = '/home/peller/cake/pisa/resources/events/pingu_v39/events__pingu__v39__runs_620-622__proc_v5.1__unjoined.hdf5'
events = Events(fname)

# CUDA
e = events['nue_cc']['reco_energy'].astype(np.float32)
cz = events['nue_cc']['reco_coszen'].astype(np.float32)
n = np.int32(len(e))

bin_edges_e = np.logspace(0,2,100).astype(np.float32)
bin_edges_cz = np.linspace(-1,1,100).astype(np.float32)

n_bins_e = np.int32(len(bin_edges_e)-1)
n_bins_cz = np.int32(len(bin_edges_cz)-1)
hist1d = np.zeros(n_bins_e).astype(np.float32)
hist2d = np.ravel(np.zeros((n_bins_e, n_bins_cz))).astype(np.float32)

print '%s events'%n

# block and grid dimensions
bdim = (256,1,1)
dx, mx = divmod(n, bdim[0])
gdim = ((dx + (mx>0)) * bdim[0], 1)

# allocate
e_gpu = cuda.mem_alloc(e.nbytes)
cz_gpu = cuda.mem_alloc(cz.nbytes)
hist1d_gpu = cuda.mem_alloc(hist1d.nbytes)
hist2d_gpu = cuda.mem_alloc(hist2d.nbytes)
bin_edges_e_gpu = cuda.mem_alloc(bin_edges_e.nbytes)
bin_edges_cz_gpu = cuda.mem_alloc(bin_edges_cz.nbytes)

# copy
cuda.memcpy_htod(e_gpu, e)
cuda.memcpy_htod(cz_gpu, cz)
cuda.memcpy_htod(hist1d_gpu, hist1d)
cuda.memcpy_htod(hist2d_gpu, hist2d)
cuda.memcpy_htod(bin_edges_e_gpu, bin_edges_e)
cuda.memcpy_htod(bin_edges_cz_gpu, bin_edges_cz)

mod = SourceModule("""
  __device__ int GetBin(float x, const int n_bins, float *bin_edges){
    int first = 0;
    int last = n_bins -1;
    int bin;
    // binary search
    while (first <= last) {
        bin = (first + last)/2;
        if (x > bin_edges[bin]){
            if (x <= bin_edges[bin+1]){
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

  __global__ void Hist1D(float *X, const int len, float *hist, const int n_bins, float *bin_edges)
  {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < len) {
        float x = X[idx];
        // check if event is even in range
        if ((x > bin_edges[0]) && (x <= bin_edges[n_bins])){
            int bin = GetBin(x, n_bins, bin_edges);
            atomicAdd(&hist[bin], 1);
        }
    }
  }
  __global__ void Hist2D(float *X, float *Y, const int len, float *hist, const int n_bins_x, const int n_bins_y, float *bin_edges_x, float *bin_edges_y)
  {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < len) {
        float x = X[idx];
        float y = Y[idx];
        // check if event is even in range
        if ((x > bin_edges_x[0]) && (x <= bin_edges_x[n_bins_x]) && (y > bin_edges_y[0]) && (y <= bin_edges_y[n_bins_y])){
            int bin_x = GetBin(x, n_bins_x, bin_edges_x);
            int bin_y = GetBin(y, n_bins_y, bin_edges_y);
            atomicAdd(&hist[bin_y + bin_x * n_bins_y], 1);
        }
    }
  }
  """)

hist1d_fun = mod.get_function("Hist1D")
hist1d_fun(e_gpu, n, hist1d_gpu, n_bins_e, bin_edges_e_gpu, block=bdim, grid=gdim)
cuda.memcpy_dtoh(hist1d, hist1d_gpu)
np_hist1d = np.histogram(e,bins=bin_edges_e)
print hist1d
assert (np.sum(hist1d - np_hist1d[0]) == 0.)

hist2d_fun = mod.get_function("Hist2D")
hist2d_fun(e_gpu, cz_gpu, n, hist2d_gpu, n_bins_e, n_bins_cz, bin_edges_e_gpu, bin_edges_cz_gpu, block=bdim, grid=gdim)
cuda.memcpy_dtoh(hist2d, hist2d_gpu)
np_hist2d = np.histogram2d(e, cz,bins=(bin_edges_e, bin_edges_cz))
hist2d = hist2d.reshape(n_bins_e, n_bins_cz)
print hist2d
assert (np.sum(hist2d - np_hist2d[0]) == 0.)
