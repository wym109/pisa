# authors: P.Eller (pde3@psu.edu)
# date:   September 2016


import os

import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from pisa import FTYPE, C_FTYPE, C_PRECISION_DEF
from pisa.utils.profiler import profile
from pisa.utils.log import logging, set_verbosity
from pisa.utils.resources import find_resource


__all__ = ['GPUhist']


# TODO: get pep8 up in heah
class GPUhist(object):
    """
    Histogramming class for GPUs

    Parameters
    ---------
    bin_edges_x : array
    bin_edges_y : array
    bin_edges_z : array (optional)

    Methods
    -------
    get_hist
        retreive weighted histogram of given events
        * n_evts : number of events
        * d_x : CUDA device array of length n_evts with x-values
        * d_y : CUDA device array of length n_evts with y-values
        * d_z : CUDA device array of length n_evts with y-values
        * d_w : CUDA device array of length n_evts with weights
    clear
        clear buffer

    """

    KERNEL_TEMPLATE = '''//CUDA//
    #define %(C_PRECISION_DEF)s
    #define fType %(C_FTYPE)s

    #include "cuda_utils.h"

    // total number of bins (must be known at comiple time)
    #define N_BINS %(N_BINS)i

    // number of events to be histogrammed per thread
    #define N_THREAD %(N_THREAD)i


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
      __global__ void Hist3D(fType *X, fType *Y, fType *Z, fType *W, const int n_evts, fType *hist, const int n_bins_x, const int n_bins_y, const int n_bins_z, fType *bin_edges_x, fType *bin_edges_y, fType *bin_edges_z)
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
              fType z = Z[idx];
              // check if event is even in range
              if ((x >= bin_edges_x[0]) && (x <= bin_edges_x[n_bins_x]) && (y >= bin_edges_y[0]) && (y <= bin_edges_y[n_bins_y]) && (z >= bin_edges_z[0]) && (z <= bin_edges_z[n_bins_z])){
                  int bin_x = GetBin(x, n_bins_x, bin_edges_x);
                  int bin_y = GetBin(y, n_bins_y, bin_edges_y);
                  int bin_z = GetBin(z, n_bins_z, bin_edges_z);
                  atomicAdd_custom(&temp_hist[bin_z + (bin_y * n_bins_z) + (bin_x * n_bins_y * n_bins_z)], W[idx]);
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
    '''

    def __init__(self, bin_edges_x, bin_edges_y, bin_edges_z=None):
        self.h3d = bool(bin_edges_z is not None)
        # events to be histogrammed per thread
        self.n_thread = 20
        self.n_bins_x = np.int32(len(bin_edges_x)-1)
        self.n_bins_y = np.int32(len(bin_edges_y)-1)
        if self.h3d:
            self.n_bins_z = np.int32(len(bin_edges_z)-1)
            self.hist = np.ravel(np.zeros((self.n_bins_x, self.n_bins_y, self.n_bins_z))).astype(FTYPE)
        else:
            self.n_bins_z = 1
            self.hist = np.ravel(np.zeros((self.n_bins_x, self.n_bins_y))).astype(FTYPE)

        # allocate
        self.d_hist = cuda.mem_alloc(self.hist.nbytes)
        self.d_bin_edges_x = cuda.mem_alloc(bin_edges_x.nbytes)
        self.d_bin_edges_y = cuda.mem_alloc(bin_edges_y.nbytes)
        if self.h3d:
            self.d_bin_edges_z = cuda.mem_alloc(bin_edges_z.nbytes)


        # copy
        cuda.memcpy_htod(self.d_hist, self.hist)
        cuda.memcpy_htod(self.d_bin_edges_x, bin_edges_x)
        cuda.memcpy_htod(self.d_bin_edges_y, bin_edges_y)
        if self.h3d:
            cuda.memcpy_htod(self.d_bin_edges_z, bin_edges_z)

        kernel_code = self.KERNEL_TEMPLATE %dict(
            C_PRECISION_DEF=C_PRECISION_DEF,
            C_FTYPE=C_FTYPE,
            N_BINS=self.n_bins_x*self.n_bins_y*self.n_bins_z,
            N_THREAD=self.n_thread
        )

        include_dirs = [
            os.path.abspath(find_resource('../utils'))
        ]

        module = SourceModule(kernel_code, include_dirs=include_dirs,
                              keep=True)
        self.hist2d_fun = module.get_function("Hist2D")
        self.hist3d_fun = module.get_function("Hist3D")

    def clear(self):
        # very dumb way to reset to zero...
        if self.h3d:
            self.hist = np.ravel(np.zeros((self.n_bins_x, self.n_bins_y, self.n_bins_z))).astype(FTYPE)
        else:
            self.hist = np.ravel(np.zeros((self.n_bins_x, self.n_bins_y))).astype(FTYPE)
        cuda.memcpy_htod(self.d_hist, self.hist)

    def get_hist(self, n_evts, d_x, d_y, d_w, d_z=None):
        """Retrive histogram, given device arrays for x&y values as well as
        weights w"""
        # block and grid dimensions
        bdim = (256,1,1)
        dx, mx = divmod(n_evts/self.n_thread+1, bdim[0])
        gdim = ((dx + (mx>0)) * bdim[0], 1)
        self.clear()
        # calculate hist
        if self.h3d:
            self.hist3d_fun(d_x, d_y, d_z, d_w, n_evts, self.d_hist, self.n_bins_x, self.n_bins_y, self.n_bins_z, self.d_bin_edges_x, self.d_bin_edges_y, self.d_bin_edges_z, block=bdim, grid=gdim)
        else:
            self.hist2d_fun(d_x, d_y, d_w, n_evts, self.d_hist, self.n_bins_x, self.n_bins_y, self.d_bin_edges_x, self.d_bin_edges_y, block=bdim, grid=gdim)
        # copy bask
        cuda.memcpy_dtoh(self.hist, self.d_hist)
        if self.h3d:
            hist = self.hist.reshape(self.n_bins_x, self.n_bins_y, self.n_bins_z)
        else:
            hist = self.hist.reshape(self.n_bins_x, self.n_bins_y)
        return hist


def test_GPUhist():
    from itertools import product
    import pycuda.autoinit

    ftypes = [FTYPE]
    nexps = [0, 1, 6]
    nbinses = [1, 10, 50]
    ft = [False, True]
    for ftype, weight, nexp, n_bins in product(ftypes, ft, nexps, nbinses):
        n_events = np.int32(10**nexp)
        logging.debug('ftype=%s, weight=%-5s, bins=%2sx2, n_events=10^%s'
                      %(ftype.__name__, weight, n_bins, nexp))

        if ftype == np.float32:
            rtol = 1e-6
        elif ftype == np.float64:
            rtol = 1e-13

        # Draw random samples from the Pareto distribution for energy values
        rs = np.random.RandomState(seed=0)
        a, m = 1., 1
        e = ((rs.pareto(a, n_events) + 1) * m).astype(ftype)
        # Ensure endpoints are in data
        e[0] = 1
        e[-1] = 80

        # Draw random samples from a uniform distribution for coszen values
        rs = np.random.RandomState(seed=1)
        cz = rs.uniform(low=-1, high=+1, size=n_events).astype(ftype)
        # Ensure endpoints are in data
        cz[0] = -1
        cz[-1] = +1

        # Draw random samples from a uniform distribution for pid values
        rs = np.random.RandomState(seed=2)
        pid = rs.uniform(low=-1, high=+2, size=n_events).astype(ftype)
        # Ensure endpoints are in data
        pid[0] = -1
        pid[-1] = +2

        if weight:
            # Draw random samples from a uniform distribution for weights
            rs = np.random.RandomState(seed=3)
            w = rs.uniform(low=0, high=1000, size=n_events).astype(ftype)
            # Ensure a weight of 0 is represented
            w[0] = 0
        else:
            w = np.ones_like(e, dtype=ftype)

        d_e = cuda.mem_alloc(e.nbytes)
        d_cz = cuda.mem_alloc(cz.nbytes)
        d_pid = cuda.mem_alloc(pid.nbytes)
        d_w = cuda.mem_alloc(w.nbytes)
        cuda.memcpy_htod(d_e, e)
        cuda.memcpy_htod(d_cz, cz)
        cuda.memcpy_htod(d_pid, pid)
        cuda.memcpy_htod(d_w, w)

        bin_edges_e = np.logspace(0, 2, n_bins+1, dtype=ftype)
        bin_edges_cz = np.linspace(-1, 1, n_bins+1, dtype=ftype)
        bin_edges_pid = np.array([-1, 0, 2], dtype=ftype)

        histogrammer = GPUhist(
            bin_edges_x=bin_edges_e,
            bin_edges_y=bin_edges_cz,
            #ftype=ftype
        )
        for i in range(3):
            hist2d = histogrammer.get_hist(
                n_evts=n_events, d_x=d_e, d_y=d_cz, d_w=d_w
            )

        np_hist2d,_,_ = np.histogram2d(
            e, cz,
            bins=(bin_edges_e, bin_edges_cz),
            weights=w
        )

        if not np.allclose(hist2d, np_hist2d, atol=0, rtol=rtol):
            logging.error('Numpy hist:\n%s' %repr(np_hist2d))
            logging.error('GPUHist hist:\n%s' %repr(hist2d))
            raise ValueError(
                '2D histogram ftype=%s, weighted=%s, n_events=%s worst fractional error is %s'
                %(ftype, weight, n_events, np.max(np.abs((hist2d-np_hist2d)/np_hist2d)))
            )

        del histogrammer

        logging.debug('ftype=%s, weight=%-5s, bins=%2sx%2sx2, n_events=10^%s'
                      %(ftype.__name__, weight, n_bins, n_bins, nexp))

        histogrammer = GPUhist(
            bin_edges_x=bin_edges_e,
            bin_edges_y=bin_edges_cz,
            bin_edges_z=bin_edges_pid,
            #ftype=ftype
        )
        for i in range(3):
            hist3d = histogrammer.get_hist(
                n_evts=n_events, d_x=d_e, d_y=d_cz, d_w=d_w, d_z=d_pid
            )

        np_hist3d, _ = np.histogramdd(
            sample=[e, cz, pid],
            bins=(bin_edges_e, bin_edges_cz, bin_edges_pid),
            weights=w
        )

        if not np.allclose(hist3d, np_hist3d, atol=0, rtol=rtol):
            logging.error('Numpy hist:\n%s' %repr(np_hist3d))
            logging.error('GPUHist hist:\n%s' %repr(hist3d))
            raise ValueError(
                '3D histogram ftype=%s, weighted=%s, n_events=%s worst fractional error is %s'
                %(ftype, weight, n_events, np.max(np.abs((hist3d-np_hist3d)/np_hist3d)))
            )

        del histogrammer

    logging.info('<< PASS : test_GPUhist >>')


if __name__ == '__main__':
    set_verbosity(2)
    test_GPUhist()
