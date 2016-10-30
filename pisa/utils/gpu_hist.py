# authors: P.Eller (pde3@psu.edu)
# date:   September 2016


__all__ = ['GPUHist']


import os

import numpy as np
from pycuda.compiler import SourceModule
import pycuda.driver as cuda

from pisa.utils.log import logging, set_verbosity
from pisa.utils.profiler import line_profile, profile
from pisa.utils.resources import find_resource


# TODO: get pep8 up in heah
# TODO: useful comments on what's going on with magical numbers and computations
class GPUHist(object):
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
    def __init__(self, bin_edges_x, bin_edges_y, bin_edges_z=None,
                 ftype=np.float64):
        self.FTYPE = ftype
        if bin_edges_z is None:
            self.h3d = False
        else:
            self.h3d = True

        self.bdim = (256, 1, 1)

        # Events to be histogrammed per thread
        self.n_threads = 20
        self.n_bins_x = np.int32(len(bin_edges_x)-1)
        self.n_bins_y = np.int32(len(bin_edges_y)-1)
        if self.h3d:
            self.n_bins_z = np.int32(len(bin_edges_z)-1)
            self.hist = np.ravel(np.zeros((self.n_bins_x, self.n_bins_y,
                                           self.n_bins_z), dtype=self.FTYPE))
        else:
            self.n_bins_z = np.int32(1)
            self.hist = np.ravel(np.zeros((self.n_bins_x, self.n_bins_y),
                                          dtype=self.FTYPE))

        # Allocate
        self.d_hist = cuda.mem_alloc(self.hist.nbytes)
        self.d_bin_edges_x = cuda.mem_alloc(bin_edges_x.nbytes)
        self.d_bin_edges_y = cuda.mem_alloc(bin_edges_y.nbytes)
        if self.h3d:
            self.d_bin_edges_z = cuda.mem_alloc(bin_edges_z.nbytes)

        # Copy
        cuda.memcpy_htod(self.d_bin_edges_x, bin_edges_x)
        cuda.memcpy_htod(self.d_bin_edges_y, bin_edges_y)
        if self.h3d:
            cuda.memcpy_htod(self.d_bin_edges_z, bin_edges_z)

        kernel_template = '''//CUDA//
        // Total number of bins (must be known at compile time)
        #define N_BINS %i

        // Number of events to be histogrammed per thread
        #define N_THREADS %i

        #include "constants.h"
        #include "utils.h"


        __device__ int GetBin(fType x, const int n_bins, fType *bin_edges)
        {
            // Search what bin an event belongs in, given the event values x,
            // the number of bins n_bins and the bin_edges array
            int first = 0;
            int last = n_bins - 1;
            int bin;

            // Binary search to speed things up and allow for arbitrary binning
            while (first <= last) {
                bin = (first + last)/2;
                if (x >= bin_edges[bin]) {
                    if ((x < bin_edges[bin+1]) || ((x <= bin_edges[n_bins])) && (bin == n_bins - 1))
                        break;
                    else
                        first = bin + 1;
                }
                else {
                    last = bin - 1;
                }
            }
            return bin;
        }


        __global__ void ClearHist(fType *hist)
        {
            // Write zeros to histogram in global memory
            int bin;
            int iterations = (N_BINS / blockDim.x) + 1;
            for (int i = 0; i < iterations; i++) {
                bin = (i * blockDim.x) + threadIdx.x;
                if (bin < N_BINS) hist[bin] = 0.0;
            }
            __syncthreads();
        }


        __global__ void Hist2D(fType *X, fType *Y, fType *W, const int n_evts,
                               fType *hist,
                               const int n_bins_x, const int n_bins_y,
                               fType *bin_edges_x, fType *bin_edges_y)
        {
            __shared__ fType temp_hist[N_BINS];

            // Zero out (reset) shared histogram buffer
            int iterations = (N_BINS / blockDim.x) + 1;
            int bin;
            for (int i = 0; i < iterations; i++) {
                bin = (i * blockDim.x) + threadIdx.x;
                if (bin < N_BINS)
                    temp_hist[bin] = 0.0;
            }
            __syncthreads();

            int idx = N_THREADS * (threadIdx.x + blockDim.x * blockIdx.x);
            for (int i = 0; i < N_THREADS; i++) {
                if (idx < n_evts) {
                    fType x = X[idx];
                    fType y = Y[idx];
                    // Check if event is even in range
                    if ((x >= bin_edges_x[0]) && (x <= bin_edges_x[n_bins_x]) && (y >= bin_edges_y[0]) && (y <= bin_edges_y[n_bins_y])) {
                        int bin_x = GetBin(x, n_bins_x, bin_edges_x);
                        int bin_y = GetBin(y, n_bins_y, bin_edges_y);
                        atomicAdd_custom(&temp_hist[bin_y + bin_x * n_bins_y],
                                         W[idx]);
                    }
                }
                idx++;
            }
            __syncthreads();

            // Write shared buffer into global memory
            for (int i = 0; i < iterations; i++) {
                bin = (i * blockDim.x) + threadIdx.x;
                if (bin < N_BINS)
                    atomicAdd_custom(&(hist[bin]), temp_hist[bin]);
            }
        }


        __global__ void Hist3D(fType *X, fType *Y, fType *Z, fType *W,
                               const int n_evts,
                               fType *hist,
                               const int n_bins_x, const int n_bins_y, const int n_bins_z,
                               fType *bin_edges_x, fType *bin_edges_y, fType *bin_edges_z)
        {
            __shared__ fType temp_hist[N_BINS];

            // Zero out (reset) shared histogram buffer
            int iterations = (N_BINS / blockDim.x) + 1;
            int bin;
            for (int i = 0; i < iterations; i++) {
                bin = (i * blockDim.x) + threadIdx.x;
                if (bin < N_BINS) temp_hist[bin] = 0;
            }
            __syncthreads();

            int idx = N_THREADS * (threadIdx.x + blockDim.x * blockIdx.x);
            for (int i = 0; i < N_THREADS; i++) {
                if (idx < n_evts) {
                    fType x = X[idx];
                    fType y = Y[idx];
                    fType z = Z[idx];

                    // Check if event is even in range
                    if ((x >= bin_edges_x[0]) && (x <= bin_edges_x[n_bins_x])
                          && (y >= bin_edges_y[0]) && (y <= bin_edges_y[n_bins_y])
                          && (z >= bin_edges_z[0]) && (z <= bin_edges_z[n_bins_z])) {
                        int bin_x = GetBin(x, n_bins_x, bin_edges_x);
                        int bin_y = GetBin(y, n_bins_y, bin_edges_y);
                        int bin_z = GetBin(z, n_bins_z, bin_edges_z);
                        atomicAdd_custom(&temp_hist[bin_z + (bin_y * n_bins_z) + (bin_x * n_bins_y * n_bins_z)], W[idx]);
                    }
                }
                idx++;
            }
            __syncthreads();

            // Write shared buffer into global memory
            for (int i = 0; i < iterations; i++) {
                bin = (i * blockDim.x) + threadIdx.x;
                if (bin < N_BINS) atomicAdd_custom(&(hist[bin]), temp_hist[bin]);
            }
        }
        '''%(self.n_bins_x*self.n_bins_y*self.n_bins_z, self.n_threads)

        include_path = os.path.abspath(
            find_resource('../stages/osc/grid_propagator/')
        )
        module = SourceModule(kernel_template, include_dirs=[include_path],
                              keep=True)

        self.clearhist_fun = module.get_function("ClearHist")
        self.hist2d_fun = module.get_function("Hist2D")
        self.hist3d_fun = module.get_function("Hist3D")

    def clear(self):
        """Clear the histogram bins on the GPU"""
        dx, mx = divmod(
            1 + (self.n_bins_x*self.n_bins_y*self.n_bins_z / self.n_threads),
            self.bdim[0]
        )
        gdim = ((dx + (mx > 0)) * self.bdim[0], 1)
        self.clearhist_fun(self.d_hist, block=self.bdim, grid=gdim)

    @profile
    def get_hist(self, n_evts, d_x, d_y, d_w, d_z=None):
        """Retrive histogram, given device arrays for x&y values as well as
        weights w"""
        # TODO: useful comments on what's going on with magical numbers and
        # computations

        # Block and grid dimensions
        dx, mx = divmod(n_evts/self.n_threads+1, self.bdim[0])
        gdim = ((dx + (mx > 0)) * self.bdim[0], 1)
        self.clear()

        # Calculate hist
        if self.h3d:
            self.hist3d_fun(
                d_x, d_y, d_z, d_w,
                n_evts,
                self.d_hist,
                self.n_bins_x, self.n_bins_y, self.n_bins_z,
                self.d_bin_edges_x, self.d_bin_edges_y, self.d_bin_edges_z,
                block=self.bdim, grid=gdim
            )
        else:
            self.hist2d_fun(
                d_x, d_y, d_w,
                n_evts,
                self.d_hist,
                self.n_bins_x, self.n_bins_y,
                self.d_bin_edges_x, self.d_bin_edges_y,
                block=self.bdim, grid=gdim
            )

        # Copy back
        cuda.memcpy_dtoh(self.hist, self.d_hist)
        if self.h3d:
            hist = self.hist.reshape(self.n_bins_x, self.n_bins_y,
                                     self.n_bins_z)
        else:
            hist = self.hist.reshape(self.n_bins_x, self.n_bins_y)

        return hist


def test_GPUHist():
    import pycuda.autoinit
    n_events = np.int32(1e6)
    n_bins = np.int32(5)

    for FTYPE in [np.float64]: #, np.float32]:
        if FTYPE == np.float32:
            rtol = 1e-7
        if FTYPE == np.float64:
            rtol = 1e-15
        # DEBUG: should not need to reset the rtol, but the GPU
        # histogrammer's not very accurate
        rtol = 1e-6

        # Draw random samples from the Pareto distribution for energy values
        rs = np.random.RandomState(seed=0)
        a, m = 1., 1
        e = ((rs.pareto(a, n_events) + 1) * m).astype(FTYPE)

        # Draw random samples from a uniform distribution for coszen values
        rs = np.random.RandomState(seed=1)
        cz = rs.uniform(low=-1, high=+1, size=n_events).astype(FTYPE)

        # Draw random samples from a uniform distribution for pid values
        rs = np.random.RandomState(seed=2)
        pid = rs.uniform(low=-1, high=+1, size=n_events).astype(FTYPE)

        # Draw random samples from a uniform distribution for weights
        rs = np.random.RandomState(seed=3)
        w = rs.uniform(low=0, high=1000, size=n_events).astype(FTYPE)
        w[0] = 0

        d_e = cuda.mem_alloc(e.nbytes)
        d_cz = cuda.mem_alloc(cz.nbytes)
        d_pid = cuda.mem_alloc(pid.nbytes)
        d_w = cuda.mem_alloc(w.nbytes)
        cuda.memcpy_htod(d_e, e)
        cuda.memcpy_htod(d_cz, cz)
        cuda.memcpy_htod(d_pid, pid)
        cuda.memcpy_htod(d_w, w)

        logging.debug('%s events'%n_events)

        bin_edges_e = np.logspace(0, 2, n_bins+1, dtype=FTYPE)
        bin_edges_cz = np.linspace(-1, 1, n_bins+1, dtype=FTYPE)
        bin_edges_pid = np.linspace(-1, 1, 2+1, dtype=FTYPE)

        histogrammer = GPUHist(bin_edges_e, bin_edges_cz, ftype=FTYPE)
        hist2d = histogrammer.get_hist(n_events, d_e, d_cz, d_w)
        hist2d = histogrammer.get_hist(n_events, d_e, d_cz, d_w)

        np_hist2d,_,_ = np.histogram2d(
            e, cz,
            bins=(bin_edges_e, bin_edges_cz),
            weights=w
        )

        logging.debug('GPU 2D histogram:\n' + repr(hist2d))
        logging.debug('Numpy 2D histogram:\n' + repr(np_hist2d))
        assert np.allclose(hist2d, np_hist2d, atol=0, rtol=rtol), str(np.max(np.abs(hist2d-np_hist2d)))

        del histogrammer

        histogrammer = GPUHist(bin_edges_x=bin_edges_e,
                               bin_edges_y=bin_edges_cz,
                               bin_edges_z=bin_edges_pid,
                               ftype=FTYPE)
        hist3d = histogrammer.get_hist(n_events, d_e, d_cz, d_w, d_pid)
        hist3d = histogrammer.get_hist(n_events, d_e, d_cz, d_w, d_pid)
        print hist3d.shape

        np_hist3d, _ = np.histogramdd(
            sample=[e, cz, pid],
            bins=(bin_edges_e, bin_edges_cz, bin_edges_pid),
            weights=w
        )

        logging.debug('GPU 3D histogram:\n' + repr(hist3d))
        logging.debug('Numpy 3D histogram:\n' + repr(np_hist3d))
        assert np.allclose(hist3d, np_hist3d, atol=0, rtol=rtol), str(np.max(np.abs(hist3d-np_hist3d)))

        del histogrammer


    logging.info('<< PASS : test_GPUHist >>')


if __name__ == '__main__':
    set_verbosity(3)
    test_GPUHist()
