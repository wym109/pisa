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
    ftype : numpy datatype (optional)

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
        self.events_per_thread = 20
        self.n_bins_x = np.int32(len(bin_edges_x)-1)
        self.n_bins_y = np.int32(len(bin_edges_y)-1)
        self.n_flat_bins = self.n_bins_x * self.n_bins_y
        if self.h3d:
            self.n_bins_z = np.int32(len(bin_edges_z)-1)
            self.hist = np.ravel(np.zeros((self.n_bins_x, self.n_bins_y,
                                           self.n_bins_z), dtype=self.FTYPE))
        else:
            self.n_bins_z = np.int32(1)
            self.hist = np.ravel(np.zeros((self.n_bins_x, self.n_bins_y),
                                          dtype=self.FTYPE))
        self.n_flat_bins *= self.n_bins_z

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
        #define N_FLAT_BINS %(n_flat_bins)i

        // Number of events to be histogrammed per thread
        #define EVENTS_PER_THREAD %(events_per_thread)i

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
            //int iterations = (N_FLAT_BINS / blockDim.x) + 1;
            for (int i = 0; i < N_FLAT_BINS; i++) {
                //bin = (i * blockDim.x) + threadIdx.x;
                if (i < N_FLAT_BINS)
                    hist[i] = 0.0;
            }
            __syncthreads();
        }


        __global__ void Hist2D(fType *X, fType *Y, fType *W, const int n_events,
                               fType *hist,
                               const int n_bins_x, const int n_bins_y,
                               fType *bin_edges_x, fType *bin_edges_y)
        {
            __shared__ fType temp_hist[N_FLAT_BINS];

            // Zero out (reset) shared histogram buffer
            int iterations = (N_FLAT_BINS / blockDim.x) + 1;
            int bin;
            for (int i = 0; i < iterations; i++) {
                bin = (i * blockDim.x) + threadIdx.x;
                if (bin < N_FLAT_BINS)
                    temp_hist[bin] = 0.0;
            }
            __syncthreads();

            int idx = EVENTS_PER_THREAD * (threadIdx.x + blockDim.x * blockIdx.x);
            for (int i = 0; i < EVENTS_PER_THREAD; i++) {
                if (idx < n_events) {
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
                if (bin < N_FLAT_BINS)
                    atomicAdd_custom(&(hist[bin]), temp_hist[bin]);
            }
        }


        __global__ void Hist3D(fType *X, fType *Y, fType *Z, fType *W,
                               const int n_events,
                               fType *hist,
                               const int n_bins_x, const int n_bins_y, const int n_bins_z,
                               fType *bin_edges_x, fType *bin_edges_y, fType *bin_edges_z)
        {
            __shared__ fType temp_hist[N_FLAT_BINS];

            // Zero out (reset) shared histogram buffer
            int iterations = (N_FLAT_BINS / blockDim.x) + 1;
            int bin;
            for (int i = 0; i < iterations; i++) {
                bin = (i * blockDim.x) + threadIdx.x;
                if (bin < N_FLAT_BINS) temp_hist[bin] = 0;
            }
            __syncthreads();

            int idx = EVENTS_PER_THREAD * (threadIdx.x + blockDim.x * blockIdx.x);
            for (int i = 0; i < EVENTS_PER_THREAD; i++) {
                if (idx < n_events) {
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
                if (bin < N_FLAT_BINS)
                    atomicAdd_custom(&(hist[bin]), temp_hist[bin]);
            }
        }
        '''%dict(n_flat_bins=self.n_flat_bins,
                 events_per_thread=self.events_per_thread)

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
            1 + (self.n_flat_bins), # / self.events_per_thread),
            self.bdim[0]
        )
        gdim = ((dx + (mx > 0)) * self.bdim[0], 1)
        self.clearhist_fun(self.d_hist, block=self.bdim, grid=gdim)

    @profile
    def get_hist(self, n_events, d_x, d_y, d_w, d_z=None):
        """Retrive histogram, given device arrays for x&y values as well as
        weights w"""
        # TODO: useful comments on what's going on with magical numbers and
        # computations

        # Block and grid dimensions
        dx, mx = divmod(n_events/self.events_per_thread+1, self.bdim[0])
        gdim = ((dx + (mx > 0)) * self.bdim[0], 1)
        self.clear()

        # Calculate hist
        if self.h3d:
            self.hist3d_fun(
                d_x, d_y, d_z, d_w,
                n_events,
                self.d_hist,
                self.n_bins_x, self.n_bins_y, self.n_bins_z,
                self.d_bin_edges_x, self.d_bin_edges_y, self.d_bin_edges_z,
                block=self.bdim, grid=gdim
            )
        else:
            self.hist2d_fun(
                d_x, d_y, d_w,
                n_events,
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
    from itertools import product
    import pycuda.autoinit

    ftypes = [np.float64]
    nexps = [0, 1, 6]
    nbinses = [1, 10, 50]
    ft = [False, True]
    for FTYPE, weight, nexp, n_bins in product(ftypes, ft, nexps, nbinses):
        n_events = np.int32(10**nexp)
        logging.debug('FTYPE=%s, weight=%-5s, bins=%2sx2, n_events=10^%s'
                      %(FTYPE.__name__, weight, n_bins, nexp))

        if FTYPE == np.float32:
            rtol = 1e-7
        elif FTYPE == np.float64:
            rtol = 1e-13

        # Draw random samples from the Pareto distribution for energy values
        rs = np.random.RandomState(seed=0)
        a, m = 1., 1
        e = ((rs.pareto(a, n_events) + 1) * m).astype(FTYPE)
        # Ensure endpoints are in data
        e[0] = 1
        e[-1] = 80

        # Draw random samples from a uniform distribution for coszen values
        rs = np.random.RandomState(seed=1)
        cz = rs.uniform(low=-1, high=+1, size=n_events).astype(FTYPE)
        # Ensure endpoints are in data
        cz[0] = -1
        cz[-1] = +1

        # Draw random samples from a uniform distribution for pid values
        rs = np.random.RandomState(seed=2)
        pid = rs.uniform(low=-1, high=+2, size=n_events).astype(FTYPE)
        # Ensure endpoints are in data
        pid[0] = -1
        pid[-1] = +2

        if weight:
            # Draw random samples from a uniform distribution for weights
            rs = np.random.RandomState(seed=3)
            w = rs.uniform(low=0, high=1000, size=n_events).astype(FTYPE)
            # Ensure a weight of 0 is represented
            w[0] = 0
        else:
            w = np.ones_like(e, dtype=FTYPE)

        d_e = cuda.mem_alloc(e.nbytes)
        d_cz = cuda.mem_alloc(cz.nbytes)
        d_pid = cuda.mem_alloc(pid.nbytes)
        d_w = cuda.mem_alloc(w.nbytes)
        cuda.memcpy_htod(d_e, e)
        cuda.memcpy_htod(d_cz, cz)
        cuda.memcpy_htod(d_pid, pid)
        cuda.memcpy_htod(d_w, w)

        bin_edges_e = np.logspace(0, 2, n_bins+1, dtype=FTYPE)
        bin_edges_cz = np.linspace(-1, 1, n_bins+1, dtype=FTYPE)
        bin_edges_pid = np.array([-1, 0, 2], dtype=FTYPE)

        histogrammer = GPUHist(
            bin_edges_x=bin_edges_e,
            bin_edges_y=bin_edges_cz,
            ftype=FTYPE
        )
        for i in range(3):
            hist2d = histogrammer.get_hist(
                n_events=n_events, d_x=d_e, d_y=d_cz, d_w=d_w
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
                '2D histogram FTYPE=%s, weighted=%s, n_events=%s worst fractional error is %s'
                %(FTYPE, weight, n_events, np.max(np.abs((hist2d-np_hist2d)/np_hist2d)))
            )

        del histogrammer

        logging.debug('FTYPE=%s, weight=%-5s, bins=%2sx%2sx2, n_events=10^%s'
                      %(FTYPE.__name__, weight, n_bins, n_bins, nexp))

        histogrammer = GPUHist(
            bin_edges_x=bin_edges_e,
            bin_edges_y=bin_edges_cz,
            bin_edges_z=bin_edges_pid,
            ftype=FTYPE
        )
        for i in range(3):
            hist3d = histogrammer.get_hist(
                n_events=n_events, d_x=d_e, d_y=d_cz, d_w=d_w, d_z=d_pid
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
                '3D histogram FTYPE=%s, weighted=%s, n_events=%s worst fractional error is %s'
                %(FTYPE, weight, n_events, np.max(np.abs((hist3d-np_hist3d)/np_hist3d)))
            )

        del histogrammer

    logging.info('<< PASS : test_GPUHist >>')


if __name__ == '__main__':
    set_verbosity(2)
    test_GPUHist()
