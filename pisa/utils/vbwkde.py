#!/usr/bin/env python
#
# J.L. Lanfranchi for the IceCube/PINGU collaboration.
#
# Based on the implementation in Matlab by Zdravko Botev, and the paper
# Z. I. Botev, J. F. Grotowski, and D. P. Kroese. Kernel density
# estimation via diffusion. The Annals of Statistics, 38(5):2916-2957, 2010.
#
# Daniel B. Smith, PhD
# Updated 1-23-2013
#
# Original BSD license, applicable *ONLY* to
#     fbw_kde
#     fixed_point
# functions since these were derived from Botev's original work (this license
# applies to any future code derived from those functions as well):
# ============================================================================
#   Copyright (c) 2007, Zdravko Botev
#   All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are
#   met:
#
#       * Redistributions of source code must retain the above copyright
#         notice, this list of conditions and the following disclaimer.
#       * Redistributions in binary form must reproduce the above copyright
#         notice, this list of conditions and the following disclaimer in
#         the documentation and/or other materials provided with the
#         distribution
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
#   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
#   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
#   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================
#
# Further modifications by author J. L. Lanfranchi:
#
# 2015-02-24: Faster via quad -> double precision, more numpy vectorized
#   functions, numexpr for a couple of the slower evaluations. Note that the
#   double precision may make this fail in some circumstances, but I haven't
#   seen it do so yet. Regardless, modifying the calls to float64 -> float128
#   and eliminating the numexpr calls (only supports doubles) should make it
#   equivalent to the original implementation.
#
# 2015-03-09: Add variable-bandwidth implementation that does the following:
#   1) compute optimal bandwidth using the improved-Sheather-Jones (ISJ)
#      algorithm described in the Botev paper cited above
#   2) Use a modified version of the variable-bandwidth algorithm described in:
#        I.S. Abramson, On bandwidth variation in kernel estimates - A square
#        root law, Annals of Stat. Vol. 10, No. 4, 1217-1223 1982.
#      The modification I made to this Ambramson paper is to force the
#      peak-density point to use the ISJ BW found in step (1). This is done by
#      dividing the inverse-square-root bandwidths by the bandwidth at the
#      peak-density point and multiplying by the ISJ BW. (This appears to do
#      quite well at both capturing the peak's characteristics and smoothing
#      out bumps in the tails, but we should be cautious if false structures
#      near the peak may arise due to densities similar to that of the peak.)
#
# 2015-03-28:
#   * Removed numexpr pieces to make an as-universal-as-possible
#     implementation, instead using far more optimized gaussian computation
#     routines in a separate Cython .pyx file if the user can compile the
#     Cython (depends upon OpenMP and Cython).
#
"""
An implementation of the kde bandwidth selection method outlined in:

Z. I. Botev, J. F. Grotowski, and D. P. Kroese. Kernel density
estimation via diffusion. The Annals of Statistics, 38(5):2916-2957, 2010.
"""


from __future__ import division

from math import exp, sqrt
import os
import threading

from numba import cuda, jit
import numpy as np
from scipy import fftpack, optimize, interpolate

from pisa import FTYPE, OMP_NUM_THREADS
from pisa.utils.log import logging, set_verbosity
from pisa.utils.profiler import line_profile, profile


__all__ = ['CUDA_PRESENT', 'gaussians', 'fbw_kde', 'vbw_kde', 'fixed_point']


CUDA_PRESENT = True
PI = np.pi
TWOPI = 2*PI
SQRTPI = np.sqrt(PI)
SQRT2PI = np.sqrt(TWOPI)
PISQ = PI**2


def gaussians(x, mu, sigma, implementation=None):
    """Sum of multiple Gaussian curves.

    Parameters
    ----------
    x : array
        Points at which to evaluate the sum of Gaussians

    mu, sigma : arrays
        Means and standard deviations of the Gaussians to accumulate

    implementation : None or string
        One of 'singlethreaded', 'multithreaded', or 'cuda'. Passing None, the
        function will try to determine which of the implementations is best to
        call.

    Returns
    -------
    outbuf : array
        Resulting sum of Gaussians

    Notes
    -----
    This function dynamically calls an appropriate implementation depending
    upon the problem size, the hardware available (multi-core CPU or a GPU),
    and whether the user specifies `implementation`.

    """
    # TODO: figure out which is the roughly the best function to call, if more
    # than one function is available to call (i.e., really small problems are
    # fastest in single core, really large `x` are probably best on GPU, and
    # if/where multithreaded CPU beats GPU is still up in the air
    if implementation is not None:
        implementation = implementation.strip().lower()
        assert implementation in ['cuda', 'singlethreaded', 'multithreaded']

    if implementation == 'cuda' or CUDA_PRESENT:
        logging.trace('Using CUDA Gaussians implementation')
        return _gaussians_cuda(x, mu, sigma)

    if implementation == 'singlethreaded' or OMP_NUM_THREADS == 1:
        logging.trace('Using single-threaded Gaussians implementation')
        outbuf = np.zeros_like(x, dtype=FTYPE)
        _gaussians_singlethreaded(outbuf, x, mu, sigma)
        return outbuf

    if implementation == 'multithreaded' or OMP_NUM_THREADS > 1:
        logging.trace('Using multi-threaded Gaussians implementation')
        return _gaussians_multithreaded(x, mu, sigma)

    raise ValueError(
        'Unhandled value(s): OMP_NUM_THREADS="%s", CUDA_PRESENT="%s"'
        % (OMP_NUM_THREADS, CUDA_PRESENT)
    )


def _gaussians_multithreaded(x, mu, sigma):
    """Sum of multiple guassians, optimized to be run in multiple threads. This
    dispatches the single-kernel threaded """
    n_points = len(x)
    outbuf = np.zeros(shape=n_points, dtype=FTYPE)
    chunklen = n_points // OMP_NUM_THREADS
    threads = []
    start = 0
    for i in range(OMP_NUM_THREADS):
        stop = n_points if i == (OMP_NUM_THREADS - 1) else start + chunklen
        thread = threading.Thread(
            target=_gaussians_singlethreaded,
            args=(outbuf[start:stop], x[start:stop], mu, sigma)
        )
        thread.start()
        threads.append(thread)
        start += chunklen
    [thread.join() for thread in threads]
    return outbuf


@jit('void({f:s}[:], {f:s}[:], {f:s}[:], {f:s}[:])'.format(f=FTYPE.__name__),
     nopython=True, nogil=False, cache=True)
def _gaussians_singlethreaded(outbuf, x, mu, sigma):
    """Sum of multiple guassians, optimized to be run in a single thread"""
    n_points = len(x)
    for mu_, sigma_ in zip(mu, sigma):
        for i in range(n_points):
            xlessmu = x[i] - mu_
            outbuf[i] += (1/(SQRT2PI*sigma_) * exp(-xlessmu*xlessmu/(2*sigma_*sigma_)))


# TODO: if this is loaded, then other CUDA (e.g. pycuda) code doesn't run. Need
# to fix this behavior.

if CUDA_PRESENT:
    def _gaussians_cuda(x, mu, sigma):
        n_points, n_gaussians = len(x), len(mu)
        n_ops = n_points * n_gaussians
        threads_per_block = 32
        blocks_per_grid = (n_points + (threads_per_block - 1)) // threads_per_block
        func = _gaussians_cuda_kernel[blocks_per_grid, threads_per_block]

        outbuf = np.empty(shape=n_points, dtype=FTYPE)

        # Create empty array on GPU
        d_outbuf = cuda.device_array(shape=n_points, dtype=FTYPE, stream=0)

        # Copy other arguments to GPU
        d_x = cuda.to_device(x)
        d_mu = cuda.to_device(mu)
        d_sigma = cuda.to_device(sigma)

        # Call the function
        func(d_outbuf, d_x, d_mu, d_sigma)

        # Copy contents of GPU result to host's outbuf
        d_outbuf.copy_to_host(ary=outbuf, stream=0)

        return outbuf


    @cuda.jit('void({f:s}[:], {f:s}[:], {f:s}[:], {f:s}[:])'.format(f=FTYPE.__name__),
              inline=True)
    def _gaussians_cuda_kernel(outbuf, x, mu, sigma):
        pt_idx = cuda.grid(1)
        tot = 0.0
        for g_idx in range(len(mu)):
            s = sigma[g_idx]
            m = mu[g_idx]
            xlessmu = x[pt_idx] - m
            tot += (
                1 / (SQRT2PI*s) * exp(-(xlessmu*xlessmu) / (2*(s*s)))
            )
        outbuf[pt_idx] = tot


@profile
def fbw_kde(data, n_dct=2**14, MIN=None, MAX=None):
    """Fixed-bandwidth (standard) Gaussian KDE using the Improved
    Sheather-Jones bandwidth.

    Code adapted for Python from the implementation in Matlab by Zdravko Botev.

    Ref: Z. I. Botev, J. F. Grotowski, and D. P. Kroese. Kernel density
    estimation via diffusion. The Annals of Statistics, 38(5):2916-2957, 2010.

    Parameters
    ----------
    data : array
    n_dct : int
        Preferably an integer power of 2 for speed purposes
    MIN : float or None
    MAX : float or None

    Returns
    -------
    bandwidth : float
    mesh : array
    density : array

    """
    assert int(n_dct) == n_dct
    n_dct = int(n_dct)

    # Parameters to set up the mesh on which to calculate
    if MIN is None or MAX is None:
        minimum = data.min()
        maximum = data.max()
        data_range = maximum - minimum
        MIN = minimum - data_range/2 if MIN is None else MIN
        MAX = maximum + data_range/2 if MAX is None else MAX

    # Range of the data
    hist_range = MAX - MIN

    # Histogram the data to get a crude first approximation of the density
    data_len = int(len(data))
    data_hist, bins = np.histogram(data, bins=n_dct, range=(MIN, MAX))
    data_hist = data_hist/data_len

    dct_data = fftpack.dct(data_hist, norm=None)

    I = np.arange(1, n_dct, dtype=np.float64)**2
    dct_data_sq = (dct_data[1:]/2)**2

    # The fixed point calculation finds the bandwidth = t_star
    failure = True
    t_star = None
    bandwidth = None
    try:
        t_star = optimize.minimize_scalar(
            fun=fixed_point,
            bounds=(np.finfo(np.float64).eps, 0.1),
            method='Bounded',
            args=(data_len, I, dct_data_sq.astype(np.float64)),
            options=dict(maxiter=1e6, xatol=1e-22),
        ).x
    except ValueError:
        pass
    else:
        failure = False
        bandwidth = np.sqrt(t_star)*hist_range

    if failure:
        if R_PRESENT:
            logging.warning('Failed to find roots in Python; will try R.')
        else:
            raise ValueError('Initial root-finding failed.')

        bandwidth = bw
        t_star = (bw/hist_range)**2

    # Smooth the DCTransformed data using t_star
    sm_dct_data = (
        dct_data*np.exp(-np.arange(n_dct)**2 * PISQ*t_star/2)
    )

    # Inverse DCT to get density
    density = fftpack.idct(sm_dct_data, norm=None)*n_dct/hist_range

    mesh = (bins[0:-1] + bins[1:]) / 2

    density = density/np.trapz(density, mesh)

    return bandwidth, mesh, density


@profile
def vbw_kde(data, n_dct=2**14, MIN=None, MAX=None, evaluate_dens=True,
            evaluate_at=None, n_addl_iter=0):
    """
    Parameters
    ----------
    data : array
        The data points for which the density estimate is sought

    n_dct : int
        Number of points with which to form regular mesh, from MIN to MAX;
        this gets DCT'd, so N should be a power of two.
        -> Default: 2**14 (16384)

    MIN : float or None
        Minimum of range over which to compute density.
        If None, defaults to min(data) - range(data)/2

    MAX : float or None
        Maximum of range over which to compute density>
        If None: max(data) + range(data)/2

    evaluate_dens : bool
        Whether to evaluate the density either at the mesh points defined by
        n_dct, MIN, and MAX, or at the points specified by the argument
        evaluate_at. If False, only the gaussians' bandwidths and the mesh
        locations (no density) are returned. Evaluating the density is a large
        fraction of total execution time, so setting this to False saves time
        if only the bandwidths are desired.
        -> Default: True

    evaluate_at
        Points at which to evaluate the density. If None is specified,
        evaluates at points on the mesh defined by MIN, MAX, and N.
        -> Default: None

    Returns
    -------
    kernel_bandwidths
        The gaussian bandwidths, one for each data point

    evaluate_at
        Locations at which the density is evaluated

    vbw_dens_est
        Density estimates at the mesh points, or None if evaluate_dens is
        False

    Notes
    -----
    Specifying the range:

        The specification of MIN and MAX are critical for obtaining a
        reasonable density estimate. If the true underlying density slowly
        decays to zero on one side or the other, like a gaussian, specifying
        too-small a range will distort the edge the VBW-KDE finds. On the
        other hand, an abrupt cut-off in the distribution should be
        accompanied by a similar cutoff in the computational range (MIN and/or
        MAX). The algorithm here will approximate such a sharp cut-off with
        roughly the same performance to the reflection method for standard
        KDE's (as the fixed-BW portion uses a DCT of the data), but note that
        this will not perform as well as polynomial-edges or other
        modifications that have been proposed in the literature.

    """
    # The pilot density estimate is given by the (fixed-bandwidth) Gaussian KDE
    # using the Improved Sheather Jones (ISJ) bandwidth
    bw_at_peak, mesh, fbw_dens_on_mesh = fbw_kde(data=data, n_dct=n_dct,
                                                 MIN=MIN, MAX=MAX)

    # Create linear interpolator for this new density then find density est. at
    # the data points' locations
    interp = interpolate.interp1d(
        x=mesh,
        y=fbw_dens_on_mesh,
        kind='linear',
        copy=False,
        bounds_error=True,
        fill_value=np.nan
    )
    fbw_dens_at_datapoints = interp(data)

    # Note below diverges from the published Ambramson method, by forcing the
    # bandwidth at the max of the density distribution to be exactly the
    # bandwidth found above with the improved Sheather-Jones BW selection
    # technique. Refs:
    #   I.S. Abramson, On bandwidth variation in kernel estimates - A square
    #       root law, Annals of Stat. Vol. 10, No. 4, 1217-1223 1982
    #   P. Hall, T. C. Hu, J. S. Marron, Improved Variable Window Kernel
    #       Estimates of Probability Densities, Annals of Statistics Vol. 23,
    #       No. 1, 1-10, 1995
    root_pknorm_fbw_dens_est = (
        np.sqrt(fbw_dens_at_datapoints / np.max(fbw_dens_at_datapoints))
    )
    kernel_bandwidths = bw_at_peak / root_pknorm_fbw_dens_est

    # TODO: feed density estimate to point just prior to taking DCT in fbw_kde
    for n in xrange(n_addl_iter):
        dens_est = np.zeros_like(data, dtype=FTYPE)
        gaussians(
            outbuf=dens_est,
            x=data,
            mu=data,
            sigma=kernel_bandwidths,
        )
        dens_est /= len(data)
        kernel_bandwidths = np.sqrt(np.sum(dens_est**2)) / dens_est

    if evaluate_at is None:
        evaluate_at = mesh

    if not evaluate_dens:
        return kernel_bandwidths, evaluate_at, None

    # Note that the array must be initialized to zeros before sending to the
    # `gaussians` function (which adds its results to the existing array)
    vbw_dens_est = gaussians(
        x=evaluate_at.astype(FTYPE),
        mu=data.astype(FTYPE),
        sigma=kernel_bandwidths.astype(FTYPE),
    )

    # Normalize distribution to have area of 1
    vbw_dens_est = vbw_dens_est / len(data)

    return kernel_bandwidths, evaluate_at, vbw_dens_est


@jit('{f:s}({f:s}, int64, {f:s}[:], {f:s}[:])'.format(f='float64'),
     nopython=True, nogil=False, cache=True)
def fixed_point(t, data_len, I, a2):
    """Fixed point algorithm for Improved Sheather Jones bandwidth
    selection.

    Implements the function t - zeta*gamma**[l](t) from The Annals of Statistics, 38(5):2916-2957, 2010.


    Parameters
    ----------
    t : float64
    data_len : int64
    I : array of float64
    a2 : array of float64


    Returns
    -------
    result : float64


    NOTES
    -----
    Original implementation by Botev et al. is quad-precision, whereas this is
    single precision only. This could cause discrepancies from the reference
    implementation.

    """
    len_i = len(I)
    l = 7

    tot = 0.0
    for idx in range(len_i):
        tot += I[idx]**l * a2[idx] * exp(-I[idx] * PISQ * t)
    f = 2 * PI**(2*l) * tot

    for s in range((l-1), 1, -1):
        k0 = np.nanprod(np.arange(1, 2*s, 2)) / SQRT2PI
        const = (1 + (0.5)**(s+0.5)) / 3.0
        time = (2 * const * k0 / data_len / f)**(2.0 / (3.0 + 2.0*s))

        tot = 0.0
        for idx in range(len_i): #I_, a2_ in zip(I, a2):
            tot += I[idx]**s * a2[idx] * exp(-I[idx] * PISQ * time)
        f = 2 * PI**(2*s) * tot

    return abs(t - (2.0 * data_len * SQRTPI * f)**(-0.4))


def speedTest():
    from time import time
    np.random.seed(10)
    times = []
    for trial in xrange(10):
        enuerr = np.random.noncentral_chisquare(df=3, nonc=1, size=int(1e3))
        min_e, max_e = 0, np.max(enuerr)
        x = np.linspace(0, 20, 1e3)
        t0 = time()
        bw, x, dens = vbw_kde(data=enuerr, n_dct=2**14, evaluate_at=x)
        times.append(time() - t0)
    #set_verbosity(1)
    #logging.info('average time to run: %f s' % np.mean(times))
    print 'average time to run: %f ms' % (np.mean(times)*1000)


if __name__ == "__main__":
    #set_verbosity(0)
    speedTest()
