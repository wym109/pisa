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

from collections import Iterable, OrderedDict
from math import exp, sqrt
import os
import threading
from time import time

NUMBA_ENV_VARS = ['NUMBA_WARNINGS', 'NUMBA_DEBUG', 'NUMBA_TRACE',
                  'NUMBA_CUDA_LOG_LEVEL']
for env_var in NUMBA_ENV_VARS:
    if not os.environ.has_key(env_var):
        os.environ[env_var] = '0'

from numba import cuda, jit
import numpy as np
from scipy import fftpack, interpolate, optimize, stats

from pisa import FTYPE, OMP_NUM_THREADS
from pisa.utils.log import logging, set_verbosity


__all__ = ['NUMBA_ENV_VARS', 'CUDA_PRESENT', 'GAUS_IMPLEMENTATIONS',
           'gaussians', 'fbw_kde', 'vbw_kde', 'isj_bandwidth', 'fixed_point',
           'test_gaussuans', 'test_fbw_kde', 'test_vbw_kde',
           'test_isj_bandwidth', 'test_fixed_point']


# TODO: if the Numba CUDA functions are defined, then other CUDA (e.g. pycuda)
# code doesn't run. Need to fix this behavior. (E.g. context that gets
# destroyed?)

# TODO: Numba CUDA info/warning/debug/etc. messages are output when PISA's
# logging is configured to output the same, but these messages are far too many
# for this behavior to be useful in general... so must figure out how to turn
# off (or on) Numba messages while leaving PISA messages on


PI = np.pi
TWOPI = 2*PI
SQRTPI = sqrt(PI)
SQRT2PI = sqrt(TWOPI)
PISQ = PI**2

CUDA_PRESENT = False
GAUS_IMPLEMENTATIONS = ['singlethreaded', 'multithreaded']
if CUDA_PRESENT:
    GAUS_IMPLEMENTATIONS.append('cuda')


def gaussians(x, mu, sigma, implementation=None):
    """Sum of multiple Gaussian curves, normalized to have area of 1.

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
        if not implementation in GAUS_IMPLEMENTATIONS:
            raise ValueError('`implementation` must be one of %s'
                             % GAUS_IMPLEMENTATIONS)

    # Convert all inputs to arrays of correct datatype
    if not isinstance(x, Iterable):
        x = [x]
    if not isinstance(mu, Iterable):
        mu = [mu]
    if not isinstance(sigma, Iterable):
        sigma = [sigma]
    x = np.asarray(x, dtype=FTYPE)
    mu = np.asarray(mu, dtype=FTYPE)
    sigma = np.asarray(sigma, dtype=FTYPE)

    n_points = len(x)

    # Instantiate an empty output buffer
    outbuf = np.empty(shape=n_points, dtype=FTYPE)

    if implementation == 'cuda' or (implementation is None and CUDA_PRESENT):
        logging.trace('Using CUDA Gaussians implementation')
        _gaussians_cuda(outbuf, x, mu, sigma)

    elif implementation == 'singlethreaded' or (implementation is None and
                                                OMP_NUM_THREADS == 1):
        logging.trace('Using single-threaded Gaussians implementation')
        _gaussians_singlethreaded(outbuf, x, mu, sigma, start=0, stop=n_points)

    elif implementation == 'multithreaded' or OMP_NUM_THREADS > 1:
        logging.trace('Using multi-threaded Gaussians implementation')
        _gaussians_multithreaded(outbuf, x, mu, sigma)

    else:
        raise ValueError(
            'Unhandled value(s): OMP_NUM_THREADS="%s", CUDA_PRESENT="%s",'
            ' `implementation`="%s"'
            % (OMP_NUM_THREADS, CUDA_PRESENT, implementation)
        )

    return outbuf


def _gaussians_multithreaded(outbuf, x, mu, sigma):
    """Sum of multiple guassians, optimized to be run in multiple threads. This
    dispatches the single-kernel threaded """
    n_points = len(x)
    chunklen = n_points // OMP_NUM_THREADS
    threads = []
    start = 0
    for i in range(OMP_NUM_THREADS):
        stop = n_points if i == (OMP_NUM_THREADS - 1) else start + chunklen
        thread = threading.Thread(
            target=_gaussians_singlethreaded,
            args=(outbuf, x, mu, sigma, start, stop)
        )
        thread.start()
        threads.append(thread)
        start += chunklen
    [thread.join() for thread in threads]


GAUS_ST_FUNCSIG = (
    (
        'void({f:s}[:], {f:s}[:], {f:s}[:], {f:s}[:], int64, int64)'
    ).format(f=FTYPE.__name__)
)

@jit(GAUS_ST_FUNCSIG, nopython=True, nogil=True, cache=True)
def _gaussians_singlethreaded(outbuf, x, mu, sigma, start, stop):
    """Sum of multiple guassians, optimized to be run in a single thread"""
    factor = 1/(SQRT2PI * len(mu))
    for i in range(start, stop):
        tot = 0.0
        for mu_j, sigma_j in zip(mu, sigma):
            xlessmu = x[i] - mu_j
            tot += (
                exp(-(xlessmu*xlessmu)/(2*(sigma_j*sigma_j))) * factor/sigma_j
            )
        outbuf[i] = tot


if CUDA_PRESENT:
    def _gaussians_cuda(outbuf, x, mu, sigma):
        n_points = len(x)
        threads_per_block = 32
        blocks_per_grid = (
            (n_points + (threads_per_block - 1)) // threads_per_block
        )
        func = _gaussians_cuda_kernel[blocks_per_grid, threads_per_block]

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

        del d_x, d_mu, d_sigma, d_outbuf


    GAUS_CUDA_FUNCSIG = (
        (
            'void({f:s}[:], {f:s}[:], {f:s}[:], {f:s}[:])'
        ).format(f=FTYPE.__name__)
    )
    @cuda.jit(GAUS_CUDA_FUNCSIG, inline=True)
    def _gaussians_cuda_kernel(outbuf, x, mu, sigma):
        pt_idx = cuda.grid(1)
        n_gaussians = len(mu)
        tot = 0.0
        factor = 1/(SQRT2PI * n_gaussians)
        for g_idx in range(n_gaussians):
            s = sigma[g_idx]
            m = mu[g_idx]
            xlessmu = x[pt_idx] - m
            tot += (
                exp(-(xlessmu*xlessmu) / (2*(s*s))) * factor / s
            )
        outbuf[pt_idx] = tot


def fbw_kde(data, n_dct=None, min=None, max=None, evaluate_dens=True,
            evaluate_at=None):
    """Fixed-bandwidth (standard) Gaussian KDE using the Improved
    Sheather-Jones bandwidth.

    Code adapted for Python from the implementation in Matlab by Zdravko Botev.

    Ref: Z. I. Botev, J. F. Grotowski, and D. P. Kroese. Kernel density
    estimation via diffusion. The Annals of Statistics, 38(5):2916-2957, 2010.

    Parameters
    ----------
    data : array

    n_dct : None or int
        Number of points with which to form a regular grid, from `min` to
        `max`; histogram values at these points are sent through a discrete
        Cosine Transform (DCT), so `n_dct` should be an integer power of 2 for
        speed purposes. If None, uses next-highest-power-of-2 above
        len(data)*10.

    min : float or None
    max : float or None
    evaluate_dens : bool
    evaluate_at : None or array

    Returns
    -------
    bandwidth : float
    mesh : array of float
    density : array of float

    """
    if n_dct is None:
        n_dct = int(2**np.ceil(np.log2(len(data)*10)))
    assert int(n_dct) == n_dct
    n_dct = int(n_dct)
    data_len = len(data)

    # Parameters to set up the mesh on which to calculate
    if min is None or max is None:
        minimum = data.min()
        maximum = data.max()
        data_range = maximum - minimum
        min = minimum - data_range/2 if min is None else min
        max = maximum + data_range/2 if max is None else max

    hist_range = max - min

    # Histogram the data to get a crude first approximation of the density
    data_hist, bins = np.histogram(
        data, bins=n_dct, range=(min, max), normed=False
    )
    data_hist = data_hist.astype(FTYPE) / FTYPE(data_len)

    isj_bw, t_star, dct_data = isj_bandwidth(
        y=data_hist, n_datapoints=data_len, x_range=hist_range
    )

    if not evaluate_dens:
        return isj_bw, evaluate_at, None

    if evaluate_at is None:
        # Smooth the discrete-cosine-transformed data using t_star
        sm_dct_data = dct_data*np.exp(-np.arange(n_dct)**2 * PISQ*t_star/2)

        # Inverse DCT to get density
        density = fftpack.idct(sm_dct_data, norm=None)*n_dct/hist_range
        mesh = (bins[0:-1] + bins[1:]) / 2
        density = density/np.trapz(density, mesh)
        return isj_bw, mesh, density

    evaluate_at = np.asarray(evaluate_at, dtype=FTYPE)
    density = gaussians(
        x=evaluate_at,
        mu=data.astype(FTYPE),
        sigma=np.full(shape=data_len, fill_value=isj_bw, dtype=FTYPE)
    )

    return isj_bw, evaluate_at, density


def vbw_kde(data, n_dct=None, min=None, max=None, n_addl_iter=0,
            evaluate_dens=True, evaluate_at=None):
    """Variable-bandwidth (standard) Gaussian KDE that uses the function
    `fbw_kde` for a pilot density estimate.

    Parameters
    ----------
    data : array
        The data points for which the density estimate is sought

    n_dct : None or int
        Number of points with which to form a regular grid, from `min` to
        `max`; histogram values at these points are sent through a discrete
        Cosine Transform (DCT), so `n_dct` should be an integer power of 2 for
        speed purposes. If None, uses next-highest-power-of-2 above
        len(data)*10.

    min : None or float
        Minimum of range over which to compute density.
        If None, defaults to min(data) - range(data)/2

    max : None or float
        Maximum of range over which to compute density>
        If None: max(data) + range(data)/2

    n_addl_iter : int >= 0
        Number of additional iterations on the Abramson VBW density *after*
        the initial VBW estimate. This can help smooth the tails of the
        distribution, at the expense of additional computational cost.

    evaluate_dens : bool
        Whether to evaluate and return the density estimate on the points
        defined by `evaluate_at`

    evaluate_at : None, float, or array of float
        Point(s) at which to evaluate the density estimate. If None,
          evaluate_at = np.linspace(min + dx/2, max - dx/2, n_dct)
        where
          dx = (max - min) / n_dct

    Returns
    -------
    kernel_bandwidths : array

    evaluate_at : array
        Locations at which the density estimates would be evaluated

    density : array
        Density estimates


    Notes
    -----
    Specifying the range:

        The specification of min and max are critical for obtaining a
        reasonable density estimate. If the true underlying density slowly
        decays to zero on one side or the other, like a gaussian, specifying
        too-small a range will distort the edge the VBW-KDE finds. On the
        other hand, an abrupt cut-off in the distribution should be
        accompanied by a similar cutoff in the computational range (min and/or
        max). The algorithm here will approximate such a sharp cut-off with
        roughly the same performance to the reflection method for standard
        KDE's (as the fixed-BW portion uses a DCT of the data), but note that
        this will not perform as well as polynomial-edges or other
        modifications that have been proposed in the literature.

    """
    if n_dct is None:
        n_dct = int(2**np.ceil(np.log2(len(data)*10)))
    assert n_addl_iter >= 0 and int(n_addl_iter) == n_addl_iter
    n_addl_iter = int(n_addl_iter)

    # Pilot density estimate for the VBW KDE comes from fixed bandwidth KDE
    # using the Improved Sheather-Jones algorithm. By specifying
    # `evaluate_at` to be None, `fbw_kde` derives a regular grid at which to
    # evaluate the points and does so without needing to do a sum of Gaussians
    # (only a freq.-domain multiply and inverse DCT)
    isj_bw, grid, pilot_dens_on_grid = fbw_kde(
        data=data, n_dct=n_dct, min=min, max=max, evaluate_dens=True,
        evaluate_at=None
    )

    # Use linear interpolation to get density at the data points
    interp = interpolate.interp1d(
        x=grid,
        y=pilot_dens_on_grid,
        kind='linear',
        copy=False,
        bounds_error=True,
        fill_value=np.nan
    )
    pilot_dens_at_datapoints = interp(data)

    n_iter = 1 + n_addl_iter
    for n in xrange(n_iter):
        # Note below diverges from the published Ambramson method, by forcing
        # the bandwidth at the max of the density distribution to be exactly
        # the bandwidth found above with the improved Sheather-Jones BW
        # selection technique. Refs:
        #   I.S. Abramson, On bandwidth variation in kernel estimates - A
        #       square root law, Annals of Stat. Vol. 10, No. 4, 1217-1223 1982
        #   P. Hall, T. C. Hu, J. S. Marron, Improved Variable Window Kernel
        #       Estimates of Probability Densities, Annals of Statistics
        #       Vol. 23, No. 1, 1-10, 1995
        kernel_bandwidths = (
            isj_bw * np.sqrt(np.max(pilot_dens_at_datapoints))
            / np.sqrt(pilot_dens_at_datapoints)
        )

        if n < n_addl_iter:
            pilot_dens_at_datapoints = gaussians(
                x=data,
                mu=data,
                sigma=kernel_bandwidths,
            )

        else: # final iteration
            if evaluate_at is None:
                evaluate_at = grid
            if evaluate_dens:
                density = gaussians(
                    x=evaluate_at,
                    mu=data,
                    sigma=kernel_bandwidths,
                )
            else:
                density = None

    return kernel_bandwidths, evaluate_at, density


def isj_bandwidth(y, n_datapoints, x_range):
    """
    Parameters
    ----------
    y : array of float
    x_range : float

    Returns
    -------
    bandwidth : float
    t_star : float
    dct_data : array of float

    """
    # Ensure double-precision datatypes are used
    y = np.asarray(y, dtype=np.float64)
    x_range = np.float64(x_range)

    n_dct = len(y)

    I = np.arange(1, n_dct, dtype=np.float64)**2

    dct_data = fftpack.dct(y, norm=None)
    dct_data_sq = (0.25 * (dct_data * dct_data))[1:]

    try:
        t_star = optimize.minimize_scalar(
            fun=fixed_point,
            bounds=(np.finfo(np.float64).eps, 0.1),
            method='Bounded',
            args=(n_datapoints, I, dct_data_sq),
            options=dict(maxiter=1e6, xatol=1e-22),
        ).x
    except ValueError:
        logging.error('ISJ root-finding failed.')
        raise

    bandwidth = sqrt(t_star)*x_range

    return bandwidth, t_star, dct_data


@jit('{f:s}({f:s}, int64, {f:s}[:], {f:s}[:])'.format(f='float64'),
     nopython=True, nogil=False, cache=True)
def fixed_point(t, data_len, I, a2):
    """Fixed point algorithm for Improved Sheather Jones bandwidth
    selection.

    Implements the fixed-point finding for the function
    ``t - zeta * gamma^{[l]}(t)``
    See The Annals of Statistics, 38(5):2916-2957, 2010.

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
    double precision only. This might cause discrepancies from the reference
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
        t_elap = (2 * const * k0 / data_len / f)**(2.0 / (3.0 + 2.0*s))

        tot = 0.0
        for idx in range(len_i): #I_, a2_ in zip(I, a2):
            tot += I[idx]**s * a2[idx] * exp(-I[idx] * PISQ * t_elap)
        f = 2 * PI**(2*s) * tot

    return abs(t - (2.0 * data_len * SQRTPI * f)**(-0.4))


def test_gaussuans():
    """Test `gaussians` function"""
    x = np.linspace(-20, 20, 1e4)
    mu_sigma_sets = [
        (0, 1),
        (np.linspace(-50, 50, 1e1), np.linspace(0.5, 100, 1e1)),
        (np.linspace(-50, 50, 1e2), np.linspace(0.5, 100, 1e2)),
        (np.linspace(-50, 50, 1e3), np.linspace(0.5, 100, 1e3)),
        (np.linspace(-20, 20, 1e4), np.logspace(-3, 3, 1e4)),
    ]
    timings = OrderedDict()
    for impl in GAUS_IMPLEMENTATIONS:
        timings[impl] = []

    for mus, sigmas in mu_sigma_sets:
        if not isinstance(mus, Iterable):
            mus = [mus]
            sigmas = [sigmas]
        for m, s in zip(mus, sigmas):
            g_i = stats.norm.pdf(x, loc=m, scale=s)
            if np.any(np.isnan(g_i)):
                logging.error('g_i: %s' % g_i)
                logging.error('m: %s, s: %s' % (m, s))
                raise Exception()
        ref = np.sum(
            [stats.norm.pdf(x, loc=m, scale=s) for m, s in zip(mus, sigmas)],
            axis=0
        )/len(mus)
        for impl in GAUS_IMPLEMENTATIONS:
            t0 = time()
            test = gaussians(x, mu=mus, sigma=sigmas, implementation=impl)
            dt = time() - t0
            timings[impl].append(np.round(dt*1000, decimals=3))
            #print len(mus), impl, dt*1000
            if not np.allclose(test, ref):
                logging.error('test: %s' % test)
                logging.error('ref : %s' % ref)
                logging.error('diff: %s' % (test - ref))
                logging.error('\nmus:%s\nsigmas: %s' % (mus, sigmas))
                logging.error('implementation: %s' % impl)
    for impl in GAUS_IMPLEMENTATIONS:
        timings_str = ', '.join([format(t, '10.3f') for t in timings[impl]])
        logging.debug('Timings, %15s (ms): %s' % (impl, timings_str))
    logging.info('<< PASS : test_gaussuans >>')


def test_fbw_kde():
    """Test speed and accuracy of fbw_kde implementation"""
    x = np.linspace(0, 20, 1e3)
    np.random.seed(0)
    times = []
    for _ in xrange(10):
        enuerr = np.random.noncentral_chisquare(df=3, nonc=1, size=int(1e3))
        t0 = time()
        fbw_kde(data=enuerr, n_dct=2**14, evaluate_at=x)
        times.append(time() - t0)
    logging.debug('average time to run fbw_kde: %f ms'
                  % (np.mean(times)*1000))
    logging.info('<< PASS : test_fbw_kde >>')


def test_vbw_kde():
    """Test speed and accuracy of vbw_kde implementations"""
    x = np.linspace(0, 20, 1e3)
    np.random.seed(0)
    times = []
    for _ in xrange(10):
        enuerr = np.random.noncentral_chisquare(df=3, nonc=1, size=int(1e3))
        t0 = time()
        vbw_kde(data=enuerr, n_dct=2**14, evaluate_at=x, n_addl_iter=2)
        times.append(time() - t0)
    logging.debug('average time to run vbw_kde: %f ms'
                  % (np.mean(times)*1000))
    logging.info('<< PASS : test_vbw_kde >>')


def test_isj_bandwidth():
    """Test function `isj_bandwidth`"""
    pass


def test_fixed_point():
    """Test function `fixed_point`"""
    pass


if __name__ == "__main__":
    set_verbosity(2)
    test_fixed_point()
    test_isj_bandwidth()
    test_fbw_kde()
    test_vbw_kde()
    test_gaussuans()
