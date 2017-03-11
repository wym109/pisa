#!/usr/bin/env python
#
# J.L. Lanfranchi for the IceCube/PINGU collaboration.
#
# Based on the implementation in Matlab by Zdravko Botev, and the paper
# Z. I. Botev, J. F. Grotowski, and D. P. Kroese. Kernel density
# estimation via diffusion. The Annals of Statistics, 38(5):2916-2957, 2010.
#
# Original BSD license, applicable *ONLY* to functions "isj_bandwidth" and
# "fixed_point" since these were derived from Botev's original work (this
# license applies to any future code derived from those functions as well):
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
"""
Implementation of the Improved Sheather Jones (ISJ) KDE bandwidth selection
method outlined in:

  Z. I. Botev, J. F. Grotowski, and D. P. Kroese. Kernel density
    estimation via diffusion. The Annals of Statistics, 38(5):2916-2957, 2010.

and extension to use this in varaible bandwidth KDE via

  I.S. Abramson, On bandwidth variation in kernel estimates - A
    square root law, Annals of Stat. Vol. 10, No. 4, 1217-1223 1982

See also

  P. Hall, T. C. Hu, J. S. Marron, Improved Variable Window Kernel
    Estimates of Probability Densities, Annals of Statistics
    Vol. 23, No. 1, 1-10, 1995
"""


# TODO: Make normalization use a relative density value normalized after ISJ
# bandwidth to take into consideration really-high densities (which can have
# narrower bandwidths than ISJ dictates) or really-low densities (which should
# have very wide bandwidths, wider than ISJ dictates)


from __future__ import division

from collections import Iterable, OrderedDict
from math import exp, sqrt
import threading
from time import time

import numpy as np
from scipy import fftpack, interpolate, optimize, stats

from pisa import FTYPE, numba_jit
from pisa.utils.gaussians import gaussians
from pisa.utils.log import logging, set_verbosity


__all__ = ['fbw_kde', 'vbw_kde', 'isj_bandwidth', 'fixed_point',
           'test_fbw_kde', 'test_vbw_kde', 'test_isj_bandwidth',
           'test_fixed_point']


PI = FTYPE(np.pi)
TWOPI = FTYPE(2*PI)
SQRTPI = FTYPE(sqrt(PI))
SQRT2PI = FTYPE(sqrt(TWOPI))
PISQ = FTYPE(PI**2)

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
    n_datapoints : int
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

    i_range = np.arange(1, n_dct, dtype=np.float64)**2

    dct_data = fftpack.dct(y, norm=None)
    dct_data_sq = (0.25 * (dct_data * dct_data))[1:]

    try:
        t_star = optimize.minimize_scalar(
            fun=fixed_point,
            bounds=(np.finfo(np.float64).eps, 0.1),
            method='Bounded',
            args=(n_datapoints, i_range, dct_data_sq),
            options=dict(maxiter=1e6, xatol=1e-22),
        ).x
    except ValueError:
        logging.error('ISJ root-finding failed.')
        raise

    bandwidth = sqrt(t_star)*x_range

    return bandwidth, t_star, dct_data


@numba_jit('{f:s}({f:s}, int64, {f:s}[:], {f:s}[:])'.format(f='float64'),
           nopython=True, nogil=False, cache=True)
def fixed_point(t, data_len, i_range, a2):
    """Fixed point algorithm for Improved Sheather Jones bandwidth
    selection.

    Implements the fixed-point finding for the function
    ``t - zeta * gamma^{[l]}(t)``
    See The Annals of Statistics, 38(5):2916-2957, 2010.

    Parameters
    ----------
    t : float64
    data_len : int64
    i_range : array of float64
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
    len_i = len(i_range)
    l = 7

    tot = 0.0
    for idx in range(len_i):
        tot += i_range[idx]**l * a2[idx] * exp(-i_range[idx] * PISQ * t)
    f = 2 * PI**(2*l) * tot

    for s in range((l-1), 1, -1):
        k0 = np.nanprod(np.arange(1, 2*s, 2)) / SQRT2PI
        const = (1 + (0.5)**(s+0.5)) / 3.0
        t_elap = (2 * const * k0 / data_len / f)**(2.0 / (3.0 + 2.0*s))

        tot = 0.0
        for idx in range(len_i): #I_, a2_ in zip(I, a2):
            tot += i_range[idx]**s * a2[idx] * exp(-i_range[idx] * PISQ * t_elap)
        f = 2 * PI**(2*s) * tot

    return abs(t - (2.0 * data_len * SQRTPI * f)**(-0.4))


def test_fbw_kde():
    """Test speed and accuracy of fbw_kde implementation"""
    n_samp = int(1e4)
    n_dct = int(2**14)
    n_eval = int(1e4)
    x = np.linspace(0, 20, n_eval)
    np.random.seed(0)
    times = []
    for _ in xrange(3):
        enuerr = np.random.noncentral_chisquare(df=3, nonc=1, size=n_samp)
        t0 = time()
        fbw_kde(data=enuerr, n_dct=n_dct, evaluate_at=x)
        times.append(time() - t0)
    logging.info('median time to run fbw_kde, %d samples %d dct,'
                 ' eval. at %d points: %f ms'
                 % (n_samp, n_dct, n_eval, np.median(times)*1000))
    logging.info('<< PASS : test_fbw_kde >>')


def test_vbw_kde():
    """Test speed and accuracy of vbw_kde implementations"""
    n_samp = int(1e4)
    n_dct = int(2**14)
    n_eval = int(1e4)
    n_addl = 1
    x = np.linspace(0, 20, n_samp)
    np.random.seed(0)
    times = []
    for _ in xrange(3):
        enuerr = np.random.noncentral_chisquare(df=3, nonc=1, size=n_eval)
        t0 = time()
        vbw_kde(data=enuerr, n_dct=n_dct, evaluate_at=x, n_addl_iter=n_addl)
        times.append(time() - t0)
    logging.info('median time to run vbw_kde, %d samples %d dct %d addl iter,'
                 ' eval. at %d points: %f ms'
                 % (n_samp, n_dct, n_addl, n_eval, np.median(times)*1000))
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
