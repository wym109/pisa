#!/usr/bin/env python
#
# Based on the implementation in Matlab by Zdravko Botev.
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

import os

R_PRESENT = False
"""Whether R (via rpy2) is usable from within Python"""
try:
    import rpy2.robjects as robjects
    from rpy2.robjects import numpy2ri
    numpy2ri.activate()
except ImportError:
    pass
else:
    R_PRESENT = True
    mydir = os.path.abspath(os.path.dirname(__file__))
    robjects.r("source('%s/isj_bw.R')" % mydir)
    isj_bw_r_func = robjects.r('isj_bw')

import numpy as np
from scipy import fftpack
from scipy import optimize
from scipy import interpolate

from pisa.utils.log import logging, set_verbosity
set_verbosity(2)
try:
    from pisa.utils.gaussians import gaussian, gaussians
except ImportError:
    logging.warning('pisa.utils.gaussians module not importable, defining'
                    ' custom (slow) functions instead')
    def gaussian(outbuf, x, mu, sigma):
        xlessmu = x-mu
        outbuf += (1./(SQRT2PI*sigma)
                   * np.exp(-xlessmu*xlessmu/(2.*sigma*sigma)))
    def gaussians(outbuf, x, mu, sigma, **kwargs):
        [gaussian(outbuf, x, mu[n], sigma[n]) for n in xrange(len(mu))]


__all__ = ['R_PRESENT', 'CHECK_BW', 'OMP_NUM_THREADS', 'fbw_kde', 'vbw_kde',
           'fixed_point']


CHECK_BW = True
"""Whether to check the bandwidth that the Python code computes against that
computed in the R code"""

PI = np.pi
SQRTPI = np.sqrt(PI)
SQRT2PI = np.sqrt(2*PI)
PISQ = PI**2

OMP_NUM_THREADS = 1
"""Number of threads OpenMP is allocated"""

if os.environ.has_key('OMP_NUM_THREADS'):
    OMP_NUM_THREADS = int(os.environ['OMP_NUM_THREADS'])


def get_isj_bw(hist, n_dft):
    dct_data = fftpack.dct(hist, norm=None)

    I = np.arange(1, n_dft)**2
    dct_data_sq = (dct_data[1:]/2)**2

    # The fixed point calculation finds the bandwidth = t_star
    failure = True
    for guess in np.logspace(-2, 5, 10):
        try:
            t_star = optimize.brentq(
                fixed_point,
                0, guess,
                args=(data_len, I, dct_data_sq),
            )
        except ValueError:
            pass
        else:
            failure = False
            bandwidth = np.sqrt(t_star)*hist_range
            break

    if failure:
        if R_PRESENT:
            logging.warning('Failed to find roots in Python; will try R.')
        else:
            raise ValueError('Initial root-finding failed.')

    if CHECK_BW or failure and R_PRESENT:
        try:
            bw = isj_bw_r_func(data, n_dft=n_dft, MIN=MIN, MAX=MAX)[0]
        except:
            raise
            if failure:
                logging.error('VBW KDE (ISJ) root-finding failed.')
                raise
        else:
            if CHECK_BW and not failure:
                pass
                #logging.info('Py ISJ bw: %e' % bandwidth)
                #logging.info(' R ISJ bw: %e' % bw)
                #logging.info(' (Py-R)/R: %e' % ((bandwidth - bw)/bw))
                #print 'Py ISJ bw : %f' % bandwidth
                #print 'R ISJ bw  : %f' % bw
                #print '(Py-R)/R  :   %e' % ((bandwidth - bw)/bw)

        if failure:
            bandwidth = bw

        bandwidth = bw
        t_star = (bandwidth/hist_range)**2

    # Smooth the DCTransformed data using t_star
    sm_dct_data = (
        dct_data*np.exp(-np.arange(n_dft)**2 * PISQ*t_star/2)
    )

    # Inverse DCT to get density
    density = fftpack.idct(sm_dct_data, norm=None)*n_dft/hist_range



def fbw_kde(data, n_dft=2**14, MIN=None, MAX=None):
    """Fixed-bandwidth (standard) Gaussian KDE using the Improved
    Sheather-Jones bandwidth.

    Code adapted for Python from the implementation in Matlab by Zdravko Botev.

    Ref: Z. I. Botev, J. F. Grotowski, and D. P. Kroese. Kernel density
    estimation via diffusion. The Annals of Statistics, 38(5):2916-2957, 2010.

    Parameters
    ----------
    data : array
    n_dft : int
        Preferably an integer power of 2 for speed purposes
    MIN : float or None
    MAX : float or None

    Returns
    -------
    bandwidth : float
    mesh : array
    density : array

    """
    assert int(n_dft) == n_dft
    n_dft = int(n_dft)

    # Parameters to set up the mesh on which to calculate
    if MIN is None or MAX is None:
        minimum = min(data)
        maximum = max(data)
        data_range = maximum - minimum
        MIN = minimum - data_range/2 if MIN is None else MIN
        MAX = maximum + data_range/2 if MAX is None else MAX

    # Range of the data
    hist_range = MAX - MIN

    # Histogram the data to get a crude first approximation of the density
    data_len = len(data)
    data_hist, bins = np.histogram(data, bins=n_dft, range=(MIN, MAX))
    data_hist = data_hist/data_len

    dct_data = fftpack.dct(data_hist, norm=None)

    I = np.arange(1, n_dft)**2
    dct_data_sq = (dct_data[1:]/2)**2

    # The fixed point calculation finds the bandwidth = t_star
    failure = True
    t_star = None
    bandwidth = None
    #for guess in np.logspace(-2, 5, 200):
    #    try:
    #        t_star = optimize.brentq(
    #            fixed_point,
    #            0, guess,
    #            args=(data_len, I, dct_data_sq),
    #        )
    #        failure = False
    #        bandwidth = np.sqrt(t_star)*hist_range
    #        break
    #    except ValueError:
    #        failure = True

    if failure:
        if R_PRESENT:
            logging.warning('Failed to find roots in Python; will try R.')
        else:
            raise ValueError('Initial root-finding failed.')

    if CHECK_BW or failure and R_PRESENT:
        try:
            bw = isj_bw_r_func(data, n_dft=n_dft, MIN=MIN, MAX=MAX)[0]
        except:
            raise
            if failure:
                logging.error('VBW KDE (ISJ) root-finding failed.')
                raise
        else:
            if CHECK_BW and not failure:
                logging.info('')
                logging.info('Py ISJ bw: %s' % bandwidth)
                logging.info(' R ISJ bw: %s' % bw)
                if bandwidth is not None:
                    logging.info(' (Py-R)/R: %s' % ((bandwidth - bw)/bw))
                #print 'Py ISJ bw : %f' % bandwidth
                #print 'R ISJ bw  : %f' % bw
                #print '(Py-R)/R  :   %e' % ((bandwidth - bw)/bw)

        #if failure:
        #    print 'failed!'
        bandwidth = bw
        t_star = (bw/hist_range)**2

    # Smooth the DCTransformed data using t_star
    sm_dct_data = (
        dct_data*np.exp(-np.arange(n_dft)**2 * PISQ*t_star/2)
    )

    # Inverse DCT to get density
    density = fftpack.idct(sm_dct_data, norm=None)*n_dft/hist_range

    mesh = (bins[0:-1] + bins[1:]) / 2

    density = density/np.trapz(density, mesh)

    return bandwidth, mesh, density


def vbw_kde(data, n_dft=2**14, MIN=None, MAX=None, evaluate_dens=True,
            evaluate_at=None, n_addl_iter=0):
    """
    Parameters
    ----------
    data : array
        The data points for which the density estimate is sought

    n_dft : int
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
        n_dft, MIN, and MAX, or at the points specified by the argument
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
    bw_at_peak, mesh, fbw_dens_on_mesh = fbw_kde(
        data=data,
        n_dft=n_dft,
        MIN=MIN,
        MAX=MAX
    )

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
        dens_est = np.zeros_like(data, dtype=np.double)
        gaussians(
            outbuf=dens_est,
            x=data,
            mu=data,
            sigma=kernel_bandwidths,
            threads=OMP_NUM_THREADS
        )
        dens_est /= len(data)
        kernel_bandwidths = np.sqrt(np.sum(dens_est**2)) / dens_est

    if evaluate_at is None:
        evaluate_at = mesh

    if not evaluate_dens:
        return kernel_bandwidths, evaluate_at, None

    # Note that the array must be initialized to zeros before sending to the
    # `gaussians` function (which adds its results to the existing array)
    vbw_dens_est = np.zeros_like(evaluate_at, dtype=np.double)
    gaussians(
        outbuf=vbw_dens_est,
        x=evaluate_at.astype(np.double),
        mu=data.astype(np.double),
        sigma=kernel_bandwidths.astype(np.double),
        threads=OMP_NUM_THREADS
    )

    # Normalize distribution to have area of 1
    vbw_dens_est = vbw_dens_est / len(data)

    return kernel_bandwidths, evaluate_at, vbw_dens_est


def fixed_point(t, data_len, i, a2):
    """Fixed point algorithm for Improved Sheather Jones bandwidth
    selection.
    
    Ref: The Annals of Statistics, 38(5):2916-2957, 2010.

    Parameters
    ----------
    t, data_len, i, a2 : float
    
    NOTES
    -----
    Original implementation by Botev et al. is quad-precision, whereas this is
    single precision only. This could cause discrepancies from the reference
    implementation.
    
    """
    l = 7
    f = 2*PISQ**l * np.sum(i**l * a2 * np.exp(-i*PISQ*t))
    for s in xrange(l, 1, -1):
        k0 = np.prod(np.arange(1, 2.*s, 2))/SQRT2PI
        const = (1 + (0.5)**(s + 0.5))/3.
        time = (2*const*k0/data_len/f)**(2./(3.+2.*s))
        x0 = i**s
        x10 = -i * PISQ * time
        x1 = np.exp(x10)
        x2 = x0 * a2 * x1
        x3 = np.sum(x2)
        f = 2*PISQ**s * x3
    return t-(2*data_len*SQRTPI*f)**(-0.4)


def speedTest():
    from time import time
    enuerr = np.random.noncentral_chisquare(df=3, nonc=1,
                                            size=int(4e3))
    min_e, max_e = 0, np.max(enuerr)

    t0 = time()
    bw, x, dens = vbw_kde(data=enuerr, n_dft=1024)
    logging.debug('time to run: %f s' % (time() - t0))


if __name__ == "__main__":
    set_verbosity(2)
    logging.info('OMP_NUM_THREADS = %d' % OMP_NUM_THREADS)
    speedTest()
