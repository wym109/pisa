# -*- coding: utf-8 -*-
# author:  J.L. Lanfanchi
#          jll1062@phys.psu.edu
#
# date:    March 28, 2015

"""
Computation of a single Guassian (function "gaussian") or the sum of multiple Guassians (function "gaussians"). Note that each function requires an
output buffer be provided as the first argument, to which the result is added
(so the user must handle initialization of the buffer).

Use of threads requires compilation with OpenMP support.
"""


cimport cython
from cython.parallel import prange
from libc.math cimport exp, fabs, sqrt, M_PI


cdef double sqrt2pi_d = <double>sqrt(2*M_PI)
cdef float sqrt2pi_s = <float>sqrt(2*M_PI)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def gaussian_d(double[::1] outbuf,
               double[::1] x,
               double mu,
               double sigma,
               int threads=1):
    """Computation of a single normalized Gaussian function at points `x`,
    given a mean `mu` and standard deviation `sigma`.

    The result is added and stored to the first argument, `outbuf`.

    Parameters
    ----------
    outbuf : array of double
        Output buffer, will be populated with values of the Gaussian function.

    x : array of double
        Points at which to evaluate the Gaussian

    mu : double
        Gaussian mean

    sigma : non-zero double
        Gaussian standard deviation

    threads : int
        Number of OpenMP threads (if compiled with support for OpenMP) to use
        for parallelizing the computation; defaults to one thread for "safe"
        operation in single-CPU cluster jobs

    Returns
    -------
    None

    """
    cdef double twosigma2 = 2*(sigma*sigma)
    cdef double sqrt2pisigma = fabs(sqrt2pi_d * sigma)
    cdef double xlessmu
    cdef Py_ssize_t i
    for i in prange(outbuf.shape[0],
                    nogil=True,
                    num_threads=threads,
                    schedule='static'):
        xlessmu = x[i] - mu
        outbuf[i] = exp(-(xlessmu * xlessmu) / twosigma2) / sqrt2pisigma


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def gaussian_s(float[::1] outbuf,
               float[::1] x,
               float mu,
               float sigma,
               int threads=1):
    """Computation of a single normalized Gaussian function at points `x`,
    given a mean `mu` and standard deviation `sigma`.

    The result is added and stored to the first argument, `outbuf`.

    Parameters
    ----------
    outbuf : array of float
        Output buffer, will be populated with values of the Gaussian function.

    x : array of float
        Points at which to evaluate the Gaussian

    mu : float
        Gaussian mean

    sigma : non-zero float
        Gaussian standard deviation

    threads : int
        Number of OpenMP threads (if compiled with support for OpenMP) to use
        for parallelizing the computation; defaults to one thread for "safe"
        operation in single-CPU cluster jobs

    Returns
    -------
    None

    """
    cdef float twosigma2 = 2*(sigma*sigma)
    cdef float sqrt2pisigma = fabs(sqrt2pi_d * sigma)
    cdef float xlessmu
    cdef Py_ssize_t i
    for i in prange(outbuf.shape[0],
                    nogil=True,
                    num_threads=threads,
                    schedule='static'):
        xlessmu = x[i] - mu
        outbuf[i] = exp(-(xlessmu * xlessmu) / twosigma2) / sqrt2pisigma


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def gaussians_d(double[::1] outbuf,
                double[::1] x,
                double[::1] mu,
                double[::1] sigma,
                int threads=1):
    """Sum of multiple, normalized Gaussian function at points `x`, given
    a mean `mu` and standard deviation `sigma`.

    The result overwrites the contents of `outbuf`.

    Parameters
    ----------
    outbuf : initialized array of double
        Populated with the sum-of-Gaussians.

    x : array of double
        Points at which to evaluate the Gaussians

    mu : array of double
        Means of the Gaussians

    sigma : array of non-zero double
        Standard deviations of the Gaussians

    threads : int
        Number of OpenMP threads (if compiled with support for OpenMP) to use
        for parallelizing the computation; defaults to one thread for "safe"
        operation in single-CPU cluster jobs

    """
    cdef double twosigma2
    cdef double norm
    cdef double xlessmu
    cdef Py_ssize_t i, gaus_n, n_gaussians
    # NOTE that the order of the loops is important, as
    # updating the outbuf is NOT thread safe!
    assert outbuf.shape[0] == x.shape[0]
    assert mu.shape[0] == sigma.shape[0]
    n_gaussians = mu.shape[0]
    for i in prange(x.shape[0],
                    nogil=True,
                    num_threads=threads,
                    schedule='static'):
        outbuf[i] = 0
        for gaus_n in xrange(mu.shape[0]):
            twosigma2 = 2*(sigma[gaus_n] * sigma[gaus_n])
            norm = fabs(sqrt2pi_d * sigma[gaus_n]) * <double>n_gaussians
            xlessmu = x[i] - mu[gaus_n]
            outbuf[i] += exp(-xlessmu * xlessmu / twosigma2) / norm


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def gaussians_s(float[::1] outbuf,
                float[::1] x,
                float[::1] mu,
                float[::1] sigma,
                int threads=1):
    """Sum of multiple, normalized Gaussian function at points `x`, given
    a mean `mu` and standard deviation `sigma`.

    The result overwrites the contents of `outbuf`.

    Parameters
    ----------
    outbuf : initialized array of float
        Populated with the sum-of-Gaussians.

    x : array of float
        Points at which to evaluate the Gaussians

    mu : array of float
        Means of the Gaussians

    sigma : array of non-zero float
        Standard deviations of the Gaussians

    threads : int
        Number of OpenMP threads (if compiled with support for OpenMP) to use
        for parallelizing the computation; defaults to one thread for "safe"
        operation in single-CPU cluster jobs

    """
    cdef float twosigma2
    cdef float norm
    cdef float xlessmu
    cdef Py_ssize_t i, gaus_n, n_gaussians
    # NOTE that the order of the loops is important, as
    # updating the outbuf is NOT thread safe!
    assert outbuf.shape[0] == x.shape[0]
    assert mu.shape[0] == sigma.shape[0]
    n_gaussians = mu.shape[0]
    for i in prange(x.shape[0],
                    nogil=True,
                    num_threads=threads,
                    schedule='static'):
        outbuf[i] = 0
        for gaus_n in xrange(mu.shape[0]):
            twosigma2 = 2*(sigma[gaus_n] * sigma[gaus_n])
            norm = fabs(sqrt2pi_s * sigma[gaus_n]) * <float>n_gaussians
            xlessmu = x[i] - mu[gaus_n]
            outbuf[i] += exp(-xlessmu * xlessmu / twosigma2) / norm
