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


cdef double sqrtpi = sqrt(M_PI)
cdef double sqrt2pi = sqrt(2*M_PI)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def gaussian(double[::1] outbuf,
             double[::1] x,
             double mu,
             double sigma,
             int threads=1):
    """Computation of a single normalized Gaussian function at points `x`,
    given a mean `mu` and standard deviation `sigma`.

    The result is added and stored to the first argument, `outbuf`.

    Parameters
    ----------
    outbuf : intialized array of double
        Output buffer, populated with the sum of the values already in the
        buffer and the newly-computed Gaussian.

        WARNING! The user of this function is responsible for initializing the
        buffer with meaningful values (e.g., zeros).

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
    cdef double sqrt2pisigma = fabs(sqrt2pi * sigma)
    cdef double xlessmu
    cdef double x1
    cdef Py_ssize_t i
    for i in prange(outbuf.shape[0],
                    nogil=True,
                    num_threads=threads,
                    schedule='static'):
        xlessmu = x[i]-mu
        x1 = -(xlessmu * xlessmu) / twosigma2
        outbuf[i] += exp(x1) / sqrt2pisigma


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def gaussians(double[::1] outbuf,
              double[::1] x,
              double[::1] mu,
              double[::1] sigma,
              int threads=1):
    """Sum of multiple, normalized Gaussian function at points `x`, given
    a mean `mu` and standard deviation `sigma`.

    The result is added and stored to the first argument, `outbuf`.

    Parameters
    ----------
    outbuf : initialized array of double
        Populated with the sum of the values already in the buffer and the
        computed sum-of-Gaussians. NOTE: The user of this function is
        responsible for initializing the buffer with meaningful values (e.g.,
        zeros).

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
    cdef double sqrt2pisigma
    cdef double xlessmu
    cdef double x1
    cdef Py_ssize_t i, gaus_n
    # NOTE that the order of the loops is important, as
    # updating the outbuf is NOT thread safe!
    assert outbuf.shape[0] == x.shape[0]
    assert mu.shape[0] == sigma.shape[0]
    for i in prange(x.shape[0],
                    nogil=True,
                    num_threads=threads,
                    schedule='static'):
        for gaus_n in xrange(mu.shape[0]):
            twosigma2 = 2*(sigma[gaus_n] * sigma[gaus_n])
            sqrt2pisigma = fabs(sqrt2pi * sigma[gaus_n])
            xlessmu = x[i] - mu[gaus_n]
            x1 = -xlessmu * xlessmu / twosigma2
            outbuf[i] += exp(x1) / sqrt2pisigma
