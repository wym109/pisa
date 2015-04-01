# -*- coding: utf-8 -*-
# 
# gaussians.pyx
# 
# Computation of a single guassian (function "gaussian") or multiple guassians,
# summed together (function "gaussians"). Note that each function requires an
# output buffer be provided as the first argument, to which the result is added
# (so the user must handle initialization of the buffer)
# 
# author:  J.L. Lanfanchi
#          jll1062@phys.psu.edu
# 
# date:    March 28, 2015
# 

cimport cython
from cython.parallel import prange
from libc.math cimport exp, sqrt, M_PI

cdef double sqrtpi = sqrt(M_PI)
cdef double sqrt2pi = sqrt(2*M_PI)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def gaussian(double[::1] outbuf,
             double[::1] x,
             double mu,
             double sigma,
             int threads=4):
    """
    Computation of a single, normalized gaussian function at points (x), given
    a mean (mu) and standard deviation (sigma).
    
    The result is added to the first argument (outbuf).
    
    Arguments
    ---------
    outbuf   Populated with the sum of the values already in the buffer and the
             computed gaussian.
             NOTE: The user of this function is responsible for initializing
             the buffer with meaningful values (e.g., zeros)
    
    x        Points at which to evaluate the gaussian
    
    mu       Gaussian mean
    
    sigma    Gaussian standard deviation (half-width at ~68.3% coverage)
    
    threads  Number of OpenMP threads to use for parallelizing the computation
    
    Returns
    -------
    None
    
    """
    cdef double twosigma2 = 2*(sigma*sigma)
    cdef double sqrt2pisigma = sqrt2pi * sigma
    cdef double xlessmu
    cdef double x1
    cdef Py_ssize_t i
    for i in prange(outbuf.shape[0],
                    nogil=True,
                    num_threads=threads,
                    schedule='static'):
        xlessmu = x[i]-mu
        x1 = -xlessmu*xlessmu / twosigma2
        outbuf[i] = exp(x1) / sqrt2pisigma


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def gaussians(double[::1] outbuf,
              double[::1] x,
              double[::1] mu,
              double[::1] sigma,
              int threads=4):
    """
    Computation of a single, normalized gaussian function at points (x), given
    a mean (mu) and standard deviation (sigma).
    
    The result is added to the first argument (outbuf).
    
    Arguments
    ---------
    outbuf : array of double
        Populated with the sum of the values already in the buffer and the
        computed sum-of-gaussians. NOTE: The user of this function is
        responsible for initializing the buffer with meaningful values (e.g.,
        zeros)
    
    x : array of double
        Points at which to evaluate the gaussians
    
    mu : array of double
        Means of each gaussian
    
    sigma : array of double
        Standard deviations (half-widths at ~68.3% coverage) of each gaussian
    
    threads : int
        Number of OpenMP threads to use for parallelizing the computation
    """
    cdef double twosigma2
    cdef double sqrt2pisigma
    cdef double xlessmu
    cdef double x1
    cdef Py_ssize_t i, gaus_n
    # NOTE that the order of the loops is important, as
    # updating the outbuf is NOT thread safe!
    for i in prange(x.shape[0],
                    nogil=True,
                    num_threads=threads,
                    schedule='static'):
        for gaus_n in xrange(mu.shape[0]):
            twosigma2 = 2*(sigma[gaus_n]*sigma[gaus_n])
            sqrt2pisigma = sqrt2pi * sigma[gaus_n]
            xlessmu = x[i]-mu[gaus_n]
            x1 = -(xlessmu*xlessmu) / twosigma2
            outbuf[i] += exp(x1) / sqrt2pisigma
