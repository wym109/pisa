#!/usr/bin/env python
"""
Unittests for functions that live in the gaussians.pyx Cython module.
"""

from itertools import product, izip

import numpy as np
from scipy.stats import norm

from pisa.utils.gaussians import gaussian, gaussians


def test_gaussian():
    x = np.linspace(-10, 10, 1e3, dtype=np.float64)

    # Place to store result of `gaussian()`
    outbuf = np.zeros_like(x, dtype=np.float64)

    # Place to store result of `scipy.stats.norm`
    refbuf = np.zeros_like(outbuf, dtype=np.float64)

    # Test negative and positive ints and floats, and test 0
    means = -2.0, -1, 0, 1, 2.0

    # TODO: scipy populates nan if given negative scale parameter (stddev);
    # should we replicate this behavior or do (as now) and take absolute value
    # of the scale parameter? Or raise ValueError?

    # Test several values for stddev (zero should yield nan results)
    stddevs = 1, 2.0, 1e10, 1e-10, 0

    # Try out the threads functionality for each result; reset the accumulation
    # buffers if any contents are NaN, such that subsequent calculations can
    # actually be tested.
    threads = 1, 2
    for mu, sigma, threads in product(means, stddevs, threads):
        gaussian(outbuf, x, mu, sigma, threads)
        refbuf += norm.pdf(x, loc=mu, scale=sigma)
        assert np.allclose(outbuf, refbuf, rtol=1e-15, atol=0, equal_nan=True),\
                str(outbuf) + '\n' + str(refbuf) + \
                '\nmu=%e, sigma=%e' %(mu,sigma)
        if np.any(np.isnan(refbuf)):
            outbuf.fill(0)
            refbuf.fill(0)


def test_gaussians():
    np.random.seed(0)
    mu = np.array(np.random.randn(1e3), dtype=np.float64)
    sigma = np.array(np.abs(np.random.randn(len(mu))), dtype=np.float64)
    np.clip(sigma, a_min=1e-20, a_max=np.inf, out=sigma)

    x = np.linspace(-10, 10, 1e4, dtype=np.float64)

    # Place to store result of `gaussians()`; zero-stuffed in the below lopp
    outbuf = np.empty_like(x, dtype=np.float64)

    # Place to store result of `scipy.stats.norm`
    refbuf = np.zeros_like(outbuf, dtype=np.float64)

    # Compute the reference result
    [refbuf.__iadd__(norm.pdf(x, loc=m, scale=s)) for m, s in izip(mu, sigma)]

    # Try out the threads functionality for each result; reset the accumulation
    # buffer each time.
    for threads in (1, 2, 32):
        outbuf.fill(0)
        gaussians(outbuf, x, mu, sigma, threads)
        assert np.allclose(outbuf, refbuf, rtol=1e-14, atol=0, equal_nan=True),\
                'outbuf=\n%s\nrefbuf=\n%s\nmu=\n%s\nsigma=\n%s\nthreads=%d' \
                %(outbuf, refbuf, mu, sigma, threads)


def speed_test_gaussians():
    import multiprocessing
    import time
    import sys

    def wstdout(msg):
        sys.stdout.write(msg)
        sys.stdout.flush()

    num_cpu = multiprocessing.cpu_count()

    np.random.seed(0)
    mu = np.array(np.random.randn(1e4), dtype=np.float64)
    sigma = np.array(np.abs(np.random.randn(len(mu))), dtype=np.float64)
    np.clip(sigma, a_min=1e-20, a_max=np.inf, out=sigma)

    x = np.linspace(-10, 10, 1e4, dtype=np.float64)

    # Place to store result of `gaussians()`; zero-stuffed in the below lopp
    outbuf = np.empty_like(x, dtype=np.float64)

    # Place to store result of `scipy.stats.norm`
    refbuf = np.zeros_like(outbuf, dtype=np.float64)

    # Compute the reference result
    [refbuf.__iadd__(norm.pdf(x, loc=m, scale=s)) for m, s in izip(mu, sigma)]

    # Try out the threads functionality for each result; reset the accumulation
    # buffer each time.
    timings = []
    wstdout('%7s %10s\n' %('Threads', 'Time'))
    for threads in range(1, num_cpu+1):
        outbuf.fill(0)
        t0 = time.time()
        gaussians(outbuf, x, mu, sigma, threads)
        timing = time.time() - t0
        timings.append((threads, timing))
        assert np.allclose(outbuf, refbuf, rtol=1e-14, atol=0, equal_nan=True),\
                'outbuf=\n%s\nrefbuf=\n%s\nmu=\n%s\nsigma=\n%s\nthreads=%d' \
                %(outbuf, refbuf, mu, sigma, threads)
        wstdout('%7d %10.3e\n' %(threads, timing))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--speed', action='store_true',
                        help='Run speed test rather than unit tests')
    args = parser.parse_args()
    if args.speed:
        speed_test_gaussians()
    else:
        test_gaussian()
        test_gaussians()
