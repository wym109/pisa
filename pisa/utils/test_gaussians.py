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


def speed_test_gaussians(num_gaussians, num_points):
    import multiprocessing
    import time
    import sys
    assert int(num_gaussians) == float(num_gaussians), \
            'must pass integral value or equivalent for `num_gaussians`'
    assert int(num_points) == float(num_points), \
            'must pass integral value or equivalent for `num_points`'

    def wstdout(msg):
        sys.stdout.write(msg)
        sys.stdout.flush()

    num_cpu = multiprocessing.cpu_count()
    wstdout('Reported #CPUs: %d (includes any hyperthreading)\n' %num_cpu)
    wstdout('Summing %d Gaussians evaluated at %d points...\n'
            %(num_gaussians, num_points))

    np.random.seed(0)
    mu = np.array(np.random.randn(num_gaussians), dtype=np.float64)
    sigma = np.array(np.abs(np.random.randn(len(mu))), dtype=np.float64)
    np.clip(sigma, a_min=1e-20, a_max=np.inf, out=sigma)

    x = np.linspace(-10, 10, num_points, dtype=np.float64)

    # Place to store result of `gaussians()`; zero-stuffed in the below lopp
    outbuf = np.empty_like(x, dtype=np.float64)

    # Place to store result of `scipy.stats.norm`
    refbuf = np.zeros_like(outbuf, dtype=np.float64)

    # Try out the threads functionality for each result; reset the accumulation
    # buffer each time.
    timings = []
    wstdout('%7s %10s %7s\n' %('Threads', 'Time (s)', 'Speedup'))
    for threads in range(1, num_cpu+1):
        outbuf.fill(0)
        t0 = time.time()
        gaussians(outbuf, x, mu, sigma, threads)
        T = time.time() - t0
        timings.append({'threads': threads, 'timing': T})

        wstdout('%7d %10.3e %7s\n'
                %(threads, T, format(timings[0]['timing']/T, '5.3f')))

    return timings


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        'Run tests on functions in gaussians.pyx; by default, runs unit tests.'
    )
    parser.add_argument(
        '-s', '--speed', action='store_true',
        help='''Run speed test rather than unit tests'''
    )
    parser.add_argument(
        '--num-gaussians', type=float, default=1e4,
        help='Number of Gaussians to sum if running speed test'
    )
    parser.add_argument(
        '--num-points', type=float, default=1e4,
        help='Number of points to evaluate if running speed test'
    )
    args = parser.parse_args()
    if args.speed:
        speed_test_gaussians(num_gaussians=args.num_gaussians,
                             num_points=args.num_points)
    else:
        test_gaussian()
        test_gaussians()
