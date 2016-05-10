# author : J.L. Lanfranchi
#          jll1062+pisa@phys.psu.edu
#
# date   : April 8, 2016
"""
Numerical utilities.
"""

from collections import Sequence, Iterable
from itertools import izip

import numpy as np
import pint; ureg = pint.UnitRegistry()


def normalizeQuantities(x, sigfigs):
    """Normalize floating point numbers, pint quantities, and sequences thereof
    such that numerical precision issues and quantities with compatible but
    differently-scaled units come out identically.

    Outputs from this function deemed to be equal by the above logic will not
    only compare to be equal, but will also hash to equal values.

    * For pint quantities (numbers with units): Convert to their base units.
    * For all floating-point numbers and sequences thereof (including the
      converted pint quantities): Round values to `sigfig` significant figures.
    * Everything else is simply returned

    A sequence or iterable at the input is returned as a list; numpy ndarrays
    are flattened and likewise iterated through, also returning a list.
    Therefore information about dimensionality is lost.

    >>> import pint
    >>> ureg = pint.UnitRegistry()
    >>> q0 = 0.1 * ureg.m
    >>> q1 = 1e-5 * ureg.um
    >>> q0.to_base_units() == q1.to_base_units()
    False
    >>> q0_approx = normalizeQuantities(q0, self.HASH_SIGFIGS)
    >>> q1_approx = normalizeQuantities(q1, self.HASH_SIGFIGS)
    >>> q0_approx == q1_approx
    True

    """
    is_pquant = False
    if isinstance(x, pint.quantity._Quantity):
        x = x.to_base_units()
        is_pquant = True

    if sigfigs is None:
        return x

    if is_pquant:
        mag = np.ceil(np.log10(np.abs(x.magnitude)))
    elif isinstance(x, float) or (isinstance(x, np.ndarray) and
                                  np.issubsctype(x, np.float)):
        mag = np.ceil(np.log10(np.abs(x)))
    else:
        return x

    if np.isscalar(x):
        if np.isfinite(x):
            return np.round(x, int(n-mag))
        return x

    if isinstance(x, np.ndarray):
        x = x.flatten()
        mag = mag.flatten()

    if isinstance(x, (Iterable, Sequence)):
        outlist = []
        for value, magnitude in izip(x.flatten(), mag.flatten()):
            if np.isfinite(value) and np.isfinite(magnitude):
                outlist.append(np.round(value, int(sigfigs-magnitude)))
            else:
                outlist.append(value)
        return outlist

    raise TypeError('Unhandled type %s' %type(x))


def test_normalizeQuantities():
    from pisa.utils.log import logging, set_verbosity
    q0 = 1e5*np.ones(10)*ureg.um
    q1 = 0.1*np.ones(10)*ureg.m
    assert not np.any(q0 == q1)
    assert not np.any(q0.to_base_units() == q1.to_base_units())
    assert not np.any(normalizeQuantities(q0, None)
                      == normalizeQuantities(q1, None))
    assert not np.any(normalizeQuantities(q0, 17)
                      == normalizeQuantities(q1, 17))
    assert np.all(normalizeQuantities(q0, 16)
                  == normalizeQuantities(q1, 16))
    assert np.all(normalizeQuantities(q0, 15)
                  == normalizeQuantities(q1, 15))
    assert np.all(normalizeQuantities(q0, 1)
                  == normalizeQuantities(q1, 1))
    assert (normalizeQuantities(np.inf, sigfigs=15)
            == normalizeQuantities(np.inf, sigfigs=15))
    assert (normalizeQuantities(-np.inf, sigfigs=15)
            == normalizeQuantities(-np.inf, sigfigs=15))
    assert (normalizeQuantities(np.inf, sigfigs=15)
            != normalizeQuantities(-np.inf, sigfigs=15))
    assert (normalizeQuantities(np.nan, sigfigs=15)
            != normalizeQuantities(np.nan, sigfigs=15))
    logging.info('<< PASSED >> test_normalizeQuantities')


if __name__ == '__main__':
    test_normalizeQuantities()
