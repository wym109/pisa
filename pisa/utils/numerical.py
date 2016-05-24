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


PREC = np.finfo(float).eps

def normQuant(x, sigfigs):
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
    >>> q0_approx = normQuant(q0, self.HASH_SIGFIGS)
    >>> q1_approx = normQuant(q1, self.HASH_SIGFIGS)
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
        mag = np.ceil(np.log10(np.abs(x.magnitude)+PREC))
    elif isinstance(x, float) or (isinstance(x, np.ndarray) and
                                  np.issubsctype(x, np.float)):
        mag = np.ceil(np.log10(np.abs(x)+PREC))
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


def engfmt(n, sigfigs=3, decimals=None, sign_always=False):
    """Format number as string in engineering format (10^(multiples-of-three)),
    including the most common metric prefixes (from atto to Exa).

    Parameters
    ----------
    n : scalar
        Number to be formatted
    sigfigs : int >= 0
        Number of significant figures to limit the result to; default=3.
    decimals : int or None
        Number of decimals to display (zeros filled out as necessary). If None,
        `decimals` is automatically determined by the magnitude of the
        significand and the specified `sigfigs`.
    sign_always : bool
        Prefix the number with "+" sign if number is positive; otherwise,
        only negative numbers are prefixed with a sign ("-")

    """
    prefixes = {-18:'a', -15:'f', -12:'p', -9:'n', -6:'u', -3:'m', 0:'',
                3:'k', 6:'M', 9:'G', 12:'T', 15:'P', 18:'E'}
    if isinstance(n, pint.quantity._Quantity):
        units = n.units
        n = n.magnitude
    else:
        units = ureg.dimensionless

    # Logs don't like negative numbers...
    sign = np.sign(n)
    n *= sign

    mag = int(np.floor(np.log10(n)))
    pfx_mag = int(np.floor(np.log10(n)/3.0)*3)

    if decimals is None:
        decimals = sigfigs-1 - (mag-pfx_mag)
    decimals = int(np.clip(decimals, a_min=0, a_max=np.inf))

    round_to = decimals
    if sigfigs is not None:
        round_to = sigfigs-1 - (mag-pfx_mag)

    scaled_rounded = np.round(n/10.0**pfx_mag, round_to)

    sign_str = ''
    if sign_always and sign > 0:
        sign_str = '+'
    num_str = sign_str + format(sign*scaled_rounded, '.'+str(decimals)+'f')

    # Very large or small quantities have their order of magnitude displayed
    # by printing the exponent rather than showing a prefix; due to my
    # inability to strip off prefix in Pint quantities (and attach my own
    # prefix), just use the "e" notation.
    if pfx_mag not in prefixes or not units.dimensionless:
        if pfx_mag == 0:
            return str.strip('{0:s} {1:~} '.format(num_str, units))
        else:
            return str.strip('{0:s}e{1:d} {2:~} '.format(num_str, pfx_mag,
                                                         units))

    # Dimensionless quantities are treated separately since Pint apparently
    # can't handle prefixed-dimensionless (e.g., simply "1 k", "2.2 M", etc.,
    # with no units attached).
    #if units.dimensionless:
    return  '{0:s} {1:s}'.format(num_str, prefixes[pfx_mag])


def test_normQuant():
    from pisa.utils.log import logging, set_verbosity
    q0 = 1e5*np.ones(10)*ureg.um
    q1 = 0.1*np.ones(10)*ureg.m
    assert not np.any(q0 == q1)
    assert not np.any(q0.to_base_units() == q1.to_base_units())
    assert not np.any(normQuant(q0, None) == normQuant(q1, None))
    assert not np.any(normQuant(q0, 17) == normQuant(q1, 17))
    assert np.all(normQuant(q0, 16) == normQuant(q1, 16))
    assert np.all(normQuant(q0, 15) == normQuant(q1, 15))
    assert np.all(normQuant(q0, 1) == normQuant(q1, 1))
    assert (normQuant(np.inf, sigfigs=15) == normQuant(np.inf, sigfigs=15))
    assert (normQuant(-np.inf, sigfigs=15) == normQuant(-np.inf, sigfigs=15))
    assert (normQuant(np.inf, sigfigs=15) != normQuant(-np.inf, sigfigs=15))
    assert (normQuant(np.nan, sigfigs=15) != normQuant(np.nan, sigfigs=15))
    logging.info('<< PASSED >> test_normQuant')


if __name__ == '__main__':
    test_normQuant()
