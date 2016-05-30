# author : J.L. Lanfranchi
#          jll1062+pisa@phys.psu.edu
#
# date   : April 8, 2016
"""
Numerical utilities.
"""

from collections import Iterable, Iterator, Mapping, Sequence
from itertools import izip

import numpy as np
import pint; ureg = pint.UnitRegistry()
from uncertainties.core import AffineScalarFunc
from uncertainties import ufloat
from uncertainties import unumpy as unp


PREC = np.finfo(float).eps

# from sympy import *
# print str(N(log(2, 10), 40))
LOG10_2 = np.float64('0.3010299956639811952137388947244930267682')


def isscalar(x):
    """Check if input is a scalar object.

    Best check found for now as to scalar-ness (works for pint,
    uncertainties, lists, tuples, numpy arrays, ...) but not tested against all
    things.

    See Also
    --------
    numpy.isscalar

    """
    return (not (hasattr(x, 'shape')
                 or isinstance(x, (Iterator, Mapping, Sequence)))
            or np.isscalar(x))


# TODO: add an arg and logic to round to a number of significand bits for more
# precise control, esp. if we decide to move to FP32 or (even more critical)
# FP16?
def normQuant(obj, sigfigs=None):
    """Normalize quantites such that two things that *should* be equal are
    returned as identical objects.

    Handles floating point numbers, pint quantities, uncertainties, and
    combinations thereof as standalone objects or in sequences, dicts, or numpy
    ndarrays. Numerical precision issues and equal quantities represented in
    differently-scaled or different systems of units come out identically.

    Outputs from this function (**not** the inputs) deemed to be equal by the
    above logic will compare to be equal (via the `==` operator and via
    `pisa.utils.comparisons.recursiveEquality`) and will also hash to equal
    values (via `pisa.utils.hash.hash_obj`).

    Parameters
    ----------
    obj
        Object to be normalized.

    sigfigs : None or int > 0
        Number of digits to which to round numbers' significands; if None, do
        not round numbers.

    Returns
    -------
    normed_obj : object roughly of same type as input `obj`
        Simple types are returned as the same type as at the input, Numpy
        ndarrays are returned in the sampe shape and representation as the
        input, Mappings (dicts) are returned as OrderdDict, and all other
        sequences or iterables are returned as (possibly nested) lists.

    Notes
    -----
    Conversion logic by `obj` type or types found within `obj`:

    * **Sequences and OrderedDicts** (but not numpy arrays) are iterated
      through recursively.
    * **Mappings without ordering** (e.g. dicts) are iterated through
      recursively after sorting their keys, and are returned as
      OrderedDicts (such that the output is always consistent when
      serialized).
    * **Sequences** (not numpy arrays) are iterated through recursively.
    * **Numpy ndarrays** are treated as the below data types (according to the
      array's dtype).
    * **Simple objects** (non-floating point / non-sequence / non-numpy / etc.)
      are returned unaltered (e.g. strings).
    * **Pint quantities** (numbers with units): Convert to their base units.
    * **Floating-point numbers** (including the converted pint quantities):
      Round values to `sigfig` significant figures.
    * **Numbers with uncertainties** (via the `uncertainties` module) have
      their nominal values rounded as above but their standard deviations are
      rounded to the same order of magnitude (*not* number of significant
      figures) as the nominal.
      Therefore passing obj=10.23+/-0.25 and sigfigs=2 returns 10+/-0.0.
      Note that **correlations are lost** in the outputs of this function, so
      equality of the output requires merely having equal nomial values and
      equal standard deviations.
      The calculations leading to these equal numbers might have used
      independent random variables to arrive at them, however, and so the
      `uncertainties` module would have evaluated them to be unequal. [1]

    To achieve rounding that masks floating point precision issues, set
    `sigfigs` to a value *less than* the number of decimal digits used for the
    significand of the calculation floating point precision.

    For reference, the IEEE 754 floating point standard [2] uses the following:

    * FP16 (half precision): **3.31** significand decimal digits (11 bits)
    * FP32 (single precision): **7.22** significand decimal digits (24 bits)
    * FP64 (double precision): **15.95** significand decimal digits (53 bits)
    * FP128 (quad precision): **34.02** significand decimal digits (113 bits)

    Logic for rounding the significand for numpy arrays was derived from
    http://stackoverflow.com/questions/18915378, in the answer by user
    BlackGriffin.

    References
    ----------
    [1] https://github.com/lebigot/uncertainties/blob/master/uncertainties/test_uncertainties.py#L436

    [2] https://en.wikipedia.org/wiki/IEEE_floating_point

    Examples
    --------
    Pint quantities hash to unequal values if specified in different scales or
    different systems of units (even if the underlying physical quantity is
    identical).

    >>> import pint; ureg = pint.UnitRegistry()
    >>> from pisa.utils.hash import hash_obj
    >>> from pisa.utils.numerical import normQuant
    >>> q0 = 1 * ureg.m
    >>> q1 = 100 * ureg.cm
    >>> q0 == q1
    True
    >>> hash_obj(q0) == hash_obj(q1)
    False

    Even the `to_base_units()` method fails for hashing to equal values, as
    `q0` is a float and `q1` is an integer.

    >>> hash_obj(q0.to_base_units()) == hash_obj(q1.to_base_units())
    False

    Even if both quantities are floating point numbers, finite precision
    effects in the `to_base_units` conversion can still cause two things which
    we "know" are equal to evaluate to be unequal.

    >>> q2 = 0.1 * ureg.m
    >>> q3 = 1e5 * ureg.um
    >>> q2 == q3
    True
    >>> q2.to_base_units() == q3.to_base_units()
    False

    `normQuant` handles all of these issues given an appropriate `sigfigs`
    argument.

    >>> q2_normed = normQuant(q2, sigfigs=12)
    >>> q3_normed = normQuant(q3, sigfigs=12)
    >>> q2_normed == q3_normed
    True
    >>> hash_obj(q2_normed) == hash_obj(q3_normed)
    True

    """
    round_result = False
    if sigfigs is not None:
        if not (int(sigfigs) == float(sigfigs) and sigfigs > 0):
            raise ValueError('`sigfigs` must be positive and integer.')
        round_result = True
        sigfigs = int(sigfigs)

    # Store kwargs for easily passing to recursive calls of this function
    kwargs = dict(sigfigs=sigfigs)

    # Recurse into dict by its (sorted) keys (or into OrderedDict using keys in
    # their defined order) and return an OrderedDict in either case.
    if isinstance(obj, Mapping):
        if isinstance(obj, OrderdDict):
            keys = obj.keys()
        else:
            keys = sorted(obj.keys())
        normed_obj = OrderdDict()
        for key in keys:
            normed_obj[key] = normQuant(obj[key], **kwargs)
        return normed_obj

    # Sequences, etc. but NOT numpy arrays (or pint quantities, which are
    # iterable) get their elements normalized and populated to a new list for
    # returning.
    if (isinstance(obj, (Iterable, Iterator, Sequence)) and not
        (isinstance(obj, np.ndarray)
         or isinstance(obj, pint.quantity._Quantity))):
        return [normQuant(x, **kwargs) for x in obj]

    # Must be a numpy array or scalar if we got here...

    # Nothing to convert for strings
    if isinstance(obj, basestring):
        return obj

    # NOTE: the order in which units (Pint module) and uncertainties
    # (uncertainties module) are handled is crucial! Essentially, it appears
    # that Pint is aware of uncertainties, but not vice versa. Hence the
    # ordering and descriptions used below.

    # The outermost "wrapper" of a number or numpy array is its Pint units. If
    # units are present, convert to base units, record the base units, and
    # strip the units off of the quantity by replacing it with its magnitude
    # (in the base units).

    has_units = False
    if isinstance(obj, pint.quantity._Quantity):
        has_units = True
        obj = obj.to_base_units()
        units = obj.units
        obj = obj.magnitude

    # The next layer possible for a number or numpy array to have is
    # uncertainties. If uncertainties are attached to `obj`, record a
    # "snapshot" (losing correlations) of the standard deviations. Then replace
    # the number or array solely with its nominal value(s).

    # NOTE: uncertainties.core.AffineScalarFunc includes such functions *and*
    # uncertainties.core.Variable objects

    has_uncertainties = False
    if isinstance(obj, AffineScalarFunc):
        has_uncertainties = True
        std_devs = obj.std_dev
        obj = obj.nominal_value
    elif isinstance(obj, np.ndarray) and np.issubsctype(obj, AffineScalarFunc):
        has_uncertainties = True
        std_devs = unp.std_devs(obj)
        obj = unp.nominal_values(obj)

    # What is done below will convert scalars into arrays, so get this info
    # before it is lost.
    is_scalar = isscalar(obj)

    if round_result:
        # frexp returns *binary* fraction (significand) and *binary* exponent
        bin_significand, bin_exponent = np.frexp(obj)
        exponent = LOG10_2 * bin_exponent
        exponent_integ = np.floor(exponent)
        exponent_fract = exponent - exponent_integ
        significand = bin_significand * 10**(exponent_fract)
        obj = np.around(significand, sigfigs-1) * 10**exponent_integ

    # Now work our way *up* through the hierarchy: First, reintroduce
    # uncertainties

    if has_uncertainties and round_result:
        std_bin_significand, std_bin_exponent = np.frexp(std_devs)
        std_exponent = LOG10_2 * std_bin_exponent
        std_exponent_integ = np.floor(std_exponent)
        std_exponent_fract = std_exponent - std_exponent_integ
        # Don't just scale magnitude by the stddev's fractional exponent; also
        # shift to be on the same scale (power-of-10) as the nominal value
        delta_order_of_mag = std_exponent_integ - exponent_integ
        std_significand = (
            std_bin_significand * 10**(std_exponent_fract + delta_order_of_mag)
        )
        # Now rounding on the stddev's significand occurs at the same order of
        # magnitude as rounding on the nominal value (and so scaling is done
        # with `exponent_integ`, NOT `std_exponent_integ`)
        std_devs = (np.around(std_significand, sigfigs-1) * 10**exponent_integ)

    if has_uncertainties:
        obj = unp.uarray(obj, std_devs)
        # If it was a scalar, it has become a len-1 array; extract the scalar
        if is_scalar:
            obj = obj[0]

    # Finally, attach units if they were present
    if has_units:
        obj = obj * units

    return obj


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


def test_isscalar():
    assert isscalar(0)
    assert isscalar('xyz')
    assert isscalar('')
    assert isscalar(np.nan)
    assert isscalar(np.inf)
    assert not isscalar(iter([]))
    assert not isscalar({})
    assert not isscalar(tuple())
    assert not isscalar([np.inf])
    assert not isscalar(np.array([np.inf]))
    assert not isscalar(np.array([]))

    a = np.array([-np.inf, np.nan, -1.1, -1, 0, 1, 1.1, np.inf])
    unp_a = unp.uarray(a, np.ones_like(a))
    pint_a = a * ureg.GeV
    pint_unp_a = unp_a * ureg.GeV
    for x in [a, unp_a, pint_a, pint_unp_a]:
        assert not isscalar(x), str(x) + ' should not evalute to scalar'

    u_fl = ufloat(1, 1)
    p_fl = 1 * ureg.GeV
    p_u_fl = ufloat(1, 1) * ureg.GeV
    for x in [u_fl, p_fl, p_u_fl]:
        assert isscalar(x), str(x) + ' should evaluate to scalar'
    print ('<< PASSED : test_isscalar >>')


def test_normQuant():
    from pisa.utils.log import logging, set_verbosity
    # TODO: test:
    # * non-numerical
    #   * single non-numerical
    #   * sequence (tuple, list) of non-numirical
    #   * np.array of non-numerical
    # * scalar
    #   * bare
    #   * with units only
    #   * with uncertainties only
    #   * with uncertanties and units
    # * multi-dimensional array
    #   * integers, bare
    #   * float32, bare
    #   * float64, bare
    #   * float64 with units only
    #   * float64 with uncertainties only
    #   * float64 with both units and uncertainties
    # * nested objects... ?
    s0 = 1e5*ureg.um
    s1 = 1e5*ureg.um
    # ...

    q0 = 1e5*np.ones(10)*ureg.um
    q1 = 0.1*np.ones(10)*ureg.m
    assert not np.any(q0 == q1)
    assert not np.any(q0.to_base_units() == q1.to_base_units())
    assert not np.any(normQuant(q0, None) == normQuant(q1, None))
    assert not np.any(normQuant(q0, 18) == normQuant(q1, 18))
    assert np.all(normQuant(q0, 16) == normQuant(q1, 16))
    assert np.all(normQuant(q0, 15) == normQuant(q1, 15))
    assert np.all(normQuant(q0, 1) == normQuant(q1, 1))
    assert (normQuant(np.inf, sigfigs=15) == normQuant(np.inf, sigfigs=15))
    assert (normQuant(-np.inf, sigfigs=15) == normQuant(-np.inf, sigfigs=15))
    assert (normQuant(np.inf, sigfigs=15) != normQuant(-np.inf, sigfigs=15))
    assert (normQuant(np.nan, sigfigs=15) != normQuant(np.nan, sigfigs=15))
    print ('<< PASSED : test_normQuant >>')


if __name__ == '__main__':
    test_isscalar()
    test_normQuant()
