
import sys
import itertools
from collections import Mapping, Sequence, OrderedDict

import numpy as np
import pint

from pisa.utils.log import logging, set_verbosity

set_verbosity(10)

NP_TYPES = (np.ndarray, np.matrix)
SEQ_TYPES = (Sequence, np.ndarray, np.matrix)
MAP_TYPES = (Mapping,)
COMPLEX_TYPES = tuple(list(NP_TYPES) + list(SEQ_TYPES) + list(MAP_TYPES))

def is_pint_quantity(x):
    #return isinstance(x, pint.quantity._Quantity)
    return hasattr(x, 'units') and hasattr(x, 'magnitude')

def recursiveEquality(x, y):
    """Recursively verify equality between two objects x and y."""
    # NOTE: The order in which types are compared matters.

    # pint units
    if isinstance(x, pint.unit._Unit) and isinstance(y, pint.unit._Unit):
        if x.dimensionality != y.dimensionality:
            logging.trace('x.dimensionality: %s' %x.dimensionality)
            logging.trace('y.dimensionality: %s' %y.dimensionality)
            return False

    # pint quantities
    elif is_pint_quantity(x) and is_pint_quantity(y):
        xunit = str(x.units)
        try:
            converted_y = y.to(xunit)
        except DimensionalityError:
            logging.trace('Incompatible units: x.units=%s, y.units=%s'
                          %(x.units, y.units))
            return False
        # Check for equality to double precision
        if not np.allclose(x.magnitude, converted_y.magnitude,
                           rtol=1e-15, atol=0):
            logging.trace('x.magnitude: %s' %x.magnitude)
            logging.trace('y.magnitude: %s' %y.magnitude)
            return False

    # simple things can be compared directly
    elif isinstance(x, basestring) or isinstance(y, basestring) or \
            (not (isinstance(x, COMPLEX_TYPES) or
                  isinstance(y, COMPLEX_TYPES))):
        if x != y and not (np.isnan(x) and np.isnan(y)):
            logging.trace('Simple types (type(x)=%s, type(y)=%s) not equal.'
                          %(type(x), type(y)))
            logging.trace('x: %s' %x)
            logging.trace('y: %s' %y)
            return False

    # Numpy types
    elif isinstance(x, NP_TYPES) or isinstance(y, NP_TYPES):
        if np.shape(x) != np.shape(y):
            logging.trace('shape(x): %s' %np.shape(x))
            logging.trace('shape(y): %s' %np.shape(y))
            return False
        if not np.all(np.equal(x, y)):
            logging.trace('x: %s' %x)
            logging.trace('y: %s' %y)
            return False

    # Dict
    elif isinstance(x, Mapping):
        xkeys = sorted(x.keys())
        if not xkeys == sorted(y.keys()):
            logging.trace('xkeys: %s' %(xkeys,))
            logging.trace('ykeys: %s' %(sorted(y.keys()),))
            return False
        else:
            for k in xkeys:
                if not recursiveEquality(x[k], y[k]):
                    logging.trace('not equal found at key: "%s"' %k)
                    return False

    # Non-numpy sequence
    elif isinstance(x, Sequence):
        if not len(x) == len(y):
            logging.trace('len(x): %s' %len(x))
            logging.trace('len(y): %s' %len(y))
            return False
        else:
            for xs, ys in itertools.izip(x, y):
                if not recursiveEquality(xs, ys):
                    logging.trace('xs: %s' %xs)
                    logging.trace('ys: %s' %ys)
                    return False

    # Unhandled
    else:
        raise TypeError('Unhandled type(s): %s, x=%s, y=%s' %
                        (type(x), str(x), str(y)))

    # Returns above only occur if comparisons evaluate to False; therefore, if
    # you make it here, everything is equal.
    return True


# TODO: Get recursiveAllclose working as recursiveEquality does.

def recursiveAllclose(x, y, *args, **kwargs):
    """Recursively verify close-equality between two objects x and y. If
    structure is different between the two objects, returns False

    args and kwargs are passed into numpy.allclose() function
    """
    # None
    if x is None:
        if y is not None:
            return False
    # Scalar
    elif np.isscalar(x):
        if not np.isscalar(y):
            return False
        # np.allclose doesn't handle some dtypes
        try:
            eq = np.allclose(x, y, *args, **kwargs)
        except TypeError:
            eq = x == y
        if not eq:
            return False
    # Dict
    elif isinstance(x, dict):
        if not isinstance(y, dict):
            return False
        xkeys = sorted(x.keys())
        if not xkeys == sorted(y.keys()):
            return False
        for k in xkeys:
            if not recursiveAllclose(x[k], y[k], *args, **kwargs):
                return False
    # Sequence
    elif hasattr(x, '__len__'):
        if not len(x) == len(y):
            return False
        if isinstance(x, list) or isinstance(x, tuple):
            # NOTE: A list is allowed to be allclose to a tuple so long
            # as the contents are allclose
            if not isinstance(y, list) or isinstance(y, tuple):
                return False
            for xs, ys in itertools.izip(x, y):
                if not recursiveAllclose(xs, ys, *args, **kwargs):
                    return False
        elif isinstance(x, np.ndarray):
            # NOTE: A numpy array only evalutes to allclose if compared to
            # another numpy array
            if not isinstance(y, np.ndarray):
                return False
            # np.allclose doesn't handle arrays of some dtypes
            # TODO: this can be rolled into the above clause, I think
            try:
                eq = np.allclose(x, y, *args, **kwargs)
            except TypeError:
                eq = np.all(x == y)
            if not eq:
                return False
        else:
            raise TypeError('Unhandled type(s): %s, x=%s, y=%s' %
                            (type(x), str(x), str(y)))
    else:
        raise TypeError('Unhandled type(s): %s, x=%s, y=%s' %
                        (type(x), str(x), str(y)))
    # If you make it to here, must be close
    return True


def test_recursiveEquality():
    d1 = {'one':1, 'two':2, 'three': None, 'four': 'four'}
    d2 = {'one':1.0, 'two':2.0, 'three': None, 'four': 'four'}
    d3 = {'one':np.arange(0, 100),
          'two':[{'three':{'four':np.arange(1, 2)}},
                 np.arange(3, 4)]}
    d4 = {'one':np.arange(0, 100),
          'two':[{'three':{'four':np.arange(1, 2)}},
                 np.arange(3, 4)]}
    d5 = {'one':np.arange(0, 100),
          'two':[{'three':{'four':np.arange(1, 3)}},
                 np.arange(3, 4)]}
    d6 = {'one':np.arange(0, 100),
          'two':[{'three':{'four':np.arange(1.1, 2.1)}},
                 np.arange(3, 4)]}
    d7 = OrderedDict()
    d7['d1'] = d1
    d7['f'] = 7.2
    d8 = OrderedDict()
    d8['d1'] = d1
    d8['f'] = 7.2
    assert recursiveEquality(d1, d2)
    assert not recursiveEquality(d1, d3)
    assert recursiveEquality(d3, d4)
    assert not recursiveEquality(d3, d5)
    assert not recursiveEquality(d4, d5)
    assert not recursiveEquality(d3, d6)
    assert not recursiveEquality(d4, d6)
    assert recursiveEquality(d7, d8)

    ureg0 = pint.UnitRegistry()
    ureg1 = pint.UnitRegistry()
    u0 = ureg0.GeV
    u1 = ureg1.MeV
    q0 = np.ones(100) * u0
    q1 = np.ones(100) * 1000.0 * u1
    assert recursiveEquality(q0, q1)
    assert not recursiveEquality(u0, u1)

    print '<< PASSED >> recursiveEquality'


if __name__ == "__main__":
    set_verbosity(10)
    test_recursiveEquality()
