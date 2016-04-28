
import sys
from collections import Mapping, Sequence
import itertools

import numpy as np


NP_TYPES = (np.ndarray, np.matrix)
SEQ_TYPES = (Sequence, np.ndarray, np.matrix)
MAP_TYPES = (Mapping,)
COMPLEX_TYPES = tuple(list(NP_TYPES) + list(SEQ_TYPES) + list(MAP_TYPES))


def recursiveEquality(x, y):
    """Recursively verify equality between two objects x and y."""
    # Python doesn't like to compare sequences together
    if not (isinstance(x, COMPLEX_TYPES) or
            isinstance(y, COMPLEX_TYPES)):
        return x == y

    # pint unit
    elif hasattr(x, 'units') and hasattr(x, 'magnitude'):
        return (x.u == y.u) and recursiveEquality(x.m, y.m)

    # Numpy types
    elif isinstance(x, NP_TYPES) or isinstance(y, NP_TYPES):
        if np.shape(x) != np.shape(y):
            return False
        if not np.all(np.equal(x, y)):
            return False

    # Dict
    elif isinstance(x, Mapping):
        xkeys = sorted(x.keys())
        if not xkeys == sorted(y.keys()):
            return False
        else:
            for k in xkeys:
                if not recursiveEquality(x[k], y[k]):
                    return False

    # Non-numpy sequence
    elif isinstance(x, Sequence):
        if not len(x) == len(y):
            return False
        else:
            for xs, ys in itertools.izip(x, y):
                if not recursiveEquality(xs, ys):
                    return False

    # Unhandled
    else:
        raise TypeError('Unhandled type(s): %s, x=%s, y=%s' %
                        (type(x), str(x), str(y)))

    return True


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
    from pisa.utils import logging
    d1 = {'one':1, 'two':2, 'three': None}
    d2 = {'one':1.0, 'two':2.0, 'three': None}
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
    assert recursiveEquality(d1, d2)
    assert not recursiveEquality(d1, d3)
    assert recursiveEquality(d3, d4)
    assert not recursiveEquality(d3, d5)
    assert not recursiveEquality(d4, d5)
    assert not recursiveEquality(d3, d6)
    assert not recursiveEquality(d4, d6)

    print '<< PASSED >> recursiveEquality'


if __name__ == "__main__":
    test_recursiveEquality()
