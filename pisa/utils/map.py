# author : J.L. Lanfranchi
#          jll1062+pisa@phys.psu.edu
#
# date   : March 25, 2016

"""
Class to contain 2D histogram, error, and metadata about the contents. Also
provides basic mathematical operations for the contained data.
"""


from collections import OrderedDict, Mapping
import re

import numpy as np

from pisa.utils.binning import Binning


def type_error(value):
    raise TypeError('Type of argument not supported: "%s"' % type(value))

def abs_var(obj):
    return None

def add_var(obj1, obj2):
    return None
    #if var_a is None:
    #    return var_b
    #if var_b is None:
    #    return var_a
    #return var_a + var_b

def add_var_stdev(var_a, var_b):
    if var_a is None:
        return var_b
    if var_b is None:
        return var_a
    val = np.sqrt(var_a) + np.sqrt(var_b)
    return val * val

def divide_var(obj1, obj2): #hist_a, var_a, hist_b, var_b):
    return None
    if var_a is not None:
        val_a = var_a / (hist_b*hist_b)
    if var_b is not None:
        val_b = var_b / (hist_a*hist_a)

    if var_a is None:
        if var_b is None:
            return None
        return val_b

    # var_a is NOT None...
    if var_b is None:
        return val_a

    # neither var_a nor var_b is None
    return val_a + val_b

def multiply_var(obj1, obj2):
    return None

def power_var(obj1, obj2):
    return None


def strip_outer_parens(value):
    value = value.strip()
    m = re.match(r'^\{\((.*)\)\}$', value)
    if m is not None:
        value = m.groups()[0]
    m = re.match(r'^\((.*)\)$', value)
    if m is not None:
        value = m.groups()[0]
    return value


class Map(object):
    """Class to contain 2D histogram, error, and metadata about the contents.
    Also provides basic mathematical operations for the contained data, and
    attempts to automatically propagate errors via the `variance` property
    (which -- as of now -- is assumed to represent Gaussian errors).

    Parameters
    ----------
    name
    hist
    binning
    hash
    variance
    tex
    full_comparison

    Properties
    ----------
    full_comparison
    hash
    hist
    name
    state
    tex
    variance

    Methods
    -------
    assert_compat
    __abs__
    __add__
    __div__
    __eq__
    __hash__
    __mul__
    __ne__
    __neg__
    __pow__
    __radd__
    __rdiv__
    __rmul__
    __rsub__
    __str__
    __sub__

    """
    slots = ('name', 'hist', 'binning', 'hash', 'variance', 'tex',
             'full_comparison')
    def __init__(self, name, hist, binning, hash=None, variance=None, tex=None,
                 full_comparison=True):
        super(Map, self).__init__()
        # Set Read/write attributes via their defined setters
        super(Map, self).__setattr__('_name', name)
        tex = r'{\rm %s}' % name if tex is None else tex
        super(Map, self).__setattr__('_tex', tex)
        super(Map, self).__setattr__('_hash', hash)
        super(Map, self).__setattr__('_full_comparison', full_comparison)

        # Do the work here to set read-only attributes
        if not isinstance(binning, Binning):
            assert isinstance(binning, Mapping)
            binning = Binning(**binning)
        super(Map, self).__setattr__('_binning', binning)
        binning.assert_array_compat(hist)
        super(Map, self).__setattr__('_hist', hist)
        if variance is not None:
            binning.assert_array_compat(variance)
        super(Map, self).__setattr__('_variance', variance)

    @property
    def state(self):
        state = OrderedDict()
        for slot in self.slots:
            state[slot] = self.__getattr__(slot)
        return state

    def assert_compat(self, other):
        if np.isscalar(other):
            return
        elif isinstance(other, np.ndarray):
            self.binning.assert_array_compat(other.hist)
        elif isinstance(other, Map):
            assert self.binning == other.binning, \
                    "(%s) incompat. with (%s)" % (other.binning, self.binning)
        else:
            assert False, 'Unrecognized type %s' % type(other)

    def __str__(self):
        return strip_outer_parens(self.name)

    def __hash__(self):
        if self.hash is not None:
            return self.hash
        raise ValueError('No hash defined.')

    def __setattr__(self, attr, value):
        """Only allow setting attributes defined in slots"""
        if attr not in self.slots:
            raise ValueError('Attribute "%s" not allowed to be set.' % attr)
        super(Map, self).__setattr__(attr, value)

    def __getattr__(self, attr):
        return super(Map, self).__getattribute__(attr)

    def __getitem__(self, idx): 
        return self.hist.__getitem__(idx) 
                     
    def __setitem__(self, idx, val):
        return self.hist.__setitem__(idx, val) 

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        assert isinstance(value, basestring)
        return super(Map, self).__setattr__('_name', value)

    @property
    def tex(self):
        return self._tex

    @tex.setter
    def tex(self, value):
        assert isinstance(value, basestring)
        return super(Map, self).__setattr__('_tex', value)

    @property
    def hash(self):
        return self._hash

    @hash.setter
    def hash(self, value):
        """Hash must be an immutable type (i.e., have a __hash__ method)"""
        #assert value.__hash__ is not None
        super(Map, self).__setattr__('_hash', value)

    @property
    def hist(self):
        return self._hist

    @property
    def binning(self):
        return self._binning

    @property
    def variance(self):
        return self._variance

    @property
    def full_comparison(self):
        return self._full_comparison

    @full_comparison.setter
    def full_comparison(self, value):
        assert isinstance(value, bool)
        super(Map, self).__setattr__('_full_comparison', value)

    # Common mathematical operators

    def __abs__(self):
        state = self.state
        state.update(dict(
            name="|%s|" % (self.name,),
            tex=r"{\left| %s \right|}" % strip_outer_parens(self.tex),
            hist=np.abs(self.hist),
            variance=abs_var(self),
        ))
        return Map(**state)

    def __add__(self, other):
        state = self.state
        if np.isscalar(other):
            state.update(dict(
                name="(%s + %s)" % (self.name, other),
                tex=r"{(%s + %s)}" % (self.tex, other),
                hist=self.hist + other,
            ))
        elif isinstance(other, np.ndarray):
            state.update(dict(
                name="(%s + array)" % self.name,
                tex=r"{(%s + X)}" % self.tex,
                hist=self.hist + other,
            ))
        elif isinstance(other, Map):
            self.assert_compat(other)
            state.update(dict(
                name="(%s + %s)" % (self.name, other.name),
                tex=r"{(%s + %s)}" % (self.tex, other.tex),
                hist=self.hist + other.hist,
                full_comparison=self.full_comparison or other.full_comparison,
            ))
        else:
            type_error(other)
        state['variance'] = add_var(self, other)
        return Map(**state)

    #def __cmp__(self, other):
    #    self.assert_compat(other)

    def __div__(self, other):
        state = self.state
        if np.isscalar(other):
            state.update(dict(
                name="(%s / %s)" % (self.name, other),
                tex=r"{(%s / %s)}" % (self.tex, other),
                hist=self.hist / other,
            ))
        elif isinstance(other, np.ndarray):
            state.update(dict(
                name="(%s / array)" % self.name,
                tex=r"{(%s / X)}" % self.tex,
                hist=self.hist / other,
            ))
        elif isinstance(other, Map):
            self.assert_compat(other)
            state.update(dict(
                name="(%s / %s)" % (self.name, other.name),
                tex=r"{(%s / %s)}" % (self.tex, other.tex),
                hist=self.hist / other.hist,
                full_comparison=self.full_comparison or other.full_comparison,
            ))
        else:
            type_error(other)
        state['variance'] = divide_var(self, other)
        return Map(**state)

    def __eq__(self, other):
        """Check if full state of maps are equal. *Not* element-by-element
        equality as for a numpy array. Call this.hist == other.hist and
        this.variance == other.variance if the latter behavior is desired.
        
        If `full_comparison` is true for *both* maps, or if either map lacks a
        hash, performs a full comparison of the contents of each map.
        
        Otherwise, simply checks that the hashes are equal.
        """
        if np.isscalar(other) or isinstance(other, np.ndarray):
            return np.all(self.hist == other)
        elif isinstance(other, Map):
            if (self.full_comparison or other.full_comparison
                or self.hash is None or other.hash is None):
                return utils.recursiveEquality(self.state, other.state)
            return self.hash == other.hash
        else:
            type_error(other)
            
    #def __ge__(self, other):
    #    self.assert_compat(other)

    #def __gt__(self, other):
    #    self.assert_compat(other)

    #def __iadd__(self, other):
    #    self.assert_compat(other)

    #def __idiv__(self, other):
    #    self.assert_compat(other)

    #def __imul__(self, other):
    #    self.assert_compat(other)

    #def __ipow__(self, other):
    #    self.assert_compat(other)

    #def __isub__(self, other):
    #    self.assert_compat(other)

    #def __le__(self, other):
    #    self.assert_compat(other)

    #def __lt__(self, other):
    #    self.assert_compat(other)

    def __mul__(self, other):
        state = self.state
        if np.isscalar(other):
            state.update(dict(
                name="%s * %s" % (self.name, other),
                tex=r"%s \times %s" % (self.tex, other),
                hist=self.hist * other,
            ))
        elif isinstance(other, np.ndarray):
            state.update(dict(
                name="%s * array" % self.name,
                tex=r"%s \times X" % self.tex,
                hist=self.hist * other,
            ))
        elif isinstance(other, Map):
            self.assert_compat(other)
            state.update(dict(
                name="%s * %s" % (self.name, other.name),
                tex=r"%s \times %s" % (self.tex, other.tex),
                hist=self.hist * other.hist,
                full_comparison=self.full_comparison or other.full_comparison,
            ))
        else:
            type_error(other)
        state['variance'] = multiply_var(self, other)
        return Map(**state)

    def __ne__(self, other):
        return not self == other

    def __neg__(self):
        state = self.state
        state.update(dict(
            name="-%s" % self.name,
            tex=r"-%s" % self.tex,
            hist=-self.hist,
        ))
        return Map(**state)

    #def __nonzero__(self):

    #def __pos__(self):

    def __pow__(self, other):
        state = self.state
        if np.isscalar(other):
            if other == 1:
                val = self.hist
            elif other == 2:
                val = self.hist * self.hist
            else:
                val = self.hist * other
            state.update(dict(
                name="%s^%s" % (self.name, other),
                tex="%s^{%s}" % (self.tex, other),
                hist=val,
            ))
        elif isinstance(other, np.ndarray):
            state.update(dict(
                name="%s^(array)" % self.name,
                tex=r"%s^{X}" % self.tex,
                hist=np.power(self.hist, other),
            ))
        elif np.isinstance(other, Map):
            self.assert_compat(other)
            state.update(dict(
                name="%s^(%s)" % (self.name, strip_outer_parens(other.name)),
                tex=r"%s^{%s}" % (self.tex, strip_outer_parens(other.tex)),
                hist=np.power(self.hist, other.hist),
                full_comparison=self.full_comparison or other.full_comparison,
            ))
        else:
            type_error(other)
        state['variance'] = power_var(self, other)
        return Map(**state)

    def __radd__(self, other):
        return self.__add__(other)

    def __rdiv__(self, other):
        if np.isinstance(other, Map):
            return other / self
        state = self.state
        if np.isscalar(other):
            state.update(dict(
                name="(%s / %s)" % (other, self.name),
                tex="{(%s / %s)}" % (other, self.tex),
                hist=other / self.hist,
            ))
        elif isinstance(other, np.ndarray):
            state.update(dict(
                name="array / %s" % self.name,
                tex="{(X / %s)}" % self.tex,
                hist=other / self.hist,
            ))
        else:
            type_error(other)
        state['variance'] = divide_var(other, self)
        return Map(**state)

    def __rmul__(self, other):
        return self.__mul__(other)

    #def __rpow__(self, other):

    def __rsub__(self, other):
        if np.isinstance(other, Map):
            return other - self
        state = self.state
        if np.isscalar(other):
            state.update(dict(
                name="(%s - %s)" % (other, self.name),
                tex="{(%s - %s)}" % (other, self.tex),
                hist=other - self.hist,
            ))
        elif isinstance(other, np.ndarray):
            state.update(dict(
                name="(array - %s)" % self.name,
                tex="{(X - %s)}" % self.tex,
                hist=other - self.hist,
            ))
        else:
            type_error(other)
        state['variance'] = add_var(other, self)
        return Map(**state)

    def __sub__(self, other):
        if np.isscalar(other):
            state.update(dict(
                name="(%s - %s)" % (self.name, other),
                tex="{(%s - %s)}" % (self.tex, other),
                hist=self.hist - other,
            ))
        elif isinstance(other, np.ndarray):
            state.update(dict(
                name="(%s - array)" % self.name,
                tex="{(%s - X)}" % self.tex,
                hist=self.hist - other,
            ))
        elif np.isinstance(other, Map):
            self.assert_compat(other)
            state.update(dict(
                name="%s - %s" % (self.name, other.name),
                tex="{(%s - %s)}" % (self.tex, other.tex),
                hist=self.hist - other.hist,
                full_comparison=self.full_comparison or other.full_comparison,
            ))
        else:
            type_error(other)
        state['variance'] = add_var(self, other)
        return Map(**state)


def test_Map():
    m1 = Map(name='x', hist=np.ones((40,20)),
             binning=dict(n_ebins=40, e_range=(1,80), e_is_log=True,
                          n_czbins=20, cz_range=(-1,0)))
    m2 = Map(name='y', hist=2*np.ones((40,20)),
             binning=dict(n_ebins=40, e_range=(1,80), e_is_log=True,
                          n_czbins=20, cz_range=(-1,0)))
    print m1, m1.binning
    print m2, m2.binning
    r = m1 + m2
    assert r == 3
    print 'm1+m2=3:', r, r[0,0]
    r = m2 + m1
    assert r == 3
    print 'm2+m1=3:', r, r[0,0]
    r = 2*m1
    assert r == 2
    print '2*m1=2:', r, r[0,0]
    r = (2*m1 + 8) / m2
    assert r == 5
    print '(2*m1 + 8) / m2=5:', r, r.hist[0,0]
    r[:,1] = 1
    r[2,:] = 2
    print r[0:5,0:5]


if __name__ == "__main__":
    test_Map()
