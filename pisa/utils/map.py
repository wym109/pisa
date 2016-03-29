# author : J.L. Lanfranchi
#          jll1062+pisa@phys.psu.edu
#
# date   : March 25, 2016

"""
Map class to contain 2D histogram, error, and metadata about the contents.
MapSet class to contain a set of maps.

Also provide basic mathematical operations that user applies directly to the
containers but that get passed down to operate on the contained data.
"""

from __future__ import division
#from types import MethodType
from operator import getitem, setitem
from collections import OrderedDict, Mapping, Sequence
import re
from copy import deepcopy

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

def log_var(obj, base):
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
    __slots = ('name', 'hist', 'binning', 'hash', 'variance', 'tex',
               'full_comparison')
    __state_attrs = __slots
    def __init__(self, name, hist, binning, hash=None, variance=None, tex=None,
                 full_comparison=True):
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
        for slot in self.__state_attrs:
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

    def index(self, idx):
        state = deepcopy(self.state)
        binning_state = deepcopy(self.binning.state)
        e_idx = idx[0]
        cz_idx = idx[1]
        ebins = self.binning.ebins[e_idx]
        czbins = self.binning.czbins[cz_idx]
        if np.isscalar(ebins):
            ebins = [ebins]
        if np.isscalar(czbins):
            czbins = [czbins]
        ebins = list(ebins)
        czbins = list(czbins)
        if isinstance(e_idx, slice):
            ebins += [self.binning.ebins[e_idx.stop]]
        elif isinstance(e_idx, int):
            ebins += [self.binning.ebins[e_idx+1]]
        elif isinstance(e_idx, Sequence):
            assert len(e_idx) == 1
            ebins += [self.binning.ebins[e_idx[0]+1]]
        else:
            raise TypeError('Unhandled e_idx type %s' % type(e_idx))

        if isinstance(cz_idx, slice):
            czbins += [self.binning.czbins[cz_idx.stop]]
        elif isinstance(cz_idx, int):
            czbins += [self.binning.czbins[cz_idx+1]]
        elif isinstance(cz_idx, Sequence):
            assert len(cz_idx) == 1
            czbins += [self.binning.czbins[cz_idx[0]+1]]
        else:
            raise TypeError('Unhandled cz_idx type %s' % type(cz_idx))

        binning_state.update(dict(
            ebins=ebins, czbins=czbins
        ))
        binning = Binning(**binning_state)
        hist = state['hist'][idx].reshape(binning.n_ebins, binning.n_czbins)
        state['hist'] = hist
        state['binning'] = binning
        if state['variance'] is not None:
            variance = state['variance'][idx].reshape(binning.n_ebins,
                                                      binning.n_czbins)
            state['variance'] = variance
        return Map(**state)

    def __str__(self):
        return strip_outer_parens(self.name)

    def __repr__(self):
        return str(self)

    def __hash__(self):
        if self.hash is not None:
            return self.hash
        raise ValueError('No hash defined.')

    def __setattr__(self, attr, value):
        """Only allow setting attributes defined in slots"""
        if attr not in self.__slots:
            raise ValueError('Attribute "%s" not allowed to be set.' % attr)
        super(Map, self).__setattr__(attr, value)

    def __getattr__(self, attr):
        return super(Map, self).__getattribute__(attr)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            raise TypeError('Map is 2D; integer indexing is ambiguous'
                            ' and therefore disallowed.')
        elif isinstance(idx, Sequence):
            if len(idx) == 2:
                return self.index(idx)
            else:
                raise ValueError('Map is 2D; %d-D indexing is disallowed' %
                                 len(idx))
        else:
            raise TypeError('Map is 2D; integer indexing is ambiguous'
                            ' and therefore disallowed.')

    def __setitem__(self, idx, val):
        return setitem(self.hist, idx, val)

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
        state = deepcopy(self.state)
        state.update(dict(
            name="|%s|" % (self.name,),
            tex=r"{\left| %s \right|}" % strip_outer_parens(self.tex),
            hist=np.abs(self.hist),
            variance=abs_var(self),
        ))
        return Map(**state)

    def __add__(self, other):
        """Add `other` to self"""
        state = deepcopy(self.state)
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
        state = deepcopy(self.state)
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

    def __truediv__(self, other):
        return self.__div__(other)

    def __floordiv__(self, other):
        raise NotImplementedError('floordiv not implemented for type Map')

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

    def log(self):
        state = deepcopy(self.state)
        state.update(dict(
            name="log(%s)" % self.name,
            tex=r"\ln\left( %s \right)" % self.tex,
            hist=np.log(self.hist),
            variance=log_var(self, np.e),
        ))
        return Map(**state)

    def log10(self):
        state = deepcopy(self.state)
        state.update(dict(
            name="log10(%s)" % self.name,
            tex=r"\log_{10}\left( %s \right)" % self.tex,
            hist=np.log10(self.hist),
            variance=log_var(self, 10),
        ))
        return Map(**state)

    def __mul__(self, other):
        state = deepcopy(self.state)
        if np.isscalar(other):
            state.update(dict(
                name="%s * %s" % (other, self.name),
                tex=r"%s \cdot %s" % (other, self.tex),
                hist=self.hist * other,
            ))
        elif isinstance(other, np.ndarray):
            state.update(dict(
                name="array * %s" % self.name,
                tex=r"X \cdot %s" % self.tex,
                hist=self.hist * other,
            ))
        elif isinstance(other, Map):
            self.assert_compat(other)
            state.update(dict(
                name="%s * %s" % (self.name, other.name),
                tex=r"%s \cdot %s" % (self.tex, other.tex),
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
        state = deepcopy(self.state)
        state.update(dict(
            name="-%s" % self.name,
            tex=r"-%s" % self.tex,
            hist=-self.hist,
        ))
        return Map(**state)

    #def __nonzero__(self):

    #def __pos__(self):

    def __pow__(self, other):
        state = deepcopy(self.state)
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
        elif isinstance(other, Map):
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
        return self + other

    def __rdiv__(self, other):
        if isinstance(other, Map):
            return other / self
        state = deepcopy(self.state)
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
        return self * other

    #def __rpow__(self, other):

    def __rsub__(self, other):
        if isinstance(other, Map):
            return other - self
        state = deepcopy(self.state)
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

    def sqrt(self):
        state = deepcopy(self.state)
        state.update(dict(
            name="sqrt(%s)" % self.name,
            tex=r"\sqrt{%s}" % self.tex,
            hist=np.sqrt(self.hist),
            variance=power_var(self, 0.5),
        ))
        return Map(**state)

    def __sub__(self, other):
        state = deepcopy(self.state)
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
        elif isinstance(other, Map):
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


class MapSet(object):
    __slots = ('_name', '_hash')
    __state_attrs = ('name', 'hash', 'maps')
    def __init__(self, maps, name=None, tex=None, hash=None,
                 collate_by_name=False):
        super(MapSet, self).__setattr__('maps', tuple(maps))
        super(MapSet, self).__setattr__('name', name)
        super(MapSet, self).__setattr__('tex', name)
        super(MapSet, self).__setattr__('hash', hash)
        super(MapSet, self).__setattr__('collate_by_name', collate_by_name)
        super(MapSet, self).__setattr__('collate_by_num', not collate_by_name)

    @property
    def name(self):
        return super(MapSet, self).__getattribute__('_name')

    @name.setter
    def name(self, name):
        return super(MapSet, self).__setattr__('_name', name)

    @property
    def hash(self):
        return super(MapSet, self).__getattribute__('_hash')

    @hash.setter
    def hash(self, hash):
        return super(MapSet, self).__setattr__('_hash', hash)

    @property
    def names(self):
        return tuple([mp.name for mp in self])

    @property
    def hashes(self):
        return tuple([mp.hash for mp in self])

    def collate_with_names(self, vals):
        ret_dict = OrderedDict()
        [setitem(ret_dict, name, val) for name, val in zip(self.names, vals)]
        return ret_dict

    def find_map(self, value):
        idx = None
        if isinstance(value, Map):
            pass
        elif isinstance(value, basestring):
            try:
                idx = self.names.index(value)
            except ValueError:
                pass
        if idx is None:
            raise ValueError('Could not find map name "%s" in %s' %
                             (value, self))
        return self[idx]

    def apply_to_maps(self, attr, *args, **kwargs):
        if len(kwargs) != 0:
            raise NotImplementedError('Keyword arguments are not handled')

        if not all([hasattr(mp, attr) for mp in self]):
            raise AttributeError('All maps do not have attribute "%s"' % attr)

        # Retrieve the corresponding callables from contained maps
        val_per_map = [getattr(mp, attr) for mp in self]
        if not all([hasattr(meth, '__call__') for meth in val_per_map]):
            # If all results are maps, populate a new map set & return that
            if all([isinstance(r, Map) for r in val_per_map]):
                return MapSet(val_per_map)
            # Otherwise put in an ordered dict with <name>: <val> pairs ordered
            # according to the map ordering in the set
            return self.collate_with_names(val_per_map)

        # Rename for clarity
        method_name = attr
        method_per_map = val_per_map

        # Create a set of args for *each* map in this map set: If an arg is a
        # MapSet, convert that arg into the map in that set corresponding to
        # the same map in this set.
        args_per_map = []
        for map_num, mp in enumerate(self):
            map_name = mp.name
            this_map_args = []
            for arg in args:
                if np.isscalar(arg) or \
                        isinstance(arg, (basestring, np.ndarray)):
                    this_map_args.append(arg)
                elif isinstance(arg, MapSet):
                    if self.collate_by_name:
                        this_map_args.append(arg[map_name])
                    elif self.collate_by_num:
                        this_map_args.append(arg[map_num])
                else:
                    raise TypeError('Unhandled arg %s / type %s' %
                                    (arg, type(arg)))
            args_per_map.append(tuple(this_map_args))

        # Make the method calls and collect returned values
        returned_vals = [meth(*args)
                         for meth, args in zip(method_per_map, args_per_map)]

        # If all results are maps, put them into a new map set & return
        if all([isinstance(r, Map) for r in returned_vals]):
            return MapSet(tuple(returned_vals))

        # If None returned by all, return a single None
        if all([(r is None) for r in returned_vals]):
            return

        # Otherwise put into an ordered dict with name:val pairs
        return self.collate_with_names(returned_vals)

    def __str__(self):
        if self.name is not None:
            my_name = "'" + self.name + "'"
        else:
            my_name = super(MapSet, self).__repr__()
        return "MapSet %s containing maps %s" % (my_name, self.names)

    def __repr__(self):
        return str(self)

    def __hash__(self):
        if self.hash is not None:
            return self.hash
        raise ValueError('No hash defined.')

    def __setattr__(self, attr, val):
        if attr in MapSet.__slots:
            object.__setattr__(attr, val)
        else:
            returned_vals = [setattr(mp, attr, val) for mp in self]
            if all([(r is None) for r in returned_vals]):
                return
            return self.collate_with_names(returned_vals)

    def __getattr__(self, attr):
        return self.apply_to_maps(attr)

    def __iter__(self):
        return iter(self.maps)

    def __getitem__(self, item):
        """Retrieve a map by name or retrieve maps' histogram values by index
        or slice.

        If `item` is a string, retrieve map by name.
        If `item is an integer or one-dim slice, retrieve maps by sequence
        If `item` is length-2 tuple or two-dim slice, retrieve value(s) of all
            contained maps, each indexed by map[`item`]. The output is returned
            in an ordered dict with format {<map name>: <values>, ...}

        """
        if isinstance(item, basestring):
            return self.find_map(item)
        elif isinstance(item, (int, slice)):
            rslt = self.maps[item]
            if hasattr(rslt, '__len__') and len(rslt) > 1:
                return MapSet(rslt)
            return rslt
        elif isinstance(item, Sequence):
            if len(item) == 1:
                return self.maps[item]
            elif len(item) == 2:
                return MapSet([getitem(m, item) for m in self])
            else:
                raise IndexError('too many indices for 2D hist') 
        #elif isinstance(item, Sequence):
        #    assert len(item) == 2, 'Maps are 2D, and so must be indexed as such'
        #    return self.collate_with_names([getitem(m, item) for m in self])
        else:
            raise TypeError('getitem does not support `item` of type %s'
                            % type(item))

    def __abs__(self):
        return self.apply_to_maps('__abs__')

    def __add__(self, val):
        return self.apply_to_maps('__add__', val)

    def __truediv__(self, val):
        return self.apply_to_maps('__truediv__', val)

    def __div__(self, val):
        return self.apply_to_maps('__div__', val)

    def log(self):
        return self.apply_to_maps('log')

    def log10(self):
        return self.apply_to_maps('log10')

    def __mul__(self, val):
        return self.apply_to_maps('__mul__', val)

    def __neg__(self):
        return self.apply_to_maps('__neg__')

    def __pow__(self, val):
        return self.apply_to_maps('__pow__', val)

    def __radd__(self, val):
        return self.apply_to_maps('__radd__', val)

    def __rdiv__(self, val):
        return self.apply_to_maps('__rdiv__', val)

    def __rmul__(self, val):
        return self.apply_to_maps('__rmul__', val)

    def __rsub__(self, val):
        return self.apply_to_maps('__rsub__', val)

    def sqrt(self):
        return self.apply_to_maps('sqrt')

    def __sub__(self, val):
        return self.apply_to_maps('__sub__', val)


## Now dynamically add all methods from Map to MapSet that don't already exist
## in MapSet (and make these methods distribute to contained maps)
##for method_name, method in sorted(Map.__dict__.items()):
#add_methods = '''__abs__ __add__ __div__ __mul__ __neg__ __pow__ __radd__
#__rdiv__ __rmul__ __rsub__ __sub__'''.split()
#
#for method_name in add_methods:
#    #if not hasattr(method, '__call__') or method_name in MapSet.__dict__:
#    #    continue
#    disallowed = ('__getattr__', '__setattr__', '__getattribute__',
#                  '__getitem__', '__eq__', '__ne__', '__str__', '__repr__')
#    if method_name in disallowed:
#        continue
#    print 'adding method "%s" to MapSet as an apply func' % method_name
#    arg_str = ', *args' # if len(args) > 0 else ''
#    eval('def {method_name}(self{arg_str}):\n'
#         '    return self.apply_to_maps({method_name}{arg_str})'.format(method_name=method_name, arg_str=arg_str))
#    #f.__doc__ = 'Apply method %s to all contained maps' % method_name
#    #method = getattr(Map, method_name)
#    #if method.__doc__:
#    #    f.__doc__ += '... ' + method.__doc__
#    setattr(MapSet, method_name, MethodType(eval(method_name), None, MapSet))


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
    print 'r[0:5,0:5].hist:', r[0:5,0:5].hist
    print 'r[0:5,0:5].binning:', r[0:5,0:5].binning
    r = m1 / m2
    assert r == 0.5
    print r, '=', r[0,0]

def test_MapSet():
    n_ebins = 5
    n_czbins = 3
    binning = Binning(n_ebins=n_ebins, e_range=(1,80), e_is_log=True,
                      n_czbins=n_czbins, cz_range=(-1,0))
    m1 = Map(name='ones', hist=np.ones((n_ebins,n_czbins)), binning=binning)
    m2 = Map(name='twos', hist=2*np.ones((n_ebins,n_czbins)), binning=binning)
    ms1 = MapSet((m1, m2))
    ms1 = MapSet((m1, m2), name='map set 1')
    ms1 = MapSet(maps=(m1, m2), name='map set 1', collate_by_name=True)
    m1 = Map(name='threes', hist=3*np.ones((n_ebins,n_czbins)), binning=binning)
    m2 = Map(name='fours', hist=4*np.ones((n_ebins,n_czbins)), binning=binning)
    ms2 = MapSet(maps=(m1, m2), name='map set 2', collate_by_name=False)
    m1 = Map(name='fives', hist=5*np.ones((n_ebins,n_czbins)), binning=binning)
    m2 = Map(name='sixes', hist=6*np.ones((n_ebins,n_czbins)), binning=binning)
    ms3 = MapSet(maps=(m1, m2), name='map set 3', collate_by_name=False)
    ms4 = MapSet(maps=(m1, m2), collate_by_name=False)
    print 'ms1.name:', ms1.name
    print 'ms1.hash:', ms1.hash
    print 'ms1.maps:', ms1.maps
    print 'ms2.maps:', ms2.maps
    print 'ms1.names:', ms1.names
    print 'ms1.tex:', ms1.tex
    print 'ms1[0].hist:', ms1[0].hist
    print 'ms1[0:2].hist:', ms1[0:2].hist
    print 'ms1[0:2,0:2].hist:', ms1[0:2,0:2].hist
    print "ms1.apply_to_maps('__add__', 1).names", ms1.apply_to_maps('__add__', 1).names
    try:
        print ms1.__add__(ms2)
    except ValueError:
        pass
    else:
        raise Exception('Should have errored out!')
    print "(ms2 + ms3).names", (ms2 + ms3).names
    print "(ms2 + ms3)[0,0].hist", (ms2 + ms3)[0,0].hist
    print "ms1['ones'][0,0]:", ms1['ones'][0,0]
    print 'ms1.__mul__(2)[0,0]:', ms1.__mul__(2)[0,0]
    print '(ms1 * 2)[0,0]:', (ms1 * 2)[0,0]
    print 'ms1.__add__(ms1)[0,0]:', ms1.__add__(ms1)[0,0]
    print '(ms1 + ms1)[0,0]:', (ms1 + ms1)[0,0]
    print ms1.names
    print '(ms1/ ms1)[0,0]:', (ms1 / ms1)[0,0]
    print '(ms1/ms1 - 1)[0,0]:', (ms1/ms1 - 1)[0,0]
    print '(ms1.log10())[0,0]:', (np.log10(ms1))[0,0]
    print 'np.log10(ms1)[0,0]:', (np.log10(ms1))[0,0]
    print 'np.log(ms1 * np.e)[0,0]:', (np.log(ms1 * np.e))[0,0]
    print 'np.log(ms1 * np.e)[0,0]:', (np.log(ms1 * np.e))[0,0]
    print 'np.sqrt(ms1)[0:4,0:2].hist:', np.sqrt(ms1)[0:4,0:2].hist
    print 'str(ms1)', str(ms1)
    print 'str(ms4)', str(ms4)
    print 'ms3', ms3
    print 'ms4', ms4

if __name__ == "__main__":
    test_Map()
    test_MapSet()


