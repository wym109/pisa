# author : J.L. Lanfranchi
#          jll1062+pisa@phys.psu.edu
#
# date   : March 25, 2016

"""
Class to carry information about 2D binning in energy and cosine-zenity, and to
provide basic operations with the binning.
"""

from operator import setitem
from copy import copy, deepcopy
import collections
from itertools import izip

import numpy as np
import pint
ureg = pint.UnitRegistry()

from pisa.utils.log import logging
from pisa.utils.comparisons import recursiveEquality


class OneDimBinning(object):
    """
    Histogram-oriented binning specialized to a single dimension.

    Parameters
    ----------
    name : str, of length > 0
    units
    prefix : str or None
    tex : str or None
    bin_edges : sequence
    is_log : bool
    is_lin : bool
    n_bins : int
    domain : length-2 sequence

    Properties
    ----------
    bin_edges
    domain
    is_irregular
    is_lin
    is_log
    midpoints
    n_bins
    visual_centers

    Methods
    -------
    is_bin_spacing_log
    is_bin_spacing_lin
    is_binning_ok
    assert_compat


    Notes
    -----
    Consistency is enforced for all redundant parameters passed to the
    constructor.

    """
    # `is_log` and `is_lin` are required so that a sub-sampling down to a
    # single bin that is then resampled to > 1 bin will retain the log/linear
    # property of the original OneDimBinning
    _state_attrs = ('name', 'units', 'prefix', 'tex', 'bin_edges', 'is_log',
                     'is_lin')
    # Convenient means for user to access info (to be attached to a container
    # such as a MultiDimBinning object)
    _prefixed_attrs = (
        ('bin_edges', '%sbin_edges'),
        ('midpoints', '%s_midpoints'),
        ('visual_centers', '%s_visual_centers'),
        ('is_log', '%s_is_log'),
        ('is_lin', '%s_is_lin'),
        ('domain', '%s_domain'),
        ('n_bins', 'n_%sbins'),
    )

    def __init__(self, name, units, prefix=None, tex=None, bin_edges=None,
                 is_log=None, is_lin=None, n_bins=None, domain=None):
        # Store metadata about naming and units
        assert isinstance(name, basestring)
        self.name = name
        if prefix is None or prefix == '':
            prefix = name
        self.prefix = prefix
        if tex is None:
            tex = name
        self.tex = tex

        if units == '' or units is None:
            units = ureg.dimensionless
        else:
            units = ureg(str(units))

        # If both `is_log` and `is_lin` are specified, both cannot be true
        # (but both can be False, in case of irregularly-spaced bins)
        if is_log and is_lin:
            raise ValueError('`is_log=%s` contradicts `is_lin=%s`' %
                                 (is_log, is_lin))

        # If no bin edges specified, the number of bins, domain, and either
        # log or linear spacing are all required to generate bins
        if bin_edges is None:
            assert n_bins is not None and domain is not None \
                    and (is_log or is_lin)
            if is_log:
                is_lin = False
                bin_edges = np.logspace(np.log10(np.min(domain)),
                                        np.log10(np.max(domain)),
                                        n_bins + 1)
            elif is_lin:
                is_log = False
                bin_edges = np.linspace(np.min(domain),
                                        np.max(domain),
                                        n_bins + 1)

        # If bin edges are pint Quantity (i.e., have units)
        if hasattr(bin_edges, 'units') and hasattr(bin_edges, 'magnitude'):
            assert bin_edges.dimensionality == units.dimensionality
            bin_edges = bin_edges.magnitude

        bin_edges = np.array(bin_edges)
        if is_lin:
            assert(self.is_bin_spacing_lin(bin_edges))
            is_log = False
        elif is_log:
            assert(self.is_binning_ok(bin_edges, True))
            is_lin = False
        else:
            is_lin = self.is_bin_spacing_lin(bin_edges)
        if not is_lin and not is_log:
            is_log = self.is_bin_spacing_log(bin_edges)

        # TODO: use list or np.array to store bin edges? A list retains the
        # same more-restrictive indexing that is used in __getitem__; an
        # np.array provides for manipulation not possible for lists,
        # though.

        # Attach units to bin edges
        self.bin_edges = bin_edges * units

        # Determine rest of unspecified parameters from passed bin_edges or
        # enforce them if specified
        if n_bins is None:
            n_bins = len(self.bin_edges) - 1
        else:
            assert n_bins == len(self.bin_edges) - 1

        self.domain = np.array([self.bin_edges[0].m,
                                self.bin_edges[-1].m])*units
        self.is_lin = is_lin
        self.is_log = is_log
        self.is_irregular = not (self.is_lin or self.is_log)
        self.midpoints = (self.bin_edges[:-1] + self.bin_edges[1:])/2.0
        self.n_bins = len(self.bin_edges) - 1
        if self.is_log:
            self.visual_centers = np.sqrt(self.bin_edges[:-1] *
                                          self.bin_edges[1:])
        else:
            self.visual_centers = self.midpoints

        # TODO: Set attributes with "nice" names according to prefix? Probably
        # too much to deal with, so don't do this (at least not here). If still
        # want this, can do it in the multi-dim binning object.
        #for attr in
        #self.n_bins =

    @property
    def units(self):
        return self.bin_edges.units

    def new_obj(original_function):
        """ decorator to deepcopy unaltered states into new object """
        def new_function(self, *args, **kwargs):
            new_state = collections.OrderedDict()
            state_updates = original_function(self, *args, **kwargs)
            for slot in self._state_attrs:
                if state_updates.has_key(slot):
                    new_state[slot] = state_updates[slot]
                elif slot == 'units':
                    new_state[slot] = copy(self.__getattr__(slot))
                else:
                    new_state[slot] = deepcopy(self.__getattr__(slot))
            return OneDimBinning(**new_state)
        return new_function

    @new_obj
    def __deepcopy__(self, memo):
        """ explicit deepcopy constructor """
        return {}

    @staticmethod
    def is_bin_spacing_log(bin_edges):
        """Check if `bin_edges` define a logarithmically-uniform bin spacing.

        Parameters
        ----------
        bin_edges : sequence
            Fewer than 2 `bin_edges` - raises ValueError
            Two `bin_edges` - returns False as a reasonable guess (spacing is
                assumed to be linear)
            More than two `bin_edges` - whether spacing is linear is computed

        Returns
        -------
        bool

        """
        bin_edges = np.array(bin_edges)
        if len(bin_edges) == 1:
            raise ValueError('Single bin edge passed; require at least 3 to'
                             ' determine nature of bin spacing.')
        if len(bin_edges) == 2:
            raise ValueError('Need at least 3 bin edges to determine nature'
                             ' of bin spacing.')
        log_spacing = bin_edges[1:] / bin_edges[:-1]
        if np.allclose(log_spacing, log_spacing[0]):
            return True
        return False

    @staticmethod
    def is_bin_spacing_lin(bin_edges):
        """Check if `bin_edges` define a linearly-uniform bin spacing.

        Parameters
        ----------
        bin_edges : sequence
            Fewer than 2 `bin_edges` - raises ValueError
            Two `bin_edges` - returns True as a reasonable guess
            More than two `bin_edges` - whether spacing is linear is computed

        Returns
        -------
        bool

        Raises
        ------
        ValueError if fewer than 2 `bin_edges` are specified.

        """
        bin_edges = np.array(bin_edges)
        if len(bin_edges) == 1:
            raise ValueError('Single bin edge passed; require at least 3 to'
                             ' determine nature of bin spacing.')
        if len(bin_edges) == 2:
            return True
        lin_spacing = np.diff(bin_edges)
        if np.allclose(lin_spacing, lin_spacing[0]):
            return True
        return False

    @staticmethod
    def is_binning_ok(bin_edges, is_log):
        """Check monotonicity and that bin spacing is logarithmically uniform
        (if `is_log=True`)

        Parameters
        ----------
        bin_edges : sequence
            Bin edges to check the validity of

        is_log : bool
            Whether binning is expected to be logarithmically uniform.

        Returns
        -------
        bool, True if binning is OK, False if not

        """
        # Must be at least two edges to define a single bin
        if len(bin_edges) < 2:
            return False
        # Bin edges must be monotonic and strictly increasing
        lin_spacing = np.diff(bin_edges)
        if not np.all(lin_spacing > 0):
            return False
        # Log binning must have equal widths in log-space (but a single bin
        # has no "spacing" or stride, so no need to check)
        if is_log and len(bin_edges) > 2:
            return OneDimBinning.is_bin_spacing_log(bin_edges)
        return True

    # TODO: refine compatibility test
    def is_compat(self, other):
        """Compatibility -- for now -- is defined by all of self's bin
        edges form a subset of other's bin edges, or vice versa, and the units
        match. This might bear revisiting, or redefining just for special
        circumstances.

        Parameters
        ----------
        other : OneDimBinning

        Returns
        -------
        bool

        """
        if self.units != other.units:
            return False
        my_edges, other_edges = set(self.ebins), set(other.ebins)
        if len(my_edges.difference(other_edges)) == 0:
            return True
        if len(other_edges.difference(my_edges)) == 0:
            return True
        return False

    @new_obj
    def oversample(self, factor):
        """Return a OneDimBinning object oversampled relative to this object's
        binning.

        Parameters
        ----------
        factor : integer
            Factor by which to oversample the binning, with `factor`-times
            as many bins (*not* bin edges) as this object has.

        Returns
        -------
        OneDimBinning object
        """
        assert factor >= 1 and factor == int(factor)
        factor = int(factor)
        if self.is_log:
            bin_edges = np.logspace(np.log10(self.domain[0].m),
                                    np.log10(self.domain[-1].m),
                                    self.n_bins * factor + 1)
        elif self.is_lin:
            bin_edges = np.linspace(self.domain[0].m, self.domain[-1].m,
                                    self.n_bins * factor + 1)
        else: # irregularly-spaced
            bin_edges = []
            for lower, upper in izip(self.bin_edges[:-1].m,
                                     self.bin_edges[1:].m):
                this_bin_new_edges = np.linspace(lower, upper, factor+1)
                # Exclude the last edge, as this will be first edge for the
                # next divided bin
                bin_edges.extend(this_bin_new_edges[:-1])
            # Final bin needs final edge
            bin_edges.append(this_bin_new_edges[-1])
        return {'bin_edges': np.array(bin_edges)*self.units}

    def __getattr__(self, attr):
        return super(OneDimBinning, self).__getattribute__(attr)

    @property
    def state(self):
        state = collections.OrderedDict()
        for attr in self._state_attrs:
            setitem(state, attr, getattr(self, attr))
        return state

    def set_prefixed_attrs(self, obj):
        for attr, spec in self._prefixed_attrs:
            setattr(obj, spec % self.prefix, getattr(self, attr))

    def __str__(self):
        domain_str = 'spanning [%s, %s] %s' %(self.bin_edges[0].magnitude,
                                              self.bin_edges[-1].magnitude,
                                              format(self.units, '~'))
        edge_str = 'with edges at ' + format(self.bin_edges, '~')

        if self.n_bins == 1:
            descr = 'one bin %s' %edge_str
            if self.is_lin:
                descr += ' (behavior is linear)'
            elif self.is_log:
                descr += ' (behavior is logarithmic)'
        elif self.is_lin:
            descr = '%d equally-sized bins %s' %(self.n_bins, domain_str)
        elif self.is_log:
            descr = '%d logarithmically-uniform bins %s' %(self.n_bins,
                                                           domain_str)
        else:
            descr = '%d irregularly-sized bins %s' %(self.n_bins, edge_str)

        return '{name:s}: {descr:s}'.format(name=self.name, descr=descr)

    # TODO: make repr return representation that can recreate binning instead
    # of str (which just looks nice for user)
    def __repr__(self):
        return str(self)

    # TODO: make this actually grab the bins specified (and be able to grab
    # disparate bins, whether or not they are adjacent)... i.e., fill in all
    # upper bin edges, and handle the case that it goes from linear or log
    # to uneven (or if it stays lin or log, keep that attribute for the
    # subselection)
    @new_obj
    def __getitem__(self, index):
        """Return a new OneDimBinning, sub-selected by `index`.

        Parameters
        ----------
        index : int, slice, or length-one Sequence
            The *bin* indices (*not* bin-edge indices) to return.

        Returns
        -------
        A new OneDimBinning but only with bins selected by `index`.

        """
        magnitude = self.bin_edges.magnitude

        # Simple to get all but final bin edge
        bin_edges = magnitude[index].tolist()

        if np.isscalar(bin_edges):
            bin_edges = [bin_edges]
        else:
            bin_edges = list(bin_edges)

        # Append final bin edge, indexed by final-bin index + 1
        if isinstance(index, slice):
            final_bin_index = index.stop
        elif isinstance(index, int):
            final_bin_index = index + 1
        elif isinstance(index, collections.Sequence):
            assert len(index) == 1
            final_bin_index = index[0] + 1
        else:
            raise TypeError('Unhandled index type %s' % type(index))
        bin_edges.append(magnitude[final_bin_index])

        # Retrieve current state; only bin_edges needs to be updated
        return {'bin_edges': np.array(bin_edges)*self.units}

    def __eq__(self, other):
        if not isinstance(other, OneDimBinning):
            return False
        for slot in self._state_attrs:
            if not self.__getattr__(slot) == other.__getattr__(slot):
                return False
        return True

class MultiDimBinning(object):
    """
    Parameters
    ----------
    *args : each a OneDimBinning or Mapping that can construct one via
        OneDimBinning Instantiated binning follows the order in which each
        OneDimBinning See OneDimBinning keys required for a Mapping that can be
        used to instantiate OneDimBinning.

    Properties
    ----------
    dimensions
    n_dimensions
    shape

    Methods
    -------
    assert_array_fits
    assert_compat
    meshgrid
    oversample
    __eq__
    __getitem__
    __repr__
    __str__

    """
    def __init__(self, *args):
        dimensions = []
        shape = []
        for arg_num, arg in enumerate(args):
            if isinstance(arg, OneDimBinning):
                one_dim_binning = arg
            elif isinstance(arg, collections.Mapping):
                one_dim_binning = OneDimBinning(**arg)
            else:
                raise TypeError('Argument #%d unhandled type: %s'
                                %(arg_num, type(arg)))
            one_dim_binning.set_prefixed_attrs(self)
            dimensions.append(one_dim_binning)
            shape.append(one_dim_binning.n_bins)

        self.dimensions = tuple(dimensions)
        self.n_dims = len(self.dimensions)
        self.shape = tuple(shape)

    @property
    def state(self):
        """Everything necessary to fully describe this object's state. Note
        that objects may be returned by reference, so to prevent external
        modification, the user must call deepcopy() separately on the returned
        tuple.

        Returns
        -------
        state tuple; can be passed to instantiate a new MultiDimBinning via
            MultiDimBinning(*state)

        """
        return tuple([d.state for d in self])

    def oversample(self, *factors):
        """Return a Binning object oversampled relative to this binning.

        Parameters
        ----------
        *factors : each factor an int
            Factors by which to oversample the binnings, with `e_factor` times
            as many energy bins and `cz_factor` times as many cosine-zenith
            bins

        """
        if len(factors) == 1:
            factors = [factors[0]]*self.n_dims
        else:
            assert len(factors) == self.n_dims
        return MultiDimBinning(*[dim.oversample(f)
                                 for dim, f in zip(self, factors)])

    def assert_array_fits(self, array):
        """Check if a 2D array of values fits into the defined bins (i.e., has
        the exact shape defined by this binning).

        Parameters
        ----------
        array : 2D array (or sequence-of-sequences)

        Returns
        -------
        bool : True if array fits, False otherwise

        """
        assert array.shape == self.shape, \
                '%s does not match self shape %s' % (array.shape, self.shape)

    def assert_compat(self, other):
        """Check if a (possibly different) binning can map onto the defined
        binning. Allows for simple re-binning schemes (but no interpolation).

        Parameters
        ----------
        `other` : Binning or container with attribute "binning"

        """
        if not isinstance(other, MultiDimBinning):
            for attr, val in other.__dict__.items():
                if isinstance(val, MultiDimBinning):
                    other = val
                    break
        assert isinstance(other, MultiDimBinning)
        if other == self:
            return True
        for my_dim, other_dim in zip(self, other):
            if not my_dim.assert_compat(other_dim):
                return False
        return True

    def meshgrid(self, which='midpoints'):
        which = which.lower().strip()
        if which == 'midpoints':
            return np.meshgrid(*tuple([d.midpoints for d in self]))
        if which == 'visual_centers':
            return np.meshgrid(*tuple([d.visual_centers for d in self]))
        if which == 'bin_edges':
            return np.meshgrid(*tuple([d.bin_edges for d in self]))
        raise ValueError('Unrecognized `which` parameter: "%s"' % which)

    def __eq__(self, other):
        if not isinstance(other, MultiDimBinning):
            return False
        return self.state == other.state

    def __str__(self):
        return '\n'.join([str(dim) for dim in self])

    def __repr__(self):
        return str(self)

    def __iter__(self):
        return iter(self.dimensions)

    def __getitem__(self, index):
        """Interpret indices as indexing bins and *not* bin edges.
        Energy index comes first, coszen index second; both must be present.

        Parameters
        ----------
        index : len-N-sequence of ints or len-N-sequence of slices

        Returns
        -------
        A new Binning object but with the bins specified by `index`.
        Whether or not spacing is logarithmic is unchanged.
        """
        if not isinstance(index, collections.Sequence):
            index = [index]
        input_dim = len(index)
        if input_dim != self.n_dims:
            raise ValueError('Binning is %dD, but %dD indexing was passed'
                             % (self.n_dims, input_dim))
        new_binning = []
        for dim, idx in zip(self.dimensions, index):
            new_binning.append(dim[idx])
        return MultiDimBinning(*new_binning)

def test_Binning():
    b1 = OneDimBinning(name='energy', units='GeV', prefix='e', n_bins=40,
                       is_log=True, domain=[1,80])
    b2 = OneDimBinning(name='coszen', units=None, prefix='cz',
                       n_bins=40, is_lin=True, domain=[-1,1])
    print 'b1:', b1
    print 'b2:', b2
    print 'b1.oversample(10):', b1.oversample(10)
    print 'b1[1:5]:', b1[1:5]


if __name__ == "__main__":
    test_Binning()
