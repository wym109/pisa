# author : J.L. Lanfranchi
#          jll1062+pisa@phys.psu.edu
#
# date   : March 25, 2016

"""
Class to carry information about 2D binning in energy and cosine-zenity, and to
provide basic operations with the binning.
"""

# TODO: include Iterables where only Sequence is allowed now?
# TODO: make indexing accessible by name?

from collections import Iterable, Mapping, OrderedDict, Sequence
from copy import copy, deepcopy
from itertools import izip
from operator import setitem

import numpy as np
import pint; ureg = pint.UnitRegistry()

from pisa.utils.comparisons import recursiveEquality
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging, set_verbosity
from pisa.utils.numerical import normalizeQuantities


HASH_SIGFIGS = 12


class OneDimBinning(object):
    """
    Histogram-oriented binning specialized to a single dimension.

    Either `domain` or `bin_edges` must be specified, but not both. `is_lin`
    and `is_log` are mutually exclusive and *must* be specified if `domain` is
    provided (along with `num_bins`), but these are optional if `bin_edges` is
    specified.

    In the case that `bin_edges` is provided and defines just a single bin, if
    this bin should be treated logarithmically (e.g. for oversampling),
    `is_log=True` must be specified (otherwise, `is_lin` will be assumed to be
    true).

    Parameters
    ----------
    name : str, of length > 0
    tex : str or None
    bin_edges : sequence
    is_log : bool
    is_lin : bool
    num_bins : int
    domain : length-2 sequence


    Attributes
    ----------
    bin_edges
    domain
    hash
    is_irregular
    is_lin
    is_log
    midpoints
    num_bins
    name
    tex
    units
    weighted_centers


    Methods
    -------
    assert_compat
    is_bin_spacing_log
    is_bin_spacing_lin
    is_binning_ok
    is_compat
    ito
    to
    oversample
    __eq__
    __getattr__
    __getitem__
    __len__
    __ne__
    __repr__
    __str__


    Notes
    -----
    Consistency is enforced for all redundant parameters passed to the
    constructor.


    Examples
    --------
    >>> import pint; ureg = pint.UnitRegistry()
    >>> ebins = OneDimBinning(name='energy', tex=r'E_\nu', is_log=True,
    ...                       num_bins=40, domain=[1, 80]*ureg.GeV)
    >>> print ebins
    energy: 40 logarithmically-uniform bins spanning [1.0, 80.0] GeV
    >>> ebins2 = ebins.to('joule')
    >>> print ebins2

    >>> czbins = OneDimBinning(name='coszen', tex=r'\cos\theta_\nu',
    ...                        is_lin=True, num_bins=4, domain=[-1, 0])
    >>> print czbins
    coszen: 4 equally-sized bins spanning [-1.0, 0.0]
    >>> czbins2 = OneDimBinning(name='coszen', tex=r'\cos\theta_\nu',
    ...                         bin_edges=[-1, -0.75, -0.5, -0.25, 0])
    >>> czbins == czbins2
    True

    """

    # `is_log` and `is_lin` are required for state alongsize bin_edges so that
    # a sub-sampling down to a single bin that is then resampled to > 1 bin
    # will retain the log/linear property of the original OneDimBinning.
    _state_attrs = ('name', 'tex', 'bin_edges', 'is_log', 'is_lin')

    def __init__(self, name, tex=None, bin_edges=None, is_log=None,
                 is_lin=None, num_bins=None, domain=None):
        assert isinstance(name, basestring), str(type(name))
        self.name = name
        if tex is None:
            tex = name
        self.tex = tex

        # Temporarily strip units (if any were provided) to make constructing
        # bins consistent (in particular, log(x) isn't valid if x has units),
        # then reattach units after bin_edges has been defined. Default units
        # are "dimensionless".
        units = ureg.dimensionless
        if isinstance(bin_edges, pint.quantity._Quantity):
            units = bin_edges.units
            bin_edges = bin_edges.magnitude

        if isinstance(domain, pint.quantity._Quantity):
            units = domain.units
            domain = domain.magnitude

        # If both `is_log` and `is_lin` are specified, both cannot be true
        # (but both can be False, in case of irregularly-spaced bins)
        if is_log and is_lin:
            raise ValueError('`is_log=%s` contradicts `is_lin=%s`'
                              %(is_log, is_lin))

        # If no bin edges specified, the number of bins, domain, and either
        # log or linear spacing are all required to generate bins
        if bin_edges is None:
            assert num_bins is not None and domain is not None \
                    and (is_log or is_lin), '%s, %s' %(num_bins, domain)
            if is_log:
                is_lin = False
                bin_edges = np.logspace(np.log10(np.min(domain)),
                                        np.log10(np.max(domain)),
                                        num_bins + 1)
            elif is_lin:
                is_log = False
                bin_edges = np.linspace(np.min(domain),
                                        np.max(domain),
                                        num_bins + 1)

        elif domain is not None:
            raise ValueError('Specify `bin_edges` or `domain`, but not both.')

        bin_edges = np.array(bin_edges)
        if is_lin:
            assert self.is_bin_spacing_lin(bin_edges), str(bin_edges)
            is_log = False
        elif is_log:
            assert self.is_binning_ok(bin_edges, is_log=True), str(bin_edges)
            is_lin = False
        else:
            is_lin = self.is_bin_spacing_lin(bin_edges)
            try:
                is_log = self.is_bin_spacing_log(bin_edges)
            except ValueError:
                is_log = False

        #if not is_lin and not is_log:
        #    is_log = self.is_bin_spacing_log(bin_edges)

        # Attach units to bin edges
        self.bin_edges = bin_edges * units

        # Define domain and attach units

        self.domain = np.array([bin_edges[0], bin_edges[-1]]) * units

        # Derive rest of unspecified parameters from bin_edges or enforce
        # them if they were specified as arguments to init
        if num_bins is None:
            num_bins = len(self.bin_edges) - 1
        else:
            assert num_bins == len(self.bin_edges) - 1, \
                    '%s, %s' %(num_bins, self.bin_edges)
        self.num_bins = num_bins

        self.is_lin = is_lin
        self.is_log = is_log
        self.is_irregular = not (self.is_lin or self.is_log)
        self.midpoints = (self.bin_edges[:-1] + self.bin_edges[1:])/2.0
        if self.is_log:
            self.weighted_centers = np.sqrt(self.bin_edges[:-1] *
                                            self.bin_edges[1:])
        else:
            self.weighted_centers = self.midpoints

        # TODO: define hash based upon conversion of things to base units (such
        # that a valid comparison can be made between indentical binnings but
        # that use different units). Be careful to round to just less than
        # double-precision limits after conversion so that hashes will work out
        # to be the same after conversion to the base units.

    @property
    def state(self):
        state = OrderedDict()
        for attr in self._state_attrs:
            setitem(state, attr, getattr(self, attr))
        return state

    @property
    def hash(self):
        """Hash value based upon less-than-double-precision-rounded
        numerical values and any other state. Rounding is done to
        `HASH_SIGFIGS` significant figures.

        Set this class attribute to None to keep full numerical precision in
        the values hashed (but be aware that this can cause equal things
        defined using different unit orders-of-magnitude to hash differently).

        """
        normalized_state = OrderedDict()
        for attr in self._state_attrs:
            val = normalizeQuantities(getattr(self, attr), HASH_SIGFIGS)
            setitem(normalized_state, attr, val)
        return hash_obj(normalized_state)

    @property
    def units(self):
        return self.bin_edges.units

    @property
    def bin_sizes(self):
        return np.diff(self.bin_edges)*self.units

    def new_obj(original_function):
        """ decorator to deepcopy unaltered states into new object """
        def new_function(self, *args, **kwargs):
            new_state = OrderedDict()
            state_updates = original_function(self, *args, **kwargs)
            for slot in self._state_attrs:
                if state_updates.has_key(slot):
                    new_state[slot] = state_updates[slot]
                else:
                    new_state[slot] = deepcopy(getattr(self, slot))
            return OneDimBinning(**new_state)
        return new_function

    def __len__(self):
        """Number of bins (*not* number of bin edges)."""
        return self.num_bins

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
        if len(bin_edges) < 3:
            raise ValueError('%d bin edge(s) passed; require at least 3 to'
                             ' determine nature of bin spacing.'
                             %len(bin_edges))
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

    # TODO: refine compatibility test to handle compatible units; as of now,
    # both upsampling and downsampling are allowed. Is this reasonable
    # behavior?
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
        if self.units.dimensionality != other.units.dimensionality:
            return False

        my_normed_bin_edges = set(normalizeQuantities(self.bin_edges))
        other_normed_bin_edges = set(normalizeQuantities(other.bin_edges))

        if len(my_normed_bin_edges.difference(other_normed_bin_edges)) == 0:
            return True
        if len(other_normed_bin_edges.difference(my_normed_bin_edges)) == 0:
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
        if factor < 1 or factor != int(factor):
            raise ValueError('`factor` must be integer >= 0; got %s' %factor)

        factor = int(factor)
        if self.is_log:
            bin_edges = np.logspace(np.log10(self.domain[0].m),
                                    np.log10(self.domain[-1].m),
                                    self.num_bins * factor + 1)

        elif self.is_lin:
            bin_edges = np.linspace(self.domain[0].m, self.domain[-1].m,
                                    self.num_bins * factor + 1)
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

    def ito(self, units):
        """Convert units in-place. Cf. Pint's `ito` method."""
        if units is None:
            units = ''
        for attr in ['bin_edges', 'domain', 'midpoints', 'weighted_centers']:
            getattr(self, attr).ito(units)

    @new_obj
    def to(self, units):
        if units is None:
            units = ''
        return {'bin_edges': self.bin_edges.to(ureg(str(units)))}

    def __getattr__(self, attr):
        return super(self.__class__, self).__getattribute__(attr)

    def __str__(self):
        domain_str = 'spanning [%s, %s] %s' %(self.bin_edges[0].magnitude,
                                              self.bin_edges[-1].magnitude,
                                              format(self.units, '~'))
        edge_str = 'with edges at [' + \
                ', '.join([str(e) for e in self.bin_edges.m]) + \
                '] ' + format(self.bin_edges.u, '~')

        if self.num_bins == 1:
            descr = 'one bin %s' %edge_str
            if self.is_lin:
                descr += ' (behavior is linear)'
            elif self.is_log:
                descr += ' (behavior is logarithmic)'
        elif self.is_lin:
            descr = '%d equally-sized bins %s' %(self.num_bins, domain_str)
        elif self.is_log:
            descr = '%d logarithmically-uniform bins %s' %(self.num_bins,
                                                           domain_str)
        else:
            descr = '%d irregularly-sized bins %s' %(self.num_bins, edge_str)

        return '{name:s}: {descr:s}'.format(name=self.name, descr=descr)

    # TODO: make repr return representation that can recreate binning instead
    # of str (which just looks nice for user); think now alos how to pickle
    # and unpickle
    def __repr__(self):
        return str(self)

    # TODO: make this actually grab the bins specified (and be able to grab
    # disparate bins, whether or not they are adjacent)... i.e., fill in all
    # upper bin edges, and handle the case that it goes from linear or log
    # to uneven (or if it stays lin or log, keep that attribute for the
    # subselection). Granted, a OneDimBinning object right now requires
    # monotonically-increasing and adjacent bins.
    @new_obj
    def __getitem__(self, index):
        """Return a new OneDimBinning, sub-selected by `index`.

        Parameters
        ----------
        index : int, slice, or length-one Sequence
            The *bin indices* (not bin-edge indices) to return. Generated
            OneDimBinning object must obey the usual rules (monotonic, etc.).

        Returns
        -------
        A new OneDimBinning but only with bins selected by `index`.

        """
        magnitude = self.bin_edges.magnitude
        units = self.bin_edges.units
        orig_index = index

        # Simple to get all but final bin edge
        bin_edges = magnitude[index].tolist()

        if np.isscalar(bin_edges):
            bin_edges = [bin_edges]
        else:
            bin_edges = list(bin_edges)

        # Convert index/indices to positive-number sequence
        if isinstance(index, slice):
            index = range(*index.indices(len(self)))
        if isinstance(index, int):
            index = [index]
        if isinstance(index, Sequence):
            if len(index) == 0:
                raise ValueError('`index` "%s" results in no bins being'
                                 ' specified.' %orig_index)
            if len(index) > 1 and not np.all(np.diff(index) == 1):
                raise ValueError('Bin indices must be monotonically'
                                 ' increasing and adjacent.')
            new_edges = set()
            for bin_index in index:
                assert(bin_index >= -len(self) and bin_index < len(self)), \
                        str(bin_index)
                edge_ind0 = bin_index % len(self)
                edge_ind1 = edge_ind0 + 1
                new_edges = new_edges.union((self.bin_edges[edge_ind0].m,
                                             self.bin_edges[edge_ind1].m))
        else:
            raise TypeError('Unhandled index type %s' %type(index))

        # Retrieve current state; only bin_edges needs to be updated
        return {'bin_edges': np.array(sorted(new_edges))*units}

    def __eq__(self, other):
        if not isinstance(other, OneDimBinning):
            print 'not OneDimBinning;', type(other)
            return False
        for slot in self._state_attrs:
            normed_self = normalizeQuantities(self.__getattr__(slot),
                                              sigfigs=HASH_SIGFIGS)
            normed_other = normalizeQuantities(other.__getattr__(slot),
                                               sigfigs=HASH_SIGFIGS)
            if not np.all(normed_other == normed_self):
                print normed_other
                print normed_self
                return False
        return True

    def __ne__(self, other):
        return not self == other


# TODO: make this able to be loaded from a pickle!!!
class MultiDimBinning(object):
    """
    Multi-dimensional binning object. This can contain one or more
    OneDimBinning objects, and all subsequent operations (e.g. slicing) will
    act on these in the order they are supplied.

    Parameters
    ----------
    *args : each a OneDimBinning or Mapping that can construct one via
        OneDimBinning Instantiated binning follows the order in which each
        OneDimBinning See OneDimBinning keys required for a Mapping that can be
        used to instantiate OneDimBinning.

    Attributes
    ----------
    dimensions
    hash
    names
    num_dims
    shape

    Methods
    -------
    assert_array_fits
    assert_compat
    meshgrid
    oversample
    to
    __eq__
    __getitem__
    __iter__
    __len__
    __ne__
    __repr__
    __str__

    """
    def __init__(self, *args):
        dimensions = []
        shape = []
        # Break out any sequences passed in
        objects = []
        for arg in args:
            if isinstance(arg, (OneDimBinning, Mapping)):
                objects.append(arg)
            elif isinstance(arg, MultiDimBinning):
                objects.extend(arg.dimensions)
            elif isinstance(arg, (Iterable, Sequence)):
                objects.extend(arg)

        for obj_num, obj in enumerate(objects):
            if isinstance(obj, OneDimBinning):
                one_dim_binning = obj
            elif isinstance(obj, Mapping):
                one_dim_binning = OneDimBinning(**obj)
            else:
                raise TypeError('Argument/object #%d unhandled type: %s'
                                %(obj_num, type(obj)))
            dimensions.append(one_dim_binning)
            shape.append(one_dim_binning.num_bins)

        self.dimensions = tuple(dimensions)
        self.num_dims = len(self.dimensions)
        self.shape = tuple(shape)

    @property
    def names(self):
        return [dim.name for dim in self.dimensions]

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
        return tuple([d.state for d in self.dimensions])

    @property
    def hash(self):
        return hash_obj(tuple([d.hash for d in self.dimensions]))

    def oversample(self, *args, **kwargs):
        """Return a Binning object oversampled relative to this binning.

        Parameters
        ----------
        *args : each factor an int
            Factors by which to oversample the binnings. There must either be
            one factor (one arg)--which will be broadcast to all dimensions--or
            there must be as many factors (args) as there are dimensions.
            If positional args arespecified (i.e., non-kwargs), then kwargs are
            forbidden.

        **kwargs : name=factor pairs

        Notes
        -----
        Can either specify oversmapling by passin gin args (ordered values, no
        keywords) or kwargs (order doesn't matter, but uses keywords), but not
        both.

        """
        factors = self._args_kwargs_to_list(*args, **kwargs)
        new_binning = [dim.oversample(f)
                       for dim, f in izip(self.dimensions, factors)]
        return MultiDimBinning(new_binning)

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
                '%s does not match self shape %s' %(array.shape, self.shape)

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
        assert isinstance(other, MultiDimBinning), str(type(other))
        if other == self:
            return True
        for my_dim, other_dim in zip(self, other):
            if not my_dim.assert_compat(other_dim):
                return False
        return True

    def _args_kwargs_to_list(self, *args, **kwargs):
        """Take either args or kwargs (but not both) and convert into a simple
        sequence of values. Broadcasts a single arg to all dimensions."""
        if not np.logical_xor(len(args), len(kwargs)):
            raise ValueError('Either args (values specified by order and not'
                             ' specified by name) or kwargs (values specified'
                             ' by name=value pairs) can be used, but not'
                             ' both.')

        if len(args) == 1:
            return [args[0]]*self.num_dims

        if len(args) > 1:
            if len(args) != self.num_dims:
                raise ValueError('Specified %s args, but binning is'
                                 ' %s-dim.' %(len(values), self.num_dims))
            return args

        if set(kwargs.keys()) != set(self.names):
            raise ValueError('Specified dimensions "%s" but this has'
                             ' dimensions "%s"' %(sorted(kwargs.keys()),
                                                  self.names))
        return [kwargs[name] for name in self.names]

    def ito(self, *args, **kwargs):
        """Convert units in-place. Cf. Pint's `ito` method."""
        units_list = self._args_kwargs_to_list(*args, **kwargs)
        [dim.ito(units) for dim, units in izip(self.dimensions, units_list)]

    def to(self, *args, **kwargs):
        """Convert the contained dimensions to the passed units."""
        units_list = self._args_kwargs_to_list(*args, **kwargs)
        new_binnings = [dim.to(units)
                        for dim, units in izip(self.dimensions, units_list)]
        return MultiDimBinning(new_binnings)

    # TODO: magnitude method that replicates Pint version's behavior (???)
    #def magnitude(self):

    def meshgrid(self, entity, attach_units=True):
        """Apply NumPy's meshgrid method on various entities of interest.


        Parameters
        ----------
        entity : string
            One of 'midpoints', 'weighted_centers', 'bin_edges', or 'bin_sizes'

        attach_units : bool
            Whether to attach units to the result (can save computation time by
            not doing so).


        Returns
        -------
        numpy ndarray or Pint quantity of the same


        See Also
        --------
        numpy.meshgrid

        """
        entity = entity.lower().strip()
        units = [d.units for d in self.dimensions]
        #stripped = self.stripped()
        if entity == 'midpoints':
            mg = np.meshgrid(*tuple([d.midpoints for d in self.dimensions]))
        elif entity == 'weighted_centers':
            mg = np.meshgrid(*tuple([d.weighted_centers
                                     for d in self.dimensions]))
        elif entity == 'bin_edges':
            mg = np.meshgrid(*tuple([d.bin_edges for d in self.dimensions]))
        elif entity == 'bin_sizes':
            mg = np.meshgrid(*tuple([d.bin_sizes for d in self.dimensions]))
        else:
            raise ValueError('Unrecognized `entity`: "%s"' %entity)

        if attach_units:
            return [m*dim.units for m, dim in izip(mg, self.dimensions)]
        return mg

    # TODO: modify technique depending upon grid size for memory concerns, or
    # even take a `method` argument to force method manually.
    def bin_sizes(self, attach_units=True):
        meshgrid = self.meshgrid(entity='bin_sizes', attach_units=False)
        volumes = reduce(lambda x,y: x*y, meshgrid)
        if attach_units:
            return volumes * reduce(lambda x,y:x*y, [ureg(str(d.units))
                                                     for d in self.dimensions])
        return volumes

    def __eq__(self, other):
        if not isinstance(other, MultiDimBinning):
            return False
        return recursiveEquality(self.state, other.state)

    def __getattr__(self, attr):
        for d in self.dimensions:
            if d.name == attr:
                return d
        return super(self.__class__, self).__getattribute__(attr)

    def __getitem__(self, index):
        """Interpret indices as indexing bins and *not* bin edges.
        Indices refer to dimensions in same order they were specified at
        instantiation, and all dimensions must be present.


        Parameters
        ----------
        index : int, len-N-sequence of ints, or len-N-sequence of slices
            If an integer is passed:
              * If num_dims is 1, `index` indexes into the bins of the sole
                OneDimBinning. The bin is returned.
              * If num_dims > 1, `index` indexes which contained OneDimBinning
                object to return.
            If a len-N-sequence of integers or slices is passed, dimensions are
            indexed by these in the order in which dimensions are stored
            internally.


        Returns
        -------
        A MultiDimBinning object new Binning object but with the bins specified
        by `index`. Whether or not behavior is logarithmic is unchanged.

        """
        if isinstance(index, basestring):
            return getattr(self, index)

        # TODO: implement a "linearization" like np.flatten() to iterate
        # through each bin individually without hassle for the user...
        #if self.num_dims == 1 and np.isscalar(index):
        #    return self.dimensions[0]

        if not isinstance(index, Sequence):
            index = [index]

        input_dim = len(index)
        if input_dim != self.num_dims:
            raise ValueError('Binning is %dD, but %dD indexing was passed'
                             %(self.num_dims, input_dim))

        new_binning = [dim[idx] for dim, idx in zip(self.dimensions, index)]

        return MultiDimBinning(*new_binning)

    #def __iter__(self):
    #    return iter(self.dimensions)

    def __len__(self):
        return self.num_dims

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return str(self)

    def __str__(self):
        return '\n'.join([str(dim) for dim in self.dimensions])


def test_OneDimBinning():
    import pickle
    b1 = OneDimBinning(name='energy', num_bins=40, is_log=True,
                       domain=[1, 80]*ureg.GeV)
    b2 = OneDimBinning(name='coszen', num_bins=40, is_lin=True,
                       domain=[-1, 1])
    logging.debug('len(b1): %s' %len(b1))
    logging.debug('b1: %s' %b1)
    logging.debug('b2: %s' %b2)
    logging.debug('b1.oversample(10): %s' %b1.oversample(10))
    logging.debug('b1.oversample(1): %s' %b1.oversample(1))
    # Slicing
    logging.debug('b1[1:5]: %s' %b1[1:5])
    logging.debug('b1[:]: %s' %b1[:])
    logging.debug('b1[-1]: %s' %b1[-1])
    logging.debug('b1[:-1]: %s' %b1[:-1])
    logging.debug('copy(b1): %s' %copy(b1))
    logging.debug('deepcopy(b1): %s' %deepcopy(b1))
    pickle.dumps(b1, pickle.HIGHEST_PROTOCOL)
    try:
        b1[-1:-3]
    except ValueError:
        pass
    else:
        assert False

    b3 = OneDimBinning(name='distance', num_bins=10, is_log=True,
                       domain=[0.1, 10]*ureg.m)
    b4 = OneDimBinning(name='distance', num_bins=10, is_log=True,
                       domain=[1e5, 1e7]*ureg.um)

    # Without rounding, converting bin edges to base units yields different
    # results due to finite precision effects
    assert np.any(normalizeQuantities(b3.bin_edges, sigfigs=None)
                  != normalizeQuantities(b4.bin_edges, sigfigs=None))

    # Normalize function should take care of this
    assert np.all(normalizeQuantities(b3.bin_edges, sigfigs=HASH_SIGFIGS)
                  == normalizeQuantities(b4.bin_edges,
                                          sigfigs=HASH_SIGFIGS))

    # And the hashes should be equal, reflecting the latter result
    assert b3.hash == b4.hash

    s = pickle.dumps(b3, pickle.HIGHEST_PROTOCOL)
    b3_loaded = pickle.loads(s)
    assert b3_loaded == b3

    logging.info('<< PASSED >> test_OneDimBinning')


def test_MultiDimBinning():
    import pickle
    b1 = OneDimBinning(name='energy', num_bins=40, is_log=True,
                       domain=[1, 80]*ureg.GeV)
    b2 = OneDimBinning(name='coszen', num_bins=40, is_lin=True,
                       domain=[-1, 1])
    mdb = MultiDimBinning(b1, b2)
    b00 = mdb[0,0]
    x0 = mdb[0:, 0]
    x1 = mdb[0:, 0:]
    x2 = mdb[0, 0:]
    x3 = mdb[-1, -1]
    logging.debug(str(mdb.energy))
    logging.debug('copy(mdb): %s' %copy(mdb))
    logging.debug('deepcopy(mdb): %s' %deepcopy(mdb))
    s = pickle.dumps(mdb, pickle.HIGHEST_PROTOCOL)
    # TODO: add these back in when we get pickle loading working!
    #mdb2 = pickle.loads(s)
    #assert mdb2 == mdb1

    binning = MultiDimBinning(
        dict(name='energy', is_log=True, domain=[1, 80]*ureg.GeV, num_bins=40),
        dict(name='coszen', is_lin=True, domain=[-1, 0], num_bins=20)
    )

    assert binning.oversample(10).shape == (400, 200)

    assert binning.oversample(10, 1).shape == (400, 20)
    assert binning.oversample(1, 3).shape == (40, 60)

    assert binning.oversample(coszen=10, energy=2).shape == (80, 200)
    assert binning.oversample(1, 1) == binning

    assert binning.to('MeV', '') == binning
    assert binning.to('MeV', '').hash == binning.hash

    mg = binning.meshgrid(entity='midpoints')
    mg = binning.meshgrid(entity='bin_edges')
    mg = binning.meshgrid(entity='weighted_centers')
    mg = binning.meshgrid(entity='midpoints')
    bv = binning.bin_sizes(attach_units=False)
    bv = binning.bin_sizes(attach_units=True)
    binning.to('MeV', None)
    binning.to('MeV', '')
    binning.to(ureg.joule, '')

    logging.info('<< PASSED >> test_MultiDimBinning')


if __name__ == "__main__":
    test_OneDimBinning()
    test_MultiDimBinning()
