#
# author: J.L. Lanfranchi
#         jll1062+pisa@phys.psu.edu
#
# date:   March 22, 2016
#


from collections import Mapping, OrderedDict, Sequence
from copy import deepcopy
from functools import total_ordering
from itertools import izip
from operator import setitem

import numpy as np
import pint
from pisa import ureg, Q_

from pisa.utils.comparisons import isbarenumeric, normQuant, recursiveEquality
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging, set_verbosity
from pisa.utils.profiler import profile

# TODO: Make property "frozen" or "read_only" so params in param set e.g.
# returned by a template maker -- which updating the values of will NOT have
# the effect the user might expect -- will be explicitly forbidden?

# TODO: units: pass in attached to values, or as separate kwarg? If e.g.
# value hasn't been set yet, then there's no implicit units to reference
# when setting the prior, range, and possibly other things (which all need to
# at least have compatible units)
@total_ordering
class Param(object):
    """Parameter class to store any kind of parameters

    Parameters
    ----------
    name : string

    value : string or pint Quantity with units

    prior : pisa.prior.Prior

    range : sequence of two numbers or Pint quantities

    is_fixed : bool

    is_discrete : bool

    tex : string

    help : string

    Notes
    -----
    In the case of a free (`is_fixed`=False) parameter, a valid range for the
    parameter should be spicfied and a prior must be assigned to compute llh and
    chi2 values.

    Examples
    --------
    >>> from pisa import ureg
    >>> from pisa.core.prior import Prior
    >>> gaussian = Prior(kind='gaussian', mean=10*ureg.meter,
    ...                  stddev=1*ureg.meter)
    >>> x = Param(name='x', value=1.5*ureg.foot, prior=gaussian,
    ...           range=[-10, 60]*ureg.foot, is_fixed=False, is_discrete=False,
    ...           tex=r'{\rm x}')
    >>> x.value
    <Quantity(1.5, 'foot')>
    >>> print x.prior_llh
    -45.532515919999994
    >>> print x.to('m')

    >>> x.value = 10*ureg.m
    >>> print x.value
    <Quantity(32.8083989501, 'foot')>
    >>> x.ito('m')
    >>> print x.value

    >>> x.prior_llh
    -1.5777218104420236e-30
    >>> p.nominal_value

    >>> x.reset()
    >>> print x.value


    """
    _slots = ('name', 'value', 'prior', 'range', 'is_fixed', 'is_discrete',
              'nominal_value', '_rescaled_value',
              '_nominal_value', '_tex', 'help','_value', '_range', '_units')
    _state_attrs = ('name', 'value', 'prior', 'range', 'is_fixed',
                     'is_discrete', 'nominal_value', 'tex', 'help')

    def __init__(self, name, value, prior, range, is_fixed, is_discrete=False,
                 nominal_value=None, tex=None, help=''):
        self._value = None
        self.value = value
        self.name = name
        self._tex = tex if tex is not None else name
        self.help = help
        self.range = range
        self.prior = prior
        self.is_fixed = is_fixed
        self.is_discrete = is_discrete
        self._nominal_value = value if nominal_value is None else nominal_value

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return recursiveEquality(self.state, other.state)

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        return self.name < other.name

    def __setattr__(self, attr, val):
        if attr not in self._slots:
            raise AttributeError('Invalid attribute: %s' % (attr,))
        object.__setattr__(self, attr, val)

    def __str__(self):
        return '%s=%s; prior=%s, range=%s, is_fixed=%s,' \
                ' is_discrete=%s; help="%s"' \
                % (self.name, self.value, self.prior, self.range,
                   self.is_fixed, self.is_discrete, self.help)

    def validate_value(self, value):
        if self.range is not None:
            if self.is_discrete:
                assert value in self.range, str(value) + ' ' + str(self.range)
            else:
                assert value >= min(self.range) and \
                        value <= max(self.range), \
                        'value=' + str(value) + '; range=' + str(self.range)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        # A number with no units actually has units of "dimensionless"
        if isbarenumeric(val):
            val = val * ureg.dimensionless
        if self._value is not None:
            if hasattr(self._value, 'units'):
                assert hasattr(val, 'units'), \
                        'Passed values must have units if the param has units'
                val = val.to(self._value.units)
            self.validate_value(val)
        self._value = val
        if hasattr(self._value, 'units'):
            self._units = self._value.units
        else:
            self._units = ureg.Unit('dimensionless')

    @property
    def magnitude(self):
        return self._value.magnitude

    @property
    def m(self):
        return self._value.magnitude

    def m_as(self, u):
        return self._value.m_as(u)

    @property
    def dimensionality(self):
        return self._value.dimensionality

    @property
    def units(self):
        return self._units

    @property
    def u(self):
        return self._units

    @property
    def range(self):
        if self._range is None:
            return None
        return tuple(self._range)

    @range.setter
    def range(self, values):
        if values is None:
            self._range = None
            return
        new_vals = []
        for val in values:
            if isbarenumeric(val):
                val = val * ureg.dimensionless
            assert type(val) == type(self.value), \
                    'Value "%s" has type %s but must be of type %s.' \
                    %(val, type(val), type(self.value))
            if isinstance(self.value, pint.quantity._Quantity):
                assert self.dimensionality == val.dimensionality, \
                    'Value "%s" units "%s" incompatible with units "%s".' \
                    %(val, val.units, self.units)

            new_vals.append(val)
        self._range = new_vals

    @property
    def _rescaled_value(self):
        if self.is_discrete:
            val = self.value
        else:
            val = (self._value.m - self.range[0].m) \
                    / (self.range[1].m-self.range[0].m)
        if hasattr(val, 'magnitude'):
            val = val.magnitude
        return val

    @_rescaled_value.setter
    def _rescaled_value(self, rval):
        self.value = ((self.range[1].m - self.range[0].m)*rval +
                      self.range[0].m)*self.units

    @property
    def tex(self):
        return '%s=%s' % (self._tex, self.value)

    @tex.setter
    def tex(self, t):
        self._tex = t if t is not None else self.name

    @property
    def nominal_value(self):
        return self._nominal_value

    @nominal_value.setter
    def nominal_value(self, value):
        self.validate_value(value)
        self._nominal_value = value

    @property
    def state(self):
        state = OrderedDict()
        for attr in self._state_attrs:
            val = getattr(self, attr)
            if hasattr(val, 'state'):
                val = val.state
            setitem(state, attr, val)
        return state

    def reset(self):
        """Reset the parameter's value to its nominal value."""
        self.value = self.nominal_value

    def set_nominal_to_current_value(self):
        """Define the nominal value to the parameter's current value."""
        self.nominal_value = self.value

    def prior_penalty(self, metric):
        if self.prior is None:
            return 0
        metric = metric.lower() if isinstance(metric, basestring) else metric
        if metric in ['llh', 'barlow_llh', 'conv_llh']:
            logging.trace('self.value: %s' %self.value)
            logging.trace('self.prior: %s' %self.prior)
            return self.prior.llh(self.value)
        elif metric in ['chi2']:
            return self.prior.chi2(self.value)
        else:
            raise ValueError('Unrecognized `metric` "%s"' %str(metric))

    def to(self, units):
        """Return an equivalent copy of param but in units of `units`.

        Parameters
        ----------
        units : string or pint.Unit

        Returns
        -------
        Param : copy of this param, but in specified `units`.

        See also
        --------
        ito
        Pint.to
        Pint.ito

        """
        new_param = Param(**deepcopy(self.state))
        new_param.ito(units)
        return new_param

    def ito(self, units):
        """Convert this param (in place) to have units of `units`.

        Parameters
        ----------
        units : string or pint.Unit

        Returns
        -------
        None

        See also
        --------
        to
        Pint.to
        Pint.ito

        """
        self._value.ito(units)

    @property
    def prior_llh(self):
        return self.prior_penalty(metric='llh')

    @property
    def prior_chi2(self):
        return self.prior_penalty(metric='chi2')

    @property
    def state_hash(self):
        return hash_obj(normQuant(self.state))


# TODO: temporary modification of parameters via "with" syntax?
class ParamSet(object):
    """Container class for a set of parameters. Most methods are passed through
    to contained params.

    Parameters
    ----------
    *args : one or more Param objects or sequences thereof

    Examples
    --------
    >>> from pisa import ureg
    >>> from pisa.core.prior import Prior

    >>> e_prior = Prior(kind='gaussian', mean=10*ureg.GeV, stddev=1*ureg.GeV)
    >>> cz_prior = Prior(kind='uniform', llh_offset=-5)
    >>> reco_energy = Param(name='reco_energy', value=12*ureg.GeV,
                            prior=e_prior, range=[1, 80]*ureg.GeV,
                            is_fixed=False, is_discrete=False,
                            tex=r'E^{\rm reco}')
    >>> reco_coszen = Param(name='reco_coszen', value=-0.2, prior=cz_prior,
                            range=[-1, 1], is_fixed=True, is_discrete=False,
                            tex=r'\cos\,\theta_Z^{\rm reco}')
    >>> param_set = ParamSet(reco_energy, reco_coszen)
    >>> print param_set
    reco_coszen=-2.0000e-01 reco_energy=+1.2000e+01 GeV

    >>> print param_set.free
    reco_energy=+1.2000e+01 GeV

    >>> print param_set.reco_energy.value
    12 gigaelectron_volt

    >>> print [p.prior_llh for p in param_set]
    [-5.0, -2]

    >>> print param_set.priors_llh
    -7.0

    >>> print param_set.values_hash
    3917547535160330856

    >>> print param_set.free.values_hash
    -7121742559130813936

    """
    def __init__(self, *args):
        param_sequence = []
        # Unpack the input args into a flat list of params
        for arg in args:
            try:
                param_sequence.extend(arg)
            except TypeError:
                param_sequence.append(arg)

        # Disallow duplicated params
        all_names = [p.name for p in param_sequence]
        unique_names = set(all_names)
        if len(unique_names) != len(all_names):
            duplicates = set([x for x in all_names if all_names.count(x) > 1])
            raise ValueError('Duplicate definitions found for param(s): ' +
                             ', '.join(str(e) for e in duplicates))

        # Elements of list must be Param type
        assert all([isinstance(x, Param) for x in param_sequence]), \
                'All params must be of type "Param"'

        self._params = sorted(param_sequence)

    @property
    def _by_name(self):
        return {obj.name: obj for obj in self._params}

    def index(self, value):
        idx = -1
        if isinstance(value, int):
            idx = value
        elif value in self.names:
            idx = self.names.index(value)
        elif isinstance(value, Param) and value in self._params:
            idx = self._params.index(value)
        if idx < 0 or idx >= len(self):
            raise ValueError('%s not found in ParamSet' % (value,))
        return idx

    def replace(self, new):
        idx = self.index(new.name)
        self._params[idx] = new

    def fix(self, x):
        if isinstance(x, (Param, int, basestring)):
            x = [x]
        my_names = self.names
        for name in x:
            self[self.index(name)].is_fixed = True

    def unfix(self, x):
        if isinstance(x, (Param, int, basestring)):
            x = [x]
        my_names = self.names
        for name in x:
            self[self.index(name)].is_fixed = False

    def update(self, obj, existing_must_match=False, extend=True):
        """Update this param set using `obj`.

        Default behavior is similar to Python's dict.update, but this can be
        modified via `existing_must_match` and `extend`.

        Parameters
        ----------
        obj : Param, ParamSet, or sequence thereof
            Param or container with params to update and/or extend this param
            set
        existing_must_match : bool
            If True, raises ValueError if param values passed in that already
            exist in this param set have differing values.
        extend : bool
            If True, params not in this param set are appended.

        """
        if isinstance(obj, Sequence) or isinstance(obj, ParamSet):
            for param in obj:
                self.update(param, existing_must_match=existing_must_match,
                            extend=extend)
            return
        if not isinstance(obj, Param):
            raise ValueError('`obj`="%s" is not a Param' % (obj))
        param = obj
        if param.name in self.names:
            if existing_must_match and (param != self[param.name]):
                raise ValueError(
                    'Param "%s" specified as\n\n%s\n\ncontradicts'
                    ' internally-stored version:\n\n%s'
                    %(param.name, param.state, self[param.name].state)
                )
            self.replace(param)
        elif extend:
            self._params.append(param)

    def extend(self, obj):
        """Append param(s) in `obj` to this param set, but ensure params in
        `obj` that are already in this param set match. Params with same name
        attribute are not duplicated.

        (Convenience method or calling `update` method with
        existing_must_match=True and extend=True.)

        """
        self.update(obj, existing_must_match=True, extend=True)

    def update_existing(self, obj):
        """Only existing params in this set are updated by that(those) param(s)
        in obj.

        Convenience method for calling `update` with
        existing_must_match=False and extend=False.

        """
        self.update(obj, existing_must_match=False, extend=False)

    def __len__(self):
        return len(self._params)

    def __setitem__(self, i, val):
        if isinstance(i, int):
            self._params[i].value = val
        elif isinstance(i, basestring):
            self._by_name[i].value = val

    def __getitem__(self, i):
        if isinstance(i, int):
            return self._params[i]
        elif isinstance(i, basestring):
            return self._by_name[i]

    def __getattr__(self, attr):
        try:
            return super(ParamSet, self).__getattr__(attr)
        except AttributeError, exc:
            try:
                return self[attr]
            except KeyError:
                raise exc

    def __setattr__(self, attr, val):
        try:
            super(ParamSet, self).__setattr__(attr, val)
        except AttributeError:
            self._params[attr].value = val

    def __iter__(self):
        return iter(self._params)

    def __str__(self):
        numfmt = '%+.4e'
        strings = []
        for p in self:
            string = p.name + '='
            if isinstance(p.value, pint.quantity._Quantity):
                string += numfmt %p.m
                full_unit_str = str(p.u)
                if full_unit_str in [str(ureg('electron_volt ** 2').u)]:
                    unit = ' eV2'
                elif full_unit_str in [str(ureg.deg)]:
                    unit = ' deg'
                elif full_unit_str in [str(ureg.rad)]:
                    unit = ' rad'
                else:
                    unit = ' ' + format(p.u, '~')
                string += unit
            else:
                try:
                    string += numfmt %p.value
                except:
                    string += '%s' %p.value
            strings.append(string.strip())
        return ' '.join(strings)

    def priors_penalty(self, metric):
        return np.sum([obj.prior_penalty(metric=metric)
                       for obj in self._params])

    def reset_all(self):
        """Reset both free and fixed parameters to their nominal values."""
        self.values = self.nominal_values

    def reset_free(self):
        """Reset only free parameters to their nominal values."""
        self.free.reset_all()

    def set_nominal_by_current_values(self):
        """Define the nominal values as the parameters' current values."""
        self.nominal_values = self.values

    @property
    def _rescaled_values(self):
        """Parameter values rescaled to be in the range [0, 1], based upon
        their defined range."""
        return tuple([param._rescaled_value for param in self._params])

    @_rescaled_values.setter
    def _rescaled_values(self, vals):
        assert len(vals) == len(self)
        for param, val in zip(self._params, vals):
            param._rescaled_value = val

    @property
    def tex(self):
        return r',\,'.join([obj.tex for obj in self._params])

    @property
    def fixed(self):
        return ParamSet([obj for obj in self._params if obj.is_fixed])

    @property
    def free(self):
        return ParamSet([obj for obj in self._params if not obj.is_fixed])

    @property
    def continuous(self):
        return ParamSet([obj for obj in self._params if not obj.is_discrete])

    @property
    def discrete(self):
        return ParamSet([obj for obj in self._params if obj.is_discrete])

    @property
    def are_fixed(self):
        return tuple([obj.is_fixed for obj in self._params])

    @property
    def are_discrete(self):
        return tuple([obj.is_discrete for obj in self._params])

    @property
    def names(self):
        return tuple([obj.name for obj in self._params])

    @property
    def values(self):
        return tuple([obj.value for obj in self._params])

    @values.setter
    def values(self, values):
        assert len(values) == len(self._params)
        [setattr(self._params[i], 'value', val) for i,val in enumerate(values)]

    @property
    def is_nominal(self):
        return np.all([(v0==v1)
                       for v0, v1 in izip(self.values, self.nominal_values)])

    @property
    def nominal_values(self):
        return [obj.nominal_value for obj in self._params]

    @nominal_values.setter
    def nominal_values(self, values):
        assert len(values) == len(self._params)
        [setattr(self._params[i], 'nominal_value', val)
         for i,val in enumerate(nominal_values)]

    @property
    def priors(self):
        return tuple([obj.prior for obj in self._params])

    @priors.setter
    def priors(self, values):
        assert len(values) == len(self._params)
        [setattr(self._params[i], 'prior', val) for i,val in enumerate(values)]

    @property
    def priors_llh(self):
        return np.sum([obj.prior_llh for obj in self._params])

    @property
    def priors_chi2(self):
        return np.sum([obj.prior_chi2 for obj in self._params])

    @property
    def ranges(self):
        return tuple([obj.range for obj in self._params])

    @ranges.setter
    def ranges(self, values):
        assert len(values) == len(self._params)
        [setattr(self._params[i], 'range', val) for i,val in enumerate(values)]

    @property
    def state(self):
        return tuple([obj.state for obj in self._params])

    @property
    def values_hash(self):
        return hash_obj(normQuant(self.values))

    @property
    def nominal_values_hash(self):
        return hash_obj(normQuant(self.nominal_values))

    @property
    def state_hash(self):
        return hash_obj(normQuant(self.state))


def test_Param():
    from scipy.interpolate import splrep, splev
    from pisa.core.prior import Prior

    uniform = Prior(kind='uniform', llh_offset=1.5)
    gaussian = Prior(kind='gaussian', mean=10*ureg.meter, stddev=1*ureg.meter)
    param_vals = np.linspace(-10, 10, 100) * ureg.meter
    llh_vals = (param_vals.magnitude)**2
    linterp_m = Prior(kind='linterp', param_vals=param_vals, llh_vals=llh_vals)
    linterp_nounits = Prior(kind='linterp', param_vals=param_vals.m,
                            llh_vals=llh_vals)

    param_vals = np.linspace(-10, 10, 100)
    llh_vals = param_vals**2
    knots, coeffs, deg = splrep(x=param_vals, y=llh_vals)

    spline = Prior(kind='spline', knots=knots, coeffs=coeffs, deg=deg)

    # Param with units, prior with compatible units
    p0 = Param(name='c', value=1.5*ureg.foot, prior=gaussian,
               range=[1,2]*ureg.foot, is_fixed=False, is_discrete=False,
               tex=r'\int{\rm c}')
    # Param with no units, prior with no units
    p1 = Param(name='c', value=1.5, prior=spline, range=[1,2],
               is_fixed=False, is_discrete=False, tex=r'\int{\rm c}')

    # Param with no units, prior with units
    try:
        p2 = Param(name='c', value=1.5, prior=linterp_m,
                   range=[1,2], is_fixed=False, is_discrete=False,
                   tex=r'\int{\rm c}')
        p2.prior_llh
        logging.debug(str(p2))
        logging.debug(str(linterp))
        logging.debug('p2.units: %s' %p2.units)
        logging.debug('p2.prior.units: %s' %p2.prior.units)
    except (TypeError, pint.DimensionalityError):
        pass
    else:
        assert False

    # Param with units, prior with no units
    try:
        p2 = Param(name='c', value=1.5*ureg.meter, prior=spline, range=[1,2],
                   is_fixed=False, is_discrete=False, tex=r'\int{\rm c}')
        p2.prior_llh
    except (TypeError, AssertionError):
        pass
    else:
        assert False
    try:
        p2 = Param(name='c', value=1.5*ureg.meter, prior=linterp_nounits,
                   range=[1,2], is_fixed=False, is_discrete=False,
                   tex=r'\int{\rm c}')
        p2.prior_llh
    except (TypeError, AssertionError):
        pass
    else:
        assert False

    # Param, range, prior with no units
    p2 = Param(name='c', value=1.5, prior=linterp_nounits,
               range=[1,2], is_fixed=False, is_discrete=False,
               tex=r'\int{\rm c}')
    p2.prior_llh

    # Param, prior with no units, range with units
    try:
        p2 = Param(name='c', value=1.5, prior=linterp_nounits,
                   range=[1,2]*ureg.m, is_fixed=False, is_discrete=False,
                   tex=r'\int{\rm c}')
        p2.prior_llh
        logging.debug(str(p2))
        logging.debug(str(linterp))
        logging.debug('p2.units: %s' %p2.units)
        logging.debug('p2.prior.units: %s' %p2.prior.units)
    except (TypeError, AssertionError):
        pass
    else:
        assert False

    nom0 = p2.nominal_value
    val0 = p2.value
    p2.value = p2.value * 1.01
    val1 = p2.value
    assert p2.value != val0
    assert p2.value == val0 * 1.01
    assert p2.value != nom0
    assert p2.nominal_value == nom0

    p2.reset()
    assert p2.value == nom0
    assert p2.nominal_value == nom0

    p2.value = val1
    p2.set_nominal_to_current_value()
    assert p2.nominal_value == p2.value
    assert p2.nominal_value == val1, \
            '%s should be %s' %(p2.nominal_value, val1)

    print '<< PASSED : test_Param >>'

# TODO: add tests for reset() and reset_all() methods
def test_ParamSet():
    p0 = Param(name='c', value=1.5, prior=None, range=[1,2],
               is_fixed=False, is_discrete=False, tex=r'\int{\rm c}')
    p1 = Param(name='a', value=2.5, prior=None, range=[1,5],
               is_fixed=False, is_discrete=False, tex=r'{\rm a}')
    p2 = Param(name='b', value=1.5, prior=None, range=[1,2],
               is_fixed=False, is_discrete=False, tex=r'{\rm b}')
    param_set = ParamSet(p0, p1, p2)
    print param_set.values
    print param_set[0]
    param_set[0].value = 1
    print param_set.values

    param_set.values = [3, 2, 1]
    print param_set.values
    print param_set.values[0]
    print param_set[0].value

    print 'priors:', param_set.priors
    print 'names:', param_set.names

    print param_set['a']
    print param_set['a'].value
    try:
        param_set['a'].value = 33
    except:
        pass
    else:
        assert False
    print param_set['a'].value

    print param_set['c'].is_fixed
    param_set['c'].is_fixed = True
    print param_set['c'].is_fixed
    print param_set.are_fixed
    param_set.fix('a')
    print param_set.are_fixed
    param_set.unfix('a')
    print param_set.are_fixed
    param_set.unfix([0,1,2])
    print param_set.are_fixed

    fixed_params = param_set.fixed
    print fixed_params.are_fixed
    free_params = param_set.free
    print free_params.are_fixed
    print param_set.free.values

    print param_set.values_hash
    print param_set.fixed.values_hash
    print param_set.free.values_hash

    print param_set[0].state
    print param_set.state_hash
    print param_set.fixed.state_hash
    print param_set.free.state_hash

    print 'fixed:', param_set.fixed.names
    print 'fixed, discrete:', param_set.fixed.discrete.names
    print 'fixed, continuous:', param_set.fixed.continuous.names
    print 'free:', param_set.free.names
    print 'free, discrete:', param_set.free.discrete.names
    print 'free, continuous:', param_set.free.continuous.names
    print 'continuous, free:', param_set.continuous.free.names
    print 'free, continuous hash:', param_set.free.continuous.values_hash
    print 'continuous, free hash:', param_set.continuous.free.values_hash

    print param_set['b'].prior_llh
    print param_set.priors_llh
    print param_set.free.priors_llh
    print param_set.fixed.priors_llh

    print param_set[0].prior_chi2
    print param_set.priors_chi2
    print '<< PASSED : test_ParamSet >>'


if __name__ == "__main__":
    test_Param()
    test_ParamSet()
