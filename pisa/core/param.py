#
# author: J.L. Lanfranchi
#         jll1062+pisa@phys.psu.edu
#
# date:   March 22, 2016
#


from functools import total_ordering
from collections import OrderedDict, Sequence, Mapping
from operator import setitem

import numpy as np

from pisa.core.prior import Prior
from pisa.utils.log import logging
import pisa.resources.resources as resources
from pisa.utils.comparisons import recursiveEquality
from pisa.utils.hash import hash_obj

# TODO: eliminate "scale" parameter, as this should be dynamically computed
# from the range for the sake of the minimizer (ranges published to minimizer
# should be in hypercube in [0,1]). Possibly introduce a "display_scale"
# parameter instead if this is desired for plots and whatnot.

# TODO: units: pass in attached to values, or
@total_ordering
class Param(object):
    """Parameter.

    Parameters
    ----------
    name
    value
    prior
    range
    is_fixed
    is_discrete
    scale
    tex
    help

    Properties
    ----------
    tex : <r>
    nominal_value : <r/w>
    state : <r>
    prior_penalty : <r>

    Methods
    -------
    validate_value
    """
    _slots = ('name', 'value', 'prior', 'range', 'is_fixed', 'is_discrete',
              'scale', '_nominal_value', '_tex', 'help', '_prior')
    _state_attrs = ('name', 'value', 'prior', 'range', 'is_fixed',
                     'is_discrete', 'scale', 'nominal_value', 'tex', 'help')

    def __init__(self, name, value, prior, range, is_fixed, is_discrete=False,
                 scale=1, nominal_value=None, tex=None, help=''):
        self.name = name
        self._tex = tex if tex is not None else name
        self.help = help
        self.range = range
        self.is_fixed = is_fixed
        self.is_discrete = is_discrete
        self.validate_value(value)
        self.value = value
        self.prior = prior
        self.scale = scale
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

    #def __getattr__(self, attr):
    #    return super(Param, self).__getattribute__(attr)

    def __str__(self):
        return '%s=%s; prior=%s, range=%s, scale=%s, is_fixed=%s,' \
                ' is_discrete=%s; help="%s"' \
                % (self.name, self.value, self.prior, self.range, self.scale,
                   self.is_fixed, self.is_discrete, self.help)

    def validate_value(self, value):
        if self.range is not None:
            if self.is_discrete:
                assert value in self.range
            else:
                assert value >= min(self.range) and value <= max(self.range)

    @property
    def prior(self):
        return self._prior

    @prior.setter
    def prior(self, prior_spec):
        if prior_spec is None or (isinstance(prior_spec, basestring) and \
                prior_spec.lower() in ['', 'none', 'uniform']):
            self._prior = Prior(kind=None)
        elif isinstance(prior_spec, Mapping):
            self._prior = Prior(**prior_spec)
        elif isinstance(prior_spec, Prior):
            self._prior = prior_spec
        else:
            raise ValueError('Unhandled `prior_spec` type "%s"'
                             %type(prior_spec))

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

    def prior_penalty(self, metric):
        metric = metric.lower()
        if metric in ['llh', 'barlow_llh', 'conv_llh']:
            return self.prior.llh(self.value)
        elif metric in ['chi2']:
            return self.prior.chi2(self.value)
        else:
            raise ValueError('Unrecognized `metric` "%s"' %metric)

    @property
    def prior_llh(self):
        return self.prior_penalty(metric='llh')

    @property
    def prior_chi2(self):
        return self.prior_penalty(metric='chi2')

    @property
    def state_hash(self):
        return hash_obj(self.state)


class ParamSet(object):
    """Container class for a set of parameters. Most methods are passed through
    to contained params.

    Parameters
    ----------
    *args : one or more Param objects or sequences thereof

    Properties
    ----------
    Note that all sequences returned are returned in the order the parameters
    are stored internally. Properties that are readable are indicated by 'r'
    and properties that are set-able are indicated by 'w'.

    Following are simple properties that describe and/or set corresponding
    properties for the contained params:

    names : <r>
        Names of all parameters
    values : <r/w>
        Get or set values of all parameters
    nominal_values : <r/w>
        Get or set "nominal" values for all parameters. These are variously
        referred to as "injected" or "asimov"; a call to `reset()` method sets
        all `values` to these `nominal_values`.
    priors : <r/w>
        Get or set priors for all parameters.
    ranges : <r/w>
        Get or set the ranges for all parameters.
    scales : <r/w>
        Get or set the scale factors used to normalize parameter values for
        minimizers, for all parameters
    tex : <r>
        LaTeX representation of the contained parameter names & values
    are_fixed : <r>
        Returns tuple of bool's; corresponding params are fixed
    are_discrete : <r>
        Tuple of bool's; corresponding params are discrete
    state : <r>
        Tuple containing the `state` OrderedDict of each contained param
    fixed : <r>
        Returns another ParamSet but with only the fixed params
    free : <r>
        Returns another ParamSet but with only the free params
    continuous : <r>
        Returns another ParamSet but with only the continuous params
    discrete : <r>
        Returns another ParamSet but with only the discrete params

    Following are properties that require computation when requested

    priors_llh : <r>
        Aggregate log-likelihood for parameter values given their priors
    priors_chi2 : <r>
        Aggregate chi-squred for all parameter values given their priors
    values_hash : <r>
        Hash on the values of all of the params; e.g.:
            param_set.values_hash
        but to just hash on free params' values:
            params_set.free_params.values_hash
    state_hash : <r>
        Hash on the complete state of the contained params, which includes
        properties such as `name`, `prior`, `tex`, and *all* other param
        properties.

    Methods
    -------
    extend(obj)
        Call `update` with existing_must_match=True and extend=True
    fix(vals)
        Set param found at each `index(val)` to be fixed.
    index(val)
        Locate and return index given `val` which can be an int (index), str
        (name), or Param object (an actual item in the set).
    replace(new)
        Replace param (by name)
    unfix(vals)
        Set param at each `index(val)` to be free.
    update(obj, existing_must_match=False, extend=True)
        Update this param set using obj (a Param, ParamSet, or sequence
        thereof), optionally enforcing existing param values to match
        those in both `obj` and self, and optionally extending the
        current param set with any new params in `obj`
    update_existing(obj)
        Call `update` with existing_must_match=False and extend=False
    __getitem__
    __iter__
    __len__
    __setitem__

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
        `obj` that are already in this param set match.

        Convenience method or calling `update` with existing_must_match=True
        and extend=True.

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
        self._params[i].value = val

    def __getitem__(self, i):
        if isinstance(i, int):
            return self._params[i]
        elif isinstance(i, basestring):
            return self._by_name[i]

    def __getattr__(self, attr):
        try:
            return super(ParamSet, self).__getattr__(attr)
        except AttributeError:
            return self[attr]

    def __setattr__(self, attr, val):
        try:
            super(ParamSet, self).__setattr__(attr, val)
        except AttributeError:
            self._params[attr].value = val

    def __iter__(self):
        return iter(self._params)

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
    def nominal_values(self):
        return tuple([obj.nominal_value for obj in self._params])

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
    def scales(self):
        return tuple([obj.scales for obj in self._params])

    @scales.setter
    def scales(self, values):
        assert len(values) == len(self._params)
        [setattr(self._params[i], 'scale', val) for i,val in enumerate(values)]

    @property
    def state(self):
        return tuple([obj.state for obj in self._params])

    @property
    def values_hash(self):
        return hash_obj(self.values)

    @property
    def state_hash(self):
        return hash_obj(self.state)


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
    param_set['a'].value = 33
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

    print param_set[0].prior_chi2
    print param_set.priors_chi2


if __name__ == "__main__":
    test_ParamSet()
