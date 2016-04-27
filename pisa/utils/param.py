#
# author: J.L. Lanfranchi
#         jll1062+pisa@phys.psu.edu
#
# date:   March 22, 2016
#


from functools import total_ordering
from collections import OrderedDict, Sequence

import numpy as np

from pisa.utils.log import logging
import pisa.resources.resources as resources
from pisa.utils import utils


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
    prior_llh : <r>
    prior_chi2 : <r>

    Methods
    -------
    validate_value
    """
    __slots = ('name', 'value', 'prior', 'range', 'is_fixed', 'is_discrete',
               'scale', '_nominal_value', '_tex', 'help')
    __state_attrs = ('name', 'value', 'prior', 'range', 'is_fixed',
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
        return utils.recursiveEquality(self.state, other.state)

    def __lt__(self, other):
        return self.name < other.name

    def __setattr__(self, attr, val):
        if attr not in self.__slots:
            raise AttributeError('Invalid attribute: %s' % (attr,))
        object.__setattr__(self, attr, val)

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
    def tex(self):
        return '%s=%s' % (self._tex, self.value)

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
        [state.__setitem__(a, self.__getattribute__(a))
         for a in self.__state_attrs]
        return state

    @property
    def prior_llh(self):
        return -100.34

    @property
    def prior_chi2(self):
        return 0.297

    @property
    def state_hash(self):
        return utils.hash_obj(self.state)


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
    fix(vals)
        Set param found at each `index(val)` to be fixed.
    index(val)
        Locate and return index given `val` which can be an int (index), str
        (name), or Param object (an actual item in the set).
    replace(old, new)
        Replace `old` with `new`. Returns `old`.
    unfix(vals)
        Set param at each `index(val)` to be free.
    update(obj)
        Update this param set using obj (if a Param, ParamSet, or sequence
        thereof)
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
        assert len(set(param_sequence)) == len(param_sequence), \
                'Duplicated params detected'
        # Elements of list must be Param type
        assert all([isinstance(x, Param) for x in param_sequence]), \
                'All params must be of type "Param"'
        self._params = param_sequence
        self._by_name = {param.name: param for param in self._params}

    def index(self, value):
        idx = -1
        if isinstance(value, int):
            idx = value
        elif value in self.names:
            idx = self.names.index(value)
        elif value in self._params:
            idx = self._params.index(value)
        if idx < 0 or idx >= len(self):
            raise ValueError('%s not found in ParamSet' % (value,))
        return idx

    def replace(self, old, new):
        idx = self.index(old)
        old = self._params[idx]
        self._params[idx] = new
        return old

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

    def update(self, obj):
        """Update this param set using `obj`.

        Parameters
        ----------
        obj : Param, ParamSet, or sequence thereof

        """
        if isinstance(obj, Sequence):
            [self.update(p) for p in obj]
            return
        if not isinstance(obj, Param):
            raise ValueError('`obj`="%s" not a Param' % (obj,))
        param = obj
        try:
            existing_idx = self.index(param.name)
            self.replace(existing_idx, param)
            return
        except ValueError:
            pass
        self._params.append(param)

    def __len__(self):
        return len(self._params)

    def __setitem__(self, i, val):
        self._params[i].value = val

    def __getitem__(self, i):
        if isinstance(i, int):
            return self._params[i]
        elif isinstance(i, basestring):
            return self._by_name[i]

    def __iter__(self):
        iter(self._params)

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
        [self._params[i].__setattr__('value', val)
         for i,val in enumerate(values)]

    @property
    def nominal_values(self):
        return tuple([obj.nominal_value for obj in self._params])

    @nominal_values.setter
    def nominal_values(self, values):
        assert len(values) == len(self._params)
        [self._params[i].__setattr__('nominal_value', val)
         for i,val in enumerate(nominal_values)]

    @property
    def priors(self):
        return tuple([obj.prior for obj in self._params])

    @priors.setter
    def priors(self, values):
        assert len(values) == len(self._params)
        [self._params[i].__setattr__('prior', val)
         for i,val in enumerate(values)]

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
        [self._params[i].__setattr__('range', val)
         for i,val in enumerate(values)]

    @property
    def scales(self):
        return tuple([obj.scales for obj in self._params])

    @scales.setter
    def scales(self, values):
        assert len(values) == len(self._params)
        [self._params[i].__setattr__('scale', val)
         for i,val in enumerate(values)]

    @property
    def state(self):
        return tuple([obj.state for obj in self._params])

    @property
    def values_hash(self):
        return utils.hash_obj(self.values)

    @property
    def state_hash(self):
        return utils.hash_obj(self.state)


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
