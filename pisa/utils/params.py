#
# author: J.L. Lanfranchi
#         jll1062+pisa@phys.psu.edu
#
# date:   March 22, 2016
#


from functools import total_ordering
from collections import OrderedDict

import numpy as np

from pisa.utils.log import logging
import pisa.resources.resources as resources


@total_ordering
class Param(object):
    __slots = ('name', 'value', 'prior', 'range', 'scale', 'is_fixed',
               'is_discrete')

    def __init__(self, name):
        self.name = name
        self.value = 0
        self.prior = None
        self.range = (-1, 1)
        self.scale = 1
        self.is_fixed = False
        self.is_discrete = False

    def __eq__(self, other):
        return self.state == other.state

    def __lt__(self, other):
        return self.name < other.name

    def __setattr__(self, attr, val):
        if attr not in self.__slots:
            raise AttributeError('Invalid attribute: %s' % (attr,))
        object.__setattr__(self, attr, val)

    @property
    def state(self):
        state = OrderedDict()
        [state.__setitem__(k, self.__getattr__(k)) for k in self.__slots]
        return state

    @property
    def prior_llh(self):
        return -100.34

    @property
    def prior_chisquare(self):
        return 0.297

    @property
    def state_hash(self):
        return hash_obj(self.state)


class ParamSet(object):
    def __init__(self, object_sequence):
        assert sorted(set(object_sequence)) == sorted(object_sequence)
        self._objs = tuple(sorted(object_sequence))
        self._by_name = {obj.name: obj for obj in self._objs}

    def index(self, value):
        idx = -1
        if isinstance(value, int):
            idx = value
        elif value in self.names:
            idx = self.names.index(value)
        elif value in self._objs:
            idx = self._objs.index(value)
        if idx < 0 or idx >= len(self):
            raise ValueError('%s is not in ParamSet' % (value,))
        return idx

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

    def __len__(self):
        return len(self._objs)

    def __setitem__(self, i, val):
        self._objs[i].value = val

    def __getitem__(self, i):
        if isinstance(i, int):
            return self._objs[i]
        elif isinstance(i, basestring):
            return self._by_name[i]

    @property
    def fixed(self):
        return ParamSet([obj for obj in self._objs if obj.is_fixed])

    @property
    def free(self):
        return ParamSet([obj for obj in self._objs if not obj.is_fixed])

    @property
    def continuous(self):
        return ParamSet([obj for obj in self._objs if not obj.is_discrete])

    @property
    def discrete(self):
        return ParamSet([obj for obj in self._objs if obj.is_discrete])

    @property
    def are_fixed(self):
        return tuple([obj.is_fixed for obj in self._objs])

    @property
    def are_discrete(self):
        return tuple([obj.is_discrete for obj in self._objs])

    @property
    def names(self):
        return tuple([obj.name for obj in self._objs])

    @property
    def values(self):
        return tuple([obj.value for obj in self._objs])

    @values.setter
    def values(self, values):
        assert len(values) == len(self._objs)
        [self._objs[i].__setattr__('value', val)
         for i,val in enumerate(values)]

    @property
    def priors(self):
        return tuple([obj.prior for obj in self._objs])

    @priors.setter
    def priors(self, values):
        assert len(values) == len(self._objs)
        [self._objs[i].__setattr__('prior', val)
         for i,val in enumerate(values)]

    @property
    def priors_llh(self):
        return np.sum([obj.prior_llh for obj in self._objs])

    @property
    def priors_chisquare(self):
        return np.sum([obj.prior_chisquare for obj in self._objs])

    @property
    def ranges(self):
        return tuple([obj.range for obj in self._objs])

    @ranges.setter
    def ranges(self, values):
        assert len(values) == len(self._objs)
        [self._objs[i].__setattr__('range', val)
         for i,val in enumerate(values)]

    @property
    def scales(self):
        return tuple([obj.scales for obj in self._objs])

    @scales.setter
    def scales(self, values):
        assert len(values) == len(self._objs)
        [self._objs[i].__setattr__('scale', val)
         for i,val in enumerate(values)]

    @property
    def state(self):
        return tuple([obj.state for obj in self._objs])

    @property
    def values_hash(self):
        return hash_obj(self.values)

    @property
    def state_hash(self):
        return hash_obj(self.state)


#def select_hierarchy(params, normal_hierarchy):
#    """Correct for a key in params being defined for both 'nh' and 'ih', and
#    return a modified dict with one value for the given hierarchy.
#    """
#    if not isinstance(normal_hierarchy, bool):
#        raise ValueError('Hierarchy selection must be boolean value')
#
#    newparams = {}
#    for key, value in params.items():
#        if key.endswith('_nh'):
#            #Don't use if doesn't match request
#            if not normal_hierarchy: continue
#            #Use simplified key
#            key = key.rsplit('_',1)[0]
#        if key.endswith('_ih'):
#            if normal_hierarchy: continue
#            key = key.rsplit('_',1)[0]
#
#        newparams[key] = value
#
#    return deepcopy(newparams)
#
#
#def get_atm_params(params):
#    """Return dictionary of just the atmospheric parameters"""
#    atm_params = ['deltam31','theta23']
#    return deepcopy({ key: value for key, value in params.items()
#             for p in atm_params if p in key })
#
#
#def fix_osc_params(params):
#    """Return dict identical to params dict but with all oscillation parameters
#    set to "fixed"=True
#    """
#    new_params = {}
#    # or initialize with new copy by dict(params)
#    osc_params = ['deltam31', 'deltam21', 'theta23', 'theta13', 'theta12',
#                  'deltacp']
#    for key,value in params.items():
#        new_params[key] = deepcopy(value)
#        for okey in osc_params:
#            if okey in key:
#                new_params[key]['fixed'] = True
#
#    return new_params
#
#
#def fix_atm_params(params):
#    """
#    Returns dict identical to params dict but with all atmospheric
#    oscillation parameters fixed.
#    """
#    new_params = {}
#    # or initialize with new copy by dict(params)
#    atm_params = ['deltam31', 'theta23']
#    for key,value in params.items():
#        new_params[key] = deepcopy(value)
#        #for akey in atm_params:
#        if (bool(re.match('^theta23', key))
#                or bool(re.match('^deltam31', key))):
#            new_params[key]['fixed'] = True
#
#    return new_params
#
#
#def fix_non_atm_params(params):
#    """Return dict identical to params dict, except that it fixes all
#    parameters besides that atmospheric mixing params: theta23/deltam31. Does
#    not modify atm mix params, leaving them fixed or not, as they were..
#    """
#    new_params = {}
#    # or initialize with new copy by dict(params)
#    for key,value in params.items():
#        new_params[key] = deepcopy(value)
#        if (bool(re.match('^theta23', key)) \
#                or bool(re.match('^deltam31', key))):
#            continue
#        else:
#            new_params[key]['fixed'] = True
#
#    return new_params


def test_params():
    c = ParamSet([Param('first'), Param('second'), Param('third')])
    print c.values
    print c[0]
    c[0].value = 1
    print c.values

    c.values = [3, 2, 1]
    print c.values
    print c.values[0]
    print c[0].value

    print 'priors:', c.priors
    print 'names:', c.names

    print c['first']
    print c['first'].value
    c['first'].value = 33
    print c['first'].value

    print c['third'].is_fixed
    c['third'].is_fixed = True
    print c['third'].is_fixed
    print c.are_fixed
    c.fix('first')
    print c.are_fixed
    c.unfix('first')
    print c.are_fixed
    c.unfix([0,1,2])
    print c.are_fixed

    fixed_params = c.fixed
    print fixed_params.are_fixed
    free_params = c.free
    print free_params.are_fixed
    print c.free.values

    print c.values_hash
    print c.fixed.values_hash
    print c.free.values_hash

    print c.state_hash
    print c.fixed.state_hash
    print c.free.state_hash

    print 'fixed:', c.fixed.names
    print 'fixed, discrete:', c.fixed.discrete.names
    print 'fixed, continuous:', c.fixed.continuous.names
    print 'free:', c.free.names
    print 'free, discrete:', c.free.discrete.names
    print 'free, continuous:', c.free.continuous.names
    print 'continuous, free:', c.continuous.free.names
    print 'free, continuous hash:', c.free.continuous.values_hash
    print 'continuous, free hash:', c.continuous.free.values_hash

    print c['second'].prior_llh
    print c.priors_llh

    print c[0].prior_chisquare
    print c.priors_chisquare
