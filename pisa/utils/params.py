#
# params.py
#
# Functions for accessing and formatting parameters taken from .json
# template settings file for analysis.
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# date:   16 October 2014
#

import scipy as sp
import numpy as np
import copy


class Prior(object):
    def __init__(self, **kwargs):
        self.constructor_args = copy.deepcopy(kwargs)
        if not kwargs.has_key('kind'):
            raise TypeError(str(self.__class__) + ' __init__ requires `kind` kwarg to be specified')
        kind = kwargs.pop('kind')
        # Dispatch the correct initialization method
        if kind.lower() in ['none', 'uniform'] or kind is None:
            Prior.__init_uniform(self, **kwargs)
        elif kind.lower() == 'gaussian':
            Prior.__init_gaussian(self, **kwargs)
        elif kind.lower() == 'linterp':
            Prior.__init_linterp(self, **kwargs)
        elif kind.lower() == 'spline':
            Prior.__init_spline(self, **kwargs)
        else:
            raise TypeError('Unknown Prior kind `' + str(kind) + '`')

    @classmethod
    def from_param(cls, param):
        '''Factory method for generating a Prior object from a param dict.'''

        if not isinstance(param, dict):
            raise TypeError("`from_param` factory method can only instantiate a Prior object from a param dict")

        # If param has no 'prior', do not create a Prior object
        # NOTE: This is probably a poor design decision, but maintains a more
        # "sparse" config file; probably should change in future.
        if 'prior' not in param:
            return None

        prior = param['prior']

        # Old-style prior specs that translate to a uniform prior
        if prior is None or (isinstance(prior, str) and prior.lower() == 'none'):
            return cls(kind='uniform')

        # Old-style prior spec that translates to a gaussian prior
        elif isinstance(prior, (int, float)):
            fiducial = param['value']
            sigma = prior
            return cls(kind='gaussian', fiducial=fiducial, sigma=sigma)

        # New-style prior is a dictionary
        elif isinstance(prior, dict):
            return cls(**prior)

        else:
            raise TypeError("Uninterpretable param dict 'prior': " + str(prior))

    def __str__(self):
        return self._str(self)

    def __repr__(self):
        return '<' + str(self.__class__) + ' ' + self.__str__() + '>'

    def build_dict(self, node_dict=None):
        if node_dict is None:
            node_dict = {}
        node_dict['prior'] = self.constructor_args
        return node_dict

    def __init_uniform(self):
        self.kind = 'uniform'
        self.llh = lambda x: 0.*x
        self.valid_range = [-np.inf, np.inf]
        self.max_at = np.nan
        self.max_at_str = "no maximum"
        self._str = lambda s: "uniform prior"

    def __init_gaussian(self, fiducial, sigma):
        self.kind = 'gaussian'
        self.fiducial = fiducial
        self.sigma = sigma
        self.llh = lambda x: -(x-self.fiducial)**2 / (2*self.sigma**2)
        self.valid_range = [-np.inf, np.inf]
        self.max_at = self.fiducial
        self.max_at_str = format(self.max_at, '6.4f')
        self._str = lambda s: "gaussian prior: sigma=%6.4f, max at %6.4f" % (self.sigma, self.fiducial)

    def __init_linterp(self, x, y):
        self.kind = 'linterp'
        self.x = np.array(x)
        self.y = np.array(y)
        self.interp = sp.interpolate.interp1d(self.x, self.y, kind='linear',
                                              copy=True, bounds_error=True)
        self.llh = lambda x_new: self.interp(x_new)
        self.valid_range = [min(self.x), max(self.x)]
        self.max_at = self.x[self.y == np.max(self.y)]
        self.max_at_str = ", ".join([format(v, '6.4f') for v in self.max_at])
        self._str = lambda s: "linearly-interpolated prior: valid in [%6.4f, %6.4f], max at %s" % (self.valid_range[0], self.valid_range[1], self.max_at_str)

    def __init_spline(self, knots, coeffs, deg):
        self.kind = 'spline'
        self.knots = knots
        self.coeffs = coeffs
        self.deg = deg
        self.llh = lambda x: sp.interpolate.splev(x, tck=(knots, coeffs, deg), ext=2)
        self.valid_range = [np.min(knots), np.max(knots)]
        self.max_at = sp.optimize.fminbound(
            func=lambda x,a: -sp.interpolate.splev(x,a),
            x1=self.valid_range[0],
            x2=self.valid_range[1],
            args=((self.knots, self.coeffs, self.deg),)
        )
        self.max_at_str = format(self.max_at, '6.4f')
        self._str = lambda s: "spline prior: deg %d, valid in [%6.4f, %6.4f], max at %s" % (self.deg, self.valid_range[0], self.valid_range[1], self.max_at_str)

    def check_range(self, x_range):
        return min(x_range) >= self.valid_range[0] and max(x_range) <= self.valid_range[1]


def get_values(params):
    """
    Takes the params dict which is of the form:
      {'param1': {
        'value': val,
        'fixed': bool
       },
       'param2': {...},...
      }
    and returns a dictionary of names/values as:
      {'param1': val,
       'param2': val,...
      }
    """
    return { key: param['value'] for key, param in sorted(params.items()) }


def select_hierarchy(params, normal_hierarchy):
    """
    Corrects for a key in params being defined for both 'nh' and 'ih',
    and returns a modified dict with one value for the given hierarchy.
    """

    if not isinstance(normal_hierarchy, bool):
        raise ValueError('Hierarchy selection must be boolean value')

    newparams = {}
    for key, value in params.items():
        if key.endswith('_nh'):
            #Don't use if doesn't match request
            if not normal_hierarchy: continue
            #Use simplified key
            key = key.rsplit('_',1)[0]
        if key.endswith('_ih'):
            if normal_hierarchy: continue
            key = key.rsplit('_',1)[0]

        newparams[key] = value

    return newparams


def get_fixed_params(params):
    """
    Finds all fixed parameters in params dict and returns them in a
    new dictionary.
    """

    return { key: value for key, value in params.items() if value['fixed']}


def get_free_params(params):
    """
    Finds all free parameters in params dict and returns them in a new
    dictionary.
    """

    return { key: value for key, value in params.items() if not value['fixed']}


def get_param_values(params):
    """
    Returns a list of parameter values
    """
    return [ val['value'] for key,val in sorted(params.items()) ]


def get_param_scales(params):
    """
    Returns a list of parameter scales
    """
    return [ val['scale'] for key,val in sorted(params.items()) ]


def get_param_bounds(params):
    """
    Returns a list of parameter bounds where elements are (min,max) pairs
    """
    return [ val['range'] for key,val in sorted(params.items()) ]


def get_param_priors(params):
    """
    Returns a list of Prior objects, one for each param.
    """
    priors = []
    for pname,param in sorted(params.items()):
        try:
            prior = Prior(**param['prior'])
        except TypeError:
            logging.error("  Check template settings format, "
                          "may have old-style priors")
            raise
        priors.append(prior)
    return priors


def get_atm_params(params):
    """
    Returns dictionary of just the atmospheric parameters
    """
    atm_params = ['deltam31','theta23']
    return { key: value for key, value in params.items()
             for p in atm_params if p in key }


def fix_osc_params(params):
    """
    Returns dict identical to params dict but with all oscillation
    parameters set to "fixed"=True
    """
    new_params = {}
    # or initialize with new copy by dict(params)
    osc_params = ['deltam31', 'deltam21', 'theta23', 'theta13', 'theta12',
                  'deltacp']
    for key,value in params.items():
        new_params[key] = value.copy()
        for okey in osc_params:
            if okey in key:
                new_params[key]['fixed'] = True
    return new_params


def fix_atm_params(params):
    """
    Returns dict identical to params dict but with all atmospheric
    oscillation parameters fixed.
    """
    new_params = {}
    # or initialize with new copy by dict(params)
    atm_params = ['deltam31', 'theta23']
    for key,value in params.items():
        new_params[key] = value.copy()
        for akey in atm_params:
            if akey in key:
                new_params[key]['fixed'] = True
    return new_params
