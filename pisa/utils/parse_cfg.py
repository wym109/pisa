#
# fileio.py
#
# A set of utility function for generic file IO
#
# author: Justin Lanfranchi
#         jll1062@phys.psu.edu
#
# date:   2015-06-13
'''  Parse a ConfigFile object into a dict, that contains values indicated by p. or param. as a param set, and all other values a s ordinary strings '''

from pisa.utils.prior import Prior
from pisa.utils.param import Param, ParamSet
from pisa.utils.log import logging
import ConfigParser
import uncertainties
from uncertainties import unumpy as unp
from uncertainties import ufloat, ufloat_fromstr
import numpy as np
import pint
units = pint.UnitRegistry()

def parse(string):
    value = string.replace(' ','')
    if 'units.' in value:
        value, unit = value.split('units.')
    else:
        unit = None
    value = value.rstrip('*')
    if '+/-' in value:
        value = ufloat_fromstr(value)
    else:
        value = ufloat(float(value),0)
    value *= units(unit)
    return value 


def parse_cfg(config):
    dict = {}
    for section in config.sections():
        dict[section] = {}
        params = []
        for name, value in config.items(section):
            if name.startswith('p.') or name.startswith('param.'):
                if name.count('.') > 1: continue
                # make param object
                _, pname = name.split('.')
                value = parse(value)
                is_fixed = True
                is_descrete = False
                prior = None
                range = None
                if config.has_option(section, name + '.fixed'):
                    is_fixed = config.getboolean(section, name + '.fixed')
                if type(value.magnitude) is uncertainties.core.AffineScalarFunc:
                    prior = Prior(kind='gaussian',fiducial=value.n, sigma = value.s)
                if config.has_option(section, name + '.prior'):
                    #ToDo
                    prior = config.get(section, name + '.prior')
                if config.has_option(section, name + '.range'):
                    range = config.get(section, name + '.range')
                    if 'nominal' in range:
                        if type(value.magnitude) is uncertainties.core.AffineScalarFunc:
                            nominal = value.n * value.units
                        else: nominal = value
                    if 'sigma' in range:
                        sigma = value.s * value.units
                    range = range.replace('[','np.array([')
                    range = range.replace(']','])')
                    range = eval(range)
                params.append(Param(name=pname, value=value.n * value.units, prior=prior, range=range, is_fixed=is_fixed))
            else:
                dict[section][name] = value
        dict[section]['params'] = ParamSet(*params)
    return dict
