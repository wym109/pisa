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
from pisa.utils.binning import OneDimBinning, MultiDimBinning

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

def list_split(string):
    list = string.split(',')
    return [x.strip() for x in list]

def parse_cfg(config):
    dict = {}
    # create binning objects
    binnings = {}
    order = list_split(config.get('binning','order'))
    binnings = list_split(config.get('binning','binnings'))
    for binning in binnings:
        bins = []
        for bin_name in order:
            args = eval(config.get(section, binning + '.' + bin_name))
            bins.append(OneDimBinning(bin_name, **args))
        binnings[binning] = MultiDimBinning(*bins)

    for section in config.sections():
        if section == 'bining': continue
        dict[section] = {}
        params = []
        if section.startswith('stage:')
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
                    if value.s != 0:
                        prior = Prior(kind='gaussian',fiducial=value.n, sigma = value.s)
                    if config.has_option(section, name + '.prior'):
                        #ToDo
                        prior = config.get(section, name + '.prior')
                    if config.has_option(section, name + '.range'):
                        range = config.get(section, name + '.range')
                        if 'nominal' in range:
                            nominal = value.n * value.units
                        if 'sigma' in range:
                            sigma = value.s * value.units
                        range = range.replace('[','np.array([')
                        range = range.replace(']','])')
                        range = eval(range)
                    params.append(Param(name=pname, value=value.n * value.units, prior=prior, range=range, is_fixed=is_fixed))
                elif 'binning' in name:
                    dict[section][name] = binning[value]
                else:
                    dict[section][name] = value
            if len(params) > 0:
                dict[section]['params'] = ParamSet(*params)
    return dict
