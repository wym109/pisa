# author: P.Eller
#         pde3+pisa@psu.edu
#
# date:   2016-04-28
"""
Parse a ConfigFile object into a dict containing a an entry for every stage,
that contains values indicated by param. each as a Param object in one
ParamSet, and the binning as binning objects.

Params that have the same name in multiple stages of the pipeline are
instantiated as references to a single param in memory, so updating one updates
all of them.

Note that this mechanism of synchronizing parameters holds only within the
scope of a single pipeline; synchronization of parameters across pipelines
is done by adding the pipelines to a single DistributionMaker object and
updating params through the DistributionMaker's update_params method.
"""
# TODO: add try: except: blocks around class instantiation calls to give
# maximally useful error info to the user (spit out a good message, but then
# re-raise the exception)

from collections import OrderedDict

import numpy as np
import uncertainties
from uncertainties import ufloat, ufloat_fromstr
from uncertainties import unumpy as unp

from pisa import ureg, Q_
from pisa.core.binning import MultiDimBinning, OneDimBinning
from pisa.core.param import Param, ParamSet
from pisa.core.prior import Prior
from pisa.utils.fileio import from_file
from pisa.utils.log import logging

# Config files use "uinits.xyz" to denote that "xyz" is a unit; therefore,
# ureg is also referred to as "units" in this context.
units = ureg


def parse_quantity(string):
    value = string.replace(' ', '')
    if 'units.' in value:
        value, unit = value.split('units.')
    else:
        unit = None
    value = value.rstrip('*')
    if '+/-' in value:
        value = ufloat_fromstr(value)
    else:
        value = ufloat(float(value), 0)
    value *= ureg(unit)
    return value


def parse_string_literal(string):
    if string.lower().strip() == 'true': return True
    elif string.lower().strip() == 'false': return False
    elif string.lower().strip() == 'none': return None
    return string


def list_split(string):
    list = string.split(',')
    return [x.strip() for x in list]


def parse_config(config):
    if isinstance(config, basestring):
        config = from_file(config)
    # create binning objects
    binning_dict = {}
    for name, value in config.items('binning'):
        if name.endswith('.order'):
            order = list_split(config.get('binning', name))
            binning, _ = name.split('.')
            bins = []
            for bin_name in order:
                kwargs = eval(config.get('binning', binning + '.' + bin_name))
                bins.append(OneDimBinning(bin_name, **kwargs))
            binning_dict[binning] = MultiDimBinning(bins)

    stage_dicts = OrderedDict()
    # find pipline setting
    pipeline_order = list_split(config.get('pipeline', 'order'))
    for item in pipeline_order:
        stage, service = item.split(':')
        section = 'stage:' + stage
        # get infos for stages
        stage_dicts[stage] = {}
        stage_dicts[stage]['service'] = service
        params = []
        if config.has_option(section, 'param_selector'):
            param_selector = config.get(section, 'param_selector')
        else:
            param_selector = ''

        for name, value in config.items(section):
            if name.startswith('param.'):
                # find parameter root
                if name.startswith('param.'+ param_selector + '.') and \
                        name.count('.') == 2:
                    _, _, pname = name.split('.')
                elif name.startswith('param.') and name.count('.') == 1:
                    _, pname = name.split('.')
                else:
                    continue

                # check if that param already exists from a stage before, and if
                # yes, make sure there are no specs, and just link to previous
                # param object that already is instantiated
                for _,stage_dict in stage_dicts.items():
                    if stage_dict.has_key('params') and \
                            pname in stage_dict['params'].names:
                        # make sure there are no other specs
                        assert config.has_option(section, name + '.range') == \
                                False
                        assert config.has_option(section, name + '.prior') == \
                                False
                        assert config.has_option(section, name + '.fixed') == \
                                False
                        params.append(stage_dict['params'][pname])
                        break
                else:

                    # defaults
                    kwargs = {'name': pname, 'is_fixed': True, 'prior': None,
                              'range': None}
                    try:
                        value = parse_quantity(value)
                        kwargs['value'] = value.n * value.units
                    except ValueError:
                        value = parse_string_literal(value)
                        kwargs['value'] = value
                    # search for explicit specifications
                    if config.has_option(section, name + '.fixed'):
                        kwargs['is_fixed'] = config.getboolean(section,
                                                               name + '.fixed')

                    if config.has_option(section, name + '.prior'):
                        if config.get(section, name + '.prior') == 'uniform':
                            kwargs['prior'] = Prior(kind='uniform')
                        elif config.get(section, name + '.prior') == 'spline':
                            priorname = pname
                            if param_selector:
                                priorname += '_' + param_selector
                            data = config.get(section, name + '.prior.data')
                            data = from_file(data)
                            data = data[priorname]
                            knots = ureg.Quantity(np.asarray(data['knots']),
                                                  data['units'])
                            knots = knots.to(value.units)
                            coeffs = np.asarray(data['coeffs'])
                            deg = data['deg']
                            kwargs['prior'] = Prior(kind='spline',
                                                    knots=knots,
                                                    coeffs=coeffs,
                                                    deg=deg)
                        elif 'gauss' in config.get(section, name + '.prior'):
                            raise Exception(
                                'Please use new style +/- notation for'
                                ' gaussian priors in config'
                            )
                        else:
                            raise Exception('Prior type unknown')
                    elif hasattr(value, 's') and value.s != 0:
                        kwargs['prior'] = Prior(kind='gaussian',
                                                mean=value.n * value.units,
                                                stddev=value.s * value.units)

                    if config.has_option(section, name + '.range'):
                        range = config.get(section, name + '.range')
                        if 'nominal' in range:
                            nominal = value.n * value.units
                        if 'sigma' in range:
                            sigma = value.s * value.units
                        range = range.replace('[', 'np.array([')
                        range = range.replace(']', '])')
                        kwargs['range'] = eval(range).to(value.units)
                    try:
                        params.append(Param(**kwargs))
                    except:
                        logging.error('Failed to instantiate new Param object'
                                      ' with kwargs %s' %kwargs)
                        raise

            elif 'binning' in name:
                stage_dicts[stage][name] = binning_dict[value]

            elif not name == 'param_selector':
                stage_dicts[stage][name] = parse_string_literal(value)

        if len(params) > 0:
            stage_dicts[stage]['params'] = ParamSet(*params)

    return stage_dicts
