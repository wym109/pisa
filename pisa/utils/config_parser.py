# author: P.Eller
#         pde3+pisa@psu.edu
#
# date:   2016-04-28
"""
Parse a ConfigFile object into a dict containing an item for every analysis stage,
that itself contains all necessary instantiation arguments/objects for that
stage.
for en example config file, please consider
`$PISA/pisa.utils/settings/pipeline_settings/pipeline_settings_example.ini`

Config File Structure:
===============

The config file is expected to contain the following sections::

    [pipeline]
    order = stageA:serviceA, stageB:serviceB

    [binning]
    binning1.order = axis1, axis2
    binning1.axis1 = {'num_bins':40, 'is_log':True, 'domain':[1,80] * units.GeV, 'tex': r'$A_1$'}
    binning1.axis2 = {'num_bins':10, 'is_lin':True, 'domain':[1,5], 'tex': r'$A_2$'}

    [stage:stageA]
    input_binning = bining1
    output_binning = binning1
    error_method = None
    debug_mode = False

    param.p1 = 0.0 +/- 0.5 * units.deg
    param.p1.fixed = False
    param.p1.range = nominal + [-2.0, +2.0] * sigma

    [stage:stageB]
    ...

* `pipeline` is the top most section that defines the hierarchy of stages and what
    services to be instatiated.

* `binning` can contain different binning definitions, that are then later
    referred to from within the stage sections.

* `stage` one such section per stage:service is necessary. It cotains some options
    that are common for all stages (`binning`, `error_method` and `debug_mode`) as
    well as all the necessary arguments and parameters for a given stage.


Param definitions:
------------------

Every key in a stage section that starts with `param.name` is interpreted and
parsed into a PISA param object. These can be strings (e.g. a filename - don't
use any quotation marks) or
quantities. The later case expects an expression that can be converted by the
`parse_quantity` function. The `+/-` notation will be interpreted as a gaussian
prior for the quantity. Units can be added by `* unit.soandso`.

Additional arguments to a parameter are passed in with the `.` notation, for
example `param.name.fixed = False`, which makes it a free parameter in the fit (by
default a parameter is fixed unless specified like this).

A range must be given for a free parameter. Either as absolute range `[x,y]` or
in conjuction with the keywords `nominal` (= nominal parameter value) and
`sigma` if the param was specified with the `+/-` notation.

`.prior` is another argument, that can take the values `uniform` or `spline`,
for the latter case a `.prior.data` will be expected, pointing to the spline
data file.

N.B.:
+++++
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
import ConfigParser
import re

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


# TODO: document code, add comments, docstrings, abide by PISA coding
# conventions
class BetterConfigParser(ConfigParser.SafeConfigParser):
    def get(self, section, option):
        result = ConfigParser.SafeConfigParser.get(self, section, option, raw=True)
        result = self.__replaceSectionwideTemplates(result)
        return result

    def items(self, section):
        list = ConfigParser.SafeConfigParser.items(self, section=section, raw=True)
        result = [(key, self.__replaceSectionwideTemplates(value)) for key,value
        in list]
        return result

    def optionxform(self, optionstr):
        """Enable case sensitive options in .ini files."""
        return optionstr

    def __replaceSectionwideTemplates(self, data):
        """Replace <section|option> with get(section,option) recursivly."""
        result = data
        findExpression = re.compile(r"((.*)\<!(.*)\|(.*)\!>(.*))*")
        groups = findExpression.search(data).groups()
        if not groups == (None, None, None, None, None): # expression not matched
            result = self.__replaceSectionwideTemplates(groups[1])
            result += self.get(groups[2], groups[3])
            result += self.__replaceSectionwideTemplates(groups[4])
        return result


# TODO: document code, add comments, docstrings, abide by PISA coding
# conventions
def parse_quantity(string):
    """Parse a string into a pint/uncertainty quantity.

    Parameters
    ----------
    string : string

    Examples
    --------
    >>> print parse_quantity('1.2 +/- 0.7 * units.meter')
    <?>

    """
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
    """Evaluate a string with special values, or return the string.

    Parameters
    ----------
    string : string

    Returns
    -------
    bool, None, or str

    Examples
    --------
    >>> print parse_string_literal('true')
    True

    >>> print parse_string_literal('False')
    False

    >>> print parse_string_literal('none')
    None

    >>> print parse_string_literal('something else')
    'something else'

    """
    if string.lower().strip() == 'true':
        return True
    if string.lower().strip() == 'false':
        return False
    if string.lower().strip() == 'none':
        return None
    return string


def split_list(string):
    """Parse a string containing a comma-separated list as a Python list of
    strings.

    Parameters
    ----------
    string : string

    Returns
    -------
    list of strings

    Examples
    --------
    >>> print split_list('one, two, three')
    ['one', 'two', 'three']

    """
    l = string.split(',')
    return [x.strip() for x in l]


def parse_pipeline_config(config):
    """Parse a PISA pipeline configuration file.

    Parameters
    ----------
    config : <?>

    Returns
    -------
    <?>

    """
    if isinstance(config, basestring) \
            and not isinstance(config, BetterConfigParser):
        config = from_file(config)
    # create binning objects
    binning_dict = {}
    for name, value in config.items('binning'):
        if name.endswith('.order'):
            order = split_list(config.get('binning', name))
            binning, _ = name.split('.')
            bins = []
            for bin_name in order:
                kwargs = eval(config.get('binning', binning + '.' + bin_name))
                bins.append(OneDimBinning(bin_name, **kwargs))
            binning_dict[binning] = MultiDimBinning(bins)

    stage_dicts = OrderedDict()
    # find pipline setting
    pipeline_order = split_list(config.get('pipeline', 'order'))
    for item in pipeline_order:
        stage, service = item.split(':')
        section = 'stage:' + stage
        # get infos for stages
        stage_dicts[stage] = {}
        stage_dicts[stage]['service'] = service
        if config.has_option(section, 'param_selectors'):
            param_selectors = eval(config.get(section, 'param_selectors'))
        else:
            param_selectors = []
        params = {'unkeyed_params':[],
                  'keyed_param_sets': [{key:[] for key in s}
                                       for s in param_selectors],
                  'selections':[]}
        for s in param_selectors:
            params['selections'].append(s[0])

        for name, value in config.items(section):
            if name.startswith('param.'):
                # find parameter root
                if any([name.startswith('param.'+ param_selector + '.')
                        for sublist in param_selectors
                        for param_selector in sublist]) \
                        and name.count('.') == 2:
                    _, set_name, pname = name.split('.')
                    set_number = [set_name in l for l in
                        param_selectors].index(True)

                elif name.startswith('param.') and name.count('.') == 1:
                    _, pname = name.split('.')
                    set_name = None
                    set_number = -1
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
                            if set_name is not None:
                                priorname += '_' + set_name
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
                        if set_name is not None:
                            params['keyed_param_sets'][set_number][set_name].append(Param(**kwargs))
                        else:
                            params['unkeyed_params'].append(Param(**kwargs))
                    except:
                        logging.error('Failed to instantiate new Param object'
                                      ' with kwargs %s' %kwargs)
                        raise

            elif 'binning' in name:
                stage_dicts[stage][name] = binning_dict[value]

            elif not name == 'param_selectors':
                stage_dicts[stage][name] = parse_string_literal(value)

        #if len(params['unkeyed_params']) > 0:
        stage_dicts[stage]['unkeyed_params'] = ParamSet(*params['unkeyed_params'])
        stage_dicts[stage]['selections'] = params['selections']
        stage_dicts[stage]['keyed_param_sets'] = []
        for pset in params['keyed_param_sets']:
            stage_dicts[stage]['keyed_param_sets'].append({})
            for key,item in pset.items():
                stage_dicts[stage]['keyed_param_sets'][-1][key] = ParamSet(*item)

    return stage_dicts

#if __name__ == '__main__':
#    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
#    parser = ArgumentParser()
#    parser.add_argument(
#        '-p', '--pipeline-settings', metavar='CONFIGFILE', type=str,
#        required=True,
#        help='File containing settings for the pipeline.'
#    )
#    args = parser.parse_args()
#    config = BetterConfigParser()
#    config.read(args.pipeline_settings)
#    cfg =  parse_pipeline_config(config)
#    for key,vals in cfg.items():
#        print key, vals
