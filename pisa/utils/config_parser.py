# author: P.Eller
#         pde3+pisa@psu.edu
#
# date:   2016-04-28
"""
Parse a ConfigFile object into a dict containing an item for every analysis
stage, that itself contains all necessary instantiation arguments/objects for
that stage. for en example config file, please consider
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

* `pipeline` is the top most section that defines the hierarchy of stages and
    what services to be instatiated.

* `binning` can contain different binning definitions, that are then later
    referred to from within the stage sections.

* `stage` one such section per stage:service is necessary. It cotains some
    options that are common for all stages (`binning`, `error_method` and
    `debug_mode`) as well as all the necessary arguments and parameters for a
    given stage.


Param definitions:
------------------

Every key in a stage section that starts with `param.name` is interpreted and
parsed into a PISA param object. These can be strings (e.g. a filename - don't
use any quotation marks) or quantities. The later case expects an expression
that can be converted by the `parse_quantity` function. The `+/-` notation will
be interpreted as a gaussian prior for the quantity. Units can be added by `*
unit.soandso`.

Additional arguments to a parameter are passed in with the `.` notation, for
example `param.name.fixed = False`, which makes it a free parameter in the fit
(by default a parameter is fixed unless specified like this).

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
scope of a single pipeline; synchronization of parameters across pipelines is
done by adding the pipelines to a single DistributionMaker object and updating
params through the DistributionMaker's update_params method.

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
from pisa.core.param import Param, ParamSelector, ParamSet
from pisa.core.prior import Prior
from pisa.utils.betterConfigParser import BetterConfigParser
from pisa.utils.fileio import from_file
from pisa.utils.log import logging


# Config files use "units.xyz" to denote that "xyz" is a unit; therefore,
# ureg is also referred to as "units" in this context.
units = ureg


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
    if string.strip().lower() == 'true':
        return True
    if string.strip().lower() == 'false':
        return False
    if string.strip().lower() == 'none':
        return None
    return string


def split(string, sep=','):
    """Parse a string containing a comma-separated list as a Python list of
    strings. Each resulting string is forced to be lower-case and surrounding
    whitespace is stripped.

    Parameters
    ----------
    string : string
        The string to be split

    sep : string
        Separator to look for

    Returns
    -------
    list of strings

    Examples
    --------
    >>> print split(' One, TWO, three ')
    ['one', 'two', 'three']

    >>> print split('one:two:three', sep=':')
    ['one', 'two', 'three']

    """
    return [x.strip().lower() for x in str.split(string, sep)]


PARAM_RE = re.compile(
    r'^param\.(?P<subfields>(([^.\s]+)(\.|$))+)',
    re.IGNORECASE
)

PARAM_ATTRS = ['range', 'prior', 'fixed']


def parse_param(config, section, selector, fullname, pname, value):
    # TODO: Are these defaults actually a good idea?
    kwargs = dict(name=pname, is_fixed=True, prior=None, range=None)
    try:
        value = parse_quantity(value)
        kwargs['value'] = value.n * value.units
    except ValueError:
        value = parse_string_literal(value)
        kwargs['value'] = value

    # Search for explicit attr specifications
    if config.has_option(section, fullname + '.fixed'):
        kwargs['is_fixed'] = config.getboolean(section, fullname + '.fixed')

    if config.has_option(section, fullname + '.prior'):
        if config.get(section, fullname + '.prior') == 'uniform':
            kwargs['prior'] = Prior(kind='uniform')
        elif config.get(section, fullname + '.prior') == 'spline':
            priorname = pname
            if selector is not None:
                priorname += '_' + selector
            data = config.get(section, fullname + '.prior.data')
            data = from_file(data)
            data = data[priorname]
            knots = ureg.Quantity(np.asarray(data['knots']), data['units'])
            knots = knots.to(value.units)
            coeffs = np.asarray(data['coeffs'])
            deg = data['deg']
            kwargs['prior'] = Prior(kind='spline', knots=knots, coeffs=coeffs,
                                    deg=deg)
        elif 'gauss' in config.get(section, fullname + '.prior'):
            raise Exception('Please use new style +/- notation for gaussian'
                            ' priors in config')
        else:
            raise Exception('Prior type unknown')

    elif hasattr(value, 's') and value.s != 0:
        kwargs['prior'] = Prior(kind='gaussian', mean=value.n * value.units,
                                stddev=value.s * value.units)

    if config.has_option(section, fullname + '.range'):
        range_ = config.get(section, fullname + '.range')
        if 'nominal' in range_:
            nominal = value.n * value.units
        if 'sigma' in range_:
            sigma = value.s * value.units
        range_ = range_.replace('[', 'np.array([')
        range_ = range_.replace(']', '])')
        kwargs['range'] = eval(range_).to(value.units)

    try:
        param = Param(**kwargs)
    except:
        logging.error('Failed to instantiate new Param object with kwargs %s'
                      %param_kwargs)
        raise

    return param


def parse_pipeline_config(config):
    """Parse a PISA pipeline configuration file.

    Parameters
    ----------
    config : <?>

    Returns
    -------
    <?>

    """
    # TODO: Why do we have to make sure it isn't a BetterConfigParser?
    if isinstance(config, basestring) \
            and not isinstance(config, BetterConfigParser):
        config = from_file(config)

    # Create binning objects
    binning_dict = {}
    for name, value in config.items('binning'):
        if name.endswith('.order'):
            order = split(config.get('binning', name))
            binning, _ = split(name, sep='.')
            bins = []
            for bin_name in order:
                kwargs = eval(config.get('binning', binning + '.' + bin_name))
                bins.append(OneDimBinning(bin_name, **kwargs))
            binning_dict[binning] = MultiDimBinning(bins)

    # Pipeline section
    section = 'pipeline'

    # Get and parse the order of the stages (and which services implement them)
    order = [split(x, ':') for x in split(config.get(section, 'order'))]

    param_selections = []
    if config.has_option(section, 'param_selections'):
        param_selections = split(config.get(section, 'param_selections'))

    # Parse [stage:<stage_name>] sections and store to stage_dicts
    stage_dicts = OrderedDict()
    for stage, service in order:
        section = 'stage:%s' %stage

        # Instantiate dict to store args to pass to this stage
        service_kwargs = dict()

        param_selector = ParamSelector()

        for name, value in config.items(section):
            param_match = PARAM_RE.match(name)
            if param_match is not None:
                param_match_dict = param_match.groupdict()
                param_subfields = param_match_dict['subfields'].split('.')

                # Figure out what the dotted fields represent...

                # Must have at least one field
                if len(param_subfields) == 0:
                    raise ValueError('No dotted fields in config name "%s"' %name)

                # If only one field, this must be the param's name
                if len(param_subfields) == 1:
                    pname = param_subfields.pop()
                    selector = None
                    attr = None
                else:
                    # Look for and remove attr field and any subsequent fields
                    attr_idx = None
                    attr = None
                    for _attr in PARAM_ATTRS:
                        if _attr in param_subfields:
                            if attr_idx is not None:
                                raise ValueError(
                                    'Found multiple attrs in config name "%s"'
                                    %name
                                )
                            attr_idx = param_subfields.index(_attr)
                            attr = []
                            # All fields from attr on pertain to the attr; remove
                            for i in range(attr_idx, len(param_subfields)):
                                a = param_subfields.pop(attr_idx)
                                assert a not in PARAM_ATTRS
                                attr.append(a)

                    # Either "pname", "selector.pname", or invalid
                    pname = param_subfields.pop()
                    if len(param_subfields) == 0:
                        selector = None
                    elif len(param_subfields) == 1:
                        selector = param_subfields.pop()
                if len(param_subfields) > 0:
                    raise ValueError('Too many dotted fields to parse in name'
                                     ' "%s"' %name)

                logging.trace('pname=%s, selector=%s, attr=%s'
                              %(pname, selector, attr))

                # If field is an attr, skip since these are located manually
                if attr is not None:
                    continue

                # Check if this param already exists in a previous stage; if
                # so, make sure there are no specs for this param, but just a
                # link to previous the param object that is already
                # instantiated.
                for kw in stage_dicts.values():
                    if not kw.has_key('params'):
                        continue
                    try:
                        param = kw['params'].get(name=pname, selector=selector)
                    except KeyError:
                        continue

                    # Make sure there are no other specs for the param defined
                    # in the current section
                    for attr_ in PARAM_ATTRS:
                        assert not config.has_option(section, name + '.' + attr_)

                    param = kw['params'][pname]
                    #print 'p copy:', param
                    break

                # Param *not* found in a previous stage (i.e., no explicit `break`
                # encountered in `for` loop above); instantiate the param.
                else:
                    param = parse_param(config, section, selector, name, pname,
                                        value)
                    #print 'p inst:', param

                print 'p.name:', param.name
                #print 'ps before:', param_selector.params.names
                #param_selector.update(p=param, selector=selector)
                #print 'ps after:', param_selector.params.names
                #print 'ps param:', param_selector.get(name=param.name,
                #                                      selector=selector).name

            elif 'binning' in name:
                service_kwargs[name] = binning_dict[value]

            else:
                service_kwargs[name] = parse_string_literal(value)

        service_kwargs['params'] = param_selector

        # Append this dict to the OrderedDict with all stage dicts
        stage_dicts[stage + '.' + service] = service_kwargs

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
