#! /usr/bin/env python
# authors: J.Lanfranchi/P.Eller
# date:   March 20, 2016

import importlib
from collections import Sequence

from pisa.core.pipeline import Pipeline
from pisa.core.param import ParamSet
from pisa.utils.log import logging, set_verbosity


class TemplateMaker(object):
    """Creates a TemplateMaker object that has several pipelines inside
    the outputs from all pipelines are added together  

    Parameters
    ----------
    *args : Pipelines, strings, OrderedDicts, or sequences thereof
        A new pipline is instantiated with each object passed. Legal objects
        are already-instantiated Pipelines, strings (which specify a resource
        location for a pipeline config file) and OrderedDicts (as returned by
        parse_config).

    Notes
    -----
    Free params with the same name in two pipelines are updated at the same
    time using the `update_params`, `set_free_params`, or
    `set_rescaled_free_params` methods.

    `*_rescaled_*` properties and methods are for interfacing with a minimizer,
    where values are linearly mapped onto the interval [0, 1] according to
    the parameter's allowed range.

    """
    def __init__(self, *args):
        self._pipelines = []
        extended_args = []
        for arg in args:
            if isinstance(arg, basestring):
                extended_args.append(arg)
            elif isinstance(arg, Sequence):
                extended_args.extend(arg)
        for arg in extended_args:
            if not isinstance(arg, Pipeline):
                arg = Pipeline(arg)
            self._pipelines.append(arg)

    def __iter__(self):
        return iter(self._pipelines)

    def compute_outputs(self, **kwargs):
        output = None
        for pipeline in self.pipelines:
            pipeline_output = pipeline.compute_outputs(**kwargs)
            if output is None:
                output = pipeline_output
            else:
                # add together
                if isinstance(pipeline_output, list):
                    output = [sum(x) for x in zip(output, pipeline_output)]
                else: 
                    output += pipeline_output
        return output

    def update_params(self, params):
        [pipeline.params.update_existing(params) for pipeline in self]

    @property
    def pipelines(self):
        return tuple(self._pipelines)

    @property
    def params(self):
        params = ParamSet()
        [params.extend(pipeline.params) for pipeline in self.pipelines]
        return params

    @property
    def free_params_values(self):
        # a simple list of param values
        return self.params.free.values

    @property
    def free_params_rescaled_values(self):
        # a simple list of idimensionless param values rescaled to (0,1)
        return [p.rescaled_value for p in self.params.free]

    @property
    def free_params_names(self):
        # a simple list of names of the free params
        return [p.name for p in self.params.free]

    def set_free_params(self, values):
        """Set free param values given a simple list

        """
        for name, value in zip(self.free_params_names, values):
            for pipeline in self.pipeline:
                if name in [p.name for p in pipeline.params.free]:
                    pipeline.params.free.value = value

    def set_rescaled_free_params(self, rvalues):
        """Set free param values given a simple list of (0,1) rescaled,
        dimensionless values

        """
        for pipeline in self.pipelines:
            fp = pipeline.params.free
            for name, rvalue in zip(self.free_params_names, rvalues):
                if name in [p.name for p in fp]:
                    fp[name].rescaled_value = rvalue
            pipeline.update_params(fp)
                    

if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    import numpy as np
    from pisa.utils.fileio import from_file, to_file
    from pisa.utils.parse_config import parse_config

    parser = ArgumentParser()
    parser.add_argument('-t', '--template-settings', type=str,
                        metavar='configfile', required=True,
                        help='''settings for the template generation''')
    parser.add_argument('-o', '--outfile', dest='outfile', metavar='FILE',
                        type=str, action='store', default="out.json",
                        help='file to store the output')
    args = parser.parse_args()

    template_maker = TemplateMaker([args.template_settings,
                                    args.template_settings])
    print template_maker.compute_outputs()
    print template_maker.compute_outputs()
