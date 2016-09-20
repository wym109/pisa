#! /usr/bin/env python
# authors: J.Lanfranchi/P.Eller
# date:   March 20, 2016

import importlib
from collections import OrderedDict, Sequence

from pisa.core.pipeline import Pipeline
from pisa.core.param import ParamSet
from pisa.utils.log import logging, set_verbosity


class DistributionMaker(object):
    """Creates a DistributionMaker object that has several pipelines inside
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
        # TODO: make this a property so that input can be validated
        self.fluctuate = None

        self._pipelines = []
        extended_args = []
        for arg in args:
            if isinstance(arg, (basestring, OrderedDict)):
                extended_args.append(arg)
            elif isinstance(arg, Sequence):
                extended_args.extend(arg)

        for arg in extended_args:
            if not isinstance(arg, Pipeline):
                arg = Pipeline(arg)
            self._pipelines.append(arg)

    def __iter__(self):
        return iter(self._pipelines)

    def get_outputs(self, seed=None, **kwargs):
        total_outputs = None
        outputs = [pipeline.get_outputs(**kwargs) for pipeline in self]
        if self.fluctuate is not None:
            outputs = outputs.fluctuate(method=self.fluctuate,
                                                    seed=seed)
        return outputs

    def get_total_outputs(self, seed=None, **kwargs):
        total_outputs = None
        outputs = [pipeline.get_outputs(**kwargs) for pipeline in self]
        total_outputs = reduce(lambda x,y: x+y, outputs)
        if self.fluctuate is not None:
            total_outputs = total_outputs.fluctuate(method=self.fluctuate,
                                                    seed=seed)
        return total_outputs
    def update_params(self, params):
        [pipeline.update_params(params) for pipeline in self]

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
    from pisa.utils.plotter import plotter    

    parser = ArgumentParser()
    parser.add_argument('-t', '--template-settings',
                        metavar='configfile', required=True,
                        action='append',
                        help='''settings for the template generation''')
    parser.add_argument(
        '-v', action='count', default=None,
        help='set verbosity level'
    )
    args = parser.parse_args()

    set_verbosity(args.v)

    template_maker = DistributionMaker(args.template_settings)
    outputs =  template_maker.get_outputs()
    my_plotter = plotter(stamp='PISA cake test',
			 outdir='.',
			 fmt='pdf', log=False,
			 annotate=False)
    my_plotter.ratio = True
    my_plotter.plot_2d_array(outputs, fname='dist_output', cmap='OrRd')
