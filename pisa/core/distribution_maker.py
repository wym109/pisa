#! /usr/bin/env python
# authors: J.Lanfranchi/P.Eller
# date:   March 20, 2016

import importlib
from collections import OrderedDict, Sequence

from pisa.core.pipeline import Pipeline
from pisa.core.param import ParamSet
from pisa.utils.config_parser import BetterConfigParser
from pisa.utils.log import logging, set_verbosity


class DistributionMaker(object):
    """Container for one or more pipelines; the outputs from all contained
    pipelines are added together to create the distribution.

    Parameters
    ----------
    pipelines : Pipeline or convertible thereto, or sequence thereof
        A new pipline is instantiated with each object passed. Legal objects
        are already-instantiated Pipelines and anything interpret-able by the
        Pipeline init method.

    fluctuate : None or string
        Apply fluctuations to outputs (as in the case of pseudo-data).
        Specifying None disables fluctuations while e.g. 'poisson' applies
        Poisson fluctuations. See `Map.fluctuate` for all valid strings.

    Notes
    -----
    Free params with the same name in two pipelines are updated at the same
    time using the `update_params`, `set_free_params`, and
    `_set_rescaled_free_params` methods.

    `_*_rescaled_*` properties and methods are for interfacing with a
    minimizer, where values are linearly mapped onto the interval [0, 1]
    according to the parameter's allowed range. Avoid interfacing with these
    except if using a minimizer, since, e.g., units are stripped and values and
    intervals are non-physical.

    """
    def __init__(self, pipelines, fluctuate=None, label=None):
        self.label = None
        # TODO: make this a property so that input can be validated
        self.fluctuate = fluctuate
        """Whether the output of the distribution maker will be fluctuated (and
        if so, by which method this is done)"""

        self._pipelines = []
        if isinstance(pipelines, (basestring, BetterConfigParser, OrderedDict,
                                  Pipeline)):
            pipelines = [pipelines]

        for pipeline in pipelines:
            if not isinstance(pipeline, Pipeline):
                pipeline = Pipeline(pipeline)
            self._pipelines.append(pipeline)

    def __iter__(self):
        return iter(self._pipelines)

    def get_outputs(self, seed=None, **kwargs):
        total_outputs = None
        outputs = [pipeline.get_outputs(**kwargs) for pipeline in self]
        total_outputs = reduce(lambda x,y: x+y, outputs)
        if self.fluctuate is not None:
            total_outputs = total_outputs.fluctuate(method=self.fluctuate,
                                                    seed=seed)
        return total_outputs

    def update_params(self, params):
        [pipeline.params.update_existing(params) for pipeline in self]

    @property
    def pipelines(self):
        return self._pipelines

    @property
    def params(self):
        params = ParamSet()
        [params.extend(pipeline.params) for pipeline in self.pipelines]
        return params

    def set_free_params(self, values):
        """Set free parameters' values.

        Parameters
        ----------
        values : list of quantities

        """
        for name, value in zip(self.free_params_names, values):
            for pipeline in self.pipeline:
                if name in [p.name for p in pipeline.params.free]:
                    pipeline.params.free.value = value

    def _set_rescaled_free_params(self, rvalues):
        """Set free param values given a simple list of (0,1) rescaled,
        dimensionless values

        """
        for pipeline in self.pipelines:
            fp = pipeline.params.free
            for name, rvalue in zip(self.free_params_names, rvalues):
                if name in [p.name for p in fp]:
                    fp[name]._rescaled_value = rvalue
            pipeline.update_params(fp)


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    import numpy as np
    from pisa.utils.fileio import from_file, to_file
    from pisa.utils.config_parser import parse_pipeline_config
    from pisa.utils.plotter import plotter

    parser = ArgumentParser()
    parser.add_argument(
        '-p', '--pipeline-settings', type=str, required=True,
        metavar='CONFIGFILE', action='append',
        help='''Settings file for each pipeline'''
    )
    parser.add_argument(
        '--outdir', type=str, action='store',
        help='Directory into which to store the output'
    )
    parser.add_argument(
        '-v', action='count', default=None,
        help='Set verbosity level'
    )
    args = parser.parse_args()

    set_verbosity(args.v)

    distribution_maker = DistributionMaker(
        pipelines=args.pipeline_settings,
        fluctuate=False
    )
    outputs = distribution_maker.get_outputs()
    if args.outdir:
        my_plotter = plotter(
            stamp='PISA cake test',
            outdir=args.outdir,
            fmt='pdf', log=False,
            annotate=False
        )
        my_plotter.ratio = True
        my_plotter.plot_2d_array(outputs, fname='dist_output', cmap='OrRd')
