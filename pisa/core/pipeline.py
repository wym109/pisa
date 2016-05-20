#! /usr/bin/env python
# authors: J.Lanfranchi/P.Eller
# date:   March 20, 2016

import importlib
import sys
from collections import OrderedDict

from pisa.core.stage import Stage
from pisa.core.param import ParamSet
from pisa.utils.parse_config import parse_config
from pisa.utils.log import logging, set_verbosity
from pisa.utils.profiler import profile

"""
Implementation of the Pipeline object, and a __main__ script to instantiate and
run a pipeline.
"""

# TODO: should we check that the output binning of a previous stage produces
# the inputs required by the current stage, or that the aggregate outputs that
# got produced by previous stages (less those that got consumed in other
# previous stages) hold what the current stage requires for inputs... or
# should we not assume either will check out, since it's possible that the
# stage requires sideband objects that are to be introduced at the top of the
# pipeline by the user (and so there's no way to verify that all inputs are
# present until we see what the user hands the pipeline as its top-level
# input)? Alternatively, the lack of apparent inputs for a stage could show
# a warning message. Or we just wait to see if it fails when the user runs the
# code.

# TODO: return an OrderedDict instead of a list if the user requests
# intermediate results? Or simply use the `outputs` attribute of each stage to
# dynamically access this?

class Pipeline(object):
    """Instantiate stages according to a parsed config object; excecute
    stages.


    Parameters
    ----------
    config : string or OrderedDict
        If string, interpret as resource location; send to the
          parse_config.parse_config() function to get a config OrderedDict.
        If OrderedDict, use directly as pipeline configuration.


    Methods
    -------
    get_outputs
        Returns output MapSet from the (final) pipeline, or all intermediate
        outputs if `return_intermediate` is specified as True.

    update_params
        Update params of all stages using values from a passed ParamSet


    Attributes
    ----------
    params : ParamSet
        All params from all stages in the pipeline

    stages : list
        All stages in the pipeline

    """
    def __init__(self, config):
        self._stages = []
        if isinstance(config, basestring):
            config = parse_config(config=config)
        assert isinstance(config, OrderedDict)
        self.config = config
        self._init_stages()

    def __iter__(self):
        return iter(self._stages)

    def _init_stages(self):
        """Stage factory: Instantiate stages specified by self.config."""

        self._stages = []
        for stage_num, stage_name in enumerate(self.config.keys()):
            logging.debug('instatiating stage %s'%stage_name)
            service = self.config[stage_name.lower()].pop('service').lower()
            # Import stage service
            module = importlib.import_module('pisa.stages.%s.%s'
                                             %(stage_name.lower(), service))
            # Get class
            cls = getattr(module, service)

            # Instantiate object, do basic type check
            stage = cls(**self.config[stage_name.lower()])
            assert isinstance(stage, Stage)

            # Make sure the input binning of this stage is compatible with the
            # output binning of the previous stage ("compatible binning"
            # includes if both are specified to be None)
            if len(self._stages) > 0:
                assert stage.input_binning == self._stages[-1].output_binning

            # Append stage to pipeline
            self._stages.append(stage)

        logging.debug(str(self.params))

    @profile
    def get_outputs(self, inputs=None, idx=None,
                    return_intermediate=False):
        """Run the pipeline to compute its outputs.

        Parameters
        ----------
        inputs : None or MapSet # TODO: other container(s)
            Optional inputs to send to the first stage of the pipeline.
        idx : None, int, or slice
            Specification of which stage(s) to run. If None is passed, all
            stages will be run.
        return_intermediate : bool
            If True,

        Returns
        -------
        outputs : list or MapSet
            MapSet output by final stage if `return_intermediate` is False, or
            list of MapSets output by each stage if `return_intermediate` is
            True.

        """
        intermediate = []
        for stage in self.stages[:idx]:
            logging.debug('Working on stage %s (%s)' %(stage.stage_name,
                                                       stage.service_name))
            try:
                outputs = stage.get_outputs(inputs=inputs)
            except:
                logging.error('Error occurred computing outputs in stage %s /'
                              ' service %s ...' %(stage.stage_name,
                                                  stage.service_name))
                raise

            logging.trace('outputs: %s' %(outputs,))

            if return_intermediate:
                intermediate.append(outputs)

            # Outputs from this stage become inputs for next stage
            inputs = outputs

        if return_intermediate:
            return intermediate

        return outputs

    def update_params(self, params):
        [stage.params.update_existing(params) for stage in self]

    @property
    def params(self):
        params = ParamSet()
        [params.extend(stage.params) for stage in self]
        return params

    @property
    def stages(self):
        return [s for s in self]


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    import numpy as np
    from pisa.core.map import Map, MapSet
    from pisa.utils.fileio import from_file, to_file
    from pisa.utils.parse_config import parse_config
    from pisa.utils.plotter import plotter

    parser = ArgumentParser()
    parser.add_argument(
        '-p', '--pipeline-settings', metavar='CONFIGFILE', type=str,
        help='File containing settings for the pipeline.'
    )
    parser.add_argument(
        '--only-stage', metavar='STAGE', type=int,
        help='''Test stage: Instantiate a single stage in the pipeline
        specification and run it in isolation (as the sole stage in a
        pipeline). If it is a stage that requires inputs, these can be
        specified with the --infile argument, or else dummy stage input maps
        (numpy.ones(...), matching the input binning specification) are
        generated for testing purposes. See also --infile and --transformfile
        arguments.'''
    )
    parser.add_argument(
        '--stop-after-stage', metavar='STAGE', type=int,
        help='''Test stage: Instantiate a pipeline up to and including
        STAGE, but stop there.'''
    )
    parser.add_argument(
        '-o', '--outputs-file', metavar='FILE', type=str,
        default='out.json',
        help='''File for storing outputs. See also --intermediate-outputs
        argument.'''
    )
    parser.add_argument(
        '-i', '--inputs-file', metavar='FILE', type=str,
        required=False,
        help='''File from which to read inputs to be fed to the pipeline.'''
    )
    parser.add_argument(
        '-T', '--transform-file', metavar='FILE', type=str,
        required=False,
        help='''File into which to store transform(s) from the pipeline.'''
    )
    parser.add_argument(
        '-I', '--intermediate', action='store_true',
        help='''Store all intermediate outputs, not just the final stage's
        outputs.'''
    )
    parser.add_argument(
        '-v', action='count', default=None,
        help='set verbosity level'
    )

    args = parser.parse_args()
    set_verbosity(args.v)
    pipeline = Pipeline(args.pipeline_settings)

    if args.only_stage is not None:
        stage = pipeline.stages[args.only_stage]
        # create dummy inputs
        if hasattr(stage, 'input_binning'):
            logging.info('building dummy input')
            input_maps = []
            for name in stage.input_names:
                hist = np.ones(stage.input_binning.shape)
                input_maps.append(Map(name=name, hist=hist,
                            binning=stage.input_binning))
            inputs = MapSet(maps=input_maps, name='ones')
        else:
            inputs = None
        m0 = stage.get_outputs(inputs=inputs)
    else:
        if args.stop_after_stage is not None:
            m0 = pipeline.get_outputs(idx=args.stop_after_stage)
        else:
            m0 = pipeline.get_outputs()
    #fp = pipeline.params.free
    #fp['test'].value *= 1.2
    #pipeline.update_params(fp)
    #m1 = pipeline.get_outputs()
    #print (m1/m0)['nue'][0,0]
    #print m0['nue']
    #print m0[m0.names[0]]
    #json = {}
    #for name in m0.names:
    #    json[name] = m0[name].hist
    #to_file(json, args.outputs_file)
    to_file(m0, args.outputs_file)
    my_plotter = plotter()
    my_plotter.add_mapset(m0)
    my_plotter.plot_2d()
