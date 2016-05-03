#! /usr/bin/env python
# authors: J.Lanfranchi/P.Eller
# date:   March 20, 2016

import importlib

from pisa.core.stage import Stage
from pisa.core.param import ParamSet
from pisa.utils.log import logging, set_verbosity


class Pipeline(object):
    """Instantiate stages according to a parsed config object; excecute stages.
    
    args:
    - config dict

    methods:
    - compute_outputs: returning output MapSet from the (final) pipeline
    stage(s)
    - update_params: update params of all stages

    attributes:
    - params: ParamSet containing all params from all stages
    """

    def __init__(self, config):
        self._stages = []
        self.config = config
        self._init_stages()

    def __iter__(self):
        return iter(self._stages)

    def _init_stages(self):
        """Stage factory: Instantiate stages specified by self.config."""
        self._stages = []
        for stage_num, stage_name in enumerate(self.config.keys()):
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

    def compute_outputs(self, inputs=None, idx=None, return_intermediate=False):
        idx = slice(None) if idx is None else idx
        intermediate = []
        for stage in self.stages[idx]:
            logging.debug('Working on stage %s (%s)' %(stage.stage_name,
                                                       stage.service_name))
            try:
                outputs = stage.compute_outputs(inputs)
            except:
                logging.error('Error occurred computing outputs in stage %s /'
                              ' service %s ...' %(stage.stage_name,
                                                  stage.service_name))
                raise

            logging.trace('outputs: %s' %(outputs,))

            if return_intermediate:
                intermediate.append(outputs)

            # Outputs from this stage become next stage's inputs
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
    from pisa.utils.fileio import from_file, to_file
    from pisa.utils.parse_config import parse_config

    parser = ArgumentParser()
    parser.add_argument(
        '-t', '--pipeline-settings', metavar='CONFIGFILE', type=str,
        help='File containing settings for the pipeline.'
    )
    parser.add_argument(
        '-s', '--test-stage', metavar='STAGE', type=str,
        help='''Test stage: Instantiate a single stage in the pipeline
        specification and run it in isolation (as the sole stage in a
        pipeline). If it is a stage that requires inputs, these can be
        specified with the --infile argument, or else dummy stage input maps
        (numpy.ones(...), matching the input binning specification) are
        generated for testing purposes. See also --infile and --transformfile
        arguments.'''
    )
    parser.add_argument(
        '-o', '--outfile', metavar='FILE', type=str,
        default='out.json',
        help='''File for storing outputs. See also --intermediate-outputs
        argument.'''
    )
    parser.add_argument(
        '-i', '--infile', metavar='FILE', type=str,
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
    args = parser.parse_args()

    pipeline_config = parse_config(from_file(args.pipeline_settings))

    pipeline = Pipeline(pipeline_config)
    m0 = pipeline.compute_outputs()
    fp = pipeline.params.free
    fp['test'].value *= 1.2
    pipeline.update_params(fp)
    m1 = pipeline.compute_outputs()
    print (m1/m0)['nue'][0,0]
