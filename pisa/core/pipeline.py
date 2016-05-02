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
        self._stages = []
        for stage_num, stage_name in enumerate(self.config.keys()):
            service = self.config[stage_name.lower()].pop('service').lower()
            # factory
            # import stage service
            module = importlib.import_module('pisa.stages.%s.%s'
                                             %(stage_name.lower(), service))
            # get class
            cls = getattr(module, service)
            # instantiate object
            stage = cls(**self.config[stage_name.lower()])
            assert isinstance(stage, Stage)
            # make sure the binnings match (including if both are specified to
            # be None)
            if len(self._stages) > 0:
                assert stage.input_binning == self._stages[-1].output_binning
            # add stage to pipeline
            self._stages.append(stage)
        print self.params

    def compute_outputs(self, inputs=None, idx=None, return_intermediate=False):
        intermediate = []
        for stage in self.stages[:idx]:
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
    parser.add_argument('-t', '--template-settings', type=str,
                        metavar='configfile', required=True,
                        help='''settings for the template generation''')
    parser.add_argument('-o', '--outfile', dest='outfile', metavar='FILE',
                        type=str, action='store', default="out.json",
                        help='file to store the output')
    args = parser.parse_args()

    template_config = parse_config(from_file(args.template_settings))

    template_nu_pipeline = Pipeline(template_config)
    m0 = template_nu_pipeline.compute_outputs()
    fp = template_nu_pipeline.params.free #free_params
    fp['test'].value*=1.2
    pipeline.update_params(fp)
    m1 = pipeline.compute_outputs()
    print (m1/m0)['nue'][0,0]
