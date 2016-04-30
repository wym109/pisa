#! /usr/bin/env python
# authors: J.Lanfranchi/P.Eller
# date:   March 20, 2016

import importlib

from pisa.core.stage import NoInputStage, InputStage
from pisa.core.param import ParamSet


class TemplateMaker(object):
    """ instantiate stages according to config; excecute stages """
    def __init__(self, config):
        self.config = config
        self.init_stages()

    def init_stages(self):
        self.stages = []
        for stage_num, stage_name in enumerate(self.config.keys()):
            service = self.config[stage_name.lower()]['service'].lower()
            # factory
            # import stage service
            module = importlib.import_module('pisa.stages.%s.%s'
                                             %(stage_name.lower(), service))
            # get class
            cls = getattr(module, service)
            # instanciate object
            stage = cls(**self.config[stage_name.lower()])
            if stage_num == 0:
                assert isinstance(stage, NoInputStage)
            else:
                assert isinstance(stage, InputStage)
                # make sure the biinings match, if there are any
                if hasattr(stage, 'input_binning'):
                    assert hasattr(self.stages[-1], 'output_binning')
                    assert stage.input_binning == \
                            self.stages[-1].output_binning
            self.stages.append(stage)

    def get_outputs(self, idx=None, return_intermediate=False):
        intermediate = []
        for stage in self.stages[:idx]:
            print 'working on stage %s' %stage.stage_name
            if isinstance(stage, NoInputStage):
                output_objs = stage.get_outputs()
            else:
                input_objs = output_objs
                output_objs = stage.get_outputs(input_objs)

            if return_intermediate:
                intermediate.append(output_objs)

        if return_intermediate:
            return intermediate

        return output_objs

    @property
    def free_params(self):
        free_params = ParamSet()
        for stage in self.stages:
            free_params.extend(stage.params.free)
        return free_params

    def update_params(self, params):
        for stage in self.stages:
            stage.params.update(params)
            

if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    import numpy as np
    from pisa.utils.fileio import from_file, to_file
    from pisa.utils.parse_cfg import parse_cfg

    parser = ArgumentParser()
    parser.add_argument('-t', '--template-settings', type=str,
                        metavar='configfile', required=True,
                        help='''settings for the template generation''')
    parser.add_argument('-o', '--outfile', dest='outfile', metavar='FILE',
                        type=str, action='store', default="out.json",
                        help='file to store the output')
    args = parser.parse_args()

    template_settings = from_file(args.template_settings)
    template_settings = parse_cfg(template_settings) 

    template_maker = TemplateMaker(template_settings)
    m0 = template_maker.get_outputs()
    fp = template_maker.free_params
    fp['test'].value*=1.2
    template_maker.update_params(fp)
    m1 = template_maker.get_outputs()
    print (m1/m0)['nue'][0,0]
