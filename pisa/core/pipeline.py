#! /usr/bin/env python
# authors: J.Lanfranchi/P.Eller
# date:   March 20, 2016
"""
Implementation of the Pipeline object, and a __main__ script to instantiate and
run a pipeline.
"""


from collections import OrderedDict
import importlib
import os
import sys

from pisa.core.stage import Stage
from pisa.core.param import ParamSet
from pisa.utils.parse_config import parse_config
from pisa.utils.log import logging, set_verbosity
from pisa.utils.hash import hash_obj
from pisa.utils.profiler import profile


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
    @profile
    def __init__(self, config):
        self._stages = []
        if isinstance(config, basestring):
            config = parse_config(config=config)
        assert isinstance(config, OrderedDict)
        self.config = config
        self._init_stages()

    def __iter__(self):
        return iter(self._stages)

    def __getitem__(self, idx):
        if isinstance(idx, basestring):
            for stage_num, stage in enumerate(self):
                if stage.stage_name == idx:
                    idx = stage_num
                    break

        if isinstance(idx, (int, slice)):
            return self.stages[idx]

        raise ValueError('Cannot locate stage "%s" in pipeline. Stages'
                         ' available are %s.' %(idx, self.stage_names))

    def __getattr__(self, attr):
        for stage in self:
            if stage.stage_name == attr:
                return stage
        raise AttributeError('Stage or attribute "%s" not in pipeline.' %attr)

    def _init_stages(self):
        """Stage factory: Instantiate stages specified by self.config."""

        self._stages = []
        for stage_num, (stage_name, settings) in enumerate(self.config.items()):
            stage_name = stage_name.lower()
            try:
                logging.debug('instantiating stage %s' % stage_name)
                print settings
                service = settings['service'].lower()
                # Import stage service
                module = importlib.import_module('pisa.stages.%s.%s'
                                                 %(stage_name, service))
                # Get class
                cls = getattr(module, service)

                # Instantiate object, do basic type check
                # NOTE: the `service` kwarg is not for the class, so leave it
                # out!
                stage = cls(**{k:v for k,v in settings.items() if k!='service'})
                assert isinstance(stage, Stage)

                # Append stage to pipeline
                self._stages.append(stage)

            except:
                logging.error('Failed to initialize stage #%d (%s).'
                              %(stage_num, stage_name))
                raise

        logging.debug(str(self.params))

    @profile
    def get_outputs(self, inputs=None, idx=None,
                    return_intermediate=False):
        """Run the pipeline to compute its outputs.

        Parameters
        ----------
        inputs : None or MapSet # TODO: other container(s)
            Optional inputs to send to the first stage of the pipeline.
        idx : None, string, or int
            Specification of which stage(s) to run. If None is passed, all
            stages will be run. If a string is passed, all stages are run up to
            and including the named stage. If int is passed, all stages are run
            up to but *not* including `idx`. Numbering follows Python
            conventions (i.e., is 0-indexed).
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
        i = 0
        if isinstance(idx, basestring):
            idx = self.stage_names.index(idx) + 1
        if idx is not None and idx <= 0:
            raise ValueError('Integer `idx` must be > 0')
        for stage in self.stages[:idx]:
            logging.info('>> Working on stage "%s" service "%s"'
                         %(stage.stage_name, stage.service_name))
            try:
                logging.trace('>>> BEGIN: get_outputs')
                outputs = stage.get_outputs(inputs=inputs)
                logging.trace('>>> END  : get_outputs')
            except:
                logging.error('Error occurred computing outputs in stage %s /'
                              ' service %s ...' %(stage.stage_name,
                                                  stage.service_name))
                raise

            logging.trace('outputs: %s' %(outputs,))

            if return_intermediate:
                intermediate.append(outputs)

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

    @property
    def stage_names(self):
        return [s.stage_name for s in self]


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    import numpy as np
    from pisa.core.map import Map, MapSet
    from pisa.utils.fileio import mkdir, to_file
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
        '-d', '--outdir', metavar='DIR', type=str,
        help='''Store all output files (data and plots) to this directory.
        Directory will be created (including missing parent directories) if it
        does not exist already. If no outdir is provided, no outputs will be
        saved.'''
    )
    #parser.add_argument(
    #    '-o', '--outname', metavar='FILENAME', type=str,
    #    default='out.json',
    #    help='''Filename for storing output data.'''
    #)
    parser.add_argument(
        '--intermediate', action='store_true',
        help='''Store all intermediate outputs, not just the final stage's
        outputs.'''
    )
    parser.add_argument(
        '--transforms', action='store_true',
        help='''Store all transforms (for stages that use transforms).'''
    )
    parser.add_argument(
        '-i', '--inputs-file', metavar='FILE', type=str,
        help='''File from which to read inputs to be fed to the pipeline.'''
    )
    # TODO: optionally store the transform sets from each stage
    #parser.add_argument(
    #    '-T', '--transform-file', metavar='FILE', type=str,
    #    help='''File into which to store transform(s) from the pipeline.'''
    #)
    parser.add_argument(
        '--pdf', action='store_true',
        help='''Produce pdf plot(s).'''
    )
    parser.add_argument(
        '--annotate', action='store_true',
        help='''Annotate pllots with counts per bin'''
    )
    parser.add_argument(
        '--png', action='store_true',
        help='''Produce png plot(s).'''
    )
    parser.add_argument(
        '-v', action='count', default=None,
        help='set verbosity level'
    )

    args = parser.parse_args()
    set_verbosity(args.v)

    if args.outdir:
        mkdir(args.outdir)
    else:
        if args.pdf or args.png:
            raise ValueError('No --outdir provided, so cannot save images.')

    # Instantiate the pipeline
    pipeline = Pipeline(args.pipeline_settings)

    for run in xrange(1):
        logging.info('')
        logging.info('## STARTING RUN %d ............' % run)
        logging.info('')
        if args.only_stage is None:
            idx = slice(0, args.stop_after_stage)
            outputs = pipeline.get_outputs(idx=args.stop_after_stage)
        else:
            assert args.stop_after_stage is None
            idx = slice(args.only_stage, args.only_stage+1)
            stage = pipeline.stages[args.only_stage]
            # create dummy inputs
            if hasattr(stage, 'input_binning'):
                logging.info('building dummy input')
                input_maps = []
                for name in stage.input_names:
                    hist = np.ones(stage.input_binning.shape)
                    input_maps.append(
                        Map(name=name, hist=hist, binning=stage.input_binning)
                    )
                inputs = MapSet(maps=input_maps, name='ones', hash=1)
            else:
                inputs = None
            outputs = stage.get_outputs(inputs=inputs)
        logging.info('')
        logging.info('## ............ finished RUN %d' % run)
        logging.info('')

    for stage in pipeline.stages[idx]:
        if not args.outdir:
            break
        stg_svc = stage.stage_name + '__' + stage.service_name
        fbase = os.path.join(args.outdir, stg_svc)
        if args.intermediate or stage == pipeline.stages[-1]:
            stage.outputs.to_json(fbase + '__output.json')
        if args.transforms and stage.use_transforms:
            stage.transforms.to_json(fbase + '__transforms.json')

        formats = OrderedDict(png=args.png, pdf=args.pdf)
        for fmt, enabled in formats.items():
            if not enabled:
                continue
            my_plotter = plotter(stamp='PISA cake test',
                                 outdir=args.outdir,
                                 fmt=fmt, log=False,
                                 annotate=args.annotate)
            my_plotter.ratio = True
            my_plotter.plot_2d_array(stage.outputs, fname=stg_svc + '__output', cmap='OrRd')
