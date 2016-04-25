#! /usr/bin/env python
# author: J.L. Lanfranchi
# date:   March 20, 2016

"""
Class to implement template-making procedure.
"""

import sys
from copy import deepcopy

import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from pisa.utils.log import logging, set_verbosity
from pisa.utils.fileio import from_file, to_file
from pisa.utils.utils import Timer
from pisa.utils.parse_cfg import parse_cfg

from pisa.flux import Flux
#from pisa.osc import Osc
#from pisa.aeff import Aeff
#from pisa.reco import Reco
#from pisa.pid import PID


class TemplateMaker:
    """Template-making data chain and methods with generated templates.

    Parameters
    ----------
    template_settings : str or utils.TemplateSettings object

    Methods
    -------
    get_template
    update_params

    Properties
    ----------
    cache_dir
    params : PramSet
    stages
    injected_hierarchy
    injected_octant
    hierarchy_params
    
    """
    def __init__(self, template_settings,
                 cache_dir='$PISA/pisa/resources/.cache'):
        self.cache_dir = resources.find_resource(self.cache_dir, is_dir=True)
        self.stages = OrderedDict()
        self.params = ParamSet()
        self.__injected_params = None
        self.instantiate_chain(template_settings)

    # TODO: Move to ParamSet object
    #@property
    #def params(self):
    #    p = OrderedDict()
    #    for stage in self.stages.values():
    #        p[stage.stage_name] = stage.params
    #    return p

    # TODO: Move to ParamSet object
    #@property
    #def free_params(self):
    #    p = OrderedDict()
    #    for stage in self.stages.values():
    #        p[stage.stage_name] = stage.free_params
    #    return p

    # TODO: Move to ParamSet object
    #@property
    #def params_hash(self):
    #    hashes = [stage.params_hash for stage in self.stages.values()]
    #    return hash_obj(tuple(hashes))

    # TODO: Move to ParamSet object
    #@property
    #def free_params_hash(self):
    #    hashes = [stage.free_params_hash for stage in self.stages.values()]
    #    return hash_obj(tuple(hashes))

    # TODO: Move to ParamSet object
    #@property
    #def num_params(self):
    #    return np.sum([stage.num_params for stage in self.stages.values()])

    # TODO: Move to ParamSet object
    #@property
    #def num_free_params(self):
    #    return np.sum([stage.num_free_params for stage in self.stages.values()])

    # TODO: Move to Analysis object
    #def match_to_data(self, data_map_set, minimizer_settings):
    #    """Use minimizer to adjust free parameters to best fit produced map set
    #    to `data_map_set`"""
    #    scales = []
    #    [scales.extend(stage.free_params_scales)
    #     for state in self.stages.values()]

    # TODO: Move to ParamSet object
    #def get_free_params_list(self):
    #    """Output all stages' free params in an ordered list"""
    #    p = []
    #    [p.extend(stage.free_params) for state in self.stages.values()]

    # TODO: Move to ParamSet object
    #def update_free_params_by_list(self, vals):
    #    """Update all stages' free params with values in list"""
    #    assert len(vals) == self.num_free_params
    #    start_ind = 0
    #    for stage in self.stages.values():
    #        stop_ind = start_ind + stage.num_free_params + 1
    #        stage.free_params = vals[start_ind:stop_ind]
    #        start_ind = stop_ind

    def instantiate_chain(self, template_settings):
        #if isinstance(template_settings, basestring):
        template_settings = loadTemplateSettings(template_settings)
        for ts_stage in template_settings.stages:
            self.stage_factory(ts_stage)
        self.injected_params = self.params

    def service_factory(self, ts_stage):
        # TODO: correct terminology for following?
        # Stage "main" files have capitalized first letter; this gets the
        # stage main as an object
        service_factory = eval('%s.service_factory' %
                               stage.stage_name.capitalize())

        # One cache *per service* for smaller, faster disk caches
        cache_fname = '%s__%s.db' % (stage.stage_name, stage.service_name)
        cache_fpath = os.path.join(self.cache_dir, cache_fname)

        self.stages[stage.stage_name] = service_factory(params=stage.params,
                                                        disk_cache=cache_fpath)

    def get_template(self, skip_stages=None, return_intermediate=False):
        """Run template-making chain.

        Parameters
        ----------
        skip_stages : str, sequence, or None
            Stage name(s) to skip (e.g. 'osc' to compute no-oscillations
            template).

        return_intermediate : bool
            Whether to return all intermediate map sets.

        Returns
        -------
        output_map_sets : OrderedDict
            Final map sets or map sets emitted from all stages (depending on
            settings of `return_intermediate`), keyed by stage names.
        """
        if skip_stages is None:
            skip_stages = []
        if isinstance(skip_stages, basestring):
            skip_stages = [skip_stages]

        input_map_set = None
        output_map_sets = OrderedDict()
        for stage in self.stages.values():
            if stage.stage_name in skip_stages:
                logging.info('Skipping stage %s' % stage.stage_name)
                continue
            with Timer() as t:
                logging.info('Computing stage %s' % stage.stage_name)
                output_map_set = stage.get_output_map_set(input_map_set)
                output_map_sets[stage.stage_name] = output_map_set
                input_map_set = output_map_set
                logging.debug('Stage %s took %s sec' % (stage.stage_name,
                                                        t.secs))

        if not return_intermediate:
            # Final stage name is set in loop above
            return OrderedDict({stage.stage_name:
                                output_map_sets[stage.stage_name]})

        # Otherwise, return all stages as a dict
        return map_sets

    @property
    def source_code_hash(self):
        hashes = [stage.source_code_hash for stage in self.stages.values()]
        return hash_obj(tuple(hashes))

    @property
    def state_hash(self):
        hashes = [stage.state_hash for stage in self.stages.values()]
        return hash_obj(tuple(hashes))


if __name__ == '__main__':
    # parser
    parser = ArgumentParser(
        description='''Runs the template making process.''',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-t', '--template_settings', type=str,
                        metavar='JSONFILE', required=True,
                        help='''settings for the template generation''')

    hselect = parser.add_mutually_exclusive_group(required=False)
    hselect.add_argument('--normal', dest='normal', default=True,
                         action='store_true',
                         help="select the normal hierarchy")
    hselect.add_argument('--inverted', dest='normal', default = False,
                         action='store_false',
                         help="select the inverted hierarchy")

    parser.add_argument('--plot', action='store_true',
                        help='plot resulting maps')
    parser.add_argument('-v', '--verbose', action='count', default=None,
                        help='set verbosity level.')
    parser.add_argument('-s', '--save_all', action='store_true', default=False,
                        help="Save all stages.")
    parser.add_argument('-o', '--outfile', dest='outfile', metavar='FILE',
                        type=str, action='store', default="template.json",
                        help='file to store the output')
    args = parser.parse_args()

    set_verbosity(args.verbose)


    with Timer() as t:
        # Load all the settings
        model_settings = from_file(args.template_settings)
        model_settings = parse_cfg(model_settings)
        
        print model_settings['osc']['params'].names
        print model_settings['osc']['params'].values


        # Select a hierarchy
        logging.info('Selected %s hierarchy' %
                     ('normal' if args.normal else 'inverted'))
        template_params_nh = select_hierarchy(
            model_settings['params'], normal_hierarchy=True
        )
        template_params_ih = select_hierarchy(
            model_settings['params'], normal_hierarchy=False
        )

        # Intialize template maker
        template_params_values_nh = get_values(template_params_nh)
        template_params_values_ih = get_values(template_params_ih)

        ebins = model_settings['binning']['ebins']
        czbins = model_settings['binning']['czbins']
        oversample_e = model_settings['binning']['oversample_e']
        oversample_cz = model_settings['binning']['oversample_cz']
        template_maker = TemplateMaker(template_params_values_nh,
                                       **model_settings['binning'])
    logging.info('  ==> elapsed time to initialize templates: %s sec'
                  % t.secs)

    # Now get the actual template (and multiple times to test caching)
    logging.info('normal...')
    with Timer(verbose=False) as t:
        stage_outputs_nh = template_maker.get_template(
            template_params_values_nh, return_intermediate=True
        )
    logging.info('==> elapsed time to get template: %s sec' % t.secs)
    with Timer(verbose=False) as t:
        stage_outputs_nh = template_maker.get_template(
            template_params_values_nh, return_intermediate=True
        )
    logging.info('==> elapsed time to get template: %s sec' % t.secs)
    with Timer(verbose=False) as t:
        stage_outputs_nh = template_maker.get_template(
            template_params_values_nh, return_intermediate=True
        )
    logging.info('==> elapsed time to get template: %s sec' % t.secs)

    logging.info('inverted...')
    with Timer(verbose=False) as t:
        stage_outputs_ih = template_maker.get_template(
            template_params_values_ih, return_intermediate=True
        )
    logging.info('==> elapsed time to get template: %s sec' % t.secs)

    logging.info('no osc...')
    with Timer(verbose=False) as t:
        final_event_rate_no_osc = template_maker.get_template(
            template_params_values_nh, no_osc=True,
        )
    logging.info('==> elapsed time to get template: %s sec' % t.secs)

    logging.info('Saving file to %s' % args.outfile)
    to_file(stage_outputs_nh, args.outfile)

    if args.plot:
        import os
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from pisa.utils import flavInt
        from pisa.utils import plot
        for k in sorted(stage_outputs_nh['final_event_rate'].keys()):
            if k == 'params':
                continue
            evtrt_nh = stage_outputs_nh['final_event_rate'][k]
            evtrt_ih = stage_outputs_ih['final_event_rate'][k]
            dist_map = plot.distinguishability_map(evtrt_ih, evtrt_nh)
            if k == 'trck':
                clim = (-0.21, 0.21)
            else:
                clim = (-0.27, 0.27)

            f = plt.figure(figsize=(24, 5), dpi=50)
            ax = f.add_subplot(131)
            plot.show_map(evtrt_nh, title=k, log=True,
                          cmap=mpl.cm.hot)

            ax = f.add_subplot(132)
            plot.show_map(evtrt_ih, log=True,
                          cmap=mpl.cm.hot)

            ax = f.add_subplot(133)
            plot.show_map(dist_map, cmap=mpl.cm.seismic)
            ax.get_children()[0].set_clim(clim)

        plt.draw()
        plt.show()
