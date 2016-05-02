#! /usr/bin/env python
# authors: J.Lanfranchi/P.Eller
# date:    March 20, 2016

from collections import Sequence

from pisa.core.pipeline import Pipeline
from pisa.utils.log import logging, set_verbosity


class Analysis(object):
    def __init__(self, pipeline_configs):
        if isinstance(pipeline_configs, basestring) \
                or not hasattr(pipeline_configs, '__iter__'):
            pipeline_configs = [pipeline_configs]
        self.pipelines = []
        for pipeline_config in pipeline_configs:
            pipeline_settings = parse_config(pipeline_config)
            self.pipelines.append(Pipeline(pipeline_settings))

    def __iter__(self):
        return iter(self.pipelines)

    def __getattr__(self, attr):
        for pipeline in self:
            if pipeline.name == attr:
                return pipeline

    def scan(self, pname, values, metric='llh'):
        metric_vals = []
        m0 = self.pipelines[0].compute_outputs()
        for val in values:
            fp = self.pipelines[1].params.free
            fp[pname].value = val
            self.pipelines[1].update_params(fp)
            m1 = self.pipelines[1].compute_outputs()
            metric_vals.append(m0.total_llh(m1))
        return metric_vals


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    import numpy as np
    import pint
    ureg = pint.UnitRegistry()
    from pisa.utils.fileio import from_file, to_file
    from pisa.utils.parse_config import parse_config

    parser = ArgumentParser()
    parser.add_argument('-d','--data-settings', type=str,
                        metavar='configfile', required=True,
                        help='settings for the generation of "data"')
    parser.add_argument('-t','--template-settings', type=str,
                        metavar='configfile', required=True,
                        help='settings for the generation of templates')
    parser.add_argument('--outfile', metavar='FILE',
                        type=str, action='store', default="out.json",
                        help='file to store the output')
    parser.add_argument('-v', action='count', default=None,
                        help='set verbosity level')
    args = parser.parse_args()

    set_verbosity(args.v)

    data_settings = from_file(args.data_settings)
    data_settings.set('stage:flux', 'param.test.fixed', 'True')
    template_settings = from_file(args.template_settings)

    ana = Analysis([data_settings, template_settings])
    logging.info('sweeping over 5 values of `test` (should affect both flux'
                 ' and osc)')
    ana.scan('test', np.arange(0,5,1)*ureg.foot, metric='llh')
    logging.info('sweeping over 5 values of `atm_delta_index` (should affect'
                 ' osc)')
    ana.scan('atm_delta_index', np.arange(-0.2,0.2,1)*ureg.dimensionless,
             metric='llh')
