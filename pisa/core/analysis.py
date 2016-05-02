#! /usr/bin/env python
# authors: J.Lanfranchi/P.Eller
# date:    March 20, 2016

from collections import Sequence

from pisa.core.template_maker import TemplateMaker
from pisa.utils.log import logging, set_verbosity


class Analysis(object):
    def __init__(self, data_maker, template_maker):
        self.data_maker = data_maker
        self.template_maker = template_maker

    def scan(self, pname, values, metric='llh'):
        metric_vals = []
        data = self.data_maker.compute_outputs()
        for val in values:
            fp = self.template_maker.params.free
            fp[pname].value = val
            self.template_maker.update_params(fp)
            template = self.template_maker.compute_outputs()
            metric_vals.append(data.total_llh(template))
        return metric_vals

    def publish_to_minimizer(self):
        return self.template_maker.free_params_rescaled_values

    def update_from_minimizer(self, valuelist):
        self.template_maker.set_rescaled_free_params(valuelist)


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
    data_cfg = parse_config(data_settings)
    data_maker = TemplateMaker([data_cfg])

    template_settings = from_file(args.template_settings)
    template_cfg = parse_config(template_settings)
    template_maker = TemplateMaker([template_cfg])

    ana = Analysis(data_maker, template_maker)

    print ''
    logging.info(
        'Sweeping over 3 values of `test` (should affect both flux and osc)'
    )
    print ''
    ana.scan('test', np.linspace(0, 5, 3)*ureg.foot, metric='llh')

    print ''
    logging.info(
        'Sweeping over 3 values of `atm_delta_index` (should only affect flux)'
    )
    print ''
    ana.scan('atm_delta_index', np.linspace(-0.2, 0.2, 3)*ureg.dimensionless,
             metric='llh')

    print ''
    logging.info(
        'Sweeping over 3 values of `theta23` (should only affect osc)'
    )
    print ''
    ana.scan('theta23', np.linspace(40, 45, 3)*ureg.degrees,
             metric='llh')
    vals = ana.publish_to_minimizer()
    vals[1]*=0.9
    ana.update_from_minimizer(vals)
