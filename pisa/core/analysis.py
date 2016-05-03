#! /usr/bin/env python
# authors: J.Lanfranchi/P.Eller
# date:    March 20, 2016

from collections import Sequence

from pisa.core.template_maker import TemplateMaker
from pisa.utils.log import logging, set_verbosity
from pisa.utils.fileio import from_file
import scipy.optimize as opt


class Analysis(object):
    '''provide scan methods, or methods to interact with a minimizer,
       
    args:
        - data_maker TemplateMaker object
        - template_maker TemplateMaker object

    data_maker is used to derive a data-like template, that is not modified
    during the analysis

    template_maker provides output templates, and e.g. free parameters, that can
    be minimized using a given metric, or scanned through, etc...
    '''
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

    def optimize_llh(self, valuelist):
        data = self.data_maker.compute_outputs()
        self.template_maker.set_rescaled_free_params(valuelist)
        template = self.template_maker.compute_outputs()
        llh = -data.total_llh(template)
        # ToDo: llh from priors
        llh -= template_maker.params.free.priors_llh
        print 'llh at %s'%llh
        return llh

    def run_l_bfgs(self, minimizer_settings):
        x0 = self.publish_to_minimizer()
        # bfgs steps outside of given bounds by 1 epsilon to evaluate gradients
        epsilon = minimizer_settings['options']['value']['epsilon']
        bounds = [(0+epsilon,1-epsilon)]*len(x0)
        a = opt.fmin_l_bfgs_b(func=self.optimize_llh,
                              x0=x0,
                              bounds=bounds,
                              **minimizer_settings['options']['value'])

        print 'found best fit parameters:'
        print self.template_maker.params.free
        return a

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
    parser.add_argument('-m','--minimizer_settings',type=str,
                        metavar='JSONFILE', required = True,
                        help='''Settings related to the optimizer used in the LLR analysis.''')
    args = parser.parse_args()

    set_verbosity(args.v)

    data_settings = from_file(args.data_settings)
    data_cfg = parse_config(data_settings)
    data_maker = TemplateMaker([data_cfg])
    test = data_maker.params['test']
    #test.value /=2.
    test.value *= 1.2
    data_maker.update_params(test)

    template_settings = from_file(args.template_settings)
    template_cfg = parse_config(template_settings)
    template_maker = TemplateMaker([template_cfg])

    ana = Analysis(data_maker, template_maker)

    #print ''
    #logging.info(
    #    'Sweeping over 3 values of `test` (should affect both flux and osc)'
    #)
    #print ''
    #ana.scan('test', np.linspace(0, 5, 3)*ureg.foot, metric='llh')

    #print ''
    #logging.info(
    #    'Sweeping over 3 values of `atm_delta_index` (should only affect flux)'
    #)
    #print ''
    #ana.scan('atm_delta_index', np.linspace(-0.2, 0.2, 3)*ureg.dimensionless,
    #         metric='llh')

    #print ''
    #logging.info(
    #    'Sweeping over 3 values of `theta23` (should only affect osc)'
    #)
    #print ''
    #ana.scan('theta23', np.linspace(40, 45, 3)*ureg.degrees,
    #         metric='llh')
    #print ''
    #logging.info(
    #    'Sweeping over 3  times the same value of "test"'
    #)
    #print ''
    #ana.scan('test', np.array([2,2,2])*ureg.meter,
    #         metric='llh')
 
    minimizer_settings  = from_file(args.minimizer_settings)

    ana.run_l_bfgs(minimizer_settings)
