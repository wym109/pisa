#!/usr/bin/env python

# authors: J.L. Lanfranchi, P.Eller
# date:    March 20, 2016

"""
Profile LLH Analysis

"""


from pisa.analysis.analysis import Analysis
from pisa.core.distribution_maker import DistributionMaker
from pisa.utils.log import logging, set_verbosity
from pisa.utils.random_numbers import get_random_state


class ProfileLLHAnalysis(Analysis):
    """
    Parameters
    ----------
    data_maker : DistributionMaker
        Generates a (pseudo)data distribution. Parameters used for generating
        this distribution are not modified from their injected values during
        the analysis, although added fluctuations may be regenerated if
        `data_fluctuations='fluctuated'` is specified (see below for more
        explanation).
    template_maker : DistributionMaker
        Provides output templates to compare against the data-like template.
        The `template_maker` provides the interface to those parameters
        that can be modifed or studied for their effects during the analysis
        process.
    metric : str
        What metric to be used for likelihood calculations / optimization
        llh or chi2
    """
    def __init__(self, data_maker, template_maker, metric, blind=False):
        assert isinstance(data_maker, DistributionMaker)
        assert isinstance(template_maker, DistributionMaker)
        self.data_maker = data_maker
        self.template_maker = template_maker
        assert isinstance(metric, basestring)
        self.metric = metric.lower()
        self.minimizer_settings = None
        self.blind = blind

        # Generate distribution
        self.asimov = self.data_maker.get_outputs()
        self.pseudodata_method = None
        self.pseudodata = None
        self.n_minimizer_calls = 0

    def generate_psudodata(self):
        if self.pseudodata_method == 'asimov':
            self.pseudodata = self.asimov
        elif self.pseudodata_method == 'poisson':
            self.pseudodata = self.asimov.fluctuate('poisson')
        else:
            raise Exception('unknown method %s'%method)

    def __init__(self, data_maker, template_maker, metric):
        assert isinstance(data_maker, DistributionMaker)
        assert isinstance(template_maker, DistributionMaker)

        self.data_maker = data_maker
        """DistributionMaker object for making data distributions"""

        self.template_maker = template_maker
        """DistributionMaker object for making template distributions to be fit
        to the data distribution"""

        assert isinstance(metric, basestring)
        self.metric = metric.lower()
        self._minimizer_settings = None

        # Generate distribution
        self.asimov = self.data_maker.get_outputs()
        self.pseudodata = None
        self.n_minimizer_calls = 0

    def profile_llh(self, param_name, values):
        """Run profile log likelihood for a single parameter.

        Parameters
        ----------
        param_name : string
            Parameter for which to run the profile log likelihood.

        values : sequence of float (?)
            Values at which to fix parameter in conditional llh

        Returns
        -------
        list of [condMLEs, globMLE]

        Notes
        -----

        Examples
        --------

        """
        # run numerator (conditional MLE)
        logging.info('fixing param %s'%param_name)
        self.template_maker.params.fix(param_name)
        condMLEs = {}
        for value in values:
            test = template_maker.params[param_name]
            test.value = value
            template_maker.update_params(test)
            condMLE = self.find_best_fit()
            condMLE[param_name] = self.template_maker.params[param_name].value
            append_results(condMLEs, condMLE)
            # report MLEs and LLH
            # also add the fixed param
        # run denominator (global MLE)
        ravel_results(condMLEs)
        logging.info('unfixing param %s'%param_name)
        self.template_maker.params.unfix(param_name)
        globMLE = self.find_best_fit()
        # report MLEs and LLH
        return [condMLEs, globMLE]

    def generate_psudodata(self, method):
        if method == 'asimov':
            self.pseudodata = self.asimov
        elif method == 'poisson':
            self.pseudodata = self.asimov.fluctuate('poisson',
                                                    random_state=random_state)
        else:
            raise ValueError('Unknown `method` "%s"' %method)
        return self.pseudodata

    @property
    def minimizer_settings(self):
        return self._minimizer_settings

    @minimizer_settings.setter
    def minimizer_settings(self, settings):
        self._minimizer_settings = settings

    @property
    def metric(self):
        return self._metric

    @metric.setter
    def metric(self, metric):
        assert isinstance(metric, basestring)
        metric = metric.lower().strip()
        # TODO: validation values here
        self._metric = metric


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    import numpy as np

    from pisa import ureg, Q_
    from pisa.utils.fileio import from_file, to_file
    from pisa.utils.config_parser import parse_pipeline_config
    from pisa.utils.format import append_results, ravel_results

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-d', '--data-settings', type=str,
        metavar='configfile', default=None,
        help='''Settings for the generation of "data" distributions; repeat
        this argument to specify multiple pipelines. If omitted, the same
        settings as specified for --template-settings are used to generate data
        distributions.'''
    )
    parser.add_argument(
        '-t', '--template-settings',
        metavar='CONFIGFILE', required=True,
        action='append',
        help='''Settings for generating template distributions; repeat
        this option to define multiple pipelines.'''
    )
    parser.add_argument(
        '-o', '--outfile', metavar='FILE',
        type=str, action='store', default='out.json',
        help='file to store the output'
    )
    parser.add_argument(
        '-n', '--num-trials', type=int, default=1,
        help='number of trials'
    )
    parser.add_argument(
        '-m', '--minimizer-settings', type=str,
        metavar='JSONFILE', required=True,
        help='''Settings related to the minimizer used in the LLR analysis.'''
    )
    parser.add_argument(
        '--metric', type=str,
        choices=['llh', 'chi2', 'conv_llh'], required=True,
        help='''Settings related to the minimizer used in the LLR analysis.'''
    )
    parser.add_argument(
        '-v', action='count', default=None,
        help='set verbosity level'
    )
    args = parser.parse_args()

    set_verbosity(args.v)

    if args.data_settings is None:
        data_settings = args.template_settings
    else:
        data_settings = args.data_settings

    data_maker = DistributionMaker(data_settings)
    template_maker = DistributionMaker(args.template_settings)

    # select inverted mass ordering
    #template_maker_settings.set('stage:osc', 'param_selector', 'ih')
    #template_maker_configurator = parse_pipeline_config(template_maker_settings)
    #template_maker_IO = DistributionMaker(template_maker_configurator)

    analysis = ProfileLLHAnalysis(data_maker=data_maker,
                                  template_maker=template_maker,
                                  metric=args.metric)

    analysis.minimizer_settings = from_file(args.minimizer_settings)

    results = []
    for i in range(args.num_trials):
        logging.info('Running trial %i'%i)
        np.random.seed()
        #analysis.generate_pseudodata('poisson')
        analysis.generate_pseudodata('asimov')

        # LLR:
        #append_results(results, analysis.llr(template_maker, template_maker_IO))

        # profile LLH:
        results.append(analysis.profile_llh(
            'nutau_cc_norm', np.linspace(0, 2, 21)*ureg.dimensionless
        ))
        #results.append(analysis.profile_llh('nutau_cc_norm', [0.]*ureg.dimensionless))

    to_file(results, args.outfile)
    logging.info('Done.')

if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    import numpy as np
    from pisa import ureg, Q_

    from pisa.utils.fileio import from_file, to_file
    from pisa.utils.parse_config import parse_config
    from pisa.utils.format import append_results, ravel_results

    parser = ArgumentParser()
    parser.add_argument('-d', '--data-settings', type=str,
                        metavar='configfile', default=None,
                        help='settings for the generation of "data"')
    parser.add_argument('-t', '--template-settings',
                        metavar='configfile', required=True,
                        action='append',
                        help='''settings for the template generation''')
    parser.add_argument('-o', '--outfile', metavar='FILE',
                        type=str, action='store', default='out.json',
                        help='file to store the output')
    parser.add_argument('-v', action='count', default=None,
                        help='set verbosity level')
    parser.add_argument('-n', '--num-trials', type=int, default=1,
                        help='number of trials')
    parser.add_argument('-b', '--blind', action='store_true',
                        help='run blindly i.e. only reporting goodness of fit, no parameter values')
    parser.add_argument('-m', '--minimizer-settings', type=str,
                        metavar='JSONFILE', required=True,
                        help='''Settings related to the optimizer used in the
                        LLR analysis.''')
    parser.add_argument('-fp', '--fix-param', type=str, default='',
                        help='''fix parameter''')
    parser.add_argument('-pd', '--pseudo-data', type=str, default='poisson',
                        choices=['poisson', 'asimov'], 
                        help='''Mode for pseudo data sampling''')
    parser.add_argument('--metric', type=str,
                        choices=['llh', 'chi2', 'conv_llh', 'mod_chi2'], required=True,
                        help='''Settings related to the optimizer used in the
                        LLR analysis.''')
    parser.add_argument('--mode', type=str,
                        choices=['H0', 'scan11', 'scan21'], default='H0',
                        help='''just run significance or whole scan''')
    args = parser.parse_args()

    set_verbosity(args.v)

    if args.data_settings is None:
        data_settings = args.template_settings
    else:
        data_settings = args.data_settings

    data_maker = DistributionMaker(data_settings)
    template_maker = DistributionMaker(args.template_settings)

    if not args.fix_param == '':
        template_maker.params.fix(args.fix_param)

    profile_llh = profile_llh(data_maker=data_maker,
                        template_maker=template_maker,
                        metric=args.metric,
                        blind=args.blind)

    profile_llh.minimizer_settings = from_file(args.minimizer_settings)
    profile_llh.pseudodata_method = args.pseudo_data

    results = []

    for i in range(args.num_trials):
        logging.info('Running trial %i'%i)
        np.random.seed()
        profile_llh.generate_psudodata()

        if args.mode == 'H0':
            results.append(profile_llh.profile('nutau_cc_norm',[0.]*ureg.dimensionless))
        elif args.mode == 'scan11':
            results.append(profile_llh.profile('nutau_cc_norm',np.linspace(0,2,11)*ureg.dimensionless))
        elif args.mode == 'scan21':
            results.append(profile_llh.profile('nutau_cc_norm',np.linspace(0,2,21)*ureg.dimensionless))

    to_file(results, args.outfile)
    logging.info('Done.')
