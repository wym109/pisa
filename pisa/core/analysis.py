#! /usr/bin/env python
# authors: J.Lanfranchi/P.Eller
# date:    March 20, 2016

from collections import Sequence
import sys
import scipy.optimize as opt
import time

from pisa.core.distribution_maker import DistributionMaker
from pisa.utils.fileio import from_file
from pisa.utils.log import logging, set_verbosity
from pisa import ureg, Q_

class Analysis(object):
    """Major tools for performing "canonical" IceCube/DeepCore/PINGU analyses.

    * "Data" distribution creation (via passed `data_maker` object)
    * Template distribution creation (via passed `distribution_maker` object)

    * Minimizer Interface (via method `_minimizer_callable`)
        Interfaces to a minimizer for modifying the free parameters of the
        `distribution_maker` to fit its output (as closely as possible) to the
        data distribution is provided. See [minimizer_settings] for

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

    Attributes
    ----------
    data_maker
    template_maker

    Methods
    -------
    optimize
    scan
    profile : profile a given param (profile LLH)
    _minimizer_callable : private method indended to be called by a minimizer

    """
    def __init__(self, data_maker, template_maker, metric):
        assert isinstance(data_maker, DistributionMaker)
        assert isinstance(template_maker, DistributionMaker)
        self.data_maker = data_maker
        self.template_maker = template_maker
        assert isinstance(metric, basestring)
        self.metric = metric.lower()
        self.minimizer_settings = None

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

    # TODO: move the complexity of defining a scan into a class with various
    # factory methods, and just pass that class to the scan method; we will
    # surely want to use scanning over parameters in more general ways, too:
    #   set (some) fixed params, then run (minimizer, scan, etc.) on free params
    #   set (some free or fixed) params, then check metric
    # where the setting of the params is done for some number of values.
    def scan(self, param_names=None, steps=None, values=None,
             outer=False):
        """Set template maker parameters named by `param_names` according to
        either values specified by `values` or number of steps specified by
        `steps`, and return the `metric` indicating how well each template
        matches the data distribution.

        Some flexibility in how the user can specify `values` is allowed, based
        upon the shapes of `param_names` and `values` and how the `outer` flag
        is set.

        Either `values` or `steps` must be specified, but not both.

        Parameters
        ----------
        param_names : None, string, or sequence of strings
            If None, assume all parameters are to be scanned; otherwise,
            specifies only the name or names of parameters to be scanned.

        steps : None, integer, or sequence of strings
            Number of steps to take within the allowed range of the parameter
            (or parameters). Value(s) specified for `steps` must be >= 2. Note
            that the endpoints of the range are always included, and numbers of
            steps higher than 2 fill in between the endpoints.
            * If integer...
                Take this many steps for each specified parameter.
            * If sequence of integers...
                Take the coresponding number of steps within the allowed range
                for each specified parameter.

        values : None, scalar, sequence of scalars, or sequence-of-sequences
          * If scalar...
                Set this value for the (one) param name in `param_names`.
          * If sequence of scalars...
              * if len(param_names) is 1, set its value to each number in the
                sequence.
              * otherwise, set each param in param_names to the corresponding
                value in `values`. There must be the same number of param names
                as values.
          * If sequence of (sequences or iterables)...
              * Each param name corresponds to one of the inner sequences, in
                the order that the param names are specified.
              * If `outer` is False, all inner sequences must have the same
                length, and there will be one template generated for each set
                of values across the inner sequences. In other words, there will
                be a total of len(inner sequenc) templates generated.
              * If `outer` is True, the lengths of inner sequences needn't be
                the same. This takes the outer product of the passed sequences
                to arrive at the permutations of the parameter values that will
                be used to produce templates (essentially nested loops over
                each parameter). E.g., if two params are scanned, for each
                value of the first param's inner sequence, a template is
                produced for every value of the second param's inner sequence.
                In total, there will be len(inner seq0) * len(inner seq1) * ...
                templates produced.

        outer : bool
            If set to True and a sequence of sequences is passed for `values`,
            the points scanned are the *outer product* of the inner sequences.
            See `values` for a more detailed explanation.

        """
        assert not (steps is not None and values is not None)
        if isinstance(param_names, basestring):
            param_names = [param_names]

        if values is not None and np.isscalar(values):
            values = np.array([values])
            nparams = len(param_names)

        metric_vals = []
        for val in values:
            fp = self.template_maker.params.free
            fp[param_names].value = val
            self.template_maker.update_params(fp)
            template = self.template_maker.get_outputs()
            metric_vals.append(self.pseudodata.metric_total(expected_values=template,
                                                      metric=self.metric))
        return metric_vals

    def _minimizer_callable(self, scaled_param_vals, pprint=True):
        """The callable interface to provide to simple minimzers e.g. those in
        scipy.optimize.

        This should *not* in general be called by users, as `scaled_param_vals`
        are stripped of their units and rescaled to [0, 1], and hence some
        validation of inputs is bypassed by this method.

        Parameters
        ----------
        scaled_param_vals : sequence of floats
            If called from a scipy.optimize minimizer, this sequence is provied
            by the minimizer itself. These values are all expected to be in the
            range [0, 1] and be simple floats (no units or uncertainties
            attached, etc.). Rescaling the parameter values to their
            original (physical) ranges (including units) is handled within this
            method.

        pprint
            Displays a single-line that updates live (assuming the entire line
            fits the width of your TTY).

        """
        sign = +1
        if self.metric in ['llh', 'conv_llh']:
            # Want to *maximize* log-likelihood but we're using a minimizer
            sign = -1

        self.template_maker.params.free._rescaled_values = scaled_param_vals

        template = self.template_maker.get_outputs()

        # Assess the fit of the template to the data distribution, and negate
        # if necessary
        metric_val = (
            self.pseudodata.metric_total(expected_values=template, metric=self.metric)
            + template_maker.params.priors_penalty(metric=self.metric)
        )

        # Report status of metric & params
        msg = '%s=%.6e | %s' %(self.metric, metric_val,
                               self.template_maker.params.free)
        if pprint:
            sys.stdout.write(msg)
            sys.stdout.flush()
            sys.stdout.write('\b' * len(msg))
        else:
            logging.debug(msg)

        self.n_minimizer_calls += 1
        return sign*metric_val

    def run_minimizer(self, pprint=True, skip=False):
        # Get initial values
        x0 = self.template_maker.params.free._rescaled_values

        # bfgs steps outside of given bounds by 1 epsilon to evaluate gradients
        try:
            epsilon = self.minimizer_settings['options']['value']['eps']
        except:
            epsilon = self.minimizer_settings['options']['value']['epsilon']
        bounds = [(0+epsilon, 1-epsilon)]*len(x0)
        logging.info('running the %s optimizer'%self.minimizer_settings['method']['value'])

        # Using scipy.opt.minimize allows a whole host of minimisers to be used
        # This set by the method value in your minimiser settings file
        self.n_minimizer_calls = 0
        if skip:
            best_fit_vals = x0
            metric_val = self._minimizer_callable(x0, False)
            dict_flags = {'warnflag':0, 'task':'skip', 'funcalls':0, 'nit':0}
        else:
            start_t = time.time()
            minim_result = opt.minimize(fun=self._minimizer_callable,
                                        x0=x0,
                                        args=(pprint,),
                                        bounds=bounds,
                                        method = self.minimizer_settings['method']['value'],
                                        options = self.minimizer_settings['options']['value'])
            
            end_t = time.time()
            if pprint:
                # clear the line
                print ''
            print '\naverage template generation time during minimizer run: %.4f ms'%((end_t - start_t) * 1000./self.n_minimizer_calls)
            best_fit_vals = minim_result.x
            metric_val = minim_result.fun
            dict_flags = {}
            dict_flags['warnflag'] = minim_result.status
            dict_flags['task'] = minim_result.message
            if minim_result.has_key('jac'):
                dict_flags['grad'] = minim_result.jac
            dict_flags['funcalls'] = minim_result.nfev
            dict_flags['nit'] = minim_result.nit
            if dict_flags['warnflag'] > 0:
                logging.warning(str(dict_flags))

        return best_fit_vals, metric_val, dict_flags

    def find_best_fit(self, check_octant=True, pprint=True, skip=False):
        """ find best fit points (max likelihood) for the free parameters and
            return likelihood + found parameter values.
        """
        # Reset free parameters to nominal values
        logging.info('resetting params')
        self.template_maker.params.reset_free()

        best_fit_vals, metric_val, dict_flags = self.run_minimizer(pprint=pprint, skip=skip)
        best_fit = {}
        best_fit[self.metric] = metric_val
        best_fit['warnflag'] = dict_flags['warnflag']
        for pname in self.template_maker.params.free.names:
            best_fit[pname] = self.template_maker.params[pname].value

        # decide wether fit for second octant is necessary
        if 'theta23' in self.template_maker.params.free.names and not skip:
            if check_octant:
                logging.info('checking other octant of theta23')
                self.template_maker.params.reset_free()
                # changing to other octant
                theta23 = self.template_maker.params['theta23']
                inflection_point = 45 * ureg.degree
                theta23.value = 2*inflection_point.to(theta23.value.units) - theta23.value
                self.template_maker.update_params(theta23)
                best_fit_vals, metric_val, dict_flags = self.run_minimizer(pprint=pprint)

                # compare results a and b, take one with lower llh
                if metric_val < best_fit[self.metric]:
                    # accept these values
                    logging.info('Accepting other octant fit')
                    best_fit[self.metric] = metric_val
                    best_fit['warnflag'] = dict_flags['warnflag']
                    for pname in self.template_maker.params.free.names:
                        best_fit[pname] = self.template_maker.params[pname].value
                    
                else:
                    logging.info('Accepting initial octant fit')

        return best_fit

    def profile(self, p_name, values):
        """Run profile log likelihood method for param `p_name`.

        Parameters
        ----------
        p_name
        values to fix parameter to in conditional llh

        """
        # run numerator (conditional MLE)
        logging.info('fixing param %s'%p_name)
        self.template_maker.params.fix(p_name)
        condMLEs = {}
        for value in values:
            test = template_maker.params[p_name]
            test.value = value
            template_maker.update_params(test)
            condMLE = self.find_best_fit()
            condMLE[p_name] = self.template_maker.params[p_name].value
            append_results(condMLEs,condMLE)
            # report MLEs and LLH
            # also add the fixed param
        ravel_results(condMLEs)
        # run denominator (global MLE)
        logging.info('unfixing param %s'%p_name)
        self.template_maker.params.unfix(p_name)
        if self.pseudodata_method == 'asimov' and self.metric in ['llh', 'chi2', 'mod_chi2']:
            # in these cases we can skip minimization
            skip = True
        else:
            skip = False
        globMLE = self.find_best_fit(skip=skip)
        # report MLEs and LLH
        return [condMLEs, globMLE]

    def llr(self, template_makerA, template_makerB):
        """ Run loglikelihood ratio for two different template makers A and B
        """
        results = []
        for template_maker in [template_makerA, template_makerB]:
            self.template_maker = template_maker
            results.append(self.find_best_fit())
        return results


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

    analysis = Analysis(data_maker=data_maker,
                        template_maker=template_maker,
                        metric=args.metric)

    analysis.minimizer_settings = from_file(args.minimizer_settings)
    analysis.pseudodata_method = args.pseudo_data

    results = []

    for i in range(args.num_trials):
        logging.info('Running trial %i'%i)
        np.random.seed()
        analysis.generate_psudodata()

        if args.mode == 'H0':
            results.append(analysis.profile('nutau_cc_norm',[0.]*ureg.dimensionless))
        elif args.mode == 'scan11':
            results.append(analysis.profile('nutau_cc_norm',np.linspace(0,2,11)*ureg.dimensionless))
        elif args.mode == 'scan21':
            results.append(analysis.profile('nutau_cc_norm',np.linspace(0,2,21)*ureg.dimensionless))

    to_file(results, args.outfile)
    logging.info('Done.')
