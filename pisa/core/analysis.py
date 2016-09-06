#! /usr/bin/env python
# authors: J.L. Lanfranchi, P.Eller, and S. Wren
# email:   jll1062+pisa@phys.psu.edu
# date:    March 20, 2016


from collections import Sequence
import sys
import time

import scipy.optimize as opt

from pisa.core.distribution_maker import DistributionMaker
from pisa.utils.fileio import from_file
from pisa.utils.log import logging, set_verbosity
from pisa.utils.random_numbers import get_random_state
from pisa import ureg, Q_


class Counter(object):
    def __init__(self, i=0):
        self._i = c0

    def __str__(self):
        return str(self._i)

    def __iadd__(self, inc):
        self._i += inc

    def reset(self):
        self._i = 0

    @property
    def count(self):
        return self._i


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
        Provides output template to compare against the data-like template.
        The `template_maker` provides the interface to those parameters
        that can be modifed or studied for their effects during the analysis
        process.

    metric : str
        What metric to be used for likelihood calculations / optimization
        llh or chi2

    """

    METRICS_TO_MAXIMIZE = ['llh', 'conv_llh']

    def __init__(self):
        #assert isinstance(data_maker, DistributionMaker)
        #assert isinstance(template_maker, DistributionMaker)

        #self.data_maker = data_maker
        #"""DistributionMaker object for making data distributions"""

        #self.template_maker = template_maker
        #"""DistributionMaker object for making template distributions to be fit
        #to the data distribution"""

        #self.metric = metric
        #if isinstance(minimizer_settings, basestring):
        #    self.minimizer_settings = from_file(minimizer_settings)
        #elif isinstance(minimizer_settings, Mapping):
        #    self.minimizer_settings = minimizer_settings
        #else:
        #    raise ValueError('Minimizer settings expected either as path to '
        #                     'file or as dictionary read from file')

        ## Generate distribution
        #self.asimov = self.data_maker.get_outputs()
        #self.pseudodata = None
        #self.n_minimizer_calls = 0

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

    def generate_psudodata(self, method):
        if method == 'asimov':
            self.pseudodata = self.asimov
        elif method == 'poisson':
            self.pseudodata = self.asimov.fluctuate('poisson',
                                                    random_state=random_state)
        else:
            raise ValueError('Unknown `method` "%s"' %method)
        return self.pseudodata

    # TODO: move the complexity of defining a scan into a class with various
    # factory methods, and just pass that class to the scan method; we will
    # surely want to use scanning over parameters in more general ways, too:
    # * set (some) fixed params, then run (minimizer, scan, etc.) on free
    #   params
    # * set (some free or fixed) params, then check metric
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
                of values across the inner sequences. In other words, there
                will be a total of len(inner sequence) templates generated.
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
            metric_vals.append(
                self.pseudodata.metric_total(
                    expected_values=template, metric=self.metric
                )
            )
        return metric_vals

    def _minimizer_callable(self, scaled_param_vals, template_maker, data,
                            metric, counter, pprint):
        """Simple callback for use by scipy.optimize minimizers.

        This should *not* in general be called by users, as `scaled_param_vals`
        are stripped of their units and scaled to the range [0, 1], and hence
        some validation of inputs is bypassed by this method.

        Parameters
        ----------
        scaled_param_vals : sequence of floats
            If called from a scipy.optimize minimizer, this sequence is
            provieded by the minimizer itself. These values are all expected to
            be in the range [0, 1] and be simple floats (no units or
            uncertainties attached, etc.). Rescaling the parameter values to
            their original (physical) ranges (including units) is handled
            within this method.

        template_maker : DistributionMaker
            Creates the per-bin expectation values per map (aka template) based
            on its param values. Free params in the `template_maker` are
            modified by the minimizer to achieve a "best" fit.

        data : MapSet
            Data distribution to be fit. Can be an actual-, Asimov-, or
            pseudo-data distribution (where the latter two are derived from
            simulation and so aren't technically "data").

        metric : str
            Metric by which to evaluate the fit. See Map

        counter : Counter
            Mutable object to keep track--outside this method--of the number of
            times this method is called.

        pprint : bool
            Displays a single-line that updates live (assuming the entire line
            fits the width of your TTY).

        """
        # Want to *maximize* e.g. log-likelihood but we're using a minimizer,
        # so flip sign of metric in those cases.
        sign = -1 if metric in self.METRICS_TO_MAXIMIZE else +1

        # Set param values from the scaled versions the minimizer works with
        template_maker.params.free._rescaled_values = scaled_param_vals

        # Get the template map set
        template = template_maker.get_outputs()

        # Assess the fit: whether the data came from the template
        metric_val = (
            data.metric_total(expected_values=template, metric=metric)
            + template_maker.params.priors_penalty(metric=metric)
        )

        # TODO: make this into a header line with param names and values
        # beneath updated, to save horizontal space (and easier to read/follow)

        # Report status of metric & params
        msg = '%s=%.6e | %s' %(metric, metric_val, template_maker.params.free)
        if pprint:
            sys.stdout.write(msg)
            sys.stdout.flush()
            sys.stdout.write('\b' * len(msg))
        else:
            logging.debug(msg)

        counter += 1

        return sign*metric_val

    # TODO: make this generic to any minimizer
    def fit_template_inner(self, template_maker, data, metric,
                           minimizer_settings, pprint=True):
        """Fitter "inner" loop: Run an arbitrary scipy minimizer to modify
        template until the data distribution is most likely to have come from
        that templated distribution.

        Note that an "outer" loop can handle discrete scanning over e.g. the
        octant for theta23; for each discrete point the "outer" loop can make a
        call to this "inner" loop.

        Parameters
        ----------
        template_maker : DistributionMaker or convertible thereto

        data : MapSet
            Data events distribution(s)

        metric : string

        minimizer_settings : string

        pprint : bool
            Whether to show live-update of minimizer progress.

        Returns
        -------

        """
        # Get starting free parameter values
        x0 = template_maker.params.free._rescaled_values

        # TODO: does this break if not using bfgs?

        # bfgs steps outside of given bounds by 1 epsilon to evaluate gradients
        minimizer_kind = minimizer_settings['options']['value']
        try:
            epsilon = minimizer_kind['eps']
        except KeyError:
            epsilon = minimizer_kind['epsilon']
        bounds = [(0+epsilon, 1-epsilon)]*len(x0)

        logging.info('Running the %s minimizer.'
                     %minimizer_settings['method']['value'])

        # Using scipy.opt.minimize allows a whole host of minimisers to be
        # used.
        counter = Counter()
        start_t = time.time()
        minim_result = opt.minimize(
            fun=self._minimizer_callable,
            x0=x0,
            args=(template_maker, data, metric, counter, pprint),
            bounds=bounds,
            method = minimizer_settings['method']['value'],
            options = minimizer_settings['options']['value']
        )
        end_t = time.time()
        if pprint:
            # clear the line
            sys.stdout.write('\n')
            sys.stdout.flush()
        logging.debug(
            'Average template generation time during minimizer run: %.4f ms'
            %((end_t - start_t) * 1000./counter.count)
        )

        # Will not assume that the minimizer left the template maker in the
        # minimized state, so set the values now (also does conversion of
        # values from [0,1] back to physical range)
        rescaled_pvals = minim_result.pop('x')
        template_maker.params._rescaled_values = rescaled_pvals

        # Get the best-fit metric value
        metric_val = minim_result.pop('fun')

        # Record minimizer metadata (all info besides 'x' and 'fun')
        metadata = OrderedDict()
        for k in sorted(minim_result.keys()):
            metadata[k] = minim_result[k]

        info = OrderedDict()
        info['metric'] = metric
        info['metric_val'] = metric_val
        info['params'] = deepcopy(template_maker.params)
        info['metadata'] = metadata

        return info

    def fit_template_outer(self, template_maker, data, metric,
                           minimizer_settings, check_octant=True, pprint=True):
        """Fitter "outer" loop: If `check_octant` is True, run
        `fit_template_inner` starting in each octant of theta23 (assuming that
        is a param in the `template_maker`). Otherwise, just run the inner
        method once.

        Parameters
        ----------
        template_maker : DistributionMaker or convertible thereto

        data : MapSet
            Data events distribution(s)

        metric : string

        minimizer_settings : string or dict

        pprint : bool
            Whether to show live-update of minimizer progress.

        Returns
        -------

        """
        # Reset free parameters to nominal values
        logging.trace('resetting params')
        template_maker.params.reset_free()

        best_fit_info = self.fit_template_inner(
            template_maker=template_maker,
            data=data,
            metric=metric,
            minimizer_settings=minimizer_settings,
            pprint=pprint
        )

        # Decide whether fit for other octant is necessary
        if 'theta23' in template_maker.params.free and check_octant:
            logging.debug('checking other octant of theta23')
            template_maker.params.reset_free()

            # Hop to other octant by reflecting about 45 deg
            theta23 = template_maker.params.theta23
            inflection_point = (45*ureg.deg).to(theta23.units)
            theta23.value = 2*inflection_point - theta23.value
            template_maker.update_params(theta23)

            # Re-run minimizer starting at new point
            new_fit_info = self.fit_template_inner(
                template_maker=template_maker,
                data=data,
                metric=metric,
                minimizer_settings=minimizer_settings,
                pprint=pprint
            )

            # Take the one with the best fit
            if metric in self.METRICS_TO_MAXIMIZE:
                it_got_better = new_fit_info['metric_val'] > \
                        best_fit_info['metric_val']
            else:
                it_got_better = new_fit_info['metric_val'] < \
                        best_fit_info['metric_val']

            if it_got_better:
                logging.debug('Accepting other-octant fit')
                best_fit_info = new_fit_info
            else:
                logging.debug('Accepting initial-octant fit')

        return best_fit_info

    def fit_hypo(self, data, hypo_maker, metric, param_selections=None):
        """
        Parameters
        ----------
        data : MapSet
        hypo_maker : DistributionMaker
        metric : None or string
        param_selections : None, string, or sequence of strings

        Returns
        -------
        FitObj containing llh

        """
        data.

    def compare_hypos(self, data, null_hypo_maker, alt_hypo_maker, metric,
                      null_hypo_param_selections=None,
                      alt_hypo_param_selections=None):
        """
        Parameters
        ----------
        data : MapSet

        null_hypo_maker : DistributionMaker

        alt_hypo_maker : DistributionMaker

        metric : None or string

        null_hypo_param_selections : None, string, or sequence of strings

        alt_hypo_param_selections : None, string, or sequence of strings


        Returns
        -------
        delta_metric, null_hypo_fit, alt_hypo_fit

        """
        null_hypo_fit = self.fit_hypo(
            data=data, hypo_maker=null_hypo_maker, metric=metric,
            param_selections=null_hypo_param_selections
        )
        alt_hypo_fit = self.fit_hypo(
            data=data, hypo_maker=alt_hypo_maker, metric=metric,
            param_selections=alt_hypo_param_selections
        )
        delta_metric = alt_hypo_fit.metric - null_hypo_fit.metric
        return delta_metric, null_hypo_fit, alt_hypo_fit

    def llr(self, data_maker, null_hypo_maker, alt_hypo_maker,
            data_name='data', null_hypo_name='null_hypo',
            alt_hypo_name='alt_hypo',
            metric='llh',
            data_param_selections=None, null_hypo_param_selections=None,
            alt_hypo_param_selections=None,
            fluctuate_data=False, fluctuate_fid_data=False,
            n_data_trials=1, n_fid_data_trials=1,
            data_start_ind=0, fid_data_start_ind=0):
        """Perform LLR analysis to determine the significance for data to have
        come from physics decribed by an alternative hypothesis versus physics
        decribed a null hypothesis.

        Note that duplicated `*_maker` specifications are _not_ instantiated
        separately, but instead are re-used for all duplicate definitions.
        `*_param_selections` allows for this reuse, whereby sets of parameters
        suffixed with the corresponding param_selectors can be switched among
        to simulate different physics using the same DistributionMaker (e.g.,
        switching between null and alt hypotheses).


        Parameters
        ----------
        data_maker : DistributionMaker or instantiable thereto

        null_hypo_maker : DistributionMaker or instantiable thereto

        alt_hypo_maker : DistributionMaker or instantiable thereto

        data_name : string

        null_hypo_name : string

        alt_hypo_name : string

        metric : string

        data_param_selections : None, string, or sequence of strings

        null_hypo_param_selections : None, string, or sequence of strings

        alt_hypo_param_selections : None, string, or sequence of strings

        fluctuate_data : bool

        fluctuate_fid_data : bool

        n_data_trials : int >= 1

        n_fid_data_trials : int >= 1

        data_start_ind : int >= 0

        fid_data_start_ind : int >= 0


        Returns
        -------
        TODO

        Notes
        -----
        LLR analysis is a very thorough (though computationally expensive)
        method to compare discrete hypotheses. In general, a total of

            n_data_trials * (2 + 4*n_fid_data_trials)

        fits must be performed (and note that for each fit, many templates
        (typically dozens or even hunreds) must be generated).

        If the "data" used in the analysis is pseudodata (i.e., `data_maker`
        uses Monte Carlo to produce its distributions, and these are then
        Poisson-fluctuated--`fluctuate_data` is True), then `n_data_trials`
        should be as large as is computationally feasible.

        Likewise, if the fiducial-fit data is to be pseudodata (i.e.,
        `fluctuate_fid_data` is True--whether or not `data_maker` is uses Monte
        Carlo), `n_fid_data_trials` should be as large as computationally
        feasible.

        Typical analyses include the following:
            * Asimov analysis of data: `data_maker` uses (actual, measured)
              data and both `fluctuate_data` and `fluctuate_fid_data` are
              False.
            * Pseudodata analysis of data: `data_maker` uses (actual, measured)
              data, `fluctuate_data` is False, and `fluctuate_fid_data` is
              True.
            * Asimov analysis of Monte Carlo: `data_maker` uses Monte Carlo to
              produce its distributions and both `fluctuate_data` and
              `fluctuate_fid_data` are False.
            * Pseudodata analysis of Monte Carlo: `data_maker` uses Monte Carlo
              to produce its distributions and both `fluctuate_data` and
              `fluctuate_fid_data` are False.


        References
        ----------
        TODO

        """
        # Identify duplicate `*_maker` specifications
        if null_hypo_maker == data_maker:
            null_maker_is_data_maker
        if alt_hypo_maker == data_maker:
            alt_maker_is_data_maker
        if alt_hypo_maker == null_hypo_maker:
            alt_maker_is_null_maker

        # Instantiate distribution makers only where necessry
        if not isinstance(data_maker, DistributionMaker):
            data_maker = DistributionMaker(data_maker)

        if not isinstance(null_hypo_maker, DistributionMaker):
            if null_maker_is_data_maker:
                null_hypo_maker = deepcopy(data_maker)
            else:
                null_hypo_maker = DistributionMaker(null_hypo_maker)

        if not isinstance(alt_hypo_maker, DistributionMaker):
            if alt_maker_is_data_maker:
                alt_hypo_maker = deepcopy(data_maker)
            elif alt_maker_is_null_maker:
                alt_hypo_maker = deepcopy(null_hypo_maker)
            else:
                alt_hypo_maker = DistributionMaker(alt_hypo_maker)

        # Select params
        data_maker.param_selections(data_param_selections)

        # Generate (unfluctuated) data
        unfluctuated_data = data_maker.get_outputs()

        for data_ind in xrange(data_start_ind, data_start_ind + n_data_trials):
            # random state is defined by:
            #   * trial-vs-supertrial = 0 : super-trial part (outer loop)
            #   * super-trial = data_ind : super-trial number
            #   * sub-trial = 0 : always 0 since not in the sub-trial part
            data_random_state = get_random_state([0, data_ind, 0])

            # Fluctuate data, if told to do so.
            data = unfluctuated_data.fluctuate(
                method=fluctuate_data,
                random_state=data_random_state
            )

            # Fiducial fits: Perform fits of the two hypotheses to `data`
            llr_fit_to_data, null_hypo_fit_to_data, alt_hypo_fit_to_data = \
                    compare_hypos(
                        data=data,
                        null_hypo_maker=null_hypo_maker,
                        alt_hypo_maker=alt_hypo_maker,
                        metric=metric,
                        null_hypo_param_selections=null_hypo_param_selections,
                        alt_hypo_param_selections=alt_hypo_param_selections
                    )

            # Retrieve event-rate maps for best fit
            fid_null_unfluctuated_data = null_hypo_fit_to_data.maps
            fid_alt_unfluctuated_data = alt_hypo_fit_to_data.maps

            for fid_data_ind in xrange(fid_data_start_ind,
                                       fid_data_start_ind + n_fid_data_trials):
                # random state is defined by:
                #   * trial-vs-supertrial = 1 : we're in the sub-trial part
                #     (i.e., inner loop)
                #   * super-trial = data_ind : super-trial number
                #   * sub-trial = fid_data_ind : which sub-trial we're on
                fid_random_state = get_random_state([1, data_ind,
                                                     fid_data_ind])

                # Fluctuate fiducial-fit data if told to do so
                fid_null_data = fid_null_unfluctuated_data.fluctuate(
                    fluctuate_fid_data,
                    random_state=fid_random_state
                )

                # (Note that the state of `random_state` will be moved
                # forward now as compared to what it was upon definition above.
                # This is the desired behavior, so the *exact* same random
                # state isn't used to fluctuate alt as was used to fluctuate
                # null.)

                fid_alt_data = fid_alt_unfluctuated_data.fluctuate(
                    fluctuate_fid_data,
                    random_state=fid_random_state
                )

                # Final fits: Perform fits of the two hypotheses to the
                # fiducial data
                llr_fit_to_fid_null, null_hypo_fit_to_fid_null, \
                        alt_hypo_fit_to_fid_null = compare_hypos(
                            data=fid_null_data,
                            null_hypo_maker=null_hypo_maker,
                            alt_hypo_maker=alt_hypo_maker,
                            metric=metric,
                            null_hypo_param_selections=null_hypo_param_selections,
                            alt_hypo_param_selections=alt_hypo_param_selections
                        )

                llr_fit_to_fid_alt, null_hypo_fit_to_fid_alt, \
                        alt_hypo_fit_to_fid_alt = compare_hypos(
                            data=fid_null_data,
                            null_hypo_maker=null_hypo_maker,
                            alt_hypo_maker=alt_hypo_maker,
                            metric=metric,
                            null_hypo_param_selections=null_hypo_param_selections,
                            alt_hypo_param_selections=alt_hypo_param_selections
                        )

    # TODO: add references, usage, docstring correctness
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


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    import numpy as np
    from pisa import ureg, Q_

    from pisa.utils.fileio import from_file, to_file
    from pisa.utils.config_parser import parse_pipeline_config
    from pisa.utils.format import append_results, ravel_results

    parser = ArgumentParser()
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

    analysis = Analysis(data_maker=data_maker,
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
