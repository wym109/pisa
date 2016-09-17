# authors: J.L. Lanfranchi, P.Eller, and S. Wren
# email:   jll1062+pisa@phys.psu.edu
# date:    March 20, 2016
"""
Common tools for performing an analysis collected into a single class
`Analysis` that can be subclassed by specific analyses.

"""


from __future__ import division

from collections import OrderedDict
from copy import deepcopy
import sys
import time

import numpy as np
import scipy.optimize as optimize

from pisa import ureg, Q_
from pisa.core.param import ParamSet
from pisa.utils.log import logging


class Counter(object):
    def __init__(self, i=0):
        self._i = i

    def __str__(self):
        return str(self._i)

    def __repr__(self):
        return str(self)

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
    * Asimov distribution creation (via passed `distribution_maker` object)
    * Minimizer Interface (via method `_minimizer_callable`)
        Interfaces to a minimizer for modifying the free parameters of the
        `distribution_maker` to fit its output (as closely as possible) to the
        data distribution is provided. See [minimizer_settings] for

    """
    METRICS_TO_MAXIMIZE = ['llh', 'conv_llh']
    def __init__(self):
        pass

    def compare_hypos(self, data,
                      h0_maker, h0_param_selections,
                      h1_maker, h1_param_selections,
                      metric, minimizer_settings, pprint=False, blind=False):
        """
        Compare how well `data` is described by each of two hypotheses: h0 and
        h1.

        Each hypothesis is represented by a distribtuion maker


        Parameters
        ----------
        data : MapSet

        h0_maker : DistributionMaker

        h0_param_selections : None, string, or sequence of strings

        h1_maker : DistributionMaker

        h1_param_selections : None, string, or sequence of strings

        metric : None or string

        minimizer_settings : string

        pprint : bool

        blind : bool


        Returns
        -------
        delta_metric, h0_fit, h1_fit

        """
        h0_fit = self.fit_hypo(
            data=data,
            hypo_maker=h0_maker,
            param_selections=h0_param_selections,
            metric=metric,
            minimizer_settings=minimizer_settings,
            pprint=pprint,
            blind=blind
        )
        h1_fit = self.fit_hypo(
            data=data,
            hypo_maker=h1_maker,
            param_selections=h1_param_selections,
            metric=metric,
            minimizer_settings=minimizer_settings,
            pprint=pprint,
            blind=blind
        )
        delta_metric = h0_fit['metric_val'] - h1_fit['metric_val']
        return delta_metric, h0_fit, h1_fit

    def fit_hypo(self, data, hypo_maker, param_selections, metric,
                 minimizer_settings, check_octant=True, pprint=True,
                 blind=False):
        """Fitter "outer" loop: If `check_octant` is True, run
        `fit_hypo_inner` starting in each octant of theta23 (assuming that
        is a param in the `hypo_maker`). Otherwise, just run the inner
        method once.

        Parameters
        ----------
        data : MapSet
            Data events distribution(s)

        hypo_maker : DistributionMaker or convertible thereto

        param_selections : None, string, or sequence of strings

        metric : string

        minimizer_settings : string or dict

        check_octant : bool

        pprint : bool
            Whether to show live-update of minimizer progress.

        blind : bool


        Returns
        -------
        OrderedDict fit_info with keys 'metric', 'metric_val', 'params',
        'asimov_dist', and 'metadata'

        """
        # Reset free parameters to nominal values
        logging.trace('Selecting and resetting params')
        hypo_maker.select_params(param_selections)
        hypo_maker.reset_free()

        best_fit_info = self.fit_hypo_inner(
            hypo_maker=hypo_maker,
            data=data,
            metric=metric,
            minimizer_settings=minimizer_settings,
            pprint=pprint,
            blind=blind
        )

        # Decide whether fit for other octant is necessary
        if check_octant and 'theta23' in hypo_maker.params.free:
            logging.debug('checking other octant of theta23')
            hypo_maker.reset_free()

            # Hop to other octant by reflecting about 45 deg
            theta23 = hypo_maker.params.theta23
            inflection_point = (45*ureg.deg).to(theta23.units)
            theta23.value = 2*inflection_point - theta23.value
            hypo_maker.update_params(theta23)

            # Re-run minimizer starting at new point
            new_fit_info = self.fit_hypo_inner(
                hypo_maker=hypo_maker,
                data=data,
                metric=metric,
                minimizer_settings=minimizer_settings,
                pprint=pprint,
                blind=blind
            )

            # Take the one with the best fit
            if metric in self.METRICS_TO_MAXIMIZE:
                it_got_better = new_fit_info['metric_val'] > \
                        best_fit_info['metric_val']
            else:
                it_got_better = new_fit_info['metric_val'] < \
                        best_fit_info['metric_val']

            if it_got_better:
                best_fit_info = new_fit_info
                if not blind:
                    logging.debug('Accepting other-octant fit')
            elif not blind:
                logging.debug('Accepting initial-octant fit')

        return best_fit_info

    def fit_hypo_inner(self, data, hypo_maker, metric, minimizer_settings,
                       pprint=True, blind=False):
        """Fitter "inner" loop: Run an arbitrary scipy minimizer to modify
        hypo dist maker's free params until the data is most likely to have
        come from this hypothesis.

        Note that an "outer" loop can handle discrete scanning over e.g. the
        octant for theta23; for each discrete point the "outer" loop can make a
        call to this "inner" loop. One such "outer" loop is implemented in the
        `fit_hypo` method.


        Parameters
        ----------
        data : MapSet
            Data events distribution(s)

        hypo_maker : DistributionMaker or convertible thereto

        metric : string

        minimizer_settings : string

        pprint : bool
            Whether to show live-update of minimizer progress.

        blind : bool


        Returns
        -------

        """
        sign = -1 if metric in self.METRICS_TO_MAXIMIZE else +1

        # Get starting free parameter values
        x0 = hypo_maker.params.free._rescaled_values

        # TODO: does this break if not using bfgs?

        # bfgs steps outside of given bounds by 1 epsilon to evaluate gradients
        minimizer_kind = minimizer_settings['options']['value']
        try:
            epsilon = minimizer_kind['eps']
        except KeyError:
            epsilon = minimizer_kind['epsilon']
        bounds = [(0+epsilon, 1-epsilon)]*len(x0)

        logging.debug('Running the %s minimizer.'
                      %minimizer_settings['method']['value'])

        # Using scipy.optimize.minimize allows a whole host of minimisers to be
        # used.
        counter = Counter()
        start_t = time.time()
        optimize_result = optimize.minimize(
            fun=self._minimizer_callable,
            x0=x0,
            args=(hypo_maker, data, metric, counter, pprint, blind),
            bounds=bounds,
            method = minimizer_settings['method']['value'],
            options = minimizer_settings['options']['value']
        )
        end_t = time.time()
        if pprint:
            # clear the line
            sys.stdout.write('\n')
            sys.stdout.flush()

        logging.info('Total time to optimize: %8.4f s;'
                     ' # of dists generated: %6d;'
                     ' avg dist gen time: %10.4f ms'
                     %(end_t-start_t,
                       counter.count,
                       (end_t - start_t)*1000./counter.count))

        # Will not assume that the minimizer left the hypo maker in the
        # minimized state, so set the values now (also does conversion of
        # values from [0,1] back to physical range)
        rescaled_pvals = optimize_result.pop('x')
        hypo_maker._set_rescaled_free_params(rescaled_pvals)

        # Record the Asimov distribution with the optimal param values
        asimov_dist = hypo_maker.get_outputs()

        # Get the best-fit metric value
        metric_val = sign * optimize_result.pop('fun')

        # Record minimizer metadata (all info besides 'x' and 'fun'; also do
        # not record some attributes if performing blinded analysis)
        metadata = OrderedDict()
        for k in sorted(optimize_result.keys()):
            if blind and k in ['jac', 'hess', 'hess_inv']:
                continue
            metadata[k] = optimize_result[k]

        fit_info = OrderedDict()
        fit_info['metric'] = metric
        fit_info['metric_val'] = metric_val
        fit_info['asimov_dist'] = asimov_dist
        if blind:
            hypo_maker.reset_free()
            fit_info['params'] = ParamSet()
        else:
            fit_info['params'] = deepcopy(hypo_maker.params)
        fit_info['metadata'] = metadata

        return fit_info

    def _minimizer_callable(self, scaled_param_vals, hypo_maker, data,
                            metric, counter, pprint, blind):
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

        hypo_maker : DistributionMaker
            Creates the per-bin expectation values per map (aka Asimov
            distribution) based on its param values. Free params in the
            `hypo_maker` are modified by the minimizer to achieve a "best" fit.

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

        blind : bool

        """
        # Want to *maximize* e.g. log-likelihood but we're using a minimizer,
        # so flip sign of metric in those cases.
        sign = -1 if metric in self.METRICS_TO_MAXIMIZE else +1

        # Set param values from the scaled versions the minimizer works with
        hypo_maker._set_rescaled_free_params(scaled_param_vals)

        # Get the Asimov map set
        try:
            asimov_dist = hypo_maker.get_outputs()
        except:
            if not blind:
                logging.error(
                    'Failed to generate Asimov distribution with free'
                    ' params %s' %(hypo_maker.params.free,)
                )
            raise

        # Assess the fit: whether the data came from the asimov_dist
        try:
            metric_val = (
                data.metric_total(expected_values=asimov_dist, metric=metric)
                + hypo_maker.params.priors_penalty(metric=metric)
            )
        except:
            if not blind:
                logging.error(
                    'Failed when computing metric with free params %s'
                    %hypo_maker.params.free
                )
            raise


        # TODO: make this into a header line with param names and values
        # beneath updated, to save horizontal space (and easier to read/follow)

        # Report status of metric & params (except if blinded)
        if blind:
            msg = 'minimizer iteration #%7d' %counter.count
        else:
            msg = '%s=%.6e | %s' %(metric, metric_val, hypo_maker.params.free)

        if pprint:
            sys.stdout.write(msg)
            sys.stdout.flush()
            sys.stdout.write('\b' * len(msg))
        else:
            logging.trace(msg)

        counter += 1

        return sign*metric_val

    # TODO: move the complexity of defining a scan into a class with various
    # factory methods, and just pass that class to the scan method; we will
    # surely want to use scanning over parameters in more general ways, too:
    # * set (some) fixed params, then run (minimizer, scan, etc.) on free
    #   params
    # * set (some free or fixed) params, then check metric
    # where the setting of the params is done for some number of values.
    def scan(self, data, hypo_maker, metric, param_names=None, steps=None,
             values=None, outer=False):
        """Set hypo maker parameters named by `param_names` according to
        either values specified by `values` or number of steps specified by
        `steps`, and return the `metric` indicating how well the data
        distribution is described by each Asimov distribution.

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
                length, and there will be one Asimov distribution generated for
                each set of values across the inner sequences. In other words,
                there will be a total of len(inner sequence) Asimov
                distribution generated.
              * If `outer` is True, the lengths of inner sequences needn't be
                the same. This takes the outer product of the passed sequences
                to arrive at the permutations of the parameter values that will
                be used to produce Asimov distributions (essentially nested
                loops over each parameter). E.g., if two params are scanned,
                for each value of the first param's inner sequence, an Asimov
                distribution is produced for every value of the second param's
                inner sequence. In total, there will be
                  len(inner seq0) * len(inner seq1) * ...
                Asimov distributions produced.

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
            fp = hypo_maker.params.free
            fp[param_names].value = val
            hypo_maker.update_params(fp)
            asimov_dist = hypo_maker.get_outputs()
            metric_vals.append(
                data.metric_total(
                    expected_values=asimov_dist, metric=metric
                )
            )
        return metric_vals


def test_Counter():
    pass
