#!/usr/bin/env python

# authors: J.L. Lanfranchi, P.Eller, and S. Wren
# email:   jll1062+pisa@phys.psu.edu
# date:    March 20, 2016
"""
Log-Likelihood-Ratio (LLR) Analysis

"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import Mapping
from copy import copy, deepcopy
import os

from pisa.analysis.analysis import Analysis
from pisa.core.distribution_maker import DistributionMaker
from pisa.utils.fileio import from_file, mkdir, to_file
from pisa.utils.log import logging, set_verbosity
from pisa.utils.resources import find_resource


class HypoTesting(Analysis):
    """Tools for testing two hypotheses against one another.

    to determine the significance for data to
    have come from
    physics described by hypothesis h0 versus physics described by hypothesis
    h1

    Note that duplicated `*_maker` specifications are _not_ instantiated
    separately, but instead are re-used for all duplicate definitions.
    `*_param_selections` allows for this reuse, whereby sets of parameters
    infixed with the corresponding param_selectors can be switched among to
    simulate different physics using the same DistributionMaker (e.g.,
    switching between h0 and h1 hypotheses).


    Parameters
    ----------
    logdir : string

    minimizer_settings : string

    data_maker : None, DistributionMaker or instantiable thereto

    data_param_selections : None, string, or sequence of strings

    data_name : string

    data : None, MapSet or instantiable thereto

    h0_name : string

    h0_maker : None, DistributionMaker or instantiable thereto

    h0_param_selections : None, string, or sequence of strings

    h0_fid_asimov_dist : None, MapSet or instantiable thereto

    h1_name : string

    h1_maker : None, DistributionMaker or instantiable thereto

    h1_param_selections : None, string, or sequence of strings

    h1_fid_asimov_dist : None, MapSet or instantiable thereto

    check_other_octant : bool

    metric : string

    blind : bool


    Notes
    -----
    LLR analysis is a very thorough (and computationally expensive) method to
    compare discrete hypotheses. In general, a total of

        num_data_trials * (2 + 4*num_fid_data_trials)

    fits must be performed (and note that for each fit, many distributions
    (typically dozens or even hundreds) must be generated).

    If the "data" used in the analysis is pseudodata (i.e., `data_maker` uses
    Monte Carlo to produce its distributions, and these are then
    Poisson-fluctuated--`fluctuate_data` is True), then `num_data_trials`
    should be as large as is computationally feasible.

    Likewise, if the fiducial-fit data is to be pseudodata (i.e.,
    `fluctuate_fid_data` is True--whether or not `data_maker` is uses Monte
    Carlo), `num_fid_data_trials` should be as large as computationally
    feasible.

    Typical analyses include the following:
        * Asimov analysis of data: `data_maker` uses (actual, measured) data
          and both `fluctuate_data` and `fluctuate_fid_data` are False.
        * Pseudodata analysis of data: `data_maker` uses (actual, measured)
          data, `fluctuate_data` is False, and `fluctuate_fid_data` is True.
        * Asimov analysis of Monte Carlo: `data_maker` uses Monte Carlo to
          produce its distributions and both `fluctuate_data` and
          `fluctuate_fid_data` are False.
        * Pseudodata analysis of Monte Carlo: `data_maker` uses Monte Carlo to
          produce its distributions and both `fluctuate_data` and
          `fluctuate_fid_data` are False.


    References
    ----------
    TODO


    Examples
    --------

    """
    def __init__(self, logdir, minimizer_settings,
                 data_is_data,
                 fluctuate_data, fluctuate_fid_data,
                 h0_name='hypo0', h0_maker=None,
                 h0_param_selections=None, h0_fid_asimov_dist=None,
                 h1_name='hypo1', h1_maker=None,
                 h1_param_selections=None, h1_fid_asimov_dist=None,
                 data_name='data', data_maker=None,
                 data_param_selections=None, data=None,
                 check_other_octant=True, metric='llh', blind=False):
        # Identify duplicate `*_maker` specifications
        self.h1_maker_is_h0_maker = False
        if h1_maker is None or h1_maker == h0_maker:
            self.h1_maker_is_h0_maker = True

        self.data_maker_is_h0_maker = False
        if data_maker is None or data_maker == h0_maker:
            self.data_maker_is_h0_maker = True

        self.data_maker_is_h1_maker = False
        if data_maker == h1_maker:
            self.data_maker_is_h1_maker = True

        # If no data maker settings AND no data param selections are provided,
        # assume that data param selections are to come from hypo h0
        if data_maker is None and data_param_selections is None:
            data_param_selections = h0_param_selections

        # If no h1 maker settings AND no h1 param selections are
        # provided, then we really can't proceed since h0 and h1 will be
        # identical in every way and there's nothing of substance to be done.
        if h1_maker is None and h1_param_selections is None:
            raise ValueError(
                'Hypotheses h0 and h1 to be generated will use the same'
                ' distribution maker configured the same way, leading to'
                ' trivial behavior. If you wish for this behavior, you'
                ' must explicitly specify `h1_maker` and/or'
                ' `h1_param_selections`.'
            )

        # If analyzing actual data, fluctuations should not be applied to the
        # data (fluctuating fiducial-fits Asimov data is still fine, though).
        if data_is_data and fluctuate_data:
            raise ValueError('Fluctuating actual data is invalid.')

        # Instantiate distribution makers only where necessary (otherwise copy)
        if not isinstance(h0_maker, DistributionMaker):
            h0_maker = DistributionMaker(h0_maker)

        if not isinstance(h1_maker, DistributionMaker):
            if self.h1_maker_is_h0_maker:
                h1_maker = h0_maker
            else:
                h1_maker = DistributionMaker(h1_maker)

        if not isinstance(data_maker, DistributionMaker):
            if self.data_maker_is_h0_maker:
                data_maker = h0_maker
            elif self.data_maker_is_h1_maker:
                data_maker = h1_maker
            else:
                data_maker = DistributionMaker(data_maker)

        # Create directory for logging results
        mkdir(logdir)
        logdir = find_resource(logdir)
        logging.info('Output will be saved to dir "%s"' %logdir)

        # Read in minimizer settings
        if isinstance(minimizer_settings, basestring):
            minimizer_settings = from_file(minimizer_settings)
        assert isinstance(minimizer_settings, Mapping)

        # Store variables to `self` for later access

        self.logdir = logdir
        self.minimizer_settings = minimizer_settings

        self.h0_maker = h0_maker
        self.h0_param_selections = h0_param_selections

        self.h1_maker = h1_maker
        self.h1_param_selections = h1_param_selections

        self.data_is_data = data_is_data
        self.data_maker = data_maker
        self.data_param_selections = data_param_selections

        self.metric = metric
        self.fluctuate_data = fluctuate_data
        self.fluctuate_fid_data = fluctuate_fid_data

        self.h0_name = h0_name
        self.h1_name = h1_name
        self.data_name = data_name

        self.blind = blind
        self.pprint = False

        # Storage for most recent Asimov (un-fluctuated) data
        self.asimov_dist = None
        self.h0_fid_asimov_dist = None
        self.h1_fid_asimov_dist = None

        # Storage for most recent "data" (either un-fluctuated--if Asimov
        # analysis being run or if actual data is being used--or fluctuated--if
        # pseudodata is being generated) data
        self.data = None
        self.h0_fid_data = None
        self.h1_fid_data = None

    def run_analysis(self, num_data_trials=1, num_fid_data_trials=1,
                     data_start_ind=0, fid_data_start_ind=0, pprint=False):
        """Run the LLR analysis."""
        self.pprint = pprint
        logging.info('Running LLR analysis.')

        # Names for purposes of stdout/stderr logging messages
        data_disp = 'pseudodata' if self.fluctuate_data \
                else 'Asimov or true data'
        fid_data_disp = 'fiducial pseudodata' if self.fluctuate_data \
                else 'fiducial Asimov data'

        # Ensure num_{fid_}data_trials is one if fluctuate_{fid_}data is False
        if not self.fluctuate_data and num_data_trials != 1:
            logging.warn(
                'More than one data trial is unnecessary because'
                ' `fluctuate_data` is False (i.e., all `num_data_trials` data'
                ' distributions will be identical). Forcing `num_data_trials`'
                ' to 1.'
            )
            num_data_trials = 1

        if not self.fluctuate_fid_data and num_fid_data_trials != 1:
            logging.warn(
                'More than one data trial is unnecessary because'
                ' `fluctuate_fid_data` is False (i.e., all'
                ' `num_fid_data_trials` data distributions will be identical).'
                ' Forcing `num_fid_data_trials` to 1.'
            )
            num_fid_data_trials = 1

        # Loop for multiple (if fluctuated) data distributions
        for data_ind in xrange(data_start_ind,
                               data_start_ind+num_data_trials):
            pct_data_complete = (
                (data_ind - data_start_ind) / num_data_trials
            )
            logging.info(
                r'%s set ID %d (will stop after %d). %0.2f%s of %s sets'
                ' completed.' %(data_disp, data_ind,
                data_start_ind+num_data_trials-1, pct_data_complete,
                '%', data_disp)
            )

            # Produce a data distribution
            self.generate_data(data_ind=data_ind)

            # Loop for multiple (fluctuated) fiducial data distributions
            for fid_data_ind in xrange(fid_data_start_ind,
                                       fid_data_start_ind+num_fid_data_trials):
                pct_fid_data_complete = (
                    (fid_data_ind - fid_data_start_ind)
                    / num_fid_data_trials
                )
                logging.info(
                    r'%s set ID %d (will stop after %d). %0.2f%s of %s sets'
                    ' completed.' %(fid_data_disp, fid_data_ind,
                    fid_data_start_ind+num_fid_data_trials-1,
                    pct_fid_data_complete, '%', fid_data_disp)
                )

                # Fit hypotheses to data and produce fiducial data
                # distributions from *each* of the hypotheses' best fits
                # (i.e., two fits are performed here)
                self.generate_fid_data(data_ind=data_ind,
                                       fid_data_ind=fid_data_ind)

                # Perform fits of the each of the two hypotheses to each of the
                # two fiducial data distributions (i.e., four fits are
                # performed here)
                self.llr_fit_to_h0_fid, self.h0_fit_to_h0_fid, \
                        self.h1_fit_to_h0_fid = \
                        self.compare_hypos(data=self.h0_fid_data)
                if self.fluctuate_fid_data:
                    self.llr_fit_to_h0_fid, self.h1_fit_to_h0_fid, \
                            self.h0_fit_to_h0_fid = \
                            self.compare_hypos(data=self.h0_fid_data)
                else:
                    self.llr_fit_to_data

                # TODO: log trial results here...
            # TODO: ... and/or here

    def compare_hypos(self, data):
        """Convenience method overriding `compare_hypos` from superclass to
        simplify arguments since this object knows almost all arguments already
        (except `data`).

        See `Analysis.compare_hypos` for more detail on the functionality of
        this method.

        Parameters
        ----------
        data : MapSet

        Returns
        -------
        delta_metric, h1_fit, h0_fit

        """
        return super(self.__class__, self).compare_hypos(
            data=data,
            h1_maker=self.h1_maker,
            h1_param_selections=self.h1_param_selections,
            h0_maker=self.h0_maker,
            h0_param_selections=self.h0_param_selections,
            metric=self.metric,
            minimizer_settings=self.minimizer_settings,
            pprint=self.pprint,
            blind=self.blind
        )

    def generate_data(self, data_ind):
        # Produce Asimov data if we don't already have it
        if self.asimov_dist is None:
            self.data_maker.select_params(self.data_param_selections)
            self.asimov_dist = self.data_maker.get_outputs()

        if self.fluctuate_data:
            # Random state for data trials is defined by:
            #   * data vs fid-data = 0  : data part (outer loop)
            #   * data trial = data_ind : data trial number (use same for data
            #                             and and fid data trials, since on the
            #                             same data trial)
            #   * fid trial = 0         : always 0 since data stays the same
            #                             for all fid trials in this data trial
            data_random_state = get_random_state([0, data_ind, 0])

            self.data = self.asimov_dist.fluctuate(
                method='poisson', random_state=data_random_state
            )

        else:
            self.data = self.asimov_dist

        return self.data

    def generate_fid_data(self, data_ind, fid_data_ind):
        """Fit both hypotheses to "data" and produce fiducial data
        distributions from *each* of the hypotheses' best fits (i.e., two fits
        are performed).

        """
        # Fiducial fits: Perform fits of the two hypotheses to `data`
        (self.llr_fit_to_data, self.h0_fit_to_data,
         self.h1_fit_to_data) = self.compare_hypos(data=self.data)

        # Retrieve event-rate maps for best fit to data with each hypo
        self.h0_fid_asimov_dist = self.h0_fit_to_data['maps']
        self.h1_fid_asimov_dist = self.h1_fit_to_data['maps']

        if self.fluctuate_fid_data:
            # Random state for data trials is defined by:
            #   * data vs fid-data = 1     : fid data part (inner loop)
            #   * data trial = data_ind    : data trial number (use same for
            #                                data and and fid data trials,
            #                                since on the same data trial)
            #   * fid trial = fid_data_ind : always 0 since data stays the same
            #                                for all fid trials in this data
            #                                trial
            fid_data_random_state = get_random_state([1, data_ind,
                                                      fid_data_ind])

            self.h1_fid_data = self.h1_fid_asimov_dist.fluctuate(
                method='poisson',
                random_state=fid_data_random_state
            )

            # (Note that the state of `random_state` will be moved
            # forward now as compared to what it was upon definition above.
            # This is the desired behavior, so the *exact* same random
            # state isn't used to fluctuate h1 as was used to fluctuate
            # h0.)

            self.h0_fid_data = self.h0_fid_asimov_dist.fluctuate(
                method='poisson',
                random_state=fid_data_random_state
            )

        else:
            self.h0_fid_data = self.h0_fid_asimov_dist
            self.h1_fid_data = self.h1_fid_asimov_dist

        return self.h1_fid_data, self.h0_fid_data

    @staticmethod
    def post_process(logdir):
        pass


def parse_args():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description='''Perform the LLR analysis for calculating the NMO
        sensitivity of the distribution made from data-settings compared with
        hypotheses generated from template-settings.

        Currently the output should be a json file containing the dictionary
        of best fit and likelihood values.'''
    )
    parser.add_argument(
        '-d', '--logdir', required=True,
        metavar='DIR', type=str,
        help='Directory into which to store results and metadata.'
    )
    parser.add_argument(
        '-m', '--minimizer-settings',
        type=str, metavar='MINIMIZER_CFG', required=True,
        help='''Settings related to the optimizer used in the LLR analysis.'''
    )
    parser.add_argument(
        '--no-octant-check',
        action='store_true',
        help='''Disable fitting hypotheses in theta23 octant opposite initial
        octant.'''
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--data-is-data', action='store_true',
        help='''Data pipeline is based upon actual, measured data. The naming
        scheme for stored results is chosen accordingly.'''
    )
    group.add_argument(
        '--data-is-mc', action='store_true',
        help='''Data pipeline is based upon Monte Carlo simulation, and not
        actual data. The naming scheme for stored results is chosen
        accordingly. If this is selected, --fluctuate-data is forced off.'''
    )
    parser.add_argument(
        '--h0-pipeline', required=True,
        type=str, action='append', metavar='PIPELINE_CFG',
        help='''Settings for the generation of hypothesis h0
        distributions; repeat this argument to specify multiple pipelines.'''
    )
    parser.add_argument(
        '--h0-param-selections',
        type=str, default=None, metavar='PARAM_SELECTOR_LIST',
        help='''Comma-separated (no spaces) list of param selectors to apply to
        hypothesis h0's distribution maker's pipelines.'''
    )
    parser.add_argument(
        '--h0-name',
        type=str, metavar='NAME', default='hypo0',
        help='''Name for hypothesis h0. E.g., "NO" for normal
        ordering in the neutrino mass ordering analysis. Note that the name
        here has no bearing on the actual process, so it's important that you
        be careful to use a name that appropriately identifies the
        hypothesis.'''
    )
    parser.add_argument(
        '--h1-pipeline',
        type=str, action='append', default=None, metavar='PIPELINE_CFG',
        help='''Settings for the generation of hypothesis h1 distributions;
        repeat this argument to specify multiple pipelines. If omitted, the
        same settings as specified for --h0-pipeline are used to generate
        hypothesis h1 distributions (and so you have to use the
        --h1-param-selections argument to generate a hypotheses distinct
        from hypothesis h0 but still use h0's distribution maker).'''
    )
    parser.add_argument(
        '--h1-param-selections',
        type=str, default=None, metavar='PARAM_SELECTOR_LIST',
        help='''Comma-separated (no spaces) list of param selectors to apply to
        hypothesis h1 distribution maker's pipelines.'''
    )
    parser.add_argument(
        '--h1-name',
        type=str, metavar='NAME', default='hypo1',
        help='''Name for hypothesis h1. E.g., "IO" for inverted
        ordering in the neutrino mass ordering analysis. Note that the name
        here has no bearing on the actual process, so it's important that you
        be careful to use a name that appropriately identifies the
        hypothesis.'''
    )
    parser.add_argument(
        '--data-pipeline',
        type=str, action='append', default=None, metavar='PIPELINE_CFG',
        help='''Settings for the generation of "data" distributions; repeat
        this argument to specify multiple pipelines. If omitted, the same
        settings as specified for --h0-pipeline are used to generate data
        distributions (i.e., data is assumed to come from hypothesis h0.'''
    )
    parser.add_argument(
        '--data-param-selections',
        type=str, default=None, metavar='PARAM_SELECTOR_LIST',
        help='''Comma-separated list of param selectors to apply to the data
        distribution maker's pipelines. If neither --data-pipeline nor
        --data-param-selections are specified, *both* are copied from
        --h0-pipeline and --h0-param-selections, respectively. However,
        if --data-pipeline is specified while --data-param-selections is not,
        then the param selections in the pipeline config file(s) specified are
        used to produce data distributions.'''
    )
    parser.add_argument(
        '--data-name',
        type=str, metavar='NAME', default='data',
        help='''Name for the data. E.g., "NO" for normal ordering in the
        neutrino mass ordering analysis. Note that the name here has no bearing
        on the actual process, so it's important that you be careful to use a
        name that appropriately identifies the hypothesis.'''
    )
    parser.add_argument(
        '--fluctuate-data',
        action='store_true',
        help='''Apply fluctuations to the data distribution. This should *not*
        be set for analyzing "real" (measured) data, and it is common to not
        use this feature even for Monte Carlo analysis. Note that if this is
        not set, --num-data-trials is forced to 1.'''
    )
    parser.add_argument(
        '--fluctuate-fid-data',
        action='store_true',
        help='''Apply fluctuations to the fiducaial data distributions. If this
        is not set, --num-fid-data-trials is forced to 1.'''
    )
    parser.add_argument(
        '--metric',
        type=str, default='llh', metavar='METRIC',
        help='''Name of metric to use for evaluating a fit.'''
    )
    parser.add_argument(
        '--num-data-trials',
        type=int, default=1,
        help='''When performing Monte Carlo analysis, set to > 1 to produce
        multiple pseudodata distributions from the data distribution maker's
        Asimov data distribution. This is overridden if --fluctuate-data is not
        set (since each data distribution will be identical if it is not
        fluctuated). This is typically left at 1 (i.e., the Asimov distribution
        is assumed to be representative.'''
    )
    parser.add_argument(
        '--data-start-ind',
        type=int, default=0,
        help='''Fluctated data set index.'''
    )
    parser.add_argument(
        '-n', '--num-fid-data-trials',
        type=int, default=1,
        help='''Number of fiducial pseudodata trials to run. In our experience,
        it takes ~10^3-10^5 fiducial psuedodata trials to achieve low
        uncertainties on the resulting significance, though that exact number
        will vary based upon the details of an analysis.'''
    )
    parser.add_argument(
        '--fid-data-start-ind',
        type=int, default=0,
        help='''Fluctated fiducial data index.'''
    )
    parser.add_argument(
        '--no-post-processing',
        action='store_true',
        help='''Do not run post-processing for the trials run. This is useful
        if the analysis is divided and run in separate processes, whereby only
        after all processes are run should post-processing be performed
        (once).'''
    )
    parser.add_argument(
        '--pprint',
        action='store_true',
        help='''Live-updating one-line vew of metric and parameter values. (The
        latter are not displayed if --blind is specified.)'''
    )
    parser.add_argument(
        '--blind',
        action='store_true',
        help='''Blinded analysis. Do not show parameter values or store to
        logfiles.'''
    )
    parser.add_argument(
        '-v', action='count', default=None,
        help='set verbosity level'
    )
    return parser.parse_args()


# TODO: make this work with Python package resources, not merely absolute
# paths! ... e.g. hash on the file or somesuch?
def normcheckpath(path, checkdir=False):
    normpath = find_resource(path)
    if checkdir:
        kind = 'dir'
        check = os.path.isdir
    else:
        kind = 'file'
        check = os.path.isfile

    if not check(normpath):
        raise IOError('Path "%s" which resolves to "%s" is not a %s.'
                      %(path, normpath, kind))
    return normpath


if __name__ == '__main__':
    args = parse_args()
    init_args_d = vars(args)

    # NOTE: Removing extraneous args that won't get passed to instantiate the
    # HypoTesting object via dictionary's `pop()` method.

    set_verbosity(init_args_d.pop('v'))
    post_process = not init_args_d.pop('no_post_processing')
    init_args_d['check_other_octant'] = not init_args_d.pop('no_octant_check')

    init_args_d['data_is_data'] = not init_args_d.pop('data_is_mc')

    # Normalize and convert `*_pipeline` filenames; store to `*_maker`
    # (which is argument naming convention that HypoTesting init accepts).
    for maker in ['h0', 'h1', 'data']:
        filenames = init_args_d.pop(maker + '_pipeline')
        if filenames is not None:
            filenames = sorted(
                [normcheckpath(fname) for fname in filenames]
            )
        init_args_d[maker + '_maker'] = filenames

        ps_name = maker + '_param_selections'
        ps_str = init_args_d[ps_name]
        if ps_str is None:
            ps_list = None
        else:
            ps_list = [x.strip().lower() for x in ps_str.split(',')]
        init_args_d[ps_name] = ps_list

    # Extract args needed by the `run_analysis` method
    run_args = ['num_data_trials', 'num_fid_data_trials', 'data_start_ind',
                'fid_data_start_ind', 'pprint']
    run_args_d = {arg:init_args_d.pop(arg) for arg in run_args}

    # Instantiate the analysis object
    hypo_testing = HypoTesting(**init_args_d)

    # Run the analysis
    hypo_testing.run_analysis(**run_args_d)

    # TODO: this.
    # Run postprocessing if called to do so
    if post_process:
        hypo_testing.post_process(args.logdir)
