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


class LLRAnalysis(Analysis):
    """LLR analysis to determine the significance for data to have come from
    physics decribed by an alternative hypothesis versus physics decribed a
    null hypothesis.

    Note that duplicated `*_maker` specifications are _not_ instantiated
    separately, but instead are re-used for all duplicate definitions.
    `*_param_selections` allows for this reuse, whereby sets of parameters
    suffixed with the corresponding param_selectors can be switched among to
    simulate different physics using the same DistributionMaker (e.g.,
    switching between null and alt hypotheses).


    Parameters
    ----------
    logdir

    minimizer_settings : string

    alt_hypo_maker : DistributionMaker or instantiable thereto

    alt_hypo_param_selections : None, string, or sequence of strings

    null_hypo_maker : DistributionMaker or instantiable thereto

    null_hypo_param_selections : None, string, or sequence of strings

    data_maker : DistributionMaker or instantiable thereto

    data_param_selections : None, string, or sequence of strings

    metric : string

    fluctuate_data : bool

    fluctuate_fid_data : bool

    num_data_trials : int > 0

    num_fid_data_trials : int > 0

    data_start_ind : int >= 0

    fid_data_start_ind : int >= 0

    alt_hypo_name : string

    null_hypo_name : string

    data_name : string


    Notes
    -----
    LLR analysis is a very thorough (and computationally expensive) method to
    compare discrete hypotheses. In general, a total of

        num_data_trials * (2 + 4*num_fid_data_trials)

    fits must be performed (and note that for each fit, many distributions
    (typically dozens or even hunreds) must be generated).

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
                 alt_hypo_maker, alt_hypo_param_selections=None,
                 null_hypo_maker=None, null_hypo_param_selections=None,
                 data_maker=None, data_param_selections=None,
                 metric='llh', fluctuate_data=False, fluctuate_fid_data=False,
                 num_data_trials=1, num_fid_data_trials=1,
                 data_start_ind=0, fid_data_start_ind=0,
                 alt_hypo_name='alt hypo', null_hypo_name='null hypo',
                 data_name='data'):
        # Identify duplicate `*_maker` specifications
        self.null_maker_is_alt_maker = False
        if null_hypo_maker is None or null_hypo_maker == alt_hypo_maker:
            self.null_maker_is_alt_maker = True

        self.data_maker_is_alt_maker = False
        if data_maker is None or data_maker == alt_hypo_maker:
            self.data_maker_is_alt_maker = True

        self.data_maker_is_null_maker = False
        if data_maker == null_hypo_maker:
            self.data_maker_is_null_maker = True

        # If no data maker settings AND no data param selections are provided,
        # assume that data param selections are to come from alt hypo
        if data_maker is None and data_param_selections is None:
            data_param_selections = alt_hypo_param_selections

        # If no null hypo maker settings AND no null hypo param selections are
        # provided, then we really can't procede since alt and null will be
        # identical.
        if null_hypo_maker is None and null_hypo_param_selections is None:
            raise ValueError(
                'Null hypothesis to be generated will use the same'
                ' distribution maker configured the same way as the'
                ' alternative hypothesis. If you wish for this behavior, you'
                ' must explicitly specify `null_hypo_maker` and/or'
                ' `null_hypo_param_selections`.'
            )

        # Ensure num_{fid_}data_trials is one if fluctuate_{fid_}data is False
        if not fluctuate_data and num_data_trials != 1:
            logging.warn(
                'More than one data trial is unnecessary because'
                ' `fluctuate_data` is False (i.e., all `num_data_trials` data'
                ' distributions will be identical). Forceing `num_data_trials`'
                ' to 1.'
            )
            num_data_trials = 1

        if not fluctuate_fid_data and num_fid_data_trials != 1:
            logging.warn(
                'More than one data trial is unnecessary because'
                ' `fluctuate_fid_data` is False (i.e., all'
                ' `num_fid_data_trials` data distributions will be identical).'
                ' Forceing `num_fid_data_trials` to 1.'
            )
            num_fid_data_trials = 1

        # Instantiate distribution makers only where necessry (otherwise copy)
        if not isinstance(alt_hypo_maker, DistributionMaker):
            alt_hypo_maker = DistributionMaker(alt_hypo_maker)

        if not isinstance(null_hypo_maker, DistributionMaker):
            if self.null_maker_is_alt_maker:
                null_hypo_maker = copy(alt_hypo_maker)
            else:
                null_hypo_maker = DistributionMaker(null_hypo_maker)

        if not isinstance(data_maker, DistributionMaker):
            if self.data_maker_is_alt_maker:
                data_maker = copy(alt_hypo_maker)
            elif self.data_maker_is_null_maker:
                data_maker = copy(null_hypo_maker)
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

        self.alt_hypo_maker = alt_hypo_maker
        self.alt_hypo_param_selections = alt_hypo_param_selections

        self.null_hypo_maker = null_hypo_maker
        self.null_hypo_param_selections = null_hypo_param_selections

        self.data_maker = data_maker
        self.data_param_selections = data_param_selections

        self.metric = metric
        self.fluctuate_data = fluctuate_data
        self.fluctuate_fid_data = fluctuate_fid_data

        self.num_data_trials = num_data_trials
        self.num_fid_data_trials = num_fid_data_trials
        self.data_start_ind = data_start_ind
        self.fid_data_start_ind = fid_data_start_ind

        self.alt_hypo_name = alt_hypo_name
        self.null_hypo_name = null_hypo_name
        self.data_name = data_name

        # Storage for most recent Asimov (unfluctuated) data
        self.asimov_data = None
        self.alt_fid_asimov_data = None
        self.null_fid_asimov_data = None

        # Storage for most recent "data" (either unfluctuated--if Asimov
        # analysis being run or if actual data is being used--or fluctuated--if
        # pseudodata is being generated) data
        self.data = None
        self.alt_fid_data = None
        self.null_fid_data = None

        # Counters
        self.data_ind = data_start_ind
        self.fid_data_ind = fid_data_start_ind

    def compare_hypos(self, data):
        """Override `compare_hypos` from Analysis to simplify arguments since
        LLRAnalysis already has knowledge of all args except `data`.

        See `Analysis.compare_hypos` for more detail on the functionality of
        this method.

        Parameters
        ----------
        data : MapSet

        Returns
        -------
        delta_metric, alt_hypo_fit, null_hypo_fit

        """
        return super(self.__class__, self).compare_hypos(
            data=data,
            alt_hypo_maker=self.alt_hypo_maker,
            alt_hypo_param_selections=self.alt_hypo_param_selections,
            null_hypo_maker=self.null_hypo_maker,
            null_hypo_param_selections=self.null_hypo_param_selections,
            metric=self.metric,
            minimizer_settings=self.minimizer_settings
        )

    def generate_data(self):
        # Produce Asimov data if we don't already have it
        if self.asimov_data is None:
            self.data_maker.select_params(self.data_param_selections)
            self.asimov_data = self.data_maker.get_outputs()

        if self.fluctuate_data:
            # Random state for data trials is defined by:
            #   * data vs fid-data = 0  : data part (outer loop)
            #   * data trial = data_ind : data trial number (use same for data
            #                             and and fid data trials, since on the
            #                             same data trial)
            #   * fid trial = 0         : always 0 since data stays the same
            #                             for all fid trials in this data trial
            data_random_state = get_random_state([0, self.data_ind, 0])

            self.data = self.asimov_data.fluctuate(
                method='poisson', random_state=data_random_state
            )

        else:
            self.data = self.asimov_data

        return self.data

    def generate_fid_data(self):
        # Fiducial fits: Perform fits of the two hypotheses to `data`
        (self.llr_fit_to_data, self.alt_hypo_fit_to_data,
         self.null_hypo_fit_to_data) = self.compare_hypos(data=self.data)

        # Retrieve event-rate maps for best fit to data with each hypo
        self.alt_hypo_maker.select_params(self.alt_hypo_param_selections)
        self.alt_hypo_maker.params.free = \
                self.alt_hypo_fit_to_data['params'].free.values
        self.alt_fid_asimov_data = self.alt_hypo_maker.get_outputs()

        self.null_hypo_maker.select_params(self.null_hypo_param_selections)
        self.null_hypo_maker.params.free = \
                self.null_hypo_fit_to_data['params'].free.values
        self.null_fid_asimov_data = self.null_hypo_maker.get_outputs()

        if self.fluctuate_fid_data:
            # Random state for data trials is defined by:
            #   * data vs fid-data = 1     : fid data part (inner loop)
            #   * data trial = data_ind    : data trial number (use same for
            #                                data and and fid data trials,
            #                                since on the same data trial)
            #   * fid trial = fid_data_ind : always 0 since data stays the same
            #                                for all fid trials in this data
            #                                trial
            fid_data_random_state = get_random_state([1, self.data_ind,
                                                      self.fid_data_ind])
            self.alt_fid_data = self.alt_fid_asimov_data.fluctuate(
                method='poisson',
                random_state=fid_data_random_state
            )

            # (Note that the state of `random_state` will be moved
            # forward now as compared to what it was upon definition above.
            # This is the desired behavior, so the *exact* same random
            # state isn't used to fluctuate alt as was used to fluctuate
            # null.)

            self.null_fid_data = self.null_fid_asimov_data.fluctuate(
                method='poisson',
                random_state=fid_data_random_state
            )

        else:
            self.alt_fid_data = self.alt_fid_asimov_data
            self.null_fid_data = self.null_fid_asimov_data

        return self.alt_fid_data, self.null_fid_data

    def run_analysis(self):
        # Loop for multiple (fluctuated) data distributions
        for self.data_ind in xrange(self.data_start_ind,
                                    self.data_start_ind
                                    + self.num_data_trials):
            # Produce a data distribution
            self.generate_data()

            # Loop for multiple (fluctuated) fiducial data distributions
            for self.fid_data_ind in xrange(self.fid_data_start_ind,
                                            self.fid_data_start_ind
                                            + self.num_fid_data_trials):
                # Fit hypotheses to data and produce fiducial data
                # distributions from *each* of the hypotheses' best fits
                # (i.e., two fits are performed here)
                self.generate_fid_data()

                # Final fits: Perform fits of the each of the two hypotheses to
                # each of the two fiducial data distributions (i.e., four fits
                # are performed here)
                self.llr_fit_to_null_fid, self.null_hypo_fit_to_null_fid, \
                        self.alt_hypo_fit_to_null_fid = \
                        self.compare_hypos(data=self.null_fid_data)
                self.llr_fit_to_alt_fid, self.null_hypo_fit_to_alt_fid, \
                        self.alt_hypo_fit_to_alt_fid = \
                        self.compare_hypos(data=self.alt_fid_data)

                # TODO: log trial results here...
            # TODO: ... and/or here

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
        '--alt-hypo-pipeline', required=True,
        type=str, action='append', metavar='PIPELINE_CFG',
        help='''Settings for the generation of alternate hypothesis
        distributions; repeat this argument to specify multiple pipelines.'''
    )
    parser.add_argument(
        '--alt-hypo-param-selections',
        type=str, default=None, metavar='PARAM_SELECTOR_LIST',
        help='''Comma-separated (no spaces) list of param selectors to apply to
        the alt hypothesis distribution maker's pipelines.'''
    )
    parser.add_argument(
        '--null-hypo-pipeline',
        type=str, action='append', default=None, metavar='PIPELINE_CFG',
        help='''Settings for the generation of null hypothesis distributions;
        repeat this argument to specify multiple pipelines. If omitted, the
        same settings as specified for --alt-hypo-pipeline are used to generate
        the null hypothesis distributions (and so you have to use the
        --null-hypo-param-selections argument to generate a hypotheses distinct
        from the alt hypothesis while using alt hypo's distribution maker).'''
    )
    parser.add_argument(
        '--null-hypo-param-selections',
        type=str, default=None, metavar='PARAM_SELECTOR_LIST',
        help='''Comma-separated (no spaces) list of param selectors to apply to
        the null hypothesis distribution maker's pipelines.'''
    )
    parser.add_argument(
        '--data-pipeline',
        type=str, action='append', default=None, metavar='PIPELINE_CFG',
        help='''Settings for the generation of "data" distributions; repeat
        this argument to specify multiple pipelines. If omitted, the same
        settings as specified for --alt-hypo-pipeline are used to generate data
        distributions (i.e., data is assumed to come from the alternate
        hypothesis.'''
    )
    parser.add_argument(
        '--data-param-selections',
        type=str, default=None, metavar='PARAM_SELECTOR_LIST',
        help='''Comma-separated list of param selectors to apply to the data
        distribution maker's pipelines. If neither --data-pipeline nor
        --data-param-selections are specified, *both* are copied from
        --alt-hypo-pipeline and --alt-param-selections, respectively. However,
        if --data-pipeline is specified while --data-param-selections is not,
        then the param selections in the pipeline config file(s) specified are
        used to produce data distributions.'''
    )
    parser.add_argument(
        '--fluctuate-data',
        action='store_true',
        help='''Apply fluctuations to the data distribution. This should *not*
        be set for analyzing "real" (measured) data, and it is common to not
        use this feature even for Monte Carlo analysis. If this is not set,
        --num-data-trials is forced to 1.'''
    )
    parser.add_argument(
        '--fluctuate-fid-data',
        action='store_true',
        help='''Apply fluctuations to the fiducaial data distributions. If this
        is not set, --num-fid-data-trials is forced to 1.'''
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
        '-n', '--num-fid-data-trials',
        type=int, default=1,
        help='''Number of fiducial pseudodata trials to run. In our experience,
        it takes ~10^3-10^5 fiducial psuedodata trials to achieve low
        uncertainties on the resulting significance, though that exact number
        will vary based upon the details of an analysis.'''
    )
    parser.add_argument(
        '--alt-hypo-name',
        type=str, metavar='NAME', default='alt hypo',
        help='''Name for the alternate hypothesis. E.g., "NO" for normal
        ordering in the neutrino mass ordering analysis. Note that the name
        here has no bearing on the actual process, so it's important that you
        be careful to use a name that appropriately identifies the
        hypothesis.'''
    )
    parser.add_argument(
        '--null-hypo-name',
        type=str, metavar='NAME', default='null hypo',
        help='''Name for the null hypothesis. E.g., "IO" for inverted
        ordering in the neutrino mass ordering analysis. Note that the name
        here has no bearing on the actual process, so it's important that you
        be careful to use a name that appropriately identifies the
        hypothesis.'''
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
        '--no-post-processing',
        action='store_true',
        help='''Do not run post-processing for the trials run. This is useful
        if the analysis is divided and run in separate processes, whereby only
        after all processes are run should post-processing be performed
        (once).'''
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

    if not check(path):
        raise IOError('Path "%s" which resolves to "%s" is not a %s.'
                      %(path, normpath, kind))
    return normpath


if __name__ == '__main__':
    args = parse_args()
    args_d = vars(args)

    # NOTE: Removing extraneous args that won't get passed to instantiate the
    # LLRAnalysis object via dictionary's `pop()` method.

    set_verbosity(args_d.pop('v'))
    post_process = not args_d.pop('no_post_processing')

    # Normalize and convert `*_pipeline` filenames; store to `*_maker`
    # (which is argument naming convention that LLRAnalysis init accepts).
    for maker in ['alt_hypo', 'null_hypo', 'data']:
        filenames = args_d.pop(maker + '_pipeline')
        if filenames is not None:
            filenames = sorted(
                [normcheckpath(fname) for fname in filenames]
            )
        args_d[maker + '_maker'] = filenames

        ps_name = maker + '_param_selections'
        ps_str = args_d[ps_name]
        ps_list = [x.strip().lower() for x in ','.split(ps_str)]
        args_d[ps_name] = ps_list

    # Instantiate the analysis object
    llr_analysis = LLRAnalysis(**args_d)

    # Run the analysis
    llr_analysis.run_analysis()

    # TODO: this.
    # Run postprocessing if called to do so
    if post_process:
        llr_analysis.post_process(args.logdir)
