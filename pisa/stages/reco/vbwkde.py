# PISA author: J.L. Lanfranchi
#              jll1062+pisa@phys.psu.edu
#
# CAKE author: Matthew Weiss
#
# date:        2016-10-01
"""
Produce a set of transforms mapping true events values (energy and coszen) onto
reconstructed values.

For each bin in true energy and true coszen, a corresponding distribution of
reconstructed energy and coszen values is estimated using a variable-bandwidth
KDE.

These transforms are used to produce reco event rate maps.

The algorithm is roughly as follows:

1. CHARACTERIZE RESOLUTIONS:
   all_kdes = characterize_resolutions()

    * PID CATEGORY NORMALIZATION: For each energy bin in each flavintgroup,
      figure out how many events PID would categorize in each PID bin.
        * events = collect_enough_events(flavint_evts, ebin)
        * KDE 'pid' distribution for collected events; store in cache
            * Yields (N_flavints * N_ebins) KDEs

    * RESOLUTIONS KDEs: For each (true-energy bin, PID category, flavintgroup),
      (allowing for expansion in true-energy to get enough events), find the
      reco-energy and reco-coszen resolutions.
        * events = collect_enough_events(flavintgroup_pid_evts, ebin)
        * KDE 'reco_energy' distribution for events; store in cache
            * Use `log(reco_energy/true_energy)`
            * Yields (N_flavints * N_pid_bins * N_ebins) energy res KDEs
        * KDE 'reco_coszen' distribution for events; store in cache
            * Use `fold_coszen_error(reco_coszen - true_coszen)`
            * Yields (N_flavints * N_pid_bins * N_ebins) coszen res KDEs

2. GENERATE SMEARING KERNELS: (assumes all energy binning is
   logarithmically-even)

   kernel = generate_smearing_kernel(all_kdes[flavints])

    * Figure out which true_energy bins fall within one reco_energy bin;
    * Scale input-energy binning into the log(ratio)-space:
      log(true_energy.bin_edges / )

"""


# TODO: nutau needn't be treated below 3.5 GeV! ...
# TODO: write "closest bin" function
# TODO: handle "closest bin" logic when infinities are involved (seems like
#         `np.clip(edges, a_min=np.ftype(FTYPE).min, a_max=np.ftype(FTYPE).max`
#       would do the trick...)

from __future__ import division

from collections import OrderedDict, Sequence, namedtuple
from copy import deepcopy
from itertools import izip
from multiprocessing import Pool

import numpy as np

from pisa import EPSILON, FTYPE, OMP_NUM_THREADS, ureg
from pisa.core.binning import MultiDimBinning, OneDimBinning
from pisa.core.events import Events
from pisa.core.stage import Stage
from pisa.core.transform import BinnedTensorTransform, TransformSet

from pisa.utils.flavInt import flavintGroupsFromString, NuFlavIntGroup
from pisa.utils.hash import hash_obj
from pisa.utils.vbwkde import vbwkde as vbwkde_func
from pisa.utils.log import logging


__all__ = ['KDE_DIM_DEPENDENCIES', 'KDE_TRUE_BINNING', 'MIN_NUM_EVENTS',
           'TGT_NUM_EVENTS', 'TGT_MAX_BINWIDTH_FACTOR',
           'KDEProfile', 'collect_enough_events', 'fold_coszen_error',
           'weight_coszen_tails', 'coszen_error_edges',
           'vbwkde']


KDE_DIM_DEPENDENCIES = OrderedDict([
    ('pid', ['true_energy']),
    ('energy', ['pid', 'true_energy']),
    ('coszen', ['pid', 'true_coszen', 'true_energy'])
])
KDE_TRUE_BINNING = {
    'pid': MultiDimBinning([
        dict(name='true_energy', num_bins=20, is_log=True,
             domain=[1, 80]*ureg.GeV,
             tex=r'E_{\rm true}')
        ]),
    'energy': MultiDimBinning([
        dict(name='true_energy', num_bins=10, is_log=True,
             domain=[1, 80]*ureg.GeV,
             tex=r'E_{\rm true}')
        ]),
    'coszen': MultiDimBinning([
        dict(name='true_energy', num_bins=5, is_log=True,
             domain=[1, 80]*ureg.GeV,
             tex=r'E_{\rm true}'),
        dict(name='true_coszen', num_bins=10, is_lin=True,
             domain=[-1, 1], #bin_edges=[-1, -0.75, 0.5, 0.75, 1],
             tex=r'\cos\,\theta_{\rm true}')
    ])
}
MIN_NUM_EVENTS = 50
TGT_NUM_EVENTS = 1000

# TODO: figure out a dynamic similarity metric such that this parameter can be
#       figured out by the software, rather than set by the user. E.g., use
#       some statistical clustering technique?
TGT_MAX_BINWIDTH_FACTOR = 0.1


KDEProfile = namedtuple('KDEProfile', ['x', 'density'])
"""namedtuple type for storing the normalized KDE profile: (x, density)"""


# TODO: revisit this heuristic with proper testing
# TODO: modify this once we have fixed the Events object to be more agnostic to
#       flavint

def collect_enough_events(events, flavint, bin,
                          min_num_events=MIN_NUM_EVENTS,
                          tgt_num_events=TGT_NUM_EVENTS,
                          tgt_max_binwidth_factor=TGT_MAX_BINWIDTH_FACTOR):
    """Heuristic to collect enough events close to the provided bin such
    that KDEs can be applied and achieve robust results.

    Events are selected via the logic:
    * If there are `tgt_num_events` or more in the energy bin, simply return
      all events in the bin.
    * Otherwise, go up to `tgt_max_binwidth_factor` times the width of the
      `bin` to find up to `tgt_num_events`. (Note that the distance is
      measured from the bin's weighted center.)
    * However, keep going as far as necessary outside the energy bin (i.e.,
      ignore `tgt_max_binwidth_factor`) to find `min_num_events` events.

    Note that in the above, _all_ events in the bin will _always_ be returned.
    The rest of the algorithm decides how far outside it has to reach to get a
    "desired" number of events (`tgt_num_events`) without going too far, but
    will reach as far as necessary to get the "minimum" number of events
    (`min_num_events').

    Distances are measured on a logarithmic scale, so events that are one-half
    and twice the energy bin's weighted center (respectively) are considered to
    be the same distance to the weighted center.

    In the case of "exact ties" as described above, _both_ events will be kept.
    Therefore, one may get _more_ than `tgt_num_events` when that threshold is
    used or more than `min_num_events` when that threshold is used.

    Finally, note that both edges of the `bin` are _inclusive_, i.e., it
    defines a closed interval.


    Parameters
    ----------
    events : pisa.core.events.Events
        Note that a deepcopy is made on this object, so it is not modified in
        this function.

    bin : pisa.core.binning.OneDimBinning

    flavint : None or convertible to pisa.utils.flavInts.NuFlavIntGroup

    min_num_events : int
        At least this many events will collected, regardless how far outside
        the boundaries of `bin` is necessary to find them.

    tgt_num_events : int >= min_num_events

    tgt_max_binwidth_factor : float >= 0

    Returns
    -------
    events_subset : pisa.core.events.Events
        The subset of the passed `events` that fulfill the criteria outlined
        above.

    """
    # For now, be strict about what is allowed, so as to keep logic simple
    assert isinstance(events, Events)
    flavintgroup = NuFlavIntGroup(flavint)
    repr_flavint = flavintgroup[0]
    logging.trace('flavintgroup=%s, repr_flavint=%s, bin=%s',
                  flavintgroup, repr_flavint, bin)

    if isinstance(bin, MultiDimBinning):
        dims = bin.dimensions
        assert len(dims) == 1
        bin = dims[0]
    assert isinstance(bin, OneDimBinning)
    edges = bin.bin_edges.m
    assert len(edges) == 2
    assert bin.is_log
    assert min_num_events <= tgt_num_events

    # ASSUMPTION: units in Events object are same as in `bin`
    bin_wtd_center = bin.weighted_centers[0].m

    if bin.is_log:
        bin_width = edges[1] / edges[0]
        bin_half_width = np.sqrt(bin_width)
    else:
        bin_width = edges[1] - edges[0]
        bin_half_width = bin_width / 2

    # Define for convenience of use elsewhere
    field_values = events[repr_flavint][bin.name]
    n_events = len(field_values)

    # If either min_num_events or tgt_num_events is greater than the total
    # number of events we have to work with, force to be equal to the number
    # of events we have.
    min_num_evts = min_num_events if min_num_events < n_events else n_events
    tgt_num_evts = tgt_num_events if tgt_num_events < n_events else n_events

    # Absolute distance from these events to the center of the bin, sorted in
    # ascending order (so events closest to bin center come first)
    if bin.is_log:
        sorted_abs_dist = np.sort(np.abs(np.log(field_values/bin_wtd_center)))
    else:
        sorted_abs_dist = np.sort(np.abs(field_values - bin_wtd_center))

    # Distance from the bin center you have to go to obtain `tgt_num_events`
    tgt_num_events_dist = np.exp(sorted_abs_dist[tgt_num_evts-1])

    # Maximum distance the  tgt_max_binwidth_factor` allows us to go in order
    # to obtain `tgt_num_evts` events
    if bin.is_log:
        tgt_max_dist = bin_half_width * (1 + tgt_max_binwidth_factor)**2
    else:
        tgt_max_dist = bin_half_width + bin_width*tgt_max_binwidth_factor

    # Define a single "target" distance taking into consideration that we
    # should neither exceed `tgt_max_dist` nor `tgt_num_evts`
    if tgt_num_events_dist < tgt_max_dist:
        tgt_dist = tgt_num_events_dist
    else:
        tgt_dist = tgt_max_dist

    # If the target distance is within the bin, simply take all events from
    # the bin and we're done, as we will get >= tgt_num_events within the bin.
    if tgt_dist <= bin_half_width:
        thresh_dist = bin_half_width

    else:
        # Figure out how far out we have to go to get `min_num_events`
        min_num_events_dist = np.exp(sorted_abs_dist[min_num_evts-1])

        # If this is _further_ than `tgt_dist`, then we have to suck it up
        # and go `min_num_events_dist` away to ensure we collect enough events
        if min_num_events_dist > tgt_dist:
            thresh_dist = min_num_events_dist

        # But if we can stop at tgt_dist away and get more events than
        # min_num_evts, we should do that to maximize our statistics
        else:
            thresh_dist = tgt_dist

    if bin.is_log:
        lower_edge = bin_wtd_center / thresh_dist
        upper_edge = bin_wtd_center * thresh_dist
    else:
        lower_edge = bin_wtd_center - thresh_dist
        upper_edge = bin_wtd_center + thresh_dist

    keep_criteria = (
        '({field:s} >= {lower:.15e}) & ({field:s} <= {upper:0.15e})'.format(
            field=bin.name, lower=lower_edge, upper=upper_edge)
    )
    events_subset = events.applyCut(keep_criteria=keep_criteria)
    logging.trace('cut criteria:                  %s', keep_criteria)
    logging.trace('total events in that group:    %s',
                  len(events[repr_flavint]['true_energy']))
    logging.trace('events in that group selected: %s',
                  len(events_subset[repr_flavint]['true_energy']))

    return events_subset


FTYPE_MIN = np.finfo(FTYPE).min
FTYPE_MAX = np.finfo(FTYPE).max
def inf2finite(x):
    return np.clip(x, a_min=FTYPE_MIN, a_max=FTYPE_MAX)


def fold_coszen_error(coszen_error, randomize=False):
    """Fold coszen error above 1 down, and below -1 up.

    Parameters
    ----------
    coszen_error
        Cosine-zenith errors, found by (coszen_reco - coszen_true).

    randomize : bool
        Randomizes the errors about 0, such that the full distribution of
        coszen errors looks good to the eye when plotted against true coszen.
        This is not necessary, though, as a computational step (the underlying
        distribution of coszen error is the same with or without
        randomization).

    Returns
    -------
    folded_coszen_error

    """
    if randomize:
        rnd = np.random.RandomState()
        random_sign = rnd.choice((-1, +1), size=coszen_error.shape)
        coszen_error = coszen_error * random_sign
        folded_coszen_error = coszen_error
    else:
        folded_coszen_error = deepcopy(coszen_error)

    mask = coszen_error > 1
    folded_coszen_error[mask] = 2 - coszen_error[mask]
    mask = coszen_error < -1
    folded_coszen_error[mask] = -2 - coszen_error[mask]

    return folded_coszen_error


def weight_coszen_tails(cz_error, cz_bin, input_weights=None):
    """Calculate weights that compensate for fewer points in the inherent tails
    of the coszen-error distribution.

    Parameters
    ----------
    cz_error : array
        Cosine-zenith erors. I.e., coszen_reco - coszen_true values

    cz_bin : OneDimBinning in true-coszen
        The true-coszen bin in which the coszen errors were computed.

    input_weights : None or array of same size as `cz_error`
        Existing weights that are to be multiplied by the tail weights to
        arrive at an overall weight for each event. If provided, must have same
        shape as `cz_error`.

    Returns
    -------
    weights : array
    error_limits : tuple of two scalars
        (error_lower_lim, error_upper_lim)

    """
    # Create all-ones weights vector if a weights field
    # hasn't been specified
    if input_weights is None:
        weights = np.ones_like(cz_error)
    else:
        weights = deepcopy(input_weights)

    # Shortcuts for accessing bin edges
    bin_lower_edge = np.min(cz_bin.bin_edges.m)
    bin_upper_edge = np.max(cz_bin.bin_edges.m)

    # Identify limits of possible error distribution
    error_lower_lim = -1 - bin_upper_edge
    error_upper_lim = +1 - bin_lower_edge
    error_limits = (error_lower_lim, error_upper_lim)

    # Identify inner limits of the tails
    lower_tail_upper_lim = -1 - bin_lower_edge
    upper_tail_lower_lim = +1 - bin_upper_edge

    # Identify tail widths
    lower_tail_width = lower_tail_upper_lim - error_lower_lim
    upper_tail_width = error_upper_lim - upper_tail_lower_lim

    # Create masks for events in the tails
    upper_tail_mask = cz_error > upper_tail_lower_lim
    lower_tail_mask = cz_error < lower_tail_upper_lim

    # Update the weights for events in the tails
    weights[lower_tail_mask] *= (
        lower_tail_width/(cz_error[lower_tail_mask] - error_lower_lim)
    )
    weights[upper_tail_mask] *= (
        upper_tail_width/(error_upper_lim - cz_error[upper_tail_mask])
    )

    return weights, error_limits


def coszen_error_edges(true_edges, reco_edges):
    """Return a list of edges in coszen-error space given 2 true-coszen edges
    and reco-coszen edges. Systematics are not implemented at thistime.

    Parameters
    ----------
    true_edges : sequence of 2 scalars
    reco_edges : sequence of scalars
    bias : scalar // NOT IMPLEMENTED YET!
    scale : scalar > 0 // NOT IMPLEMENTED YET!

    Returns
    -------
    all_dcz_binedges : list of scalars
        The interleaved coszen-error (delta-cz) bin edges found both from the
        full reco range possible (given by `true_edges` and assuming reco can
        be from -1 to +1), and from the spans due to `reco_edges`.

    reco_indices : tuple with 2 lists of scalars
        Each list contains `len(reco_edges - 1)` scalars. Thesea are the
        indices for locating the edge in `all_binedges`, corresponding to
        `(reco_edges[:-1], reco_dges[1:])`.

    """
    reco_lower_binedges = reco_edges[:-1]
    reco_upper_binedges = reco_edges[1:]
    t_lower_binedge, t_upper_binedge = true_edges
    full_reco_range_lower_binedge = -1 - true_edges[0]
    full_reco_range_upper_binedge = +1 - true_edges[1]

    czerr_lower_binedges = []
    czerr_upper_binedges = []
    for reco_lower_binedge, reco_upper_binedge in zip(reco_lower_binedges,
                                                      reco_upper_binedges):
        czerr_lower_binedges.append(reco_lower_binedge - true_upper_binedge)
        czerr_upper_binedges.append(reco_upper_binedge - true_lower_binedge)

    all_dcz_binedges = czerr_lower_binedges + czerr_upper_binedges
    all_dcz_binedges.sort()

    # Make sure the full-reco-range edges are included in "all" bin edges
    if full_reco_range_lower_binedge != all_dcz_binedges[0]:
        all_dcz_binedges.insert(0, full_reco_range_lower_binedge)
    if full_reco_range_upper_binedge != all_dcz_binedges[-1]:
        all_dcz_binedges.insert(-1, full_reco_range_upper_binedge)

    # We know the indices of the full-range edges since they're the extrema
    full_reco_range_lower_binedge_idx = 0
    full_reco_range_upper_binedge_idx = len(all_dcz_binedges) - 1

    # Find the indices corresponding to the lower and upper bin edges
    czerr_lower_binedge_indices, czerr_upper_binedge_indices = [], []
    for lower_binedge, upper_binedge in zip(czerr_lower_binedges,
                                            czerr_upper_binedges):
        czerr_lower_binedge_indices.append(
            all_dcz_binedges.index(lower_binedge)
        )
        czerr_upper_binedge_indices.append(
            all_dcz_binedges.index(upper_binedge)
        )

    reco_indices = (czerr_lower_binedge_indices, czerr_upper_binedge_indices)

    return all_dcz_binedges, reco_indices



# TODO: the below logic does not generalize to muons, but probably should
# (rather than requiring an almost-identical version just for muons). For
# example, an input arg can dictate neutrino or muon, which then sets the
# input_names and output_names.

class vbwkde(Stage):
    """
    From simulated events, a set of transforms are created which map
    bins of true events onto distributions of reconstructed events using
    variable-bandwidth kernel density estimation. These transforms can be
    accessed by [true_energy][true_coszen][reco_energy][reco_coszen].
    These distributions represent the probability that a true event
    (true_energy, true_coszen) with be reconstructed as (pid, reco_energy,
    reco_coszen).

    From these transforms and "true" event rate maps as inputs, calculates
    the reconstructed even rate maps.

    Parameters
    ----------
    params : ParamSet
        Must exclusively have parameters:

        reco_events : string or Events
            PISA Events object or filename/path to use to derive transforms, or
            a string specifying the resource location of the same.

        reco_weights_name : None or string
            Field to use in MC events to apply MC weighting for the computation

        transform_events_keep_criteria : None or string
            Additional cuts that are applied to events prior to computing
            transforms with them. E.g., "true_coszen <= 0" removes all MC-true
            downgoing events. See `pisa.core.events.Events` class for details
            on cut specifications.

        res_scale_ref : string
            One of "mean", "mode", or "zero". This is the reference point about
            which resolutions are scaled. "zero" scales about the zero-error
            point (i.e., the bin midpoint), "mean" scales about the mean of the
            KDE, and "mode" scales about the peak of the KDE.

        e_res_scale : float
            A scaling factor for energy resolutions.

        cz_res_scale : float
            A scaling factor for coszen resolutions.

        e_reco_bias : float

        cz_reco_bias : float

    particles : string
        Must be one of 'neutrinos' or 'muons' (though only neutrinos are
        supported at this time).

    input_names : string or list of strings
        Names of inputs expected. These should follow the standard PISA
        naming conventions for flavor/interaction types OR groupings
        thereof. Note that this service's outputs are named the same as its
        inputs. See Conventions section in the documentation for more info.

    transform_groups : string
        Specifies which particles/interaction types to combine together in
        computing the transforms. See Notes section for more details on how
        to specify this string

    sum_grouped_flavints : bool

    input_binning : MultiDimBinning or convertible thereto
        Input binning is in true variables, with names prefixed by "true_".
        Each must match a corresponding dimension in `output_binning`.

    output_binning : MultiDimBinning or convertible thereto
        Output binning is in reconstructed variables, which can include pid.

    disk_cache : bool, string, etc.
        Simplest to set to True or False for enabling/disabling disk caching,
        respectively, but other inputs are possible (see docs for
        `pisa.core.stage.Stage` class for more info). The KDE profiles are
        cached to disk by this service, _not_ the full transform (since the
        latter can be multiple GB, depending on input/output binning).

    transforms_cache_depth : int >= 0
        Default is 1 since transforms for this service can be huge (gigabytes)

    outputs_cache_depth : int >= 0
        Default is 20 since the outputs from this stage are generally not too
        large.

    memcache_deepcopy : bool

    debug_mode : None, bool, or string
        Whether to store extra debug info for this service.

    Notes
    -----
    The `transform_groups` string is interpreted (and therefore defined) by
    pisa.utils.flavInt.flavint_groups_string. E.g. commonly one might use:

    'nue cc + nuebar cc, numu_cc+numubar_cc, nutau_cc+nutaubar_cc, nuall_nc+nuallbar_nc'

    Any particle type not explicitly mentioned in `transform_groups` is taken
    as a singleton group.

    Plus signs add types to a group, while groups are separated by commas.
    Whitespace is ignored.

    Input "true" event rate maps should be binned finely enough that the
    smearing that is removed when we lump all events in the kde_binning
    together and KDE the reco error is not an issue.

    """
    def __init__(self, params, particles, input_names, transform_groups,
                 sum_grouped_flavints, input_binning, output_binning,
                 error_method=None, disk_cache=True, transforms_cache_depth=1,
                 outputs_cache_depth=20, memcache_deepcopy=False,
                 debug_mode=None):
        assert particles in ['neutrinos', 'muons']
        self.particles = particles
        if isinstance(transform_groups, basestring):
            self.transform_groups = flavintGroupsFromString(transform_groups)
        else:
            self.transform_groups = transform_groups
        self.sum_grouped_flavints = sum_grouped_flavints

        # All of the following params (and no more) must be passed via the
        # `params` argument.
        expected_params = (
            'reco_events', 'reco_weights_name',
            'transform_events_keep_criteria',
            'res_scale_ref', 'e_res_scale', 'cz_res_scale',
            'e_reco_bias', 'cz_reco_bias'
        )

        if isinstance(input_names, basestring):
            input_names = (''.join(input_names.split(' '))).split(',')

        # Define the names of objects expected in inputs and produced as
        # outputs
        if self.particles == 'neutrinos':
            if self.sum_grouped_flavints:
                output_names = [str(g) for g in self.transform_groups]
            else:
                output_names = input_names
        elif self.particles == 'muons':
            raise NotImplementedError('`particles` = %s not implemented.'
                                      % self.particles)
        else:
            raise ValueError('Invalid value for `particles`: "%s"'
                             % self.particles)

        # Invoke the init method from the parent class, which does a lot of
        # work for you.
        super(self.__class__, self).__init__(
            use_transforms=True,
            params=params,
            expected_params=expected_params,
            input_names=input_names,
            output_names=output_names,
            error_method=error_method,
            disk_cache=disk_cache,
            outputs_cache_depth=outputs_cache_depth,
            transforms_cache_depth=transforms_cache_depth,
            memcache_deepcopy=memcache_deepcopy,
            input_binning=input_binning,
            output_binning=output_binning,
            debug_mode=debug_mode
        )

        #
        # Can do the below now that binning has been set up in call to Stage's
        # init
        #

        self.validate_binning()

        # We have some number of dimensions to characterize (KDE). Each one of
        # these has "dimension dependencies," whereby we need to isolate events
        # in _those_ dimensions before we can characterize the dimension. By
        # isolating events as such, we reduce or remove correlations among the
        # involved dimensions. Ego we reduce the dimensionality that needs to
        # be characterized and so can get away with fewer events for the
        # characterization. (This also means that we can simply use
        # one-dimensional KDE for the characterization.)

        self.kde_binning = OrderedDict()
        """OrderedDict formatted as `{out_dim.basename: MultiDimBinning}`, used
        for binning events for characterizing resolutions in the output
        dimensions"""

        # Only characterize dimensions that are to be output
        for out_dim in output_binning:
            dep_binnings = []
            for dep_dim_name in KDE_DIM_DEPENDENCIES[out_dim.basename]:
                # Use KDE_TRUE_BINNING for KDE binning where possible
                if dep_dim_name in KDE_TRUE_BINNING[out_dim.basename]:
                    dep_binnings.append(
                        KDE_TRUE_BINNING[out_dim.basename][dep_dim_name]
                    )

                # Otherwise (e.g. pid) must be the same as output_binning, so
                # get binning spec from there
                elif dep_dim_name in output_binning:
                    dep_binnings.append(output_binning[dep_dim_name])

                else:
                    raise ValueError('Dimension "%s" is not handled.'
                                     % out_dim.name)

            self.kde_binning[out_dim.basename] = MultiDimBinning(dep_binnings)

        self.include_attrs_for_hashes('particles')
        self.include_attrs_for_hashes('transform_groups')
        self.include_attrs_for_hashes('kde_binning')
        self.include_attrs_for_hashes('sum_grouped_flavints')

        self.kde_info = OrderedDict()
        """OrderedDict containing KDEProfile's. Structure is:
            {dim_basename: {flavintgroup: {(Coord): (KDEProfile)}}}
        For example:
            {'pid': {
                numu_cc: {
                    (true_energy=0): (x=[...], density=[...]),
                    (true_energy=1): (x=[...], density=[...])
                },
                numubar_cc: {
                    (true_energy=0): (x=[...], density=[...]),
                    (true_energy=1): (x=[...], density=[...])
                },
             'energy': {
                numu_cc: {
                    (true_energy=0): (x=[...], density=[...]),
                    (true_energy=1): (x=[...], density=[...])
                },
                numubar_cc: {
                    (true_energy=0): (x=[...], density=[...]),
                    (true_energy=1): (x=[...], density=[...])}
                },
            }}
        """

        self._kde_hashes = OrderedDict()

    def validate_binning(self):
        """Require input dimensions of "true_energy" and "true_coszen" (in any
        order).

        Require output dimensions of "reco_energy" and "reco_coszen", and
        optionally allow output dimension of "pid"; can be in any order.

        """
        input_names = set(self.input_binning.names)
        assert input_names == set(['true_energy', 'true_coszen']), \
                str(input_names)

        output_names = set(self.output_binning.names)
        outs1 = set(['reco_energy', 'reco_coszen'])
        outs2 = set(['reco_energy', 'reco_coszen', 'pid'])
        assert output_names == outs1 or output_names == outs2

        input_basenames = set(self.input_binning.basenames)
        output_basenames = set(self.output_binning.basenames)
        for base_d in input_basenames:
            assert base_d in output_basenames

    def _compute_transforms(self):
        """Generate reconstruction smearing kernels by estimating the
        distribution of reconstructed events corresponding to each bin of true
        events using VBW-KDE.

        The resulting transform is an MxN-dimensional histogram, where M is the
        dimensionality of the input binning and N is the dimensionality of the
        output binning. The transform maps the truth bin counts to the
        reconstructed bin counts.

        I.e., for the case of 1D input binning, the ith element of the
        reconstruction kernel will be a map showing the distribution of events
        over all the reco space from truth bin i. This will be normalised to
        the total number of events in truth bin i.

        """
        self.load_events(self.params.reco_events)
        self.cut_events(self.params.transform_events_keep_criteria)

        # Compute the KDEs for each (pid, E) bin (this is then propagated to
        # each (pid, E, cz) bin, as the transform is assumed to not be
        # cz-dependent)
        self.characterize_resolutions()


        # Apply scaling factors and figure out the area per bin for each KDE
        xforms = []
        for xform_flavints in self.transform_groups:
            # Generate the kernel just once for all flavints grouped together
            # for computing the transform
            reco_kernel = self.generate_smearing_kernel(xform_flavints)

            if self.sum_grouped_flavints:
                xform_input_names = []
                for input_name in self.input_names:
                    input_flavs = NuFlavIntGroup(input_name)
                    if len(set(xform_flavints).intersection(input_flavs)) > 0:
                        xform_input_names.append(input_name)

                for output_name in self.output_names:
                    if output_name not in xform_flavints:
                        continue
                    xform = BinnedTensorTransform(
                        input_names=xform_input_names,
                        output_name=output_name,
                        input_binning=self.input_binning,
                        output_binning=self.output_binning,
                        xform_array=reco_kernel.hist,
                        sum_inputs=self.sum_grouped_flavints
                    )
                    xforms.append(xform)
            else:
                for input_name in self.input_names:
                    if input_name not in xform_flavints:
                        continue
                    xform = BinnedTensorTransform(
                        input_names=input_name,
                        output_name=input_name,
                        input_binning=self.input_binning,
                        output_binning=self.output_binning,
                        xform_array=reco_kernel.hist,
                        sum_inputs=self.sum_grouped_flavints
                    )
                    xforms.append(xform)

        return TransformSet(transforms=xforms)

    def characterize_resolutions(self):
        """Compute the KDEs for each (pid, E) bin. If PID is not present, this
        is just (E). The results are propagated to each (pid, E, cz) bin, as
        the transforms are assumed to not be cz-dependent.

        The results are cached to disk and simply loaded from that cache to
        avoid re-computation.

        Returns
        -------
        all_kde_info : OrderedDict with format:
            {
                '<flavint group 1>': kde_info,
                '<flavint group 2>': kde_info,
                ...
            }
            where the format of kde_info is defined in `compute_kdes`

        """
        weights_name = self.params.reco_weights_name.value

        # TODO: add sourcecode hash for pisa.utils.vbwkde module (entire module
        #       is probably safest, due to all the functions there)

        hash_items = [self.source_code_hash, self.events.hash]

        for kde_dim, dep_dims_binning in self.kde_binning.items():
            logging.debug('Working on KDE dimension "%s"', kde_dim)
            new_hash = hash_obj(deepcopy(hash_items) + [dep_dims_binning.hash])

            # See if we already have correct kde_info for this dim
            if (kde_dim in self._kde_hashes
                    and new_hash == self._kde_hashes[kde_dim]):
                logging.debug('  > Already have KDEs for "%s"', kde_dim)
                continue

            # Try to load from disk cache
            if self.disk_cache is not None and new_hash in self.disk_cache:
                logging.debug('  > Loading KDEs for "%s" from disk cache',
                              kde_dim)
                self.kde_info[kde_dim] = self.disk_cache[new_hash]
                self._kde_hashes[kde_dim] = new_hash
                continue

            # Reset the hash for this dim so if anything fails below, the wrong
            # info won't be loaded
            self._kde_hashes[kde_dim] = None

            # Clear out all previous kde info
            self.kde_info[kde_dim] = OrderedDict()
            for flavintgroup in self.transform_groups:
                self.kde_info[kde_dim][flavintgroup] = OrderedDict()

            # First element in the list is the full set of events. Then the
            # next element is the previous element's events but with inbounds
            # cut from the first dim applied, and the next element has second
            # dim's inbounds cut applied to the first dim's inbounds cut, and
            # so forth.
            #
            # The final dimensional dependency is _not_ cut on, allowing for
            # events to be collected exceeding its boundaries so that enough
            # statistics can be acquired.
            cut_events = [self.events] * len(dep_dims_binning)

            for bin_num, bin_binning in enumerate(dep_dims_binning.iterbins()):
                bin_dims = bin_binning.dims
                bin_coord = dep_dims_binning.index2coord(bin_num)
                logging.debug('  > characterizing bin %s (%d of %d)',
                              bin_coord, bin_num+1, dep_dims_binning.size)

                # Apply cuts for all but the last dimension dependency
                for dim_num, dim in enumerate(bin_dims[:-1]):
                    if dim.inbounds_criteria not in cut_events[dim_num + 1]:
                        # Apply this dim cut to "parent" dim's events. Note
                        # that indices are off by 1, so this dim's events are
                        # stored in element dim_num + 1 and its parent is at
                        # [dim_num]
                        cut_events[dim_num + 1] = (
                            cut_events[dim_num].applyCut(dim.inbounds_criteria)
                        )

                for flavintgroup in self.transform_groups:
                    logging.trace('    > flavintgroup = %s', flavintgroup)
                    repr_flavint = flavintgroup[0]

                    flav_events = collect_enough_events(
                        events=cut_events[-1], flavint=repr_flavint,
                        bin=bin_dims[-1]
                    )[repr_flavint]

                    weights_specified = False
                    if weights_name in flav_events:
                        weights_specified = True
                        weights = flav_events[weights_name]
                        weights = weights * (len(weights)/np.sum(weights))
                    else:
                        weights = None

                    # TODO: adjust `n_dct`, may want to revise down or separate
                    #       out `n_dct` from `n_eval` by manually setting
                    #       `evaluate_at`

                    if kde_dim == 'pid':
                        feature = flav_events['pid']
                        fmin, fmax = min(feature), max(feature)
                        half_width = (fmax - fmin)/2
                        lowerlim = fmin - half_width
                        upperlim = fmax + half_width
                        vbwkde_kwargs = dict(
                            n_dct=int(2**6),
                            #min=lowerlim, max=upperlim,
                            evaluate_at=np.linspace(lowerlim, upperlim,
                                                    int(5e3))
                        )

                    elif kde_dim == 'energy':
                        feature = np.log(flav_events['reco_energy']
                                         / flav_events['true_energy'])
                        fmin, fmax = min(feature), max(feature)
                        half_width = (fmax - fmin)/2
                        lowerlim = fmin - half_width
                        upperlim = fmax + half_width
                        vbwkde_kwargs = dict(
                            n_dct=int(2**6),
                            min=lowerlim, max=upperlim,
                            evaluate_at=np.linspace(lowerlim, upperlim,
                                                    int(5e3))
                        )

                    elif kde_dim == 'coszen':
                        feature = (flav_events['reco_coszen']
                                   - flav_events['true_coszen'])
                        weights, error_limits = weight_coszen_tails(
                            cz_error=feature, cz_bin=bin_binning.true_coszen,
                            input_weights=weights
                        )
                        weights = weights * (len(weights)/np.sum(weights))
                        error_width = error_limits[1] - error_limits[0]
                        error_mid = 0.5*(error_limits[0] + error_limits[1])

                        vbwkde_kwargs = dict(
                            n_dct=int(2**6),
                            min=error_limits[0],
                            max=error_limits[1],
                            evaluate_at=np.linspace(error_limits[0],
                                                    error_limits[1],
                                                    int(5e3))
                        )

                    else:
                        raise NotImplementedError('Applying KDEs to dimension'
                                                  ' "%s" is not implemented.'
                                                  % kde_dim)

                    _, x, density = vbwkde_func(feature, weights=weights,
                                                **vbwkde_kwargs)

                    normalized_density = density / np.sum(density)

                    self.kde_info[kde_dim][flavintgroup][bin_coord] = (
                        KDEProfile(x=x, density=normalized_density)
                    )

            self._kde_hashes[kde_dim] = new_hash

            if self.disk_cache is not None:
                self.disk_cache[new_hash] = self.kde_info[kde_dim]

    def generate_smearing_kernel(self, flavintgroup):
        """Construct a smearing kernel for the flavintgroup specified.

        The resulting array can be indexed for clarity using two indexes,
        one for input and one for output dimensions. E.g.:
           kernel[true_energy_i, true_coszen_j][reco_energy_k, reco_coszen_l]
        or if PID is included,
           kernel[true_energy_i, true_coszen_j][reco_energy_k, reco_coszen_l, pid_m]
        where ordering within the two sets of brackets is arbitrary.

        In other words, the indices point from a single MC-true histogram bin
        indexed by (i,j) to a single reco histogram bin indexed by (k,l{,m}).

        Parameters
        ----------
        flavintgroup

        Returns
        -------
        kernel : MxN-dimensional array of float
            Mapping from the number of events in each bin of the 2D
            MC-true-events histogram to the number of events reconstructed in
            each bin of the 2D reconstructed-events histogram. Dimensions are
                input_binning.shape x output_binning.shape
            Note that PID binning can optionally be included, and behaves as
            another output (reco) dimension.

        """
        logging.debug('Generating smearing kernel for %s', flavintgroup)

        pid_kde_profiles = self.kde_info['pid'][flavintgroup]
        e_kde_profiles = self.kde_info['energy'][flavintgroup]
        dcz_kde_profiles = self.kde_info['coszen'][flavintgroup]

        kde_binning = self.kde_binning

        # Events in each input bin will be smeared out into all output bins.
        # To characterize the smearing, get the KDE profile from each input
        # dimension that was created from events closest to this input bin.

        kernel_binning = self.input_binning * self.output_binning
        kernel = kernel_binning.empty(name='kernel')

        # Shortcut names
        true_energy = self.input_binning.true_energy
        true_coszen = self.input_binning.true_coszen
        reco_energy = self.output_binning.reco_energy
        reco_coszen = self.output_binning.reco_coszen
        pid_binning = self.output_binning.pid

        # Get the following once so we don't have to repeat within the loops

        pid_kde_e_centers = inf2finite(
            kde_binning['pid'].true_energy.weighted_centers.m
        )
        e_kde_e_centers = kde_binning['energy'].true_energy.weighted_centers.m
        cz_kde_e_centers = kde_binning['coszen'].true_energy.weighted_centers.m
        cz_kde_cz_centers = kde_binning['coszen'].true_coszen.weighted_centers.m

        num_pid_bins = len(pid_binning)
        num_reco_cz_bins = len(reco_coszen)

        e_res_scale = self.params.e_res_scale.value.m
        cz_res_scale = self.params.cz_res_scale.value.m
        cz_reco_bias = self.params.cz_reco_bias.value.m

        pid_edges = inf2finite(pid_binning.bin_edges.m)
        # NOTE: when we get scaling-about-the-mode working, will have to change
        # this.
        reco_e_edges = (
            np.log(inf2finite(reco_energy.bin_edges.m)
                   - self.params.e_reco_bias.value.m) / e_res_scale
        )
        reco_cz_edges = reco_coszen.bin_edges.m
        reco_cz_lower_edges = reco_cz_edges[:-1]
        reco_cz_upper_edges = reco_cz_edges[1:]

        # Compute info for the delta-coszen bin edges, for each true-coszen
        # input bin and each reco-coszen output bin; also get the bin edges
        # across the entire _possible_ reco-coszen range, for purposes of
        # normalization.

        # NOTE: reco-coszen systematics will be applied to the reco-coszen
        # bin edges, but (for proper treatment) must know the mode location of
        # the coszen KDE profile (which is only known in the innermost loop).
        # The bias isn't costly to do, but scaling will require shifting the
        # mode to 0, applying the scale factor, and then shifting back (or
        # shifting by the bias). So we need to compute as much here as possible
        # (since it's relatively cheap) but we may still need to do some of the
        # work within the innermost loop (yuck).

        #if reco_coszen.is_lin:
        #    rcz_min = reco_cz_edges[0]
        #    rcz_max = reco_cz_edges[-1]
        #    dcz = np.mean(np.diff(reco_cz_edges))
        #    def czbinfunc(x, weights):
        #        inside_mask = (x >= rcz_min) & (x <= rcz_max)
        #        intx = np.int64((x[inside_mask] - reco_cz_edges[0]) / dcz)
        #        return np.bincount(intx, weights[inside_mask])
        #else:
        #    def czbinfunc(x, weights):
        #        out, _ = np.histogram(x, weights=weights, bins=reco_cz_edges)
        #        return out

        true_e_centers = inf2finite(true_coszen.weighted_centers.m)
        true_cz_centers = true_coszen.weighted_centers.m
        true_cz_edges = true_coszen.bin_edges.m
        true_cz_edge_pairs = [(e0, e1) for e0, e1 in zip(true_cz_edges[:-1],
                                                         true_cz_edges[1:])]

        allbins_dcz_edge_info = []
        for true_cz_binedges in izip(true_cz_centers,
                                     true_coszen.iteredgetuples()):
            all_dcz_binedges, cz_reco_indices = coszen_error_edges(
                true_cz_edges, reco_cz_edges
            )
            allbins_dcz_edge_info.append(
                dict(all_dcz_binedges=all_dcz_binedges,
                     cz_reco_indices=cz_reco_indices)
            )

        cz_closest_cz_indices = [
            np.argmin(np.abs(fold_coszen_error(cz_kde_cz_centers
                                               - true_cz_center)))
            for true_cz_center in true_cz_centers
        ]

        # Define only what needs to be done to coszen KDE profile based on
        # systematics that actually have an effect
        op_scale_shift_cz_kde = 'cz_kde_profile.x'
        if cz_res_scale != 1:
            op_scale_shift_cz_kde += ' * cz_res_scale'
        if cz_reco_bias != 0:
            op_scale_shift_cz_kde += ' + (cz_reco_bias + true_cz_center)'
        else:
            op_scale_shift_cz_kde += ' + true_cz_center'

        for true_e_bin_num, true_e_center in enumerate(true_e_centers):
            logging.debug('  > Working on true_e_bin_num %d of %d',
                          true_e_bin_num+1, true_energy.size)

            idx = np.argmin(np.abs(np.log(true_e_center/pid_kde_e_centers)))
            pid_closest_kde_coord = kde_binning['pid'].Coord(true_energy=idx)

            # Figure out PID fractions
            pid_kde_profile = pid_kde_profiles[pid_closest_kde_coord]
            pid_fractions, _ = np.histogram(
                pid_kde_profile.x, weights=pid_kde_profile.density,
                bins=pid_edges, density=False
            )

            for pid_bin_num in xrange(num_pid_bins):
                pid_fraction = pid_fractions[pid_bin_num]

                energy_indexer = kernel_binning.defaults_indexer(
                    true_energy=true_e_bin_num,
                    pid=pid_bin_num
                )

                # If PID is zero, no need to figure out anything further for
                # (..., true_energy=this, ..., pid=this, ...) bins
                if not np.any(np.abs(pid_fraction) > EPSILON):
                    kernel[energy_indexer] = 0
                    continue

                # Get the energy smearing for this (PID, true-energy) bin
                closest_e_idx = np.argmin(
                    np.abs(np.log(true_e_center / e_kde_e_centers))
                )
                e_closest_kde_coord = kde_binning['energy'].Coord(
                    pid=pid_bin_num,
                    true_energy=closest_e_idx
                )
                e_kde_profile = e_kde_profiles[e_closest_kde_coord]

                # TODO: scale about the mode of the KDE! i.e., implement
                #       `res_scale_ref`

                # Tranform reco-energy bin edges into the log-ratio space
                # where we characterized the energy resolutions
                e_edges = reco_e_edges - np.log(true_e_center)/e_res_scale

                energy_fractions, _ = np.histogram(
                    e_kde_profile.x, weights=e_kde_profile.density,
                    bins=e_edges, density=False
                )

                kernel[energy_indexer] = pid_fraction * energy_fractions

                # Do this just once for the energy bin, prior to looping over
                # coszen
                cz_closest_e_idx = np.argmin(np.abs(
                    np.log(true_e_center / cz_kde_e_centers)
                ))

                # TODO: implement `res_scale_ref` and `cz_reco_bias`!

                # Get the closest coszen smearing for this
                # (PID, true-coszen, true-energy) bin
                for true_cz_bin_num, (true_cz_lower_edge, true_cz_upper_edge) \
                        in enumerate(true_cz_edge_pairs):
                    cz_closest_cz_idx = cz_closest_cz_indices[true_cz_bin_num]
                    cz_closest_kde_coord = kde_binning['coszen'].Coord(
                        pid=pid_bin_num,
                        true_coszen=cz_closest_cz_idx,
                        true_energy=cz_closest_e_idx
                    )

                    # Get KDE profile (in "delta-cz" space)
                    dcz_kde_profile = dcz_kde_profiles[cz_closest_kde_coord]

                    dcz_edge_info = allbins_dcz_edge_info[true_cz_bin_num]

                    hist, _ = np.histogram(
                        cz_kde_profile.x, weights=cz_kde_profile.weights,
                        bins=dcz_edge_info['all_dcz_binedges'], density=False
                    )

                    # Collect the relevant hist sections to describe each
                    # quantity of interest, starting with normalization
                    norm = 1/np.sum(hist)
                    reco_indices = dcz_edge_info['cz_reco_indices']
                    reco_cz_fractions = []
                    for reco_lower, reco_upper in izip(reco_indices):
                        reco_cz_fractions.append(
                            norm * np.sum(hist[reco_lower:reco_upper])
                        )

                    coszen_indexer = kernel_binning.defaults_indexer(
                        true_energy=true_e_bin_num,
                        true_coszen=true_cz_bin_num,
                        pid=pid_bin_num
                    )
                    kernel.hist[coszen_indexer] *= coszen_fractions

        return kernel
