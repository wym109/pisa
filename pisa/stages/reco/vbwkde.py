# PISA author: Timothy C. Arlen
#              tca3@psu.edu
#
# CAKE author: Steven Wren
#              steven.wren@icecube.wisc.edu
#
# date:   2016-05-27

"""This reco service produces a set of transforms mapping true
events values (energy and coszen) onto reconstructed values.

For each bin in true energy and true coszen, a corresponding distribution of
reconstructed energy and coszen values is estimated using a variable-bandwidth
KDE.

These transforms are used to produce reco event rate maps.
"""


from __future__ import division

from collections import OrderedDict
from copy import deepcopy
from string import ascii_lowercase

import numpy as np
from scipy.interpolate import interp1d

from pisa.core.binning import MultiDimBinning
from pisa.core.events import Events
from pisa.core.param import Param, ParamSet
from pisa.core.stage import Stage
from pisa.core.transform import BinnedTensorTransform, TransformSet
from pisa.utils import kde, confInterval
from pisa.utils.flavInt import flavintGroupsFromString, NuFlavIntGroup
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging, set_verbosity
from pisa.utils.profiler import profile, line_profile

EPSILON = 1e-9

# NOTE: If we want to scale the resolutions about some reference point
# (x_r) and then shift them by `shift`:
#
#   x1 = x_r + (x - x_r)*scale + shift
#
# but we should take the sample points as fixed, and instead sample
# from locations we desire but transformed to the correct locatons on
# the original curve
#
#    x = (x1 - x_r - shift)/scale + x_r
#

## TODO: move these functions elsewhere in the codebase (utils.something)
def abs2rel(abs_coords, abs_bin_midpoint, rel_scale_ref, scale, abs_obj_shift):
    """Viewing an object that is defined in a relative coordinate space, """
    return (
        (abs_coords - abs_bin_midpoint - abs_obj_shift + rel_scale_ref)/scale
    )


def rel2abs(rel_coords, abs_bin_midpoint, rel_scale_ref, scale, abs_obj_shift):
    """Convert coordinates defined relative to a bin to absolute
    coordinates.

    """
    return (
        rel_coords*scale - rel_scale_ref + abs_bin_midpoint + abs_obj_shift
    )
    #return (
    #    (rel_coords - rel_scale_ref)*scale + rel_scale_ref
    #    + abs_bin_midpoint + abs_obj_shift
    #)

def test_abs2rel():
    xabs = np.array([-2, -1, 0, 1, 2])

    # The identity transform
    xrel = abs2rel(abs_coords=xabs, abs_bin_midpoint=0, rel_scale_ref=0,
                   scale=1, abs_obj_shift=0)
    assert np.all(xrel == xabs)

    # Absolute bin midpoint: if the bin is centered at 2, then the relative
    # coordinates should range from -4 to 0
    xrel = abs2rel(abs_coords=xabs, abs_bin_midpoint=2, rel_scale_ref=0,
                   scale=1, abs_obj_shift=0)
    assert xrel[0] == -4 and xrel[-1] == 0

    # Scale: an object that is 4 units wide absolute space scaled by 2
    # should appear to be 2 units wide in relative space... or in other words,
    # the relative coordinates should be spaced more narrowly to one another by
    # a factor of 2 than coordinates in the absolute space
    xrel = abs2rel(abs_coords=xabs, abs_bin_midpoint=0, rel_scale_ref=0,
                   scale=2, abs_obj_shift=0)
    assert (xabs[1]-xabs[0]) == 2*(xrel[1]-xrel[0])

    # Relative scale reference point: If an object living in relative space is
    # centered at 1 and scaled by 2 with rel_scale_ref=1, then it should still
    # be centered at 1 but be wider by a factor of 2. This means that all
    # coordinates must scale relative to 1: 2 gets 2x closer to 1, 0 gets 2x
    # closer to 1, etc
    # if stuff stuff
    xrel = abs2rel(abs_coords=xabs, abs_bin_midpoint=0, rel_scale_ref=1,
                   scale=2, abs_obj_shift=0)
    assert (xabs[1]-xabs[0]) == 2*(xrel[1]-xrel[0])
    assert np.all(xrel == np.array([-0.5, 0, 0.5, 1, 1.5]))
    xrel = abs2rel(abs_coords=xabs, abs_bin_midpoint=0, rel_scale_ref=-2,
                   scale=2, abs_obj_shift=0)
    assert np.all(xrel == np.array([-2, -1.5, -1, -0.5, 0]))

    # Shift: an object that lives in the relative space centered at 0 should
    # now be centered at 1 in absolute space. Relative coordinates should be
    # shifted to the left such that object appears to be shifted to the right.
    xrel = abs2rel(abs_coords=xabs, abs_bin_midpoint=0, rel_scale_ref=0,
                   scale=1, abs_obj_shift=1)
    assert xrel[0] == -3 and xrel[-1] == 1

    logging.info('<< PASSED : test_abs2rel >>')

def test_rel2abs():
    xabs = np.array([-2, -1, 0, 1, 2])
    kwargs = dict(abs_bin_midpoint=12, rel_scale_ref=-3.3, scale=5.4,
                  abs_obj_shift=19)
    xrel = abs2rel(xabs, **kwargs)
    assert np.allclose(rel2abs(abs2rel(xabs, **kwargs), **kwargs), xabs)
    logging.info('<< PASSED : test_rel2abs >>')


# TODO: the below logic does not generalize to muons, but probably should
# (rather than requiring an almost-identical version just for muons). For
# example, an input arg can dictate neutrino or muon, which then sets the
# input_names and output_names.

class vbwkde(Stage):
    """
    From the simulation file, a set of 4D transforms are created which map
    bins of true events onto distributions of reconstructed events using
    variable-bandwidth kernel density estimation. These transforms can be
    accessed by [true_energy][true_coszen][reco_energy][reco_coszen].
    These distributions represent the probability that a true event
    (true_energy, true_coszen) with be reconstructed as (reco_energy,
    reco_coszen).

    From these transforms and the true event rate maps, calculates
    the reconstructed even rate templates.

    Parameters
    ----------
    params : ParamSet
        Must exclusively have parameter:

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
            One of 'mean', 'mode', or 'zero'. This is the reference point about
            which resolutions are scaled. 'zero' scales about the zero-error
            point (i.e., the bin midpoint), 'mean' scales about the mean of the
            KDE, and 'mode' scales about the peak of the KDE.

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
        Output binning is in reconstructed variables, with names (traditionally
        in PISA but not necessarily) prefixed by "reco_". Each must match a
        corresponding dimension in `input_binning`.

    disk_cache

    transforms_cache_depth

    outputs_cache_depth

    debug_mode : None, bool, or string
        Whether to store extra debug info for this service.

    Notes
    -----
    The `transform_groups` string is interpreted (and therefore defined) by
    pisa.utils.flavInt.flavint_groups_string. E.g. commonly one might use:

    'nue_cc+nuebar_cc; numu_cc+numubar_cc; nutau_cc+nutaubar_cc; nuall_nc+nuallbar_nc'

    Any particle type not explicitly mentioned is taken as a singleton group.
    Commas and plus signs add types to a group, while groups are separated by
    semicolons. Whitespace is ignored, so add whitespace to the string for
    readability.

    """
    def __init__(self, params, particles, input_names, transform_groups,
                 sum_grouped_flavints, input_binning, output_binning,
                 error_method=None, disk_cache=None, transforms_cache_depth=20,
                 outputs_cache_depth=20, memcache_deepcopy=True,
                 debug_mode=None):
        assert particles in ['neutrinos', 'muons']
        self.particles = particles
        """Whether stage is instantiated to process neutrinos or muons"""

        self.transform_groups = flavintGroupsFromString(transform_groups)
        """Particle/interaction types to group for computing transforms"""

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

        # Can do these now that binning has been set up in call to Stage's init
        self.validate_binning()
        self.include_attrs_for_hashes('particles')
        self.include_attrs_for_hashes('transform_groups')
        self.include_attrs_for_hashes('sum_grouped_flavints')

        self._kde_hash = None
        self.instantiate_disk_cache()

    def validate_binning(self):
        assert set(['energy', 'coszen']) == set(self.input_binning.basenames)
        assert self.input_binning.basenames  == self.output_binning.basenames

    @profile
    def _compute_transforms(self):
        """Generate reconstruction "smearing kernels" by estimating the
        distribution of reconstructed events corresponding to each bin of true
        events using VBWKDE.

        The resulting transform is a 2N-dimensional histogram, where N is the
        dimensionality of the input binning. The transform maps the truth bin
        counts to the reconstructed bin counts.

        I.e., for the case of 1D input binning, the ith element of the
        reconstruction kernel will be a map showing the distribution of events
        over all the reco space from truth bin i. This will be normalised to
        the total number of events in truth bin i.

        """
        self.load_events(self.params.reco_events)
        self.cut_events(self.params.transform_events_keep_criteria)

        # Computational units must be the following for compatibility with
        # events file
        comp_units = dict(
            true_energy='GeV', true_coszen=None, true_azimuth='rad',
            reco_energy='GeV', reco_coszen=None, reco_azimuth='rad'
        )

        # Select only the units in the input/output binning for conversion
        # (can't pass more than what's actually there)
        in_units = {dim: unit for dim, unit in comp_units.items()
                    if dim in self.input_binning}
        out_units = {dim: unit for dim, unit in comp_units.items()
                     if dim in self.output_binning}

        # These binnings will be in the computational units defined above
        input_binning = self.input_binning.to(**in_units)
        output_binning = self.output_binning.to(**out_units)

        self.get_all_kde_info()

        # Apply scaling factors and figure out the area per bin for each KDE
        transforms = []
        for xform_flavints in self.transform_groups:
            reco_kernel = self.compute_kernel(
                kde_info=self.all_kde_info[str(xform_flavints)],
                binning=self.input_binning,
                e_res_scale=self.params.e_res_scale,
                cz_res_scale=self.params.cz_res_scale,
                e_reco_bias=self.params.e_reco_bias,
                cz_reco_bias=self.params.cz_reco_bias,
                res_scale_ref='mode'
            )

            if self.sum_grouped_flavints:
                xform_input_names = []
                for input_name in self.input_names:
                    input_flavs = NuFlavIntGroup(input_name)
                    if len(set(xform_flavints).intersection(input_flavs)) > 0:
                        xform_input_names.append(input_name)
                for output_name in self.output_names:
                    if not output_name in xform_flavints:
                        continue
                    xform = BinnedTensorTransform(
                        input_names=xform_input_names,
                        output_name=output_name,
                        input_binning=self.input_binning,
                        output_binning=self.output_binning,
                        xform_array=reco_kernel,
                        sum_inputs=self.sum_grouped_flavints
                    )
                    transforms.append(xform)
            else:
                for input_name in self.input_names:
                    if input_name not in xform_flavints:
                        continue
                    xform = BinnedTensorTransform(
                        input_names=input_name,
                        output_name=input_name,
                        input_binning=self.input_binning,
                        output_binning=self.output_binning,
                        xform_array=reco_kernel,
                    )
                    transforms.append(xform)

        return TransformSet(transforms=transforms)

    @profile
    def get_all_kde_info(self):
        """Load from cache or recompute."""
        kde_hash = hash_obj([self.source_code_hash,
                             self.input_binning.hash,
                             self.params.reco_events.value,
                             self.params.transform_events_keep_criteria,
                             self.transform_groups,
                             self.sum_grouped_flavints])
        logging.debug('kde_hash = %s' %kde_hash)
        if (self._kde_hash is not None and kde_hash == self._kde_hash
            and hasattr(self, 'all_kde_info')):
            return

        logging.trace('no match')
        logging.trace('self._kde_hash: %s' % self._kde_hash)
        logging.trace('kde_hash: %s' % kde_hash)
        logging.trace('hasattr: %s' % hasattr(self, 'all_kde_info'))

        try:
            self.all_kde_info = self.disk_cache[kde_hash]
            self._kde_hash = kde_hash
            return
        except KeyError:
            pass

        self.all_kde_info = OrderedDict()
        self.all_extra_info = OrderedDict()
        for xform_flavints in self.transform_groups:
            logging.debug("Working on %s reco kernels" %xform_flavints)
            repr_flav_int = xform_flavints.flavints()[0]
            kde_info, extra_info = self.compute_kdes(
                events=self.remaining_events[repr_flav_int],
                binning=self.output_binning
            )
            self.all_kde_info[str(xform_flavints)] = kde_info
            if self.debug_mode:
                self.all_extra_info[str(xform_flavints)] = extra_info
        self.disk_cache[kde_hash] = self.all_kde_info
        self._kde_hash = kde_hash

    @profile
    def compute_kdes(self, events, binning):
        """Construct a 4D kernel set from MC events using VBWKDE.

        Given a set of MC events and binning (which serves as both input and
        output binning), construct a KDE estimate and draw samples from this.
        From the samples, a linear interpolant is generated. There are one
        energy and one coszen interpolant genreated for each energy bin.

        NOTE: Actual limits in energy used to group events into a single "true"
        bin may be extended beyond the bin edges defined by self.ebins in order
        to gather enough events to successfully apply VBWKDE.


        Parameters
        ----------
        events : Events
            Must contain each of {true|reco}_{energy|coszen} fields

        binning : MultiDimBinning
            Must contain each of {true|reco}_{energy|coszen} dimensions

        Returns
        -------
        kde_info

        """
        # Constants. Can turn into stage args or params if that makes more
        # sense.
        OVERFIT_FACTOR = 1.0
        MIN_NUM_EVENTS = 100
        TGT_NUM_EVENTS = 300
        EPSILON = 1e-10
        ENERGY_RANGE = [0, 501] # GeV

        # TODO: handle units consistency here when Events object gets units
        e_true = events['true_energy']
        e_reco = events['reco_energy']
        cz_true = events['true_coszen']
        cz_reco = events['reco_coszen']
        ebins = binning.reco_energy
        ebin_edges = ebins.bin_edges.m_as('GeV')
        czbins = binning.reco_coszen
        czbin_edges = czbins.bin_edges.m_as('dimensionless')

        # NOTE: below defines bin centers on linear scale; other logic
        # in this method assumes this to be the case, so
        # **DO NOT USE** utils.utils.get_bin_centers in this method, which
        # may return logarithmically-defined centers instead.
        left_ebin_edges = ebin_edges[0:-1]
        right_ebin_edges = ebin_edges[1:]

        n_events = len(e_true)

        if MIN_NUM_EVENTS > n_events:
            MIN_NUM_EVENTS = n_events
        if TGT_NUM_EVENTS > n_events:
            TGT_NUM_EVENTS = n_events

        kde_info = OrderedDict()
        extra_info = OrderedDict()
        for ebin_n in range(ebins.num_bins):
            ebin_min = left_ebin_edges[ebin_n]
            ebin_max = right_ebin_edges[ebin_n]
            ebin_mid = (ebin_min+ebin_max)/2.0
            ebin_wid = ebin_max-ebin_min

            logging.debug(
                'Processing true-energy bin_n=' + format(ebin_n, 'd') + ' of ' +
                format(ebins.num_bins-1, 'd') + ', E_{nu,true} in ' +
                '[' + format(ebin_min, '0.3f') + ', ' +
                format(ebin_max, '0.3f') + '] ...'
            )

            # Absolute distance from these events' re-centered reco energies to
            # the center of this energy bin; sort in ascending-distance order
            abs_enu_dist = np.abs(e_true - ebin_mid)
            sorted_abs_enu_dist = np.sort(abs_enu_dist)

            # Grab the distance the number-"TGT_NUM_EVENTS" event is from the
            # bin center
            tgt_thresh_enu_dist = sorted_abs_enu_dist[TGT_NUM_EVENTS-1]

            # Grab the distance the number-"MIN_NUM_EVENTS" event is from the
            # bin center
            min_thresh_enu_dist = sorted_abs_enu_dist[MIN_NUM_EVENTS-1]

            # TODO: revisit the below algorithm with proper testing

            # Make threshold distance (which is half the total width) no more
            # than 4x the true-energy-bin width in order to capture the
            # "target" number of points (TGT_NUM_EVENTS) but no less than half
            # the bin width (i.e., the bin should be at least be as wide as the
            # pre-defined bin width).
            #
            # HOWEVER, allow the threshold distance (bin half-width) to expand
            # to as much as 4x the original bin full-width in order to capture
            # the "minimum" number of points (MIN_NUM_EVENTS).
            thresh_enu_dist = \
                    max(min(max(tgt_thresh_enu_dist, ebin_wid/2),
                            4*ebin_wid),
                        min_thresh_enu_dist)

            # Grab all events within the threshold distance
            in_ebin_ind = abs_enu_dist <= thresh_enu_dist
            n_in_bin = len(in_ebin_ind)

            # Record lowest/highest energies that are included in the bin
            actual_left_ebin_edge = min(ebin_min, min(e_true[in_ebin_ind]))
            actual_right_ebin_edge = max(ebin_max, max(e_true[in_ebin_ind]))

            # Extract just the neutrino-energy/coszen error columns' values for
            # succinctness
            enu_err = e_reco[in_ebin_ind] - e_true[in_ebin_ind]
            cz_err = cz_reco[in_ebin_ind] - cz_true[in_ebin_ind]

            #==================================================================
            # Neutrino energy resolution for events in this energy bin
            #==================================================================
            e_err_min = min(enu_err)
            e_err_max = max(enu_err)
            e_err_range = e_err_max-e_err_min

            # Want the lower limit of KDE evaluation to be located at the most
            # negative of
            # * 2x the distance between the bin midpoint to the reco with
            #   most-negative error
            # * 4x the bin width to the left of the midpoint
            e_lowerlim = min([
                e_err_min*4,
                -ebin_wid*4
            ])
            e_upperlim = max([
                e_err_max*4,
                ebin_wid*4
            ])
            #e_upperlim = max((np.max(ebin_edges)-ebin_mid)*1.5, e_err_max+e_err_range*0.5)
            egy_kde_lims = np.array([e_lowerlim, e_upperlim])

            # Use at least min_num_pts points and at most the next-highest
            # integer-power-of-two that allows for at least 10 points in the
            # smallest energy bin
            min_num_pts = 2**14
            min_bin_width = np.min(ebin_edges[1:]-ebin_edges[:-1])
            min_pts_smallest_bin = 5.0
            kde_range = np.diff(egy_kde_lims)
            num_pts0 = kde_range/(min_bin_width/min_pts_smallest_bin)
            kde_num_pts = int(max(min_num_pts, 2**np.ceil(np.log2(num_pts0))))
            logging.debug(
                '  N_evts=' + str(n_in_bin) + ', taken from [' +
                format(actual_left_ebin_edge, '0.3f') + ', ' +
                format(actual_right_ebin_edge, '0.3f') + ']' + ', VBWKDE lims=' +
                str(egy_kde_lims) + ', VBWKDE_N: ' + str(kde_num_pts)
            )

            ## Exapnd range of sample points for future axis scaling
            #e_factor = 1

            #low_lim_shift = egy_kde_lims[0] * (e_factor - 1)
            #upp_lim_shift = egy_kde_lims[1] * (e_factor - 1)

            #egy_kde_lims_ext = np.copy(egy_kde_lims)
            #if low_lim_shift > 0:
            #    egy_kde_lims_ext[0] = (
            #        egy_kde_lims[0] - low_lim_shift * (1./e_factor)
            #    )
            #if upp_lim_shift < 0:
            #    egy_kde_lims_ext[1] = (
            #        egy_kde_lims[1] - upp_lim_shift * (1./e_factor)
            #    )

            ## Adjust kde_num_points accordingly
            #e_kde_num_pts_ext = int(
            #    kde_num_pts*((egy_kde_lims_ext[1] - egy_kde_lims_ext[0])
            #    / (egy_kde_lims[1] - egy_kde_lims[0]))
            #)

            logging.trace('MIN/MAX = %s' %egy_kde_lims_ext)

            # Compute variable-bandwidth KDEs
            enu_bw, enu_mesh, enu_pdf = kde.vbw_kde(
                data=enu_err,
                overfit_factor=OVERFIT_FACTOR,
                MIN=egy_kde_lims[0],
                MAX=egy_kde_lims[1],
                N=e_kde_num_pts
            )

            if np.min(enu_pdf) < 0:
                # Only issue warning if the most-negative value is negative
                # beyond specified acceptable-numerical-precision threshold
                # (EPSILON)
                if np.min(enu_pdf) <= -EPSILON:
                    logging.warn(
                        "np.min(enu_pdf) < 0: Minimum value is " +
                        str(np.min(enu_pdf)) +
                        "; forcing all negative values to 0."
                    )
                # Otherwise, just quietly clip any negative values at 0
                enu_pdf = np.clip(a=enu_pdf, a_min=0, a_max=np.inf)

            assert np.min(enu_pdf) >= 0, str(np.min(enu_pdf))
            #assert np.max(enu_pdf) < 1, str(np.max(enucz_pdf))

            # Create linear interpolator for the PDF (relative to bin midpoint)
            e_interp = interp1d(
                x=enu_mesh, y=enu_pdf, kind='linear',
                copy=True, bounds_error=False, fill_value=0
            )

            #==================================================================
            # Neutrino coszen resolution for events in this energy bin
            #==================================================================
            cz_err_min = min(cz_err)
            cz_err_max = max(cz_err)
            cz_err_range = cz_err_max-cz_err_min

            # NOTE the limits are 1 less than / 1 greater than the limits that
            # the error will actually take on, so as to allow for any smooth
            # roll-off at edges of data. The calculation of areas below
            # captures all of the area, though, by reflecting bins defined in
            # [-1, 1] about the points -1 and 1, thereby capturing any
            # densities in the range [-3, +3]. This is not necessarily
            # accurate, but it's better than throwing that info out entirely.

            # NOTE also that since reco events as of now are only in range -1
            # to 0, though, that there are "gaps" in the capture range, but
            # this is due to densities being in the upper-hemisphere which we
            # are intentionally ignoring, rather than the code here not taking
            # them into account. Normalization is based upon *all* events,
            # whether or not they fall within a bin specified above.

            # Number of points in the mesh used for VBWKDE; must be large
            # enough to capture fast changes in the data but the larger the
            # number, the longer it takes to compute the densities at all the
            # points. Here, just choosing a fixed number regardless of the data
            # or binning
            N_cz_mesh = 2**10

            # Data range for VBWKDE to consider
            cz_kde_min = -3
            cz_kde_max = +2

            # Adjust range of kde for future axis scaling
            cz_factor = 4

            low_lim_shift = cz_kde_min * (cz_factor - 1)
            upp_lim_shift = cz_kde_max * (cz_factor - 1)

            cz_kde_min_ext = cz_kde_min
            cz_kde_max_ext = cz_kde_max
            if low_lim_shift > 0:
                cz_kde_min_ext = cz_kde_min - low_lim_shift * (1./cz_factor)
            if upp_lim_shift < 0:
                cz_kde_max_ext = cz_kde_max - upp_lim_shift * (1./cz_factor)

            # Adjust kde_num_points accordingly
            N_cz_mesh_ext = int(
                N_cz_mesh * (
                    (cz_kde_max_ext - cz_kde_min_ext) / (cz_kde_max - cz_kde_min)
                )
            )

            cz_kde_failed = False
            previous_fail = False
            for n in xrange(3):
                try:
                    cz_bw, cz_mesh, cz_pdf = kde.vbw_kde(
                        data=cz_err,
                        overfit_factor=OVERFIT_FACTOR,
                        MIN=cz_kde_min_ext,
                        MAX=cz_kde_max_ext,
                        N=N_cz_mesh_ext
                    )
                # TODO: only catch specific exception
                except:
                    cz_kde_failed = True
                    if n == 0:
                        logging.trace('(cz vbwkde ')
                    logging.trace('fail, ')
                    # If failure occurred in vbw_kde, expand the data range it
                    # takes into account; this usually helps
                    cz_kde_min -= 1
                    cz_kde_max += 1

                    low_lim_shift = cz_kde_min * (cz_factor - 1)
                    upp_lim_shift = cz_kde_max * (cz_factor - 1)

                    cz_kde_min_ext = cz_kde_min
                    cz_kde_max_ext = cz_kde_max
                    if low_lim_shift > 0:
                        cz_kde_min_ext = (
                            cz_kde_min - low_lim_shift * (1./cz_factor)
                        )
                    if upp_lim_shift < 0:
                        cz_kde_max_ext = (
                            cz_kde_max - upp_lim_shift * (1./cz_factor)
                        )

                    # Adjust kde_num_points accordingly
                    N_cz_mesh_ext = int(
                        N_cz_mesh* ((cz_kde_max_ext - cz_kde_min_ext)
                        / (cz_kde_max - cz_kde_min))
                    )
                else:
                    if cz_kde_failed:
                        previous_fail = True
                        logging.trace('success!')
                    cz_kde_failed = False
                finally:
                    if previous_fail:
                        logging.trace(')')
                    previous_fail = False
                    if not cz_kde_failed:
                        break

            if cz_kde_failed:
                logging.warn('Failed to fit VBWKDE!')
                continue

            if np.min(cz_pdf) < 0:
                logging.warn("np.min(cz_pdf) < 0: Minimum value is " +
                             str(np.min(cz_pdf)) +
                             "; forcing all negative values to 0.")
                np.clip(a=cz_mesh, a_min=0, a_max=np.inf)

            assert np.min(cz_pdf) >= 0, str(np.min(cz_pdf))
            #assert np.max(cz_pdf) < 1, str(np.max(cz_pdf))

            # coszen interpolant is centered about the 0-error point--i.e., the
            # bin's midpoint
            cz_interp = interp1d(
                x=cz_mesh, y=cz_pdf, kind='linear',
                copy=True, bounds_error=False, fill_value=0
            )

            thisbin_kde_info = dict(
                e_interp=e_interp, cz_interp=cz_interp,
                enu_err=enu_err, cz_err=cz_err
            )

            thisbin_extra_info = dict(
                enu_err=enu_err,
                cz_err=cz_err,
                enu_bw=enu_bw,
                cz_bw=cz_bw,
                e_kde_min=e_kde_min,
                e_kde_max=e_kde_max,
                cz_kde_min_ext=cz_kde_min_ext,
                cz_kde_max_ext=cz_kde_max_ext
            )

            thisbin_key = (ebin_min, ebin_mid, ebin_max)
            kde_info[thisbin_key] = thisbin_kde_info
            extra_info[thisbin_key] = thisbin_extra_info

        return kde_info, extra_info

    @profile
    def compute_kernel(self, kde_info, binning, e_res_scale,
                       cz_res_scale, e_reco_bias, cz_reco_bias,
                       res_scale_ref='mode'):
        """Construct a 4D kernel from linear interpolants describing the
        density of reconstructed events.

        The resulting 4D array can be indexed for clarity using
           kernel4d[e_true_i, cz_true_j][e_reco_k, cz_reco_l]
        where the 4 indices point from a single MC-true histogram bin (i,j) to
        a single reco histogram bin (k,l). (Or flip e with cz if the `binning`
        specifies them in this order.)


        Parameters
        ----------
        kde_info : OrderedDict
        binning : MultiDimBinning
        e_res_scale : scalar
        cz_res_scale : scalar
        e_reco_bias : scalar Quantity
        cz_reco_bias : scalar Quantity
        res_scale_ref : string


        Returns
        -------
        kernel4d : 4D array of float
            Mapping from the number of events in each bin of the 2D
            MC-true-events histogram to the number of events reconstructed in
            each bin of the 2D reconstructed-events histogram. Dimensions are
              len(ebins)-1 x len(czbins)-1 x len(ebins)-1 x
              len(czbins)-1
            since ebins and czbins define the histograms' bin edges.

        """
        SAMPLES_PER_BIN = 500
        """Number of samples for computing area in a bin (via np.trapz)."""

        if isinstance(e_res_scale, Param):
            e_res_scale = e_res_scale.value.m_as('dimensionless')
        if isinstance(cz_res_scale, Param):
            cz_res_scale = cz_res_scale.value.m_as('dimensionless')
        if isinstance(e_reco_bias, Param):
            e_reco_bias = e_reco_bias.value.m_as('GeV')
        if isinstance(cz_reco_bias, Param):
            cz_reco_bias = cz_reco_bias.value.m_as('dimensionless')
        if isinstance(res_scale_ref, Param):
            res_scale_ref = res_scale_ref.value.strip().lower()
        assert res_scale_ref in ['zero', 'mean', 'mode']

        e_dim_num = binning.index('energy', use_basenames=True)
        cz_dim_num = binning.index('coszen', use_basenames=True)

        energy_first = True if e_dim_num == 0 else False

        ebins = binning.dims[e_dim_num]
        czbins = binning.dims[cz_dim_num]

        # Upsample to get coordinates at which to evaluate trapezoidal-rule
        # integral for each bin; convert to scalars in compuational units
        e_oversamp = ebins.oversample(SAMPLES_PER_BIN-1).bin_edges.m_as('GeV')
        cz_oversamp = czbins.oversample(SAMPLES_PER_BIN-1).bin_edges.m_as('dimensionless')

        # Object in which to store the 4D kernels: np 4D array
        kernel4d = np.zeros(binning.shape * 2)

        for ebin_n, (ebinpoints, interpolants) in enumerate(kde_info.iteritems()):
            ebin = ebins[ebin_n]
            emin, emid, emax = ebinpoints
            e_interp = interpolants['e_interp']
            cz_interp = interpolants['cz_interp']

            if res_scale_ref in ['zero']:
                rel_e_ref = 0

            elif res_scale_ref == 'mean':
                # Find the mean by locating where the CDF is equal to 0.5,
                # i.e., by evaluating the quantile func at 0.5.
                cum_sum = np.cumsum(e_interp.y)
                cdf = cum_sum / cum_sum[-1]
                quantile_func = interp1d(cdf, e_interp.x)
                rel_e_ref = quantile_func(0.5)

            elif res_scale_ref == 'mode':
                # Approximate the mode by the highest point in the (sampled)
                # PDF
                rel_e_ref = e_interp.x[e_interp.y == np.max(e_interp.y)][0]

            # Figure out what points we need to sample in the relative space
            # (this is where the interpolant is defined) given our dense
            # sampling in absolute coordinate space and our desire to scale and
            # shift the resolutions by some amount.
            rel_e_coords = abs2rel(
                abs_coords=e_oversamp, abs_bin_midpoint=emid,
                rel_scale_ref=rel_e_ref, scale=e_res_scale,
                abs_obj_shift=e_reco_bias
            )

            # NOTE: We don't need to account for area lost in tail below 0 GeV
            # so long as our analysis doesn't go to 0: We can just assume all
            # events below our lower threshold end up below the threshold but
            # above 0, and then we have no "missing" area.

            # Divide by e_res_scale to keep the PDF area normalized to one when
            # we make it wider/narrower.
            e_pdf = e_interp(rel_e_coords) / e_res_scale

            ebin_areas = []
            for n in xrange(ebins.num_bins):
                sl = slice(n*SAMPLES_PER_BIN, (n+1)*SAMPLES_PER_BIN + 1)
                ebin_area = np.trapz(x=rel_e_coords[sl], y=e_pdf[sl])
                assert ebin_area > -EPSILON, 'bin %d ebin_area=%e' %(n, ebin_area)
                ebin_areas.append(ebin_area)

            # Sum the individual bins' areas
            tot_ebin_area = np.sum(ebin_areas)

            #==================================================================
            # Neutrino coszen resolution for events in this energy bin
            #==================================================================
            for czbin_n in range(czbins.num_bins):
                czbin = czbins[czbin_n]

                czbin_min, czbin_max = czbin.bin_edges.m_as('dimensionless')
                czbin_mid = czbin.midpoints[0].m_as('dimensionless')

                if res_scale_ref in ['zero']:
                    rel_cz_ref = 0

                elif res_scale_ref == 'mean':
                    # Find the mean by locating where the CDF is equal to 0.5,
                    # i.e., by evaluating the quantile func at 0.5.
                    cum_sum = np.cumsum(cz_interp.y)
                    cdf = cum_sum / cum_sum[-1]
                    quantile_func = interp1d(cdf, cz_interp.x)
                    rel_cz_ref = quantile_func(0.5)

                elif res_scale_ref == 'mode':
                    # Approximate the mode by the highest point in the (sampled)
                    # PDF
                    rel_cz_ref = cz_interp.x[cz_interp.y == np.max(cz_interp.y)][0]

                # Interpolant was defined in relative space (to bin center);
                # translate this to absolute CZ coords, taking this bin's
                # center as the one about which it is defined (and take into
                # account any resoltuions scaling / bias shifting we are
                # applying).
                cz_interpolant_limits = rel2abs(
                    rel_coords=cz_interp.x[0::len(cz_interp.x)-1],
                    abs_bin_midpoint=czbin_mid,
                    rel_scale_ref=rel_cz_ref,
                    scale=cz_res_scale,
                    abs_obj_shift=cz_reco_bias
                )

                # Account for all area, including area under aliased PDF
                negative_aliases = 0
                positive_aliases = 0
                if cz_interpolant_limits[0] < -1:
                    negative_aliases = int(np.abs(np.floor(
                        (cz_interpolant_limits[0] + 1) / 2.0
                    )))
                if cz_interpolant_limits[1] > +1:
                    positive_aliases = int(np.abs(np.ceil(
                        (cz_interpolant_limits[1] - 1) / 2.0
                    )))

                czbin_areas = np.zeros(czbins.num_bins)
                for alias_n in range(-negative_aliases, 1 + positive_aliases):
                    if alias_n == 0:
                        abs_cz_coords = cz_oversamp
                    elif alias_n % 2 == 0:
                        abs_cz_coords = cz_oversamp + alias_n
                    else:
                        # NOTE: need to flip order such that it's monotonically
                        # increasing (else trapz returns negative areas)
                        abs_cz_coords = (-cz_oversamp + 1+alias_n)[::-1]

                    rel_cz_coords = abs2rel(
                        abs_coords=abs_cz_coords,
                        abs_bin_midpoint=czbin_mid,
                        rel_scale_ref=rel_cz_ref,
                        scale=cz_res_scale,
                        abs_obj_shift=cz_reco_bias
                    )
                    cz_pdf = cz_interp(rel_cz_coords) / cz_res_scale
                    assert np.all(cz_pdf >= 0), str(cz_pdf)

                    areas = []
                    for n in xrange(czbins.num_bins):
                        sl = slice(n*SAMPLES_PER_BIN, (n+1)*SAMPLES_PER_BIN+1)
                        area = np.trapz(x=rel_cz_coords[sl], y=cz_pdf[sl])
                        #if n < 0:
                        #    area = -area
                        if area <= -EPSILON:
                            logging.error('x  = %s' %rel_cz_coords[sl])
                            logging.error('y  = %s' %cz_pdf[sl])
                            logging.error('sl = %s' %sl)
                            logging.error('alias %d czbin %d area=%e' %(alias_n, n, area))
                            raise ValueError()

                        areas.append(area)

                    czbin_areas += np.array(areas)

                tot_czbin_area = np.sum(czbin_areas)

                if energy_first:
                    x, y = ebin_n, czbin_n
                    kernel4d[x, y, :, :] = np.outer(ebin_areas, czbin_areas)
                else:
                    x, y = czbin_n, ybin_n
                    kernel4d[x, y, :, :] = np.outer(czbin_areas, ebin_areas)

                d = (np.sum(kernel4d[x,y])-tot_ebin_area*tot_czbin_area)
                assert (d < EPSILON), 'd: %s, epsilon: $s' %(d, epsilon)

        check_areas = kernel4d.sum(axis=(2,3))

        #assert np.max(check_areas) < 1 + EPSILON, str(np.max(check_areas))
        assert np.min(check_areas) > 0 - EPSILON, str(np.min(check_areas))

        return kernel4d


if __name__ == '__main__':
    set_verbosity(3)
    test_abs2rel()
    test_rel2abs()
