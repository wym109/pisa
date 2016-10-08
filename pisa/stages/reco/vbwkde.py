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
"""


from __future__ import division

from collections import OrderedDict
from copy import deepcopy
import os

import numpy as np
from scipy.interpolate import interp1d

from pisa.core.stage import Stage
from pisa.core.transform import BinnedTensorTransform, TransformSet
from pisa.utils.confInterval import MLConfInterval
from pisa.utils.coords import abs2rel, rel2abs
from pisa.utils.fileio import mkdir
from pisa.utils.flavInt import flavintGroupsFromString, NuFlavIntGroup
from pisa.utils.hash import hash_obj
from pisa.utils.kde import vbw_kde
from pisa.utils.log import logging, set_verbosity
from pisa.utils.profiler import profile, line_profile
from pisa.utils.resources import find_resource

EPSILON = 1e-9

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

        transform_events_keep_criteria : None, string, or sequence of strings

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

    transforms_cache_depth : int >= 0

    outputs_cache_depth : int >= 0

    memcache_deepcopy : bool

    debug_mode : None, bool, or string
        Whether to store extra debug info for this service.

    Notes
    -----
    The `transform_groups` string is interpreted (and therefore defined) by
    pisa.utils.flavInt.flavint_groups_string. E.g. commonly one might use:

    'nue_cc+nuebar_cc, numu_cc+numubar_cc, nutau_cc+nutaubar_cc, nuall_nc+nuallbar_nc'

    Any particle type not explicitly mentioned is taken as a singleton group.
    Plus signs add types to a group, while groups are separated by commas.
    Whitespace is ignored, so add whitespace for readability.

    """
    E_CONVOLUTION = True
    CZ_CONVOLUTION = True

    def __init__(self, params, particles, input_names, transform_groups,
                 sum_grouped_flavints, input_binning, output_binning,
                 error_method=None, transforms_cache_depth=20,
                 outputs_cache_depth=20, memcache_deepcopy=True,
                 debug_mode=None):
        assert particles in ['neutrinos', 'muons']
        self.particles = particles
        self.transform_groups = flavintGroupsFromString(transform_groups)
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
        """Require input dimensions of "true_energy" and "true_coszen" (in any
        order).

        Require output dimensions of "reco_energy" and "reco_coszen", and
        optionally allow output dimension of "pid"; can be in any order.

        """
        input_names = set(self.input_binning.names)
        assert input_names == set(['true_energy', 'true_coszen']), str(input_names)

        output_names = set(self.output_binning.names)
        outs1 = set(['reco_energy', 'reco_coszen'])
        outs2 = set(['reco_energy', 'reco_coszen', 'pid'])
        assert output_names == outs1 or output_names == outs2

        input_basenames = set(self.input_binning.basenames)
        output_basenames = set(self.output_binning.basenames)
        for base_d in input_basenames:
            assert base_d in output_basenames

    @profile
    def _compute_transforms(self):
        """Generate reconstruction "smearing kernels" by estimating the
        distribution of reconstructed events corresponding to each bin of true
        events using VBW-KDE.

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
            reco_energy='GeV', reco_coszen=None, reco_azimuth='rad', pid=None
        )

        # Compute the KDEs for each (pid, E) bin (this is then propagated to
        # each (pid, E, cz) bin, as the transform is assumed to not be
        # cz-dependent)
        all_kde_info = self.get_all_kde_info()

        # Apply scaling factors and figure out the area per bin for each KDE
        xforms = []
        for xform_flavints in self.transform_groups:
            reco_kernel = self.compute_resolutions(
                e_cz_kde_info=all_kde_info[str(xform_flavints)],
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
                        xform_array=reco_kernel,
                        sum_inputs=self.sum_grouped_flavints
                    )
                    xforms.append(xform)

        return TransformSet(transforms=xforms)

    @profile
    def get_all_kde_info(self):
        """Compute the KDEs for each (pid, E) bin. If PID is not present, this
        is just (E). The results are propagated to each (pid, E, cz) bin, as
        the transforms are assumed to not be cz-dependent.

        The results are cached to disk and simply loaded from that cache to
        avoid re-computation.

        Returns
        -------
        all_kde_info

        """
        # Only hash on the things that could be relevant to the KDE's:
        # 1. source code: if the code changes at all
        # 2. input_binning: only input binning matters for the KDE's (output
        #       binning only comes into play when computing the actual
        #       transform kernels)
        # 3. reco_events: event used to compute the transforms, including any
        #       cuts applied
        # 4. transform_events_keep_criteria: redundant with the cuts registered
        #       with the events? not sure... so using it anyway
        # 5. transform_groups: which flav/ints are grouped for computing KDE
        kde_hash = hash_obj([self.source_code_hash,
                             self.input_binning.hash,
                             self.params.reco_events.value,
                             self.params.transform_events_keep_criteria.value,
                             self.transform_groups])
        logging.trace('kde_hash = %s' %kde_hash)
        if (self._kde_hash is not None and kde_hash == self._kde_hash
            and hasattr(self, 'all_kde_info')):
            return self.all_kde_info

        logging.trace('no match')
        logging.trace('self._kde_hash: %s' % self._kde_hash)
        logging.trace('kde_hash: %s' % kde_hash)
        logging.trace('hasattr: %s' % hasattr(self, 'all_kde_info'))

        if not self.debug_mode:
            try:
                self.all_kde_info = self.disk_cache[kde_hash]
                self._kde_hash = kde_hash
                return self.all_kde_info
            except KeyError:
                pass

        if self.debug_mode:
            outdir = os.path.join(find_resource('debug'),
                                  self.stage_name,
                                  self.service_name)
            mkdir(outdir)

        self.all_kde_info = OrderedDict()
        self.all_extra_info = OrderedDict()
        for xform_flavints in self.transform_groups:
            logging.debug("Working on %s reco kernels" %xform_flavints)
            repr_flav_int = xform_flavints.flavints()[0]
            kde_info, extra_info = self.compute_kdes(
                events=self.remaining_events[repr_flav_int],
                binning=self.input_binning
            )
            self.all_kde_info[str(xform_flavints)] = kde_info
            self.all_extra_info[str(xform_flavints)] = extra_info

            if self.debug_mode:
                plot_kde_detail(flavints=xform_flavints,
                                kde_info=kde_info,
                                extra_info=extra_info,
                                binning=self.input_binning,
                                outdir=outdir)

        self._kde_hash = kde_hash

        if self.debug_mode and kde_hash in self.disk_cache:
            del self.disk_cache[kde_hash]

        self.disk_cache[kde_hash] = self.all_kde_info

        return self.all_kde_info

    @profile
    def compute_kdes(self, events, binning):
        """Construct a 4D kernel set from MC events using VBW-KDE.

        Given a set of MC events and binning (which serves as both input and
        output binning), construct a KDE estimate and draw samples from this.
        From the samples, a linear interpolant is generated. There are one
        energy and one coszen interpolant genreated for each energy bin.

        NOTE: Actual limits in energy used to group events into a single "true"
        bin may be extended beyond the bin edges defined by ebins in order
        to gather enough events to successfully apply VBW-KDE.


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
        true_energy = events['true_energy']
        reco_energy = events['reco_energy']
        true_coszen = events['true_coszen']
        reco_coszen = events['reco_coszen']
        ebins = binning.true_energy
        ebin_edges = ebins.bin_edges.m_as('GeV')
        czbins = binning.true_coszen
        czbin_edges = czbins.bin_edges.m_as('dimensionless')

        # NOTE: below defines bin centers on linear scale; other logic
        # in this method assumes this to be the case, so
        # **DO NOT USE** utils.utils.get_bin_centers in this method, which
        # may return logarithmically-defined centers instead.
        left_ebin_edges = ebin_edges[0:-1]
        right_ebin_edges = ebin_edges[1:]

        n_events = len(true_energy)

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
            abs_enu_dist = np.abs(true_energy - ebin_mid)
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
            in_ebin_mask = abs_enu_dist <= thresh_enu_dist
            n_in_bin = np.sum(in_ebin_mask)

            # Record lowest/highest energies that are included in the bin
            actual_left_ebin_edge = min(
                ebin_min, min(true_energy[in_ebin_mask])
            )
            actual_right_ebin_edge = max(
                ebin_max, max(true_energy[in_ebin_mask])
            )

            # Extract just the neutrino-energy/coszen error columns' values for
            # succinctness
            enu_err = reco_energy[in_ebin_mask] - true_energy[in_ebin_mask]

            # TODO: figure out zenith angle error here, and then map this to
            # coszen error for each bin when we compute the actual kernels
            cz_err = reco_coszen[in_ebin_mask] - true_coszen[in_ebin_mask]

            # NOTE: the following is a bad idea. The spike at 0 (error) screws
            # up KDE in the bins where we're having issues, and we continue to
            # have 0 event reco-ing inbounds from these bins (or it gets even
            # worse).
            ## Clip enu error to imply reco >= 0 GeV
            #np.clip(enu_err, a_min=-ebin_mid, a_max=np.inf, out=enu_err)

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
            e_kde_lims = np.array([e_lowerlim, e_upperlim])

            # Use at least min_num_pts points and at most the next-highest
            # integer-power-of-two that allows for at least 10 points in the
            # smallest energy bin
            min_num_pts = 2**12
            min_bin_width = np.min(ebin_edges[1:]-ebin_edges[:-1])
            min_pts_smallest_bin = 5.0
            kde_range = np.diff(e_kde_lims)
            #num_pts0 = kde_range/(min_bin_width/min_pts_smallest_bin)
            num_pts0 = kde_range/(ebin_wid/min_pts_smallest_bin)
            kde_num_pts = int(max(min_num_pts, 2**np.ceil(np.log2(num_pts0))))
            logging.debug(
                '  N_evts=' + str(n_in_bin) + ', taken from [' +
                format(actual_left_ebin_edge, '0.3f') + ', ' +
                format(actual_right_ebin_edge, '0.3f') + ']' + ', VBWKDE lims=' +
                str(e_kde_lims) + ', VBWKDE_N: ' + str(kde_num_pts)
            )

            ## Exapnd range of sample points for future axis scaling
            #e_factor = 1

            #low_lim_shift = e_kde_lims[0] * (e_factor - 1)
            #upp_lim_shift = e_kde_lims[1] * (e_factor - 1)

            #e_kde_lims_ext = np.copy(e_kde_lims)
            #if low_lim_shift > 0:
            #    e_kde_lims_ext[0] = (
            #        e_kde_lims[0] - low_lim_shift * (1./e_factor)
            #    )
            #if upp_lim_shift < 0:
            #    e_kde_lims_ext[1] = (
            #        e_kde_lims[1] - upp_lim_shift * (1./e_factor)
            #    )

            # Adjust kde_num_points accordingly
            e_kde_num_pts = int(
                kde_num_pts*((e_kde_lims[1] - e_kde_lims[0])
                / (e_kde_lims[1] - e_kde_lims[0]))
            )

            logging.trace('e_kde_num_pts = %s; MIN/MAX = %s' %(e_kde_num_pts,
                                                               e_kde_lims))

            # Compute variable-bandwidth KDEs
            enu_bw, enu_mesh, enu_pdf = vbw_kde(
                data=enu_err,
                overfit_factor=OVERFIT_FACTOR,
                MIN=e_kde_lims[0],
                MAX=e_kde_lims[1],
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
                # Otherwise, just quietly clip any negative values to 0
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

            # Number of points in the mesh used for VBW-KDE; must be large
            # enough to capture fast changes in the data but the larger the
            # number, the longer it takes to compute the densities at all the
            # points. Here, just choosing a fixed number regardless of the data
            # or binning
            N_cz_mesh = 2**12

            # Data range for VBW-KDE to consider
            cz_kde_min = -9
            cz_kde_max = +8

            # Adjust range of kde for future axis scaling
            cz_factor = 4

            low_lim_shift = cz_kde_min * (cz_factor - 1)
            upp_lim_shift = cz_kde_max * (cz_factor - 1)

            #cz_kde_min_ext = cz_kde_min
            #cz_kde_max_ext = cz_kde_max
            #if low_lim_shift > 0:
            #    cz_kde_min_ext = cz_kde_min - low_lim_shift * (1./cz_factor)
            #if upp_lim_shift < 0:
            #    cz_kde_max_ext = cz_kde_max - upp_lim_shift * (1./cz_factor)

            ## Adjust kde_num_points accordingly
            #N_cz_mesh_ext = int(
            #    N_cz_mesh * (
            #        (cz_kde_max_ext - cz_kde_min_ext)/(cz_kde_max - cz_kde_min)
            #    )
            #)

            cz_kde_failed = False
            previous_fail = False
            for n in xrange(3):
                try:
                    cz_bw, cz_mesh, cz_pdf = vbw_kde(
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
                    cz_kde_lims = np.array([cz_kde_min, cz_kde_max])
                    if not cz_kde_failed:
                        break

            if cz_kde_failed:
                logging.error('Failed to fit VBW-KDE!')
                raise Exception('Failed to fit VBW-KDE!')

            if np.min(cz_pdf) < 0:
                logging.warn("np.min(cz_pdf) < 0: Minimum value is " +
                             str(np.min(cz_pdf)) +
                             "; forcing all negative values to 0.")
                np.clip(a=cz_mesh, a_min=0, a_max=np.inf)

            assert np.min(cz_pdf) >= 0, str(np.min(cz_pdf))

            #print '='*80
            #print cz_mesh
            #print '='*80
            #print cz_pdf
            #print '='*80

            total_area = np.trapz(y=cz_pdf, x=cz_mesh)
            logging.trace('Bin %4d cz trapz area = %e'
                          %(ebin_n, total_area))
            cz_pdf /= total_area
            total_area /= total_area
            logging.trace('Bin %4d cz area after trapz renorm = %e'
                          %(ebin_n, total_area))
            assert np.min(cz_pdf) >= 0, str(np.min(cz_pdf))

            # coszen interpolant is centered about the 0-error point--i.e., the
            # bin's midpoint
            cz_interp = interp1d(
                x=cz_mesh, y=cz_pdf, kind='linear',
                copy=True, bounds_error=False, fill_value=0
            )

            thisbin_kde_info = dict(
                e_interp=e_interp, cz_interp=cz_interp,
            )

            thisbin_extra_info = dict(
                enu_err=enu_err,
                cz_err=cz_err,
                actual_ebin_edges=[actual_left_ebin_edge,
                                   actual_right_ebin_edge],
                enu_bw=enu_bw,
                cz_bw=cz_bw,
                enu_mesh=enu_mesh,
                cz_mesh=cz_mesh,
                e_kde_lims=e_kde_lims,
                cz_kde_lims=cz_kde_lims,
            )

            thisbin_key = (ebin_min, ebin_mid, ebin_max)
            kde_info[thisbin_key] = thisbin_kde_info
            extra_info[thisbin_key] = thisbin_extra_info

        return kde_info, extra_info

    @profile
    def compute_resolutions(self, e_cz_kde_info):
        """Construct a kernel from linear interpolants describing the
        normalized density of reconstructed events.

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
        e_cz_kde_info : OrderedDict
            Object returned by method `compute_kdes`.


        Returns
        -------
        kernel : 4D array of float
            Mapping from the number of events in each bin of the 2D
            MC-true-events histogram to the number of events reconstructed in
            each bin of the 2D reconstructed-events histogram. Dimensions are
              len(ebins) x len(czbins) x len(ebins) x len(czbins).

        """
        SAMPLES_PER_BIN = 500
        """Number of samples for computing each 1D area in a bin (using
        np.trapz)."""

        e_res_scale = self.params.e_res_scale.value.m_as('dimensionless')
        cz_res_scale = self.params.cz_res_scale.value.m_as('dimensionless')
        e_reco_bias = self.params.e_reco_bias.value.m_as('GeV')
        cz_reco_bias = self.params.cz_reco_bias.value.m_as('dimensionless')
        res_scale_ref = self.params.res_scale_ref.value.strip().lower()

        e_dim_num = self.input_binning.index('true_energy')
        cz_dim_num = self.input_binning.index('true_coszen')

        energy_first = True if e_dim_num < cz_dim_num else False

        transform_binning = self.input_binning * self.output_binning

        output_ebins = self.output_binning.reco_energy
        output_czbins = self.output_binning.reco_coszen

        input_czbins = self.input_binning.true_coszen

        # Upsample to get coordinates at which to evaluate trapezoidal-rule
        # integral for each output bin
        e_oversamp_binned = output_ebins.oversample(SAMPLES_PER_BIN-1)
        cz_oversamp_binned = output_czbins.oversample(SAMPLES_PER_BIN-1)

        # Convert to scalars in compuational units
        e_oversamp_binned = e_oversamp_binned.bin_edges.m_as('GeV')
        cz_oversamp_binned = cz_oversamp_binned.bin_edges.m_as('dimensionless')

        # Object in which to store the 4D kernels
        kernel = np.full(shape=transform_binning.shape, fill_value=np.nan,
                         dtype=np.float64)

        for input_ebin_n, item in enumerate(e_cz_kde_info.iteritems()):
            input_ebinpoints, interpolants = item
            input_ebin_min, input_ebin_mid, input_ebin_max = input_ebinpoints
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

            else:
                raise ValueError('`res_scale_ref` unrecognized: "%s"'
                                 %res_scale_ref)

            # Convolve the input bin (represented by a boxcar normalized to
            # have summed area of 1) with the resolution function's shape,
            # since by binning the input (truth dimensions) we must (roughlY)
            # assume that an event in the bin could have come from anywhere in
            # that bin with equal probability. Therfore the resolution function
            # could be applied at the left edge of the bin, the right edge of
            # the bin, or anywhere in between. This is a convolution of the two
            # PDFs: a boxcar for the bin, and the KDE resolution function.
            # Note that a function other than a boxcar might be appropriate as
            # well, but simplicity rules at the moment.

            if self.E_CONVOLUTION:
                # 1. Determine bin width in relative coordinates (taking res
                #    scaling into consideration)
                input_ebin_rel_width = np.abs(np.diff(abs2rel(
                    abs_coords=np.array([input_ebin_min, input_ebin_max]),
                    abs_bin_midpoint=input_ebin_mid,
                    rel_scale_ref=rel_e_ref, scale=e_res_scale,
                    abs_obj_shift=e_reco_bias
                ))[0])

                # 2. Determine how many samples wide the bin is in relative coords;
                #    clip to 2 or else normalization of the boxcar by ts "area"
                #    (done below) is undefined
                dx_e_interp = e_interp.x[1] - e_interp.x[0]
                input_ebin_n_rel_samples = int(np.clip(
                    np.round(input_ebin_rel_width / dx_e_interp),
                    a_min=2, a_max=np.inf
                ))

                # 3. Construct the boxcar func for the input bin width
                input_ebin_pdf = np.full(
                    shape=(input_ebin_n_rel_samples,),
                    fill_value=1.0/(input_ebin_n_rel_samples))

                # 4. Perform the convolution to smear the resolution function over
                #    the extents of the bin
                smeared_y = np.convolve(a=e_interp.y, v=input_ebin_pdf,
                                        mode='same')

                # 5. Create an interpolant with the smeared resolution function
                smeared_e_interp = interp1d(
                    x=e_interp.x, y=smeared_y, kind='linear',
                    copy=True, bounds_error=False, fill_value=0
                )
            else:
                smeared_e_interp = e_interp

            # Figure out what points we need to sample in the relative space
            # (this is where the interpolant is defined) given our dense
            # sampling in absolute coordinate space and our desire to scale and
            # shift the resolutions by some amount.
            rel_e_coords_binned = abs2rel(
                abs_coords=e_oversamp_binned, abs_bin_midpoint=input_ebin_mid,
                rel_scale_ref=rel_e_ref, scale=e_res_scale,
                abs_obj_shift=e_reco_bias
            )

            # Divide by e_res_scale to keep the PDF area normalized to one when
            # we make it wider/narrower (i.e., while interpolant lives in
            # relative-space, e_pdf_binned lives in absolute-space, so the
            # latter needs to be scaled vertically since it is rescaled
            # horizontally).
            e_pdf_binned = smeared_e_interp(rel_e_coords_binned) / e_res_scale

            binned_area = np.abs(np.trapz(y=e_pdf_binned, x=e_oversamp_binned))
            logging.trace('Bin %4d binned E area before any renorm = %e'
                          %(input_ebin_n, binned_area))

            # Compute total area under curve (since KDE is a sum of normalized
            # Gaussians divided by the number of Gaussians, the area should be
            # exactly 1. Only deviations should be due to finite sampling of
            # the curve and the use of linear interpolation between these
            # samples (and note that since we're doing so, trapz gives the
            # "correct" area under this curve).
            total_trapz_area = np.trapz(y=smeared_e_interp.y,
                                        x=smeared_e_interp.x)
            logging.trace('Bin %4d total trapz E area = %e'
                          %(input_ebin_n, total_trapz_area))

            # Normalize e_pdf_binned so that the entire PDF (including
            # points outside of those that are binned) will have area of 1.
            # (Necessitated due to finite sampling; see notes above.)
            e_pdf_binned /= total_trapz_area
            binned_area /= total_trapz_area
            logging.trace('Bin %4d binned E area after trapz renorm = %e'
                          %(input_ebin_n, binned_area))

            if False:
                # Now figure out the "invalid" area under the PDF. Since we
                # draw events that are > than the bin midpoint, but we
                # effectively interpret their reco as coming from an event with
                # true-energy at the bin center, the reco can be < 0 GeV. While
                # this gives the KDE a "better" shape (compared to e.g. not
                # using these events at all), it does leave us with a tail that
                # extends more or less (but always some) below the valid
                # range--i.e., below 0 GeV.

                # Proposed solution: Add up this area, and rescale the PDF to
                # be larger to compensate for this "wasted," non-physical area.

                # Figure out relative coordinate corresponding to 0 GeV
                zero_in_rel_coords = abs2rel(
                    abs_coords=0, abs_bin_midpoint=input_ebin_mid,
                    rel_scale_ref=rel_e_ref, scale=e_res_scale,
                    abs_obj_shift=e_reco_bias
                )

                # The only point we can use to start the integration is the
                # lower limit of the energy intpolant as we must assume
                # (rightly or wrongly) that we covered the complete range of
                # where there might be any appreciable area under the curve.

                # Find the absolute coordinate of this lowest-energy sample
                # point
                abs_e_samp_min = rel2abs(
                    rel_coords=np.min(smeared_e_interp.x),
                    abs_bin_midpoint=input_ebin_mid,
                    rel_scale_ref=rel_e_ref,
                    scale=e_res_scale,
                    abs_obj_shift=e_reco_bias
                )

                if np.min(smeared_e_interp.x) < zero_in_rel_coords:
                    # Identify all interpolant x-coords that are less than 0
                    # GeV in absolute space
                    lt_zero_mask = smeared_e_interp.x < zero_in_rel_coords

                    # Integrate the area including the points less than zero
                    # and the 0 GeV point; normalize by the same
                    # total_trapz_area that we had to normalize by above.
                    x = np.concatenate(
                        (smeared_e_interp.x[lt_zero_mask],
                         [zero_in_rel_coords])
                    )
                    y = np.concatenate(
                        (smeared_e_interp.y[lt_zero_mask],
                         [smeared_e_interp(zero_in_rel_coords)])
                    )
                    invalid_e_area = np.trapz(y=y, x=x) / total_trapz_area

                    logging.trace('Bin %4d invalid E area = %0.4e'
                                  %(input_ebin_n, invalid_e_area))
                    e_pdf_binned /= 1 - invalid_e_area
                    binned_area /= 1 - invalid_e_area
                else:
                    logging.trace('Bin %4d abs_e_samp_min = %s'
                                  %(input_ebin_n, abs_e_samp_min))

                logging.trace('Bin %4d binned E area after invalid renorm = %e'
                              %(input_ebin_n, binned_area))

            output_ebin_areas = []
            for output_ebin_n in xrange(output_ebins.num_bins):
                sl = slice(output_ebin_n*SAMPLES_PER_BIN,
                           (output_ebin_n+1)*SAMPLES_PER_BIN + 1)
                ebin_area = np.trapz(y=e_pdf_binned[sl],
                                     x=e_oversamp_binned[sl])
                assert ebin_area > -EPSILON, \
                        'Bin %d ebin_area=%e' %(output_ebin_n, ebin_area)
                ebin_area = np.clip(ebin_area, a_min=0, a_max=np.inf)
                output_ebin_areas.append(ebin_area)

            # Sum the area in each bin
            tot_output_ebin_area = np.sum(output_ebin_areas)

            #==================================================================
            # Neutrino coszen resolution for events in this energy bin
            #==================================================================
            for input_czbin_n, input_czbin in enumerate(input_czbins):
                input_czbin_min, input_czbin_max = \
                        input_czbin.bin_edges.m_as('dimensionless')
                input_czbin_mid = \
                        input_czbin.midpoints[0].m_as('dimensionless')

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
                    idx = cz_interp.y == np.max(cz_interp.y)
                    rel_cz_ref = cz_interp.x[idx][0]

                else:
                    raise ValueError('`res_scale_ref` unrecognized: "%s"'
                                     %res_scale_ref)

                if self.CZ_CONVOLUTION:
                    # 1. Determine bin width in relative coordinates (taking res
                    #    scaling into consideration)
                    input_czbin_rel_width = np.abs(np.diff(abs2rel(
                        abs_coords=np.array([input_czbin_min, input_czbin_max]),
                        abs_bin_midpoint=input_czbin_mid,
                        rel_scale_ref=rel_e_ref, scale=e_res_scale,
                        abs_obj_shift=e_reco_bias
                    ))[0])

                    # 2. Determine how many samples wide the bin is in relative
                    #    coords; clip to 2 or else normalization of the boxcar by
                    #    ts "area" (done below) is undefined
                    dx_cz_interp = cz_interp.x[1] - cz_interp.x[0]
                    input_czbin_n_rel_samples = int(np.clip(
                        np.round(input_czbin_rel_width / dx_cz_interp),
                        a_min=2, a_max=np.inf
                    ))

                    # 3. Construct the boxcar func for the input bin width
                    input_czbin_pdf = np.full(
                        shape=(input_czbin_n_rel_samples,),
                        fill_value=1.0/(input_czbin_n_rel_samples))

                    # 4. Perform the convolution to smear the resolution function
                    #    over the extents of the bin
                    smeared_y = np.convolve(a=cz_interp.y, v=input_czbin_pdf,
                                            mode='same')

                    total_trapz_area = np.trapz(y=smeared_y, x=cz_interp.x)
                    #logging.trace('Input czbin %4d total trapz area = %e'
                    #              %(input_czbin_n, total_trapz_area))

                    # 5. Create an interpolant with the smeared resolution function
                    smeared_cz_interp = interp1d(
                        x=cz_interp.x, y=smeared_y/total_trapz_area, kind='linear',
                        copy=True, bounds_error=False, fill_value=0
                    )
                else:
                    smeared_cz_interp = cz_interp
                    total_trapz_area = np.trapz(y=smeared_cz_interp.y,
                                                x=smeared_cz_interp.x)

                # Interpolant was defined in relative space (to bin center);
                # translate this to absolute CZ coords, taking this bin's
                # center as the one about which it is defined (and take into
                # account any resoltuions scaling / bias shifting we are
                # applying).
                cz_interpolant_limits = rel2abs(
                    rel_coords=smeared_cz_interp.x[0::len(smeared_cz_interp.x)-1],
                    abs_bin_midpoint=input_czbin_mid,
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

                output_czbin_areas = np.zeros(output_czbins.num_bins)
                for alias_n in range(-negative_aliases, 1 + positive_aliases):
                    if alias_n == 0:
                        abs_cz_coords = cz_oversamp_binned
                    elif alias_n % 2 == 0:
                        abs_cz_coords = cz_oversamp_binned + alias_n
                    else:
                        # NOTE: need to flip order such that it's monotonically
                        # increasing (else trapz returns negative areas)
                        abs_cz_coords = (-cz_oversamp_binned + 1+alias_n)[::-1]

                    rel_cz_coords = abs2rel(
                        abs_coords=abs_cz_coords,
                        abs_bin_midpoint=input_czbin_mid,
                        rel_scale_ref=rel_cz_ref,
                        scale=cz_res_scale,
                        abs_obj_shift=cz_reco_bias
                    )
                    cz_pdf = smeared_cz_interp(rel_cz_coords) / cz_res_scale
                    assert np.all(cz_pdf >= 0), str(cz_pdf)

                    areas = []
                    for n in xrange(output_czbins.num_bins):
                        sl = slice(n*SAMPLES_PER_BIN, (n+1)*SAMPLES_PER_BIN+1)
                        area = np.trapz(y=cz_pdf[sl], x=abs_cz_coords[sl])
                        #if n < 0:
                        #    area = -area
                        if area <= -EPSILON:
                            logging.error('x  = %s' %rel_cz_coords[sl])
                            logging.error('y  = %s' %cz_pdf[sl])
                            logging.error('sl = %s' %sl)
                            logging.error('alias %d czbin %d area=%e'
                                          %(alias_n, n, area))
                            raise ValueError()

                        areas.append(area)

                    output_czbin_areas += np.array(areas)

                # How much area is spread from this czbin across all others
                tot_output_czbin_area = np.sum(output_czbin_areas)

                #logging.trace('Bin %4d binned cz area = %e'
                #              %(input_ebin_n, tot_output_czbin_area))

                # TODO: if full cz range [-1,1] is covered, then we can force
                # renormalization to 1. Otherwise, all bets are off.

                #output_czbin_areas /= tot_output_czbin_area
                #tot_output_czbin_area /= tot_output_czbin_area
                #logging.trace('Bin %4d tot cz area after renorm = %e'
                #              %(input_ebin_n, tot_output_czbin_area))

                # Coszen must reconstruct somewhere, so area must be 1 if
                # binning includes all coszen; otherwise we can just say it
                # must be less than or equal to 1.
                assert tot_output_czbin_area <= 1+EPSILON, str(tot_output_czbin_area)

                if energy_first:
                    i, j = input_ebin_n, input_czbin_n
                    kernel[i, j, :, :] = np.outer(output_ebin_areas,
                                                  output_czbin_areas)
                else:
                    i, j = input_czbin_n, input_ebin_n
                    kernel[i, j, :, :] = np.outer(output_czbin_areas,
                                                  output_ebin_areas)

                d = (np.sum(kernel[i,j]) -
                     tot_output_ebin_area*tot_output_czbin_area)
                assert (np.abs(d) < EPSILON), 'd: %s, epsilon: %s' %(d, epsilon)

        check_areas = kernel.sum(axis=(2,3))

        #assert np.max(check_areas) < 1 + EPSILON, str(np.max(check_areas))
        assert np.min(check_areas) > 0 - EPSILON, str(np.min(check_areas))

        return kernel


def plot_kde_detail(flavints, kde_info, extra_info, binning, outdir,
                    ebin_n=None):
    """

    Parameters
    ----------
    kde_info : OrderedDict
        KDE info recorded for a single flav/int

    extra_info : OrderedDict
        Extra info (in same order as `kde_info` recorded for a single flav/int

    binning

    outdir

    ebin_n : None, int, or slice
        Index used to pick out a particular energy bin (or bins) to plot. Default
        (None) plots all energy bins.

    """
    import matplotlib as mpl
    mpl.use('pdf')
    import matplotlib.pyplot as plt
    #from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle

    def rugplot(a, y0, dy, ax, **kwargs):
        return ax.plot([a,a], [y0, y0+dy], **kwargs)

    label = str(flavints)
    flavint_tex = flavints.tex()

    ebins = binning.true_energy
    ebin_edges = ebins.bin_edges.m_as('GeV')

    plt.close(1)
    plt.close(2)
    plt.close(3)
    #plot_fname = os.path.join(outdir, label + '.pdf')
    TOP = 0.925
    BOTTOM = 0.05
    RIGHT = 0.97
    LEFT = 0.07
    HSPACE = 0.12
    LABELPAD = 0.058
    #AXISBG = (0.5, 0.5, 0.5)
    AXISBG = (1.0, 1.0, 1.0)
    DARK_RED =  (0.7, 0.0, 0.0)
    HIST_PP = dict(
        facecolor=(1,0.5,0.5), edgecolor=DARK_RED,
        histtype='stepfilled', alpha=0.7, linewidth=2.0,
        label=r'$\mathrm{Histogram}$'
    )
    N_HBINS = 25
    DIFFUS_PP = dict(
        color=(0.0, 0.0, 0.0), linestyle='-', marker=None, alpha=0.6,
        linewidth=2.0, label=r'$\mathrm{VBWKDE}$'
    )
    TITLEFONTSIZE = 14
    LEGTITLEFONTSIZE = 12
    LEGFNTSIZE = 12
    #RUG_PP = dict(color=(1.0, 1.0, 1.0), linewidth=0.4, alpha=0.5)
    RUG_PP = dict(color=(0.8, 0.0, 0.0), linewidth=0.4, alpha=0.5)
    RUG_LAB =r'$\mathrm{Rug\,plot}$'
    LEGFNTCOL = (0,0,0)
    #LEGFACECOL = (0.2,0.2,0.2)
    LEGFACECOL = (0.8,0.8,0.8)
    LEGALPHA = 0.5
    GRIDCOL = (0.4, 0.4, 0.4)
    #pdfpgs = PdfPages(plot_fname)

    if ebin_n is None:
        idx = slice(0, None)
    elif isinstance(ebin_n, int):
        idx = ebin_n
    elif isinstance(ebin_n, slice):
        idx = ebin_n
    else:
        raise ValueError('Unhadled type for `ebin_n`: %s' %type(ebin_n))

    bin_numbers = range(len(kde_info))
    binfos = kde_info.keys()
    kinfos = kde_info.values()
    einfos = extra_info.values()
    for (bin_n, bin_info, kde_info, extra_info) in zip(bin_numbers[idx],
                                                       binfos[idx],
                                                       kinfos[idx],
                                                       einfos[idx]):
        plot_fname = os.path.join(outdir, label + format(bin_n, '03d') + '.pdf')

        ebin_min, ebin_mid, ebin_max = bin_info
        ebin_wid = ebin_max - ebin_min

        e_interp = kde_info['e_interp']
        cz_interp = kde_info['cz_interp']

        enu_err = extra_info['enu_err']
        cz_err = extra_info['cz_err']
        n_in_bin = len(enu_err)

        actual_left_ebin_edge, actual_right_ebin_edge = \
                extra_info['actual_ebin_edges']

        e_err_min, e_err_max = min(enu_err), max(enu_err)
        e_err_range = e_err_max - e_err_min

        cz_err_min, cz_err_max = min(cz_err), max(cz_err)
        cz_err_range = cz_err_max - cz_err_min

        enu_bw = extra_info['enu_bw']
        cz_bw = extra_info['cz_bw']

        enu_mesh = extra_info['enu_mesh']
        cz_mesh = extra_info['cz_mesh']

        e_kde_lims = extra_info['e_kde_lims']
        cz_kde_lims = extra_info['cz_kde_lims']

        enu_pdf = e_interp(enu_mesh)
        cz_pdf = cz_interp(cz_mesh)

        fig1 = plt.figure(1, figsize=(8,10), dpi=90)
        fig1.clf()
        ax1 = fig1.add_subplot(211, axisbg=AXISBG)

        # Retrieve region where VBWKDE lives
        ml_ci = MLConfInterval(x=enu_mesh, y=enu_pdf)
        #for conf in np.logspace(np.log10(0.999), np.log10(0.95), 50):
        #    try:
        #        lb, ub, yopt, r = ml_ci.findCI_lin(conf=conf)
        #    except:
        #        pass
        #    else:
        #        break
        #xlims = (min(-ebin_mid*1.5, lb),
        #         max(min(ub, 6*ebin_mid),2*ebin_mid))
        lb, ub, yopt, r = ml_ci.findCI_lin(conf=0.98)
        xlims = (lb, #min(-ebin_mid*1.5, lb),
                 max(min(ub, 6*ebin_mid),2*ebin_wid))

        #xlims = (
        #    -ebin_wid*1.5,
        #    ebin_wid*1.5
        #)
        #    min(ebin_mid*2, ebin_edges[-1]+(ebin_edges[-1]-ebin_edges[0])*0.1)
        #)

        # Histogram of events' reco error
        e_hbins = np.linspace(
            e_err_min-0.02*e_err_range,
            e_err_max+0.02*e_err_range,
            N_HBINS*np.round(e_err_range/ebin_mid)
        )
        hvals, e_hbins, hpatches = ax1.hist(
            enu_err, bins=e_hbins, normed=True, **HIST_PP
        )

        # Plot the VBWKDE
        ax1.plot(enu_mesh, enu_pdf, **DIFFUS_PP)
        axlims = ax1.axis('tight')
        ax1.set_xlim(xlims)
        ymax = axlims[3]*1.05
        ax1.set_ylim(0, ymax)

        # Grey-out regions outside binned region, so it's clear what
        # part of tail(s) will be thrown away
        width = -ebin_mid+ebin_edges[0]-xlims[0]
        unbinned_region_tex = r'$\mathrm{Unbinned}$'
        if width > 0:
            ax1.add_patch(Rectangle((xlims[0],0), width, ymax, #zorder=-1,
                                    alpha=0.30, facecolor=(0.0 ,0.0, 0.0),
                                    fill=True,
                                    ec='none'))
            ax1.text(xlims[0]+(xlims[1]-xlims[0])/40., ymax/10.,
                     unbinned_region_tex, fontsize=14, ha='left',
                     va='bottom', rotation=90, color='k')

        width = xlims[1] - (ebin_edges[-1]-ebin_mid)
        if width > 0:
            ax1.add_patch(Rectangle((xlims[1]-width,0), width, ymax,
                                    alpha=0.30, facecolor=(0, 0, 0),
                                    fill=True, ec='none'))
            ax1.text(xlims[1]-(xlims[1]-xlims[0])/40., ymax/10.,
                     unbinned_region_tex, fontsize=14, ha='right',
                     va='bottom', rotation=90, color='k')

        # Rug plot of events' reco energy errors
        ylim = ax1.get_ylim()
        dy = ylim[1] - ylim[0]
        ruglines = rugplot(enu_err, y0=ylim[1], dy=-dy/40., ax=ax1,
                           **RUG_PP)
        ruglines[-1].set_label(RUG_LAB)

        # Legend
        leg_title_tex = r'$\mathrm{Normalized}\,E\mathrm{-err.\,distr.}$'
        x1lab = ax1.set_xlabel(
            r'$E_{\mathrm{reco}}-E_{\mathrm{true}}\;' +
            r'(\mathrm{GeV})$', labelpad=LABELPAD
        )
        leg = ax1.legend(loc='upper right', title=leg_title_tex,
                         frameon=True, framealpha=LEGALPHA,
                         fancybox=True, bbox_to_anchor=[1,0.975])

        # Other plot details
        ax1.xaxis.set_label_coords(0.9, -LABELPAD)
        ax1.xaxis.grid(color=GRIDCOL)
        ax1.yaxis.grid(color=GRIDCOL)
        leg.get_title().set_fontsize(LEGTITLEFONTSIZE)
        leg.get_title().set_color(LEGFNTCOL)
        [t.set_color(LEGFNTCOL) for t in leg.get_texts()]
        [t.set_fontsize(LEGFNTSIZE) for t in leg.get_texts()]
        frame = leg.get_frame()
        frame.set_facecolor(LEGFACECOL)
        frame.set_edgecolor(None)

        #
        # Coszen plot
        #

        ax2 = fig1.add_subplot(212, axisbg=AXISBG)
        cz_hbins = np.linspace(
            cz_err_min-0.02*cz_err_range,
            cz_err_max+0.02*cz_err_range,
            N_HBINS*3
        )
        hvals, cz_hbins, hpatches = ax2.hist(
            cz_err, bins=cz_hbins, normed=True, **HIST_PP
        )
        ax2.plot(cz_mesh, cz_pdf, **DIFFUS_PP)
        fci = MLConfInterval(x=cz_mesh, y=cz_pdf)
        lb, ub, yopt, r = fci.findCI_lin(conf=0.995)
        axlims = ax2.axis('tight')
        ax2.set_xlim(lb, ub)
        ax2.set_ylim(0, axlims[3]*1.05)

        ylim = ax2.get_ylim()
        dy = ylim[1] - ylim[0]
        ruglines = rugplot(cz_err, y0=ylim[1], dy=-dy/40., ax=ax2, **RUG_PP)
        ruglines[-1].set_label(r'$\mathrm{Rug\,plot}$')

        x2lab = ax2.set_xlabel(
            r'$\cos\,\theta_{\mathrm{reco}}-\cos\,\theta_{\,\mathrm{true}}$',
            labelpad=LABELPAD
        )
        ax2.xaxis.set_label_coords(0.9, -LABELPAD)
        ax2.xaxis.grid(color=GRIDCOL)
        ax2.yaxis.grid(color=GRIDCOL)
        leg_title_tex = r'$\mathrm{Normalized}\,\cos\,\theta \mathrm{-err.\,distr.}$'
        leg = ax2.legend(loc='upper right', title=leg_title_tex,
                         frameon=True, framealpha=LEGALPHA, fancybox=True,
                         bbox_to_anchor=[1,0.975])
        leg.get_title().set_fontsize(LEGTITLEFONTSIZE)
        leg.get_title().set_color(LEGFNTCOL)
        [t.set_color(LEGFNTCOL) for t in leg.get_texts()]
        [t.set_fontsize(LEGFNTSIZE) for t in leg.get_texts()]
        frame = leg.get_frame()
        frame.set_facecolor(LEGFACECOL)
        frame.set_edgecolor(None)

        actual_bin_tex = ''
        if ((actual_left_ebin_edge != ebin_min)
            or (actual_right_ebin_edge != ebin_max)):
            actual_bin_tex = r'E_{\mathrm{true}}\in [' + \
                    format(actual_left_ebin_edge, '0.2f') + r',\,' + \
                    format(actual_right_ebin_edge, '0.2f') + r'] \mapsto '
        stt = r'$\mathrm{Resolutions,\,' + flavint_tex + r'}$' + '\n' + \
                r'$' + actual_bin_tex + r'\mathrm{Bin}_{' + format(bin_n, 'd') + r'}\equiv E_{\mathrm{true}}\in [' + format(ebin_min, '0.2f') + \
                r',\,' + format(ebin_max, '0.2f') + r']\,\mathrm{GeV}' + \
                r',\,N_\mathrm{events}=' + format(n_in_bin, 'd') + r'$'

        fig1.subplots_adjust(top=TOP, bottom=BOTTOM, left=LEFT, right=RIGHT, hspace=HSPACE)
        suptitle = fig1.suptitle(stt)
        suptitle.set_fontsize(TITLEFONTSIZE)
        suptitle.set_position((0.5,0.98))
        logging.trace('plot_fname = %s' %plot_fname)
        fig1.savefig(plot_fname, format='pdf')

