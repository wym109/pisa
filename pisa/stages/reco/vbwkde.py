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

# TODO reco_scale params
# In PISA v2, e_reco_scale and cz_reco_scale where only allowed to be 1.
# Enforce this in _compute_nominal_transforms.

from copy import deepcopy
from string import ascii_lowercase

import numpy as np
from scipy.interpolate import interp1d

from pisa.core.binning import MultiDimBinning
from pisa.core.events import Events
from pisa.core.stage import Stage
from pisa.core.transform import BinnedTensorTransform, TransformSet
from pisa.utils import kde, confInterval
from pisa.utils.flavInt import flavintGroupsFromString
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging, set_verbosity


def rel2abs(rel_coord, abs_bin_midpoint, rel_scale_ref, rel_to_abs_scale,
            obj_abs_shift):
    """Convert coordinates in a space relative to a bin center to the
    equivalent coordinate in absolute coordinate space, allowing for shifting
    and scaling after the transformation.


    Parameters
    ----------
    rel_coord
        Relative coordinate
    abs_bin_midpoint
        Midpoint (in *absolute* coordinate space) of the bin about which the
        relative coordinates are defined
    rel_scale_ref
        Coordinate (in *relative* coordinate space) about which scaling will be
        performed
    rel_to_abs_scale
        Scaling factor; >1 implies coordinates in the absolute space are closer
        together--i.e., a shape in the absolute coordinate shape is wider than
        in the relative coordinate space
    obj_abs_shift
        Shift. This is the amount (in *absolute coordinates*) that a shape
        is shifted after it is scaled and transformed to absolute
        coordinates.

    """
    assert rel_to_abs_scale > 0
    return (
        rel_scale_ref + (rel_coord - rel_scale_ref)/rel_to_abs_scale
        + abs_bin_midpoint
        - obj_abs_shift
    )


def abs2rel(abs_coord, abs_bin_midpoint, abs_scale_ref, rel_to_abs_scale,
            obj_abs_shift):
    """Convert coordinates in absolute coordinate spaceto the equivalent
    coordinate in coordinate space relative about a bin's midpoint.
    Additionally, coordinates can be shifted and/or scaled before the
    transformation to relative coordinate space.


    Parameters
    ----------
    abs_coord
        Absolute coordinate
    abs_bin_midpoint
        Midpoint (in *absolute* coordinate space) of the bin about which the
        relative coordinates are to be defined
    abs_scale_ref
        Coordinate (in *absolute* coordinate space) about which scaling will be
        performed
    rel_to_abs_scale
        Scaling factor; >1 implies coordinates in the absolute space are closer
        together--i.e., a shape in the absolute coordinate shape is wider than
        in the relative coordinate space
    obj_abs_shift
        Shift. This is the amount (in *absolute coordinates*) that a shape
        will be shifted after it is scaled but *prior to* translation to the
        relative coordinate space.

    """
    assert scale > 0
    return (
        abs_scale_ref + (abs_coord - abs_scale_ref)*rel_to_abs_scale
        - abs_bin_midpoint
        + obj_abs_shift
    )


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

        e_res_scale : float
            A scaling factor for energy resolutions.

        cz_res_scale : float
            A scaling factor for coszen resolutions.

        transform_events_keep_criteria : None or string
            Additional cuts that are applied to events prior to computing
            transforms with them. E.g., "true_coszen <= 0" removes all MC-true
            downgoing events. See `pisa.core.events.Events` class for details
            on cut specifications.

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

    # TODO: sum_grouped_flavints : bool

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

        if sum_grouped_flavints:
            raise NotImplementedError('`sum_grouped_flavints` must be False.')
        self.sum_grouped_flavints = sum_grouped_flavints

        # All of the following params (and no more) must be passed via the
        # `params` argument.
        expected_params = (
            'reco_events', 'e_res_scale', 'cz_res_scale',
            'transform_events_keep_criteria'
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

    def validate_binning(self):
        assert self.input_binning == self.output_binning
        #assert self.input_binning.num_dims == self.output_binning.num_dims

    def reflect1d(self, x, refl):
        """Reflect a point x in 1D about another point, refl"""
        return 2*refl - x

    @staticmethod
    def compute_kdes(events, binning):
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
        e_true = events[repr_flav_int]['true_energy']
        e_reco = events[repr_flav_int]['reco_energy']
        cz_true = events[repr_flav_int]['true_coszen']
        cz_reco = events[repr_flav_int]['reco_coszen']
        ebins = binning.true_energy.bin_edges.m_as('GeV')
        czbins = binning.true_coszen.bin_edges.m_as('dimensionless')

        # NOTE: below defines bin centers on linear scale; other logic
        # in this method assumes this to be the case, so
        # **DO NOT USE** utils.utils.get_bin_centers in this method, which
        # may return logarithmically-defined centers instead.
        bin_edges = ebins
        left_ebin_edges = ebin_edges[0:-1]
        right_ebin_edges = ebin_edges[1:]
        ebin_centers = (left_ebin_edges+right_ebin_edges)/2.0
        ebin_range = ebin_edges[-1] - ebin_edges[0]
        n_ebins = len(ebin_centers)

        czbin_edges = czbins
        left_czbin_edges = czbin_edges[0:-1]
        right_czbin_edges = czbin_edges[1:]
        czbin_centers = (left_czbin_edges+right_czbin_edges)/2.0
        n_czbins = len(czbin_centers)

        n_events = len(e_true)

        if MIN_NUM_EVENTS > n_events:
            MIN_NUM_EVENTS = n_events
        if TGT_NUM_EVENTS > n_events:
            TGT_NUM_EVENTS = n_events

        kde_info = OrderedDict()
        for ebin_n in range(n_ebins):
            ebin_min = left_ebin_edges[ebin_n]
            ebin_max = right_ebin_edges[ebin_n]
            ebin_mid = (ebin_min+ebin_max)/2.0
            ebin_wid = ebin_max-ebin_min

            logging.debug(
                'Processing true-energy bin_n=' + format(ebin_n, 'd') + ' of ' +
                format(n_ebins-1, 'd') + ', E_{nu,true} in ' +
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
            dmin = min(enu_err)
            dmax = max(enu_err)
            drange = dmax-dmin

            e_lowerlim = min(ENERGY_RANGE[0]-ebin_mid*1.5, dmin-drange*0.5)
            e_upperlim = max((np.max(ebin_edges)-ebin_mid)*1.5, dmax+drange*0.5)
            egy_kde_lims = np.array([e_lowerlim, e_upperlim])

            # Use at least min_num_pts points and at most the next-highest
            # integer-power-of-two that allows for at least 10 points in the
            # smallest energy bin
            min_num_pts = 2**12
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

            # Exapnd range of sample points for future axis scaling
            e_factor = 4

            low_lim_shift = egy_kde_lims[0] * (e_factor - 1)
            upp_lim_shift = egy_kde_lims[1] * (e_factor - 1)

            egy_kde_lims_ext = np.copy(egy_kde_lims)
            if low_lim_shift > 0:
                egy_kde_lims_ext[0] = (
                    egy_kde_lims[0] - low_lim_shift * (1./e_factor)
                )
            if upp_lim_shift < 0:
                egy_kde_lims_ext[1] = (
                    egy_kde_lims[1] - upp_lim_shift * (1./e_factor)
                )

            # Adjust kde_num_points accordingly
            kde_num_pts_ext = int(
                kde_num_pts*((egy_kde_lims_ext[1] - egy_kde_lims_ext[0])
                / (egy_kde_lims[1] - egy_kde_lims[0]))
            )

            # Compute variable-bandwidth KDEs
            enu_bw, enu_mesh, enu_pdf = kde.vbw_kde(
                data=enu_err,
                overfit_factor=OVERFIT_FACTOR,
                MIN=egy_kde_lims_ext[0],
                MAX=egy_kde_lims_ext[1],
                N=kde_num_pts_ext
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

            offset_enu_mesh = enu_mesh + ebin_mid

            # Create linear interpolator for the PDF
            e_interp = interp1d(
                x=offset_enu_mesh, y=enu_pdf, kind='linear',
                copy=True, bounds_error=True, fill_value=np.nan
            )

            #==================================================================
            # Neutrino coszen resolution for events in this energy bin
            #==================================================================
            dmin = min(cz_err)
            dmax = max(cz_err)
            drange = dmax-dmin

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
            N_cz_mesh_ext = int(N_cz_mesh * ((cz_kde_max_ext - cz_kde_min_ext)
                                             / (cz_kde_max - cz_kde_min)))

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

            assert np.min(cz_pdf) >= -EPSILON, str(np.min(cz_pdf))

            # coszen interpolant is centered about "0-error"--i.e., the bin
            # center
            cz_interp = interp1d(
                x=cz_mesh, y=cz_pdf, kind='linear',
                copy=True, bounds_error=True, fill_value=np.nan
            )

            kde_info[(ebin_min, ebin_mid, ebin_max)] = dict(
                e_interp=e_interp, cz_interp=cz_interp
            )

        return kde_info

    def compute_kernels_from_kdes(self, kde_info, binning, e_res_scale=1,
                                  cz_res_scale=1, e_shift=None, cz_shift=None,
                                  ref_point='mode'):
        """Construct a 4D kernel set from linear interpolants describing the
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
        e_reco_shift : scalar Quantity
        cz_reco_shift : scalar Quantity
        ref_point : string


        Returns
        -------
        kernel4d : 4D array of float
            Mapping from the number of events in each bin of the 2D
            MC-true-events histogram to the number of events reconstructed in
            each bin of the 2D reconstructed-events histogram. Dimensions are
              len(self.ebins)-1 x len(self.czbins)-1 x len(self.ebins)-1 x
              len(self.czbins)-1
            since ebins and czbins define the histograms' bin edges.

        """
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

        SAMPLES_PER_BIN = 50
        """Number of samples for computing area in a bin (via np.trapz)."""

        if isinstance(e_res_scale, Param):
            e_res_scale = e_res_scale.value.m_as('dimensionless')
        if isinstance(e_res_scale, Param):
            cz_res_scale = cz_res_scale.value.m_as('dimensionless')
        if isinstance(e_shift, Param):
            e_shift = e_shift.value.m_as('GeV')
        if isinstance(cz_shift, Param):
            cz_shift = cz_shift.value.m_as('dimensionless')
        if isinstance(ref_point, basestring):
            ref_point = ref_point.strip().lower()
        assert ref_point in [0, 'zero', 'mean', 'mode']

        e_dim_num = binning.index('energy', use_basename=True)
        cz_dim_num = binning.index('coszen', use_basename=True)

        ebins = binning.dims[e_dim_num]
        czbins = binning.dims[cz_dim_num]

        e_oversamp = ebins.oversample(SAMPLES_PER_BIN-1)
        cz_oversamp = czbins.oversample(SAMPLES_PER_BIN-1)

        # Convert to computational units
        e_oversamp = e_oversamp.bin_edges.m_as('GeV')
        cz_oversamp = cz_oversamp.bin_edges.m_as('dimensionless')

        # Object in which to store the 4D kernels: np 4D array
        kernel4d = np.zeros(binning.shape * 2)

        for ebin_n, (k, v) in enumerate(kde_info.iteritems()):
            ebin = ebins[ebin_n]
            recorded_emin, recorded_emid, recorded_emax = k
            e_interp, cz_interp = v

            if ref_point in [0, 'zero']:
                e_ref = recorded_emid
            elif ref_point == 'mean':
                cum = np.cumsum(e_interp.y)
                cdf = cum / cum[-1]
                # Find the mean: locate where CDF is 0.5
                quantile_func = interp1d(cdf, e_interp.x)
                e_ref = quantile_func(0.5)
            elif ref_point == 'mode':
                e_ref = e_interp.x[e_interp.y == np.max(e_interp.y)][0]

            e_scale_shift = \
                    (e_oversamp - e_ref - e_reco_shift)/e_res_scale + e_ref

            # NOTE: We don't need to account for area lost in tail below 0 GeV
            # so long as our analysis doesn't go to 0: We can just assume all
            # events below our lower threshold end up below the threshold but
            # above 0, and then we have no "missing" area.

            # Divide by e_res_scale to keep the PDF area normalized to 1.0
            # when we make it wider/narrower.
            e_pdf = e_interp(e_scale_shift) / e_res_scale

            ebin_areas = []
            for n in xrange(ebins.num_bins):
                sl = slice(n*SAMPLES_PER_BIN, (n+1)*SAMPLES_PER_BIN + 1)
                ebin_areas.append(np.trapz(x=e_oversamp[sl], y=e_pdf[sl]))

            # Sum the individual bins' areas
            tot_ebin_area = np.sum(ebin_areas)

            #==================================================================
            # Neutrino coszen resolution for events in this energy bin
            #==================================================================
            for czbin_n in range(n_czbins):
                czbin = czbins[czbin_n]

                czbin_min, czbin_max = czbin.bin_edges.m_as('dimensionless')
                czbin_mid = czbin.midpoints[0].m_as('dimensionless')

                # Figure out the limits of the interpolant in absolute cz-space
                # (whereas the interpolant is relative to a cz bin's
                # midpoint)
                cz_interp_min = np.min(cz_interp.x)
                cz_interp_max = np.max(cz_interp.x)

                # Figure out where all bin edges lie in this re-centered
                # distribution (some bins may be repeated since bins in [-1,0]
                # and err in [-2,1]:
                #
                # 1. Find limits of mesh values..
                mmin = offset_cz_mesh[0]
                mmax = offset_cz_mesh[-1]

                # 2. Map all bin edges into the full mesh-value range,
                # reflecting about -1 and +1. If the reflected edge is outside
                # the mesh range, use the exceeded limit of the mesh range as
                # the bin edge instead.
                #
                # This maps every bin edge {i} to 3 new edges, indexed
                # new_edges[i][{0,1,2}]. Bins are formed by adjacent indices
                # and same-subindices, so what started as, e.g., bin 3 now is
                # described by (left, right) edges at
                #   (new_edges[3][0], new_edges[4][0]),
                #   (new_edges[3][1], new_edges[4][1]), and
                #   (new_edges[3][2], new_edges[4][2]).

                # NOTE / TODO: It's tempting to dynamically set the number of
                # reflections to minimize computation time, but I think it
                # breaks the code. Just set to a reasonably large number for
                # now and accept the performance penalty. ALSO: if you change
                # the parity of the number of reflections, the code below that
                # has either (wrap_n % 2 == 0) or (wrap_n+1 % 2 == 0) must be
                # swapped!!!
                n_left_reflections = 4
                n_right_reflections = 4

                new_czbin_edges = []
                for edge in czbin_edges:
                    edges_refl_left = []
                    for n in xrange(n_left_reflections):
                        edge_refl_left = self.reflect1d(edge, -1-(2*n))
                        if edge_refl_left < mmin:
                            edge_refl_left = mmin
                        edges_refl_left.append(edge_refl_left)
                    edges_refl_right = []
                    for n in xrange(n_right_reflections):
                        edge_refl_right = self.reflect1d(edge, +1+(2*n))
                        if edge_refl_right > mmax:
                            edge_refl_right = mmax
                        edges_refl_right.append(edge_refl_right)
                    # Include all left-reflected versions of this bin edge, in
                    # increasing-x order + this bin edge + right-reflected
                    # versions of this bin edge
                    new_czbin_edges.append(edges_refl_left[::-1] + [edge]
                                           + edges_refl_right)

                # Record all unique bin edges
                edge_locs = set()
                [edge_locs.update(edges) for edges in new_czbin_edges]

                # Throw away bin edges that are already in the mesh
                [edge_locs.remove(edge) for edge in list(edge_locs)
                 if edge in offset_cz_mesh]

                # Make into sorted list
                edge_locs = sorted(edge_locs)

                # Record the total area under the curve
                int_val0 = np.trapz(y=cz_pdf_scaled, x=offset_cz_mesh)

                # Insert the missing bin edge locations & pdf-values into
                # the mesh & pdf, respectively
                temp = np.copy(offset_cz_mesh)

                edge_pdfs = interp(edge_locs)
                insert_ind = np.searchsorted(offset_cz_mesh, edge_locs)
                offset_cz_mesh = np.insert(offset_cz_mesh, insert_ind,
                                           edge_locs)
                offset_cz_pdf = np.insert(cz_pdf_scaled, insert_ind, edge_pdfs)
                assert np.min(offset_cz_pdf) > -EPSILON

                # Check that this total of all the bins is equal to the total
                # area under the curve (i.e., check there is no overlap between
                # or gaps between bins)
                int_val = np.trapz(y=offset_cz_pdf, x=offset_cz_mesh)
                assert np.abs(int_val-1) < EPSILON

                # Renormalize if it's not exactly 1
                if int_val != 1.0:
                    offset_cz_pdf = offset_cz_pdf / int_val

                # Add up the area in the bin and areas that are "reflected"
                # into this bin
                new_czbin_edges = np.array(new_czbin_edges)
                czbin_areas = np.zeros(np.shape(new_czbin_edges)[0]-1)
                for wrap_n in range(np.shape(new_czbin_edges)[1]):
                    bin_edge_inds = np.searchsorted(offset_cz_mesh,
                                                    new_czbin_edges[:,wrap_n])
                    lbinds = bin_edge_inds[0:-1]
                    rbinds = bin_edge_inds[1:]
                    # Make sure indices that appear first are less than indices
                    # that appear second in a pair of bin indices
                    if (wrap_n+1) % 2 == 0:
                        bininds = zip(rbinds, lbinds)
                    else:
                        bininds = zip(lbinds, rbinds)
                    tmp_areas = []
                    for (binind_left_edge, binind_right_edge) in bininds:
                        if binind_left_edge == binind_right_edge:
                            tmp_areas.append(0)
                            continue
                        this_bin_area = np.array(np.trapz(
                            y=offset_cz_pdf[binind_left_edge:binind_right_edge+1],
                            x=offset_cz_mesh[binind_left_edge:binind_right_edge+1]
                        ))
                        tmp_areas.append(this_bin_area)
                    czbin_areas += np.array(tmp_areas)

                assert np.min(czbin_areas) > -EPSILON

                tot_czbin_area = np.sum(czbin_areas)
                assert tot_czbin_area < int_val + EPSILON

                if energy_first:
                    x, y = ebin_n, czbin_n
                    kernel4d[x, y] = np.outer(ebin_areas, czbin_areas)
                else:
                    x, y = czbin_n, ybin_n
                    kernel4d[x, y] = np.outer(czbin_areas, ebin_areas)

                d = (np.sum(kernel4d[x,y])-tot_ebin_area*tot_czbin_area)
                assert (d < EPSILON), d, epsilon

        check_areas = kernel4d.sum(axis=(2,3))

        assert np.max(check_areas) < 1 + EPSILON, str(np.max(check_areas))
        assert np.min(check_areas) > 0 - EPSILON, str(np.min(check_areas))

        return kernel4d

    # TODO arbitrary dimensions (currently must be energy and coszen)
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

        # TODO: attempt to load kde_info from disk cache or store new to disk;
        # hash value from relevant params: `reco_events` and
        # `transform_events_keep_criteria`...

        self.all_kde_info = OrderedDict()
        for xform_flavints in self.transform_groups:
            logging.debug("Working on %s reco kernels" %xform_flavints)

            # TODO: `repr_flav_int` is due to repetition of flavors/interaction
            # types in events files that are "grouped" together. Someday that
            # will change, and then this has to change, too.
            repr_flav_int = xform_flavints.flavints()[0]
            self.all_kde_info[xform_flavints] = compute_kdes(
                events=self.remaining_events, binning=self.output_binning
            )

        # Apply scaling factors and figure out the area per bin for each KDE
        trnsforms = []
        for xform_flavints in self.transform_groups:
            reco_kernel = self.kernels_from_kdes(kde_info=kde_info,
                                                 binning=input_binning)

            # Swap axes according to specified binning order
            if self.input_binning.names[0] == 'true_coszen':
                reco_kernel = np.swapaxes(reco_kernel, 0, 1)
            if self.output_binning.names[0] == 'reco_coszen':
                reco_kernel = np.swapaxes(reco_kernel, 2, 3)

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
                nominal_transforms.append(xform)

        return TransformSet(transforms=nominal_transforms)

    def scale_resolutions(self):
        """Apply scaling factors to the nominal resolutions."""
        e_res_scale = self.params.e_res_scale.value.m_as('dimensionless')
        cz_res_scale = self.params.cz_res_scale.value.m_as('dimensionless')
