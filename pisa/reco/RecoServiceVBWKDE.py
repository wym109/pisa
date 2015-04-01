#
# RecoServiceVBWKDE.py
#
# author: J. L. Lanfranchi
#         jll1062@phys.psu.edu
#
# date:   March 31, 2015
#


import copy
import itertools
import numpy as np
from scipy import interpolate

from pisa.reco.RecoServiceBase import RecoServiceBase
from pisa.resources.resources import find_resource
from pisa.utils import kde, hdf, utils
from pisa.utils.log import logging


def reflect1d(x, refl):
    """Reflect a point x in 1D about another point, refl"""
    return 2*refl - x


class RecoServiceVBWKDE(RecoServiceBase):
    """Reco service which creates reconstruction kernels using VBWKDEs

    The variable-bandwidth kernel density estimation (VBWKDE) technique defined
    in pisa.utils.kde shows successful and fast fitting to event densities with
    lower variance than histograms (particularly in low-statistics situations);
    superior fitting characteristics than fixed-bandwidth KDEs for skewed
    and heavy-tailed distributions (as we encounter in our V36 MC data);
    and perfectly reproducible results at a tiny fraction of the time as
    compared with the hand-tuned sum-of-2-gaussians method.

    It is expected that the _get_reco_kernels method is called

    """
    def __init__(self, ebins, czbins, reco_vbwkde_evts_file, **kwargs):
        """Initializtion

        Parameters
        ----------
        ebins : sequence
            Neutrino energy histogram bin edges

        czbins : sequence
            Cosine-of-zenith histogram bin edges

        reco_vbwkde_evts_file : str or dict
            Resource location of HDF5 file containing event reconstruction
            information for each of the neutrino flavors and interaction types.
            If an HDF5 file name is specified, the method utils.hdf.from_hdf
            converts the contents to a nested dictionary. With either type of
            input, the dictionary must have the form
              eventsdict[flavor][int_type][{true|reco}_{energy|coszen}] =
                  np.array([...], dtype=float64)
            where each array of the same flavor and int_type must be of the
            same length.


        """
        self.kernels = None
        self.reco_events_hash = ''
        self.duplicate_nu_bar_cc = False
        self.duplicate_nc = False

        self.MIN_NUM_EVENTS = 100
        self.TGT_NUM_EVENTS = 300
        self.EPSILON = 1e-10
        self.ENERGY_RANGE = [0, 501]

        RecoServiceBase.__init__(self, ebins=ebins, czbins=czbins,
                                 reco_vbwkde_evts_file=reco_vbwkde_evts_file,
                                 **kwargs)

    def _get_reco_kernels(self, reco_vbwkde_evts_file=None, evts_dict=None,
                          **kwargs):
        """Given a reco events resource (resource file name or dictionary),
        retrieve data from it then serialize and hash the data. If the object
        attribute kernels were computed from the same source data, simply
        return those. Otherwise, compute the kernels anew and return them.

        Arguments
        ---------
        NOTE: One--and only one--of the two arguments must be specified.

        reco_vbwkde_evts_file : str (or dict)
            Name or path to file containing event reco info. See doc for
            __init__ method for details about contents. If a dict is passed
            in, it is automatically populated to evts_dict (see below).

        evts_dict : dict
            Dictionary containing event reco info. Allows user to pass in a
            non-string-object to avoid re-loading a file to check whether the
            contents have changed each time. See doc for __init__ method for
            details about the dictionary's format.

        """
        REMOVE_SIM_DOWNGOING = True

        if (reco_vbwkde_evts_file is not None) and (evts_dict is not None):
            raise TypeError(
                'One--and only one--of {reco_vbwkde_evts_file|evts_dict} ' +
                'may be specified'
            )

        if isinstance(reco_vbwkde_evts_file, dict):
            evts_dict = reco_vbwkde_evts_file
            evts_dict = None

        if isinstance(reco_vbwkde_evts_file, str):
            logging.info('Constructing VBWKDEs from event true & reco ' +
                         'info in file: %s' % reco_vbwkde_evts_file)
            fpath = find_resource(reco_vbwkde_evts_file)
            eventsdict = hdf.from_hdf(fpath)
            new_hash = utils.hash_file(fpath)
        elif isinstance(evts_dict, dict):
            eventsdict = evts_dict
            new_hash = utils.hash_obj(eventsdict)
        else:
            raise TypeError('A {reco_vbwkde_evts_file|evts_dict} must be' +
                            'provided, where the former must be a str ' +
                            'and the latter must be a dict.')

        if (self.kernels is not None) and (new_hash == self.reco_events_hash):
            return self.kernels

        self.kernels = self.all_kernels_from_events(
            eventsdict=eventsdict, remove_sim_downgoing=REMOVE_SIM_DOWNGOING
        )
        self.reco_events_hash = new_hash

        return self.kernels

    def all_kernels_from_events(self, eventsdict, remove_sim_downgoing):
        """Given a reco events dictionary, retrieve reco/true information from
        it, group MC data by flavor & interaction type, and return VBWKDE-based
        PISA reco kernels for all flavors/types. Checks are performed if
        duplicate data has already been computed, in which case a (deep) copy
        of the already-computed kernels are populated.

        Arguments
        ---------
        eventsdict : dict
            Dictionary containing event reco info. See docstr for __init__ for
            details.

        remove_sim_downgoing : bool
            Whether to remove MC-true downgoing events prior to computing
            resolutions.

        """
        all_flavors = \
                ['nue', 'nue_bar', 'numu', 'numu_bar', 'nutau', 'nutau_bar']
        all_ints = ['cc', 'nc']
        flav_ints = itertools.product(all_flavors, all_ints)

        kernels = {f:{} for f in all_flavors}
        kernels['ebins'] = self.ebins
        kernels['czbins'] = self.czbins
        computed_datahashes = {}
        for flavor, int_type in flav_ints:
            logging.info("Working on %s/%s kernels" % (flavor, int_type))
            e_true = eventsdict[flavor][int_type]['true_energy']
            e_reco = eventsdict[flavor][int_type]['reco_energy']
            cz_true = eventsdict[flavor][int_type]['true_coszen']
            cz_reco = eventsdict[flavor][int_type]['reco_coszen']

            if remove_sim_downgoing:
                logging.info("Removing simulated downgoing " +
                              "events in KDE construction.")
                keep_inds = np.where(cz_true < 0.0)
                e_true = e_true[keep_inds]
                e_reco = e_reco[keep_inds]
                cz_true = cz_true[keep_inds]
                cz_reco = cz_reco[keep_inds]

            datahash = utils.hash_obj((e_true.tolist(), e_reco.tolist(),
                                       cz_true.tolist(), cz_reco.tolist()))
            if datahash in computed_datahashes:
                ref_flavor, ref_int_type = computed_datahashes[datahash]
                logging.info("   > Found duplicate source data; " +
                              "copying kernels already computed for " +
                              "%s/%s to %s/%s."
                              % (ref_flavor, ref_int_type, flavor, int_type))
                kernels[flavor][int_type] = copy.deepcopy(
                    kernels[ref_flavor][ref_int_type]
                )
                continue

            kernels[flavor][int_type] = self.single_kernel_set(
                e_true=e_true, cz_true=cz_true,
                e_reco=e_reco, cz_reco=cz_reco
            )
            computed_datahashes[datahash] = (flavor, int_type)

        return kernels

    def single_kernel_set(self, e_true, cz_true, e_reco, cz_reco):
        """Construct a 4D kernel set from MC events using VBWKDE.

        Given a set of MC events and each of their {energy{true, reco},
        coszen{true, reco}}, generate a 4D NumPy array that maps a 2D true-flux
        histogram onto the corresponding 2D reco-flux histogram.

        The resulting 4D array can be indexed logically using
          kernel4d[e_true_i, cz_true_j][e_reco_k, cz_reco_l]
        where the 4 indices point from a single MC-true histogram bin (i,j) to
        a single reco histogram bin (k,l).

        Binning of both MC-true and reco histograms is the same and is given by
        the values in self.ebins and self.czbins which define the bin *edges*
        (not the bin centers; hence, len(self.ebins) is one greater than the
        number of bins, etc.).

        NOTE: Actual limits in energy used to group events into a single "true"
        bin may be extended beyond the bin edges defined by self.ebins in order
        to gather enough events to successfully apply VBWKDE.

        Parameters
        ----------
        e_true : sequence
            MC-true neutrino energies, one per event
        cz_true : sequence
            MC-true neutrino coszen, one per event
        e_reco : sequence
            Reconstructed neutrino energies, one per event
        cz_reco : sequence
            Reconstructed neutrino coszen, one per event

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
        OVERFIT_FACTOR = 1.0

        assert np.min(np.diff(self.ebins)) > 0, \
            "Energy bin edges not monotonically increasing."
        assert np.min(np.diff(self.czbins)) > 0, \
            "coszen bin edges not monotonically increasing."

        # NOTE: below defines bin centers on linear scale; other logic
        # in this method assumes this to be the case, so
        # **DO NOT USE** utils.utils.get_bin_centers in this method, which
        # may return logarithmically-defined centers instead.

        ebin_edges = np.array(self.ebins)
        left_ebin_edges = ebin_edges[0:-1]
        right_ebin_edges = ebin_edges[1:]
        ebin_centers = (left_ebin_edges+right_ebin_edges)/2.0
        n_ebins = len(ebin_centers)

        czbin_edges = np.array(self.czbins)
        left_czbin_edges = czbin_edges[0:-1]
        right_czbin_edges = czbin_edges[1:]
        czbin_centers = (left_czbin_edges+right_czbin_edges)/2.0
        n_czbins = len(czbin_centers)

        n_events = len(e_true)

        if self.MIN_NUM_EVENTS > n_events:
            self.MIN_NUM_EVENTS = n_events
        if self.TGT_NUM_EVENTS > n_events:
            self.TGT_NUM_EVENTS = n_events

        # Object with which to store the 4D kernels: np 4D array
        kernel4d = np.zeros((n_ebins, n_czbins, n_ebins, n_czbins))

        # Object with which to store the 2D "aggregate_map": the total number
        # of events reconstructed into a given (E, CZ) bin, used for sanity
        # checks
        aggregate_map = np.zeros((n_ebins, n_czbins))
        for ebin_n in range(n_ebins):
            ebin_min = left_ebin_edges[ebin_n]
            ebin_max = right_ebin_edges[ebin_n]
            ebin_mid = (ebin_min+ebin_max)/2.0
            ebin_wid = ebin_max-ebin_min

            logging.debug(
                '  processing true-energy bin_n=' + str(ebin_n) + ' of ' +
                str(n_ebins-1) + ', E_{nu,true} in ' +
                '[' + str(ebin_min) + ', ' + str(ebin_max) + '] ...'
            )

            # Absolute distance from these events' re-centered reco energies to
            # the center of this energy bin; sort in ascending-distance order
            abs_enu_dist = sorted(np.abs(e_true - ebin_mid))

            # Grab the distance the number-"TGT_NUM_EVENTS" event is from the
            # bin center
            tgt_thresh_enu_dist = abs_enu_dist[self.TGT_NUM_EVENTS-1]

            # Grab the distance the number-"MIN_NUM_EVENTS" event is from the
            # bin center
            min_thresh_enu_dist = abs_enu_dist[self.MIN_NUM_EVENTS-1]

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
            in_ebin_ind = np.where(abs_enu_dist <= thresh_enu_dist)[0]
            n_in_bin = len(in_ebin_ind)

            # Extract just the neutrino-energy/coszen error columns' values for
            # succinctness
            enu_err = e_reco[in_ebin_ind] - e_true[in_ebin_ind]
            cz_err = cz_reco[in_ebin_ind] - cz_true[in_ebin_ind]

            #==================================================================
            # Neutrino energy resolutions
            #==================================================================
            dmin = min(enu_err)
            dmax = max(enu_err)
            drange = dmax-dmin

            e_lowerlim = min(self.ENERGY_RANGE[0]-ebin_mid*1.5, dmin-drange*0.5)
            e_upperlim = max((np.max(ebin_edges)-ebin_mid)*1.5, dmax+drange*0.5)
            egy_kde_lims = np.array([e_lowerlim, e_upperlim])

            # Use at least 2**10 points and at most the next-highest integer-
            # power-of-two that allows for at least two points in the smallest
            # energy bin
            min_bin_width = np.min(ebin_edges)
            min_pts_smallest_bin = 2.0
            kde_range = np.diff(egy_kde_lims)
            num_pts0 = kde_range/(min_bin_width/min_pts_smallest_bin)
            kde_num_pts = int(max(2**10, 2**np.ceil(np.log2(num_pts0))))
            logging.debug(
                ' Nevts=' + str(n_in_bin) + ' taken from [' +
                str(ebin_mid-thresh_enu_dist) + ', ' +
                str(ebin_mid+thresh_enu_dist) + ']' + ', KDE lims=' +
                str(kde_range) + ', KDE_N: ' + str(kde_num_pts)
            )

            # Compute variable-bandwidth KDEs
            enu_bw, enu_mesh, enu_pdf = kde.vbw_kde(
                data           = enu_err,
                overfit_factor = OVERFIT_FACTOR,
                MIN            = egy_kde_lims[0],
                MAX            = egy_kde_lims[1],
                N              = kde_num_pts
            )

            if np.min(enu_pdf) < 0:
                # Only issue warning if the most-negative value is negative
                # beyond specified acceptable-numerical-precision threshold
                # (EPSILON)
                if np.min(enu_pdf) <= -self.EPSILON:
                    logging.warn(
                        "np.min(enu_pdf) < 0: Minimum value is " +
                        str(np.min(enu_pdf)) +
                        "; forcing all negative values to 0."
                    )
                # Otherwise, just quietly clip any negative values at 0
                enu_pdf = np.clip(a=enu_pdf, a_min=0, a_max=np.inf)

            assert np.min(enu_pdf) >= 0, str(np.min(enu_pdf))

            # Re-center distribution at the center of the energy bin for which
            # errors were computed
            offset_enu_mesh = enu_mesh+ebin_mid
            offset_enu_pdf = enu_pdf

            # Get reference area under the PDF, for checking after interpolated
            # values are added.
            #
            # NOTE There should be NO normalization because any events lost due
            # to cutting off tails outside the binned region are actually going
            # to be lost, and so should penalize the total area.
            int_val0 = np.trapz(y=offset_enu_pdf,
                                x=offset_enu_mesh)

            # Create linear interpolator for the PDF
            interp = interpolate.interp1d(
                x             = offset_enu_mesh,
                y             = offset_enu_pdf,
                kind          = 'linear',
                copy          = True,
                bounds_error  = True,
                fill_value    = np.nan,
                assume_sorted = True
            )

            # Insert all bin edges' exact locations into the mesh (For accurate
            # accounting of area in each bin, must include values out to bin
            # edges)
            edge_locs = [be for be in
                         np.concatenate((left_ebin_edges, right_ebin_edges))
                         if not(be in offset_enu_mesh)]
            edge_locs.sort()
            edge_pdfs = interp(edge_locs)
            insert_ind = np.searchsorted(offset_enu_mesh, edge_locs)
            offset_enu_mesh = np.insert(offset_enu_mesh, insert_ind, edge_locs)
            offset_enu_pdf = np.insert(offset_enu_pdf, insert_ind, edge_pdfs)

            int_val = np.trapz(y=offset_enu_pdf, x=offset_enu_mesh)

            assert np.abs(int_val - int_val0) < self.EPSILON

            # Chop off distribution at extrema of energy bins
            valid_ind = np.where(
                (offset_enu_mesh >= np.min(ebin_edges)) &
                (offset_enu_mesh <= np.max(ebin_edges))
            )[0]
            offset_enu_mesh = offset_enu_mesh[valid_ind]
            offset_enu_pdf = offset_enu_pdf[valid_ind]

            # Check that there are no negative density values (after inserts)
            assert np.min(offset_enu_pdf) > 0-self.EPSILON, \
                str(np.min(offset_enu_pdf))

            # Record the integrated area after removing parts outside binned
            # range
            tot_ebin_area0 = np.trapz(y=offset_enu_pdf,
                                      x=offset_enu_mesh)

            # Check that it integrates to <= 1, sanity check
            assert tot_ebin_area0 < 1+self.EPSILON, str(tot_ebin_area0)

            # Identify indices encapsulating the defined energy bins' ranges,
            # and find the area of each bin
            lbinds = np.searchsorted(offset_enu_mesh, left_ebin_edges)
            rbinds = np.searchsorted(offset_enu_mesh, right_ebin_edges)
            bininds = zip(lbinds, rbinds)
            ebin_areas = [np.trapz(y=offset_enu_pdf[l:r+1],
                                   x=offset_enu_mesh[l:r+1])
                          for (l, r) in bininds]

            # Check that no bins have negative areas
            assert np.min(ebin_areas) >= 0

            # Sum the individual bins' areas
            tot_ebin_area = np.sum(ebin_areas)

            # Check that this total of all the bins is equal to the total area
            # under the curve (i.e., make sure there is no overlap or gaps
            # between bins)
            assert np.abs(tot_ebin_area-tot_ebin_area0) < self.EPSILON, \
                    'tot_ebin_area=' + str(tot_ebin_area) + \
                    ' should equal tot_ebin_area0=' + str(tot_ebin_area0)

            #==================================================================
            # Neutrino coszen resolutions
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
            #
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
            N_cz_mesh = 2**9

            # Data range for VBWKDE to consider
            cz_gaus_kde_min = -3
            cz_gaus_kde_max = +2

            cz_gaus_kde_failed = False
            previous_fail = False
            for n in xrange(3):
                # TODO: only catch specific exception
                try:
                    cz_bw, cz_mesh, cz_pdf = kde.vbw_kde(
                        data           = cz_err,
                        overfit_factor = OVERFIT_FACTOR,
                        MIN            = cz_gaus_kde_min,
                        MAX            = cz_gaus_kde_max,
                        N              = N_cz_mesh
                    )
                except:
                    cz_gaus_kde_failed = True
                    if n == 0:
                        logging.trace('(cz vbwkde ')
                    logging.trace('fail, ')
                    # If failure occurred in vbw_kde, expand the data range it
                    # takes into account; this usually helps
                    cz_gaus_kde_min -= 1
                    cz_gaus_kde_max += 1
                else:
                    if cz_gaus_kde_failed:
                        previous_fail = True
                        logging.trace('success!')
                    cz_gaus_kde_failed = False
                finally:
                    if previous_fail:
                        logging.trace(')')
                    previous_fail = False
                    if not cz_gaus_kde_failed:
                        break

            if cz_gaus_kde_failed:
                logging.warn('Failed to fit VBWKDE!')
                continue

            if np.min(cz_pdf) < 0:
                logging.warn("np.min(cz_pdf) < 0: Minimum value is " +
                             str(np.min(cz_pdf)) +
                             "; forcing all negative values to 0.")
                np.clip(a=cz_mesh, a_min=0, a_max=np.inf)

            assert np.min(cz_pdf) >= -self.EPSILON, \
                str(np.min(cz_pdf))

            for czbin_n in range(n_czbins):
                czbin_mid = czbin_centers[czbin_n]

                # Re-center distribution at the center of the current cz bin
                offset_cz_mesh = cz_mesh + czbin_mid

                # Create interpolation object, used to fill in bin edge values
                interp = interpolate.interp1d(
                    x             = offset_cz_mesh,
                    y             = cz_pdf,
                    kind          = 'linear',
                    copy          = True,
                    bounds_error  = False,
                    fill_value    = 0,
                    assume_sorted = True,
                )

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
                        edge_refl_left = reflect1d(edge, -1-(2*n))
                        if edge_refl_left < mmin:
                            edge_refl_left = mmin
                        edges_refl_left.append(edge_refl_left)
                    edges_refl_right = []
                    for n in xrange(n_right_reflections):
                        edge_refl_right = reflect1d(edge, +1+(2*n))
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
                int_val0 = np.trapz(y=cz_pdf, x=offset_cz_mesh)

                # Insert the missing bin edge locations & pdf-values into
                # the mesh & pdf, respectively
                edge_pdfs = interp(edge_locs)
                insert_ind = np.searchsorted(offset_cz_mesh, edge_locs)
                offset_cz_mesh = np.insert(offset_cz_mesh, insert_ind,
                                           edge_locs)
                offset_cz_pdf = np.insert(cz_pdf, insert_ind, edge_pdfs)
                assert np.min(offset_cz_pdf) > -self.EPSILON

                # Check that this total of all the bins is equal to the total
                # area under the curve (i.e., check there is no overlap between
                # or gaps between bins)
                int_val = np.trapz(y=offset_cz_pdf, x=offset_cz_mesh)
                assert np.abs(int_val-1) < self.EPSILON

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

                assert np.min(czbin_areas) > -self.EPSILON

                tot_czbin_area = np.sum(czbin_areas)
                assert tot_czbin_area < int_val + self.EPSILON

                kernel4d[ebin_n, czbin_n] = np.outer(ebin_areas, czbin_areas)
                assert (np.sum(kernel4d[ebin_n, czbin_n]) -
                        tot_ebin_area*tot_czbin_area) < self.EPSILON

        check_areas = kernel4d.sum(axis=(2,3))

        assert np.max(check_areas) < 1 + self.EPSILON, str(np.max(check_areas))
        assert np.min(check_areas) > 0 - self.EPSILON, str(np.min(check_areas))

        return kernel4d
