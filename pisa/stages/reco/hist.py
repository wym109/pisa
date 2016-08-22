
# PISA author: Timothy C. Arlen
#              tca3@psu.edu
#
# CAKE author: Steven Wren
#              steven.wren@icecube.wisc.edu
#
# date:   2016-05-27

"""
This reco service creates the pdfs of the reconstructed energy and coszen
from the true parameters. Provides reco event rate maps using these pdfs.
"""


from copy import deepcopy
from string import ascii_lowercase

import numpy as np

from pisa.core.binning import MultiDimBinning
from pisa.core.stage import Stage
from pisa.core.transform import BinnedTensorTransform, TransformSet
from pisa.utils.events import Events
from pisa.utils.flavInt import flavintGroupsFromString
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging, set_verbosity


# TODO: the below logic does not generalize to muons, but probably should
# (rather than requiring an almost-identical version just for muons). For
# example, an input arg can dictate neutrino or muon, which then sets the
# input_names and output_names.

class hist(Stage):
    """
    From the simulation file, creates 4D histograms of
    [true_energy][true_coszen][reco_energy][reco_coszen] which act as
    2D pdfs for the probability that an event with (true_energy,
    true_coszen) will be reconstructed as (reco_energy,reco_coszen).

    From these histograms and the true event rate maps, calculates
    the reconstructed even rate templates.

    Parameters
    ----------
    params : ParamSet
        Must exclusively have parameter:

        reco_weight_file : string or Events
            PISA events file to use to derive transforms, or a string
            specifying the resource location of the same.

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

    input_binning : MultiDimBinning or convertible thereto
        Input binning is in true variables, with names prefixed by "true_".
        Each must match a corresponding dimension in `output_binning`.

    output_binning : MultiDimBinning or convertible thereto
        Output binning is in reconstructed variables, with names (traditionally
        in PISA but not necessarily) prefixed by "reco_". Each must match a
        corresponding dimension in `input_binning`.

    disk_cache

    transforms_cache_depth : int >= 0

    outputs_cache_depth : int >= 0

    memcache_deepcopy : bool

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
                 input_binning, output_binning, error_method=None,
                 disk_cache=None, transforms_cache_depth=20,
                 outputs_cache_depth=20, memcache_deepcopy=True,
                 debug_mode=None):
        self.events_hash = None
        """Hash of events file or Events object used"""

        assert particles in ['neutrinos', 'muons']
        self.particles = particles
        """Whether stage is instantiated to process neutrinos or muons"""

        self.transform_groups = flavintGroupsFromString(transform_groups)
        """Particle/interaction types to group for computing transforms"""

        # All of the following params (and no more) must be passed via the
        # `params` argument.
        expected_params = (
            'reco_weight_file', 'reco_weights_name'
            # NOT IMPLEMENTED: 'e_reco_scale', 'cz_reco_scale'
        )

        if isinstance(input_names, basestring):
            input_names = (''.join(input_names.split(' '))).split(',')

        # Define the names of objects that get produced by this stage
        # The output combines nu and nubar together (just called nu)
        # All of the NC events are joined (they look the same in the detector).
        output_names = input_names

        # Invoke the init method from the parent class, which does a lot of
        # work for you.
        super(self.__class__, self).__init__(
            use_transforms=True,
            stage_name='reco',
            service_name='hist',
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

    def validate_binning(self):
        assert self.input_binning.num_dims == self.output_binning.num_dims

    def load_events(self):
        evts = self.params.reco_weight_file.value
        this_hash = hash_obj(evts)
        if this_hash == self.events_hash:
            return
        logging.debug('Extracting events from Events obj or file: %s' %evts)
        self.events = Events(evts)
        self.events_hash = this_hash

    def _compute_nominal_transforms(self):
        """Generate reconstruction "smearing kernels" by histogramming true and
        reconstructed variables from a Monte Carlo events file.

        The resulting transform is a 2N-dimensional histogram, where N is the
        dimensionality of the input binning. The transform maps the truth bin
        counts to the reconstructed bin counts.

        I.e., for the case of 1D input binning, the ith element of the
        reconstruction kernel will be a map showing the distribution of events
        over all the reco space from truth bin i. This will be normalised to
        the total number of events in truth bin i.

        Notes
        -----
        In the current implementation these histograms are made
        **UN**weighted. This is probably quite wrong...

        """
        self.load_events()

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

        # First N dimensions of the transform are the input dimensions; last N
        # dimensions are the output dimensions. So concatenate the two into a
        # single 2N-dimensional binning object to work with.
        transform_binning = MultiDimBinning(input_binning.dimensions
                                            + output_binning.dimensions)
        true_and_reco_bin_edges = [dim.bin_edges.magnitude
                                   for dim in transform_binning.dimensions]
        true_and_reco_binning_cols = transform_binning.names
        true_only_bin_edges = [dim.bin_edges.magnitude
                               for dim in input_binning.dimensions]
        true_only_binning_cols = input_binning.names

        nominal_transforms = []
        for flav_int_group in self.transform_groups:
            logging.debug("Working on %s reco kernels" %flav_int_group)

            # True+reco (2N-dimensional) histogram is the basis for the
            # transformation
            reco_kernel = self.events.histogram(
                kinds=flav_int_group,
                binning=true_and_reco_bin_edges,
                binning_cols=true_and_reco_binning_cols,
                weights_col=self.params['reco_weights_name'].value
            )

            # This takes into account the correct kernel normalization:
            # What this means is that we have to normalise the reco map
            # to the number of events in the truth bin.
            #
            # I.e., we have N events from the truth bin which then become
            # spread out over the whole map due to reconstruction.
            # The normalisation is dividing this map by N.
            #
            # Previously this was hard-coded for 2 dimensions, but I have tried
            # to generalise it to arbitrary dimensionality.

            # Truth-only (N-dimensional) histogram will be used for
            # normalization (so transform is in terms of fraction-of-events in
            # input--i.e. truth--bin).
            true_event_counts = self.events.histogram(
                kinds=flav_int_group,
                binning=true_only_bin_edges,
                binning_cols=true_only_binning_cols,
                weights_col=self.params['reco_weights_name'].value
            )

            # If there weren't any events in the input (true_*) bin, make this
            # bin have no effect -- i.e., populate all output bins
            # corresponding to the input bin with zeros via `nan_to_num`.
            with np.errstate(divide='ignore', invalid='ignore'):
                norm_factors = np.nan_to_num(1.0 / true_event_counts)

            # Numpy broadcasts lower-dimensional things to higher dimensions
            # from last dimension to first; if we simply mult the reco_kernel
            # by norm_factors, this will apply the normalization to the
            # __output__ dimensions rather than the input dimensions. Add
            # "dummy" dimensions to norm_factors where we want the "extra
            # dimensions": at the end.
            for dim in self.output_binning.dimensions:
                norm_factors = np.expand_dims(norm_factors, axis=-1)

            # Apply the normalization to the kernels
            reco_kernel *= norm_factors

            #assert np.all(reco_kernel >= 0), 'number of elements less than 0 = %d' % np.sum(reco_kernel < 0)
            #ndims = len(reco_kernel.shape)
            #sum_over_axes = tuple(range(-int(ndims/2),0))
            #totals = np.sum(reco_kernel, axis=sum_over_axes)
            #assert np.all(totals <= 1+1e-14), 'max = ' + str(np.max(totals)-1)

            # Now populate this transform to each input for which it applies.

            # NOTES:
            # * Output name is same as input name
            # * Use `self.input_binning` and `self.output_binning` so maps are
            #   returned in user-defined units (rather than computational
            #   units, which are attached to the non-`self` versions of these
            #   binnings).
            for input_name in self.input_names:
                if input_name not in flav_int_group:
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

    def _compute_transforms(self):
        """There are no systematics in this stage, so the transforms are just
        the nominal transforms. Thus, this function just returns the nominal
        transforms, computed by `_compute_nominal_transforms`..

        """
        return self.nominal_transforms
