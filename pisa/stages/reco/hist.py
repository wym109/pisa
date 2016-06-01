
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

import numpy as np

from pisa.core.stage import Stage
from pisa.core.transform import BinnedTensorTransform, TransformSet
from pisa.utils.events import Events
from pisa.utils.flavInt import NuFlavInt, ALL_NUFLAVINTS
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging, set_verbosity
from pisa.utils.profiler import profile

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
        The only param supported at this time is `reco_weight_file`, which is a
        PISA events file or resource path specifying one.

    transform_groups : string
        Specifies which particles/interaction types to combine together in
        computing the transforms. See Notes section for more details on how
        to specify this string

    input_names : string or list of strings
        Names of inputs expected. These should follow the standard PISA naming
        conventions for flavor/interaction types OR groupings thereof. Note
        that this service's outputs are named the same as its inputs.

    input_binning : MultiDimBinning or convertible thereto
        Input binning is in true variables, with names prefixed by "true_".
        Each must match a corresponding dimension in `output_binning`.

    output_binning : MultiDimBinning or convertible thereto
        Output binning is in reconstructed variables, with names prefixed by
        "reco_". Each must match a corresponding dimension in `input_binning`.

    particles : string
        Must be one of 'neutrinos' or 'muons' (though only neutrinos are
        supported at this time).

    disk_cache

    transforms_cache_depth

    outputs_cache_depth

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
    def __init__(self, params, transform_groups, input_names,
                 input_binning, output_binning,
                 particles='neutrinos', disk_cache=None,
                 transforms_cache_depth=20, outputs_cache_depth=20):
        assert particles in ['neutrinos', 'muons']

        self.transform_groups = [NuFlavIntGroup(g) for g in transform_groups]

        # All of the following params (and no more) must be passed via the
        # `params` argument.
        expected_params = (
            'reco_weight_file', # NOT IMPLEMENTED! 'e_reco_scale', 'cz_reco_scale'
        )

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
            disk_cache=disk_cache,
            outputs_cache_depth=outputs_cache_depth,
            transforms_cache_depth=transforms_cache_depth,
            input_binning=input_binning,
            output_binning=output_binning
        )

    @profile
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
        # Computational units must be the following for compatibility with
        # events file
        comp_units = dict(
            true_energy='GeV', true_coszen=None, true_azimuth='rad',
            reco_energy='GeV', reco_coszen=None, reco_azimuth='rad'
        )

        # Only works if (true/reco)_energy is in (input/output)_binning
        if 'true_energy' not in self.input_binning:
            raise ValueError('Input binning must contain "true_energy"'
                             ' dimension, but does not.')
        if 'reco_energy' not in self.output_binning:
            print self.output_binning
            raise ValueError('Output binning must contain "reco_energy"'
                             ' dimension, but does not.')

        # Any dimension in input (true_*) must have its reconstructed version
        # in the output (reco_*).
        for dim in self.input_binning.dimensions:
            in_dim_name = dim.name
            out_dim_name = in_dim_name.replace('true_', 'reco_')
            if out_dim_name not in self.output_binning:
                raise ValueError(
                    'Input dimension name "%s" requires corresponding'
                    ' dimension "%s" be in output; however, output only'
                    ' contains dimensions %s.'
                    %(in_dim_name, out_dim_name,
                      ', '.join(["%s"%d for d in self.output_binning.names]))
                )

        logging.info('Extracting events from file: %s'
                     %(self.params.reco_weight_file.value))
        events = Events(self.params.reco_weight_file.value)

        # Select only the units in the input/output binning for conversion
        # (can't pass more than what's actually there)
        in_units = {dim: unit for dim, unit in comp_units.items()
                    if dim in self.input_binning}
        out_units = {dim: unit for dim, unit in comp_units.items()
                     if dim in self.output_binning}

        # These will be in the computational units
        input_binning = self.input_binning.to(**in_units)
        output_binning = self.output_binning.to(**out_units)

        # TODO: take events object as an input instead of as a param that
        # specifies a file? Or handle both cases?

        # TODO: include here the logic from the make_events_file.py script so
        # we can go directly from a (reasonably populated) icetray-converted
        # HDF5 file (or files) to a nominal transform, rather than having to
        # rely on the intermediate step of converting that HDF5 file (or files)
        # to a PISA HDF5 file that has additional column(s) in it to account
        # for the combinations of flavors, interaction types, and/or simulation
        # runs. Parameters can include which groupings to use to formulate an
        # output.

        # This gets used in innermost loop, so produce it just once here...
        # However, we actually need 2N dimensional binning here
        # i.e. once for truth and once for reco

        # First N dimensions of the transform are the input dimensions; last N
        # dimensions are the output dimensions
        transform_binning = input_binning.dimensions + output_binning.dimensions
        all_bin_edges = transform_binning.bin_edges.magnitude
        input_bin_edges = input_binning.bin_edges.magnitude

        nominal_transforms = []
        for flav_int in ALL_NUFLAVINTS:
            # Extract the columns' data into a list for histogramming

            # Full sample for computing transform
            full_sample = [events[flav_int][name] for name in transform_binning.names]

            # Truth-only data will be used for normalization (so transform is
            # in terms of fraction-of-events in input bin).
            true_sample = [events[flav_int][name] for name in input_binning.names]

            reco_kernel, _ = np.histogramdd(
                sample=full_sample,
                bins=all_bin_edges
            )

            # This takes into account the correct kernel normalization:
            # What this means is that we have to normalise the reco map
            # to the number of events in the truth bin.
            # i.e. we have N events from the truth bin which then become
            # spread out over the whole map due to reconstruction.
            # The normalisation is dividing this map by N
            # Previously this was hard-coded for 2 dimensions
            # I have tried to generalise it to arbitrary dimensionality

            # First, make a histogram of the truth data
            truth_map, _ = np.histogramdd(
                sample=true_sample,
                bins=all_bin_edges
            )

            # The bins of this should be the set of N described in the
            # above comment. So I should be able to divide these
            # two entities. However, after much trial and error I found
            # I need to do this to get the right answer:
            while truth_map.ndim != reco_kernel.ndim:
                truth_map = np.expand_dims(truth_map,axis=-1)

            # My understanding is that what this essentially does
            # is it puts everything in the right place so that numpy
            # is dividing what I want by what I want when I call
            norm_reco_kernel = np.nan_to_num(reco_kernel / truth_map)

            # NOTE - Here I have used np.nan_to_num to avoid cases where
            # there were no events in truth bin i and so this division
            # returns NaN. I assume that in such cases we should have a
            # set of zeroes in the reco_kernel anyway, since the
            # contribution to the map from truth bin i must be zero if the
            # content of truth bin i was also zero. This logic makes sense
            # to me but please let me know if you disagree!

            # Construct the BinnedTensorTransform
            xform = BinnedTensorTransform(
                input_names=str(flav_int),
                output_name=str(flav_int),
                input_binning=input_binning,
                output_binning=self.output_binning,
                xform_array=norm_reco_kernel,
            )
            nominal_transforms.append(xform)

        return TransformSet(transforms=nominal_transforms)

    def _compute_transforms(self):
        """There are no systematics in this stage, so the transforms are just
        the nominal transforms. Thus, this function just returns the nominal
        transforms, computed by `_compute_nominal_transforms`..

        """
        return self.nominal_transforms
