
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
from pisa.utils.flavInt import NuFlavInt
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

    From these histograms, and the true event rate maps, calculates
    the reconstructed even rate templates.

    Parameters
    ----------
    params
    input_binning
    output_binning
    particles : string
        Must be one of 'neutrinos' or 'muons'
    disk_cache
    transforms_cache_depth
    outputs_cache_depth

    """
    def __init__(self, params, input_binning, output_binning,
                 particles='neutrinos', disk_cache=None,
                 transforms_cache_depth=20, outputs_cache_depth=20):
        assert particles in ['neutrinos', 'muons']

        # All of the following params (and no more) must be passed via the
        # `params` argument.
        expected_params = (
            'reco_weight_file',
            'e_reco_scale', 'cz_reco_scale'
        )

        # Define the names of objects that are required by this stage (objects
        # will have the attribute "name": i.e., obj.name)
        # This expects events separated by both flavour and interaction
        # Matter and antimatter are NOT joined yet
        # Also, all NC events are still separate
        input_names = (
            'nue_cc', 'numu_cc', 'nutau_cc', 'nuebar_cc', 'numubar_cc',
            'nutaubar_cc',
            'nue_nc', 'numu_nc', 'nutau_nc', 'nuebar_nc', 'numubar_nc',
            'nutaubar_nc'
        )

        # Define the names of objects that get produced by this stage
        # The output combines nu and nubar together (just called nu)
        # All of the NC events are joined (they look the same in the detector).
        output_names = (
            'nue_cc', 'numu_cc', 'nutau_cc', 'nuebar_cc', 'numubar_cc',
            'nutaubar_cc',
            'nue_nc', 'numu_nc', 'nutau_nc', 'nuebar_nc', 'numubar_nc',
            'nutaubar_nc'
        )

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

    def _compute_nominal_transforms(self):
        """
        Reads in the PISA HDF5 events file and constructs a reconstruction kernel by histogramming this directly.
        The result is a 2N dimensional histogram, where N is the dimensionality of the input binning, which maps the truth bin counts to the reconstructed bin counts.
        i.e. For the case of 1D input binning, the ith element of the reconstruction kernel will be a map showing the distribution of events over all the reco space from truth bin i. 
        This will be normalised to the total number of events in truth bin i.

        NOTE - In the current implementation these histograms are made ***UN***weighted.
        This is probably quite wrong...
        """
        # Units must be the following for correctly converting a sum-of-
        # OneWeights-in-bin to an average effective area across the bin.
        comp_units = dict(energy='GeV', coszen=None, azimuth='rad')

        # Only works if energy is in input_binning
        if 'energy' not in self.input_binning:
            raise ValueError('Input binning must contain "energy" dimension,'
                             ' but does not.')

        # coszen and azimuth are both optional, but no further dimensions are
        excess_dims = set(self.input_binning.names).difference(comp_units.keys())
        if len(excess_dims) > 0:
            raise ValueError('Input binning has extra dimension(s): %s'
                             %sorted(excess_dims))

        # TODO: not handling rebinning in this stage or within Transform
        # objects; implement this! (and then this assert statement can go away)
        assert self.input_binning == self.output_binning

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

        # This gets used in innermost loop, so produce it just once here
        all_bin_edges = [edges.magnitude for edges in output_binning.bin_edges]
        # However, we actually need 2N dimensional binning here
        # i.e. once for truth and once for reco
        # Create the list once more (we need the single list too!)
        double_bin_edges = [edges.magnitude for edges in output_binning.bin_edges]
        # Then re-add it again
        # Has to be done explicitly again
        for edges in output_binning.bin_edges:
            double_bin_edges.append(edges.magnitude)

        nominal_transforms = []
        for flav in ['nue','nue_bar','numu','numu_bar','nutau','nutau_bar']:
            logging.debug("Working on %s kernels"%flav)
            for interaction in ['cc', 'nc']:
                # Flavor+interaction type naming convention used in the PISA
                # HDF5 files
                flav_int = NuFlavInt(flav, interaction)

                # "MC-True" field naming convention in PISA HDF5
                true_var_names = ['true_%s' %bin_name
                                  for bin_name in output_binning.names]
                # "MC-Reco" field naming convention in PISA HDF5
                reco_var_names = ['reco_%s' %bin_name
                                  for bin_name in output_binning.names]

                # Extract the columns' data into a list for histogramming
                # First take the truth data
                true_sample = [events[flav_int][tvn] for tvn in true_var_names]
                # Now make it once more since we need this true sample separately
                full_sample = [events[flav_int][tvn] for tvn in true_var_names]
                # Then append the reco data
                for rvn in reco_var_names:
                    full_sample.append(events[flav_int][rvn])

                reco_kernel, _ = np.histogramdd(
                    sample=full_sample,
                    bins=double_bin_edges
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
        """
        There are no systematics in this stage, so the transforms are just the nominal transforms. 
        Thus, this function just calls the previous function.
        """

        return self.nominal_transforms
