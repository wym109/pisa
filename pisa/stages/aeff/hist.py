# authors: J.Lanfranchi/P.Eller
# date:   March 20, 2016


import copy

import numpy as np

from pisa.core.stage import Stage
from pisa.core.transform import BinnedTensorTransform, TransformSet
from pisa.utils.events import Events
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging, set_verbosity


# TODO: the below logic does not generalize to muons, but probably should
# (rather than requiring an almost-identical version just for muons). For
# example, an input arg can dictate neutrino or muon, which then sets the
# input_names and output_names.

class hist(Stage):
    """Example stage with maps as inputs and outputs, and no disk cache. E.g.,
    histogrammed oscillations stages will work like this.

    """
    def __init__(self, params, input_binning, output_binning, disk_cache=None,
                 transforms_cache_depth=20, outputs_cache_depth=20):
        # All of the following params (and no more) must be passed via the
        # `params` argument.
        expected_params = (
            'aeff_weight_file', 'livetime', 'aeff_scale'
        )

        # Define the names of objects that are required by this stage (objects
        # will have the attribute "name": i.e., obj.name)
        input_names = (
            'nue', 'numu', 'nutau', 'nuebar', 'numubar', 'nutaubar'
        )

        # Define the names of objects that get produced by this stage
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
            stage_name='aeff',
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
        logging.info('Extracting events from file: %s'
                     %(self.params.aeff_weight_file.value))
        events = Events(self.params.aeff_weight_file.value)

        # TODO: convert energy, coszen, and/or azimuth bin edges (if present)
        # to the expected units (GeV, None/dimensionless, and rad,
        # respectively) so that bin area computation is correct for converting
        # sum-of-OneWeights-in-bin to average effective area across bin.

        # TODO: More flexible handling of E, CZ, and/or azimuth (+ other
        # dimensions that don't enter directly into OneWeight normalization):
        # Start with defaults for each (energy, coszen, and azimuth default
        # "widths" are the full simulated ranges for each, given the events
        # file and MC sim run info for each); then, loop through the binning.
        # If it is found that binning is done in one of these three, then the
        # bin sizes are modified from the full range to the new widths.
        # Finally, allow binning to be done in variables *other* than these
        # (which does not change a bin width for computing aeff from OneWeight,
        # but does add some complexity for handling).

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

        nominal_transforms = []
        for flav in self.input_names:
            for interaction in ['cc', 'nc']:
                xform_input_names = [flav]
                flav_int = '%s_%s'%(flav, interaction)
                bin_names = self.output_binning.names
                var_names = ['true_%s'%bin_name for bin_name in bin_names]

                logging.debug("Working on %s effective areas" %flav_int)
                aeff_hist, _, _ = np.histogram2d(
                    events[flav_int][var_names[0]],
                    events[flav_int][var_names[1]],
                    weights=events[flav_int]['weighted_aeff'],
                    bins=(self.output_binning[bin_names[0]].bin_edges.m,
                          self.output_binning[bin_names[1]].bin_edges.m)
                )

                # Divide histogram by
                #   (energy bin width x coszen bin width x azimuth bin width)
                # to convert from sum-of-OneWeights-in-bin to effective area
                delta0 = self.output_binning[bin_names[0]].bin_sizes
                delta1 = self.output_binning[bin_names[1]].bin_sizes
                bin_areas = np.abs(delta0[:, None] * delta1 * 2. * np.pi)
                aeff_hist /= bin_areas

                dimensionality = list(self.input_binning.shape)

                # Construct the BinnedTensorTransform
                xform = BinnedTensorTransform(
                    input_names=xform_input_names,
                    output_name=flav_int,
                    input_binning=self.input_binning,
                    output_binning=self.output_binning,
                    xform_array=aeff_hist,
                )
                nominal_transforms.append(xform)

        return TransformSet(transforms=nominal_transforms)

    def _compute_transforms(self):
        """Compute new oscillation transforms"""
        # Read parameters in in the units used for computation
        aeff_scale = self.params.aeff_scale.value.magnitude
        livetime_s = self.params.livetime.value.to('sec').magnitude
        logging.trace('livetime = %s --> %s sec'
                      %(self.params.livetime.value, livetime_s))

        # TODO: make the following syntax work by implementing the guts in
        #       TransformSet/Transform objects, as is done for MapSet/Map
        #       objects:
        #return self.nominal_transforms * (aeff_scale * livetime_s)

        new_transforms = []
        for xform in self.nominal_transforms:
            new_xform = copy.deepcopy(xform)
            new_xform.xform_array *= aeff_scale * livetime_s
            new_transforms.append(new_xform)

        return TransformSet(new_transforms)
