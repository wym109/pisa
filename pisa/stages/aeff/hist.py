# authors: J.Lanfranchi/P.Eller
# date:   March 20, 2016


import copy

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
    """Example stage with maps as inputs and outputs, and no disk cache. E.g.,
    histogrammed oscillations stages will work like this.

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
                     %(self.params.aeff_weight_file.value))
        events = Events(self.params.aeff_weight_file.value)

        # Select only the units in the input/output binning for conversion
        # (can't pass more than what's actually there)
        in_units = {dim: unit for dim, unit in comp_units.items()
                    if dim in self.input_binning}
        out_units = {dim: unit for dim, unit in comp_units.items()
                     if dim in self.output_binning}

        # These will be in the computational units
        input_binning = self.input_binning.to(**in_units)
        output_binning = self.output_binning.to(**out_units)

        # Account for "missing" dimension(s) (dimensions OneWeight expects for
        # computation of bin volume), and accommodate with a factor equal to
        # the full range. See IceCube wiki/documentation for OneWeight for
        # more info.
        missing_dims_vol = 1
        if 'azimuth' not in input_binning:
            missing_dims_vol *= 2*np.pi
        if 'coszen' not in input_binning:
            missing_dims_vol *= 2

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

        nominal_transforms = []
        for flav in self.input_names:
            for interaction in ['cc', 'nc']:
                # Flavor+interaction type naming convention used in the PISA
                # HDF5 files
                flav_int = NuFlavInt(flav, interaction)

                logging.debug("Working on %s effective areas" %flav_int)

                # "MC-True" field naming convention in PISA HDF5
                var_names = ['true_%s' %bin_name
                             for bin_name in output_binning.names]

                # Extract the columns' data into a list for histogramming
                sample = [events[flav_int][vn] for vn in var_names]

                aeff_hist, _ = np.histogramdd(
                    sample=sample,
                    weights=events[flav_int]['weighted_aeff'],
                    bins=all_bin_edges
                )

                # Divide histogram by
                #   (energy bin width x coszen bin width x azimuth bin width)
                # volumes to convert from sums-of-OneWeights-in-bins to
                # effective areas. Note that volume correction factor for
                # missing dimensions is applied here.
                bin_volumes = output_binning.bin_volumes(attach_units=False)
                aeff_hist /= (bin_volumes * missing_dims_vol)

                # Construct the BinnedTensorTransform
                xform = BinnedTensorTransform(
                    input_names=flav,
                    output_name=str(flav_int),
                    input_binning=input_binning,
                    output_binning=self.output_binning,
                    xform_array=aeff_hist,
                )
                nominal_transforms.append(xform)

        return TransformSet(transforms=nominal_transforms)

    def _compute_transforms(self):
        """Compute new oscillation transforms"""
        # Read parameters in in the units used for computation
        aeff_scale = self.params.aeff_scale.value.m_as('dimensionless')
        livetime_s = self.params.livetime.value.m_as('sec')
        logging.trace('livetime = %s --> %s sec'
                      %(self.params.livetime.value, livetime_s))

        # TODO: make the following syntax work by implementing the guts in
        #       TransformSet/Transform objects, as is done for MapSet/Map
        #       objects:
        #return self.nominal_transforms * (aeff_scale * livetime_s)

        new_transforms = []
        for xform in self.nominal_transforms:
            new_xform = BinnedTensorTransform(
                input_names=xform.input_names,
                output_name=xform.output_name,
                input_binning=xform.input_binning,
                output_binning=xform.output_binning,
                xform_array=xform.xform_array * (aeff_scale * livetime_s)
            )
            new_transforms.append(new_xform)

        return TransformSet(new_transforms)
