# authors: J.Lanfranchi/P.Eller
# date:   March 20, 2016


import copy
from itertools import product

import numpy as np

from pisa.core.stage import Stage
from pisa.core.transform import BinnedTensorTransform, TransformSet
from pisa.utils.events import Events
from pisa.utils.flavInt import flavintGroupsFromString
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
    params : ParamSet
        Must exclusively have parameters:

        aeff_weight_file
        livetime
        aeff_scale

    particles : string
        Must be one of 'neutrinos' or 'muons' (though only neutrinos are
        supported at this time).

    transform_groups : string
        Specifies which particles/interaction types to combine together in
        computing the transforms. (See Notes.)

    combine_grouped_flavints : bool
        Whether to combine the event-rate maps for the flavint groupings
        specified by `transform_groups`.

    input_binning : MultiDimBinning or convertible thereto
        Input binning is in true variables, with names prefixed by "true_".
        Each must match a corresponding dimension in `output_binning`.

    output_binning : MultiDimBinning or convertible thereto
        Output binning is in reconstructed variables, with names (traditionally
        in PISA but not necessarily) prefixed by "reco_". Each must match a
        corresponding dimension in `input_binning`.

    debug_mode : None, bool, or string
        Whether to store extra debug info for this service.

    disk_cache
    transforms_cache_depth
    outputs_cache_depth

    Notes
    -----
    Example input names would be:
    See Conventions section in the documentation for more informaton on
    particle naming scheme in PISA. As an example

    """
    def __init__(self, params, particles, transform_groups,
                 combine_grouped_flavints, input_binning, output_binning,
                 error_method=None, disk_cache=None, transforms_cache_depth=20,
                 outputs_cache_depth=20, debug_mode=None):
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
            'aeff_weight_file', 'livetime', 'aeff_scale', 'nutau_cc_norm'
        )

        # Define the names of objects expected in inputs and produced as
        # outputs
        if self.particles == 'neutrinos':
            input_names = (
                'nue', 'numu', 'nutau', 'nuebar', 'numubar', 'nutaubar'
            )
            if combine_grouped_flavints:
                output_names = tuple([str(g) for g in self.transform_groups])

            else:
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
            error_method=error_method,
            disk_cache=disk_cache,
            outputs_cache_depth=outputs_cache_depth,
            transforms_cache_depth=transforms_cache_depth,
            input_binning=input_binning,
            output_binning=output_binning,
            debug_mode=debug_mode
        )

        # Can do these now that binning has been set up in call to Stage's init
        self.include_attrs_for_hashes('particles')
        self.include_attrs_for_hashes('transform_groups')

    def load_events(self):
        evts = self.params.aeff_weight_file.value
        this_hash = hash_obj(evts)
        if this_hash == self.events_hash:
            return
        logging.debug('Extracting events from Events obj or file: %s' %evts)
        self.events = Events(evts)
        self.events_hash = this_hash

    @profile
    def _compute_nominal_transforms(self):
        self.load_events()
        # Units must be the following for correctly converting a sum-of-
        # OneWeights-in-bin to an average effective area across the bin.
        comp_units = dict(true_energy='GeV', true_coszen=None,
                          true_azimuth='rad')

        # Only works if energy is in input_binning
        if 'true_energy' not in self.input_binning:
            raise ValueError('Input binning must contain "true_energy"'
                             ' dimension, but does not.')

        # coszen and azimuth are both optional, but no further dimensions are
        excess_dims = set(self.input_binning.names).difference(
            comp_units.keys()
        )
        if len(excess_dims) > 0:
            raise ValueError('Input binning has extra dimension(s): %s'
                             %sorted(excess_dims))

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
        if 'true_azimuth' not in input_binning:
            missing_dims_vol *= 2*np.pi
        if 'true_coszen' not in input_binning:
            missing_dims_vol *= 2

        # This gets used in innermost loop, so produce it just once here
        all_bin_edges = [edges.magnitude for edges in output_binning.bin_edges]

        nominal_transforms = []
        for flav_int_group in self.transform_groups:
            logging.debug("Working on %s effective areas" %flav_int_group)

            aeff_transform = self.events.histogram(
                kinds=flav_int_group,
                binning=all_bin_edges,
                binning_cols=self.input_binning.names,
                weights_col='weighted_aeff',
                errors=(self.error_method is not None)
            )

            # Divide histogram by
            #   (energy bin width x coszen bin width x azimuth bin width)
            # volumes to convert from sums-of-OneWeights-in-bins to
            # effective areas. Note that volume correction factor for
            # missing dimensions is applied here.
            bin_volumes = input_binning.bin_volumes(attach_units=False)
            aeff_transform /= (bin_volumes * missing_dims_vol)

            flav_names = [str(flav) for flav in flav_int_group.flavs()]
            for input_name in self.input_names:
                if input_name not in flav_names:
                    continue
                for output_name in self.output_names:
                    if output_name not in flav_int_group:
                        continue
                    xform = BinnedTensorTransform(
                        input_names=input_name,
                        output_name=output_name,
                        input_binning=self.input_binning,
                        output_binning=self.output_binning,
                        xform_array=aeff_transform,
                    )
                    nominal_transforms.append(xform)

        return TransformSet(transforms=nominal_transforms)

    @profile
    def _compute_transforms(self):
        """Compute new oscillation transforms"""
        # Read parameters in in the units used for computation
        aeff_scale = self.params.aeff_scale.value.m_as('dimensionless')
        livetime_s = self.params.livetime.value.m_as('sec')
        logging.trace('livetime = %s --> %s sec'
                      %(self.params.livetime.value, livetime_s))

        new_transforms = []
        for flav_int_group in self.transform_groups:
            repr_flav_int = flav_int_group[0]
            flav_names = [str(flav) for flav in flav_int_group.flavs()]
            aeff_transform = None
            for transform in self.nominal_transforms:
                if transform.input_names[0] in flav_names \
                        and transform.output_name in flav_int_group:
                    if aeff_transform is None:
                        aeff_transform = transform.xform_array * (aeff_scale *
                                                                  livetime_s)
                        if transform.output_name in ['nutau_cc', 'nutaubar_cc']:
                            aeff_transform = aeff_transform * self.params.nutau_cc_norm.value.m
                    new_xform = BinnedTensorTransform(
                        input_names=transform.input_names,
                        output_name=transform.output_name,
                        input_binning=transform.input_binning,
                        output_binning=transform.output_binning,
                        xform_array=aeff_transform
                    )
                    new_transforms.append(new_xform)

        return TransformSet(new_transforms)
