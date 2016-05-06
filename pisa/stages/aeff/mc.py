# authors: J.Lanfranchi/P.Eller
# date:   March 20, 2016

import numpy as np
import copy

from pisa.core.stage import Stage
from pisa.core.transform import BinnedTensorTransform, TransformSet
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging, set_verbosity
from pisa.utils.events import Events


class mc(Stage):
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
            service_name='mc',
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

        # for now here, ToDo: replace by method invoked in baseclass + disk
        # caching
        self._compute_nominal_transforms()

    def _compute_nominal_transforms(self):
        logging.info('Extracting events from file: %s' %
                (self.params.aeff_weight_file.value))
        evts = Events(self.params.aeff_weight_file.value)
       
        # ToDO: assert that bin edges are in expected units (probably) GeV and
        # dimesnionless

        nominal_transforms = []
        for flav in self.input_names:
            for interaction in ['cc', 'nc']:
                xform_input_names = [flav]
                flav_int = '%s_%s'%(flav, interaction)
                bin_names = self.output_binning.names
                var_names = ['true_%s'%bin_name for bin_name in bin_names]
                logging.debug("Working on %s effective areas" %flav_int)
                aeff_hist, _, _ = np.histogram2d(
                    evts[flav_int][var_names[0]],
                    evts[flav_int][var_names[1]],
                    weights=evts[flav_int]['weighted_aeff'],
                    bins=(self.output_binning[bin_names[0]].bin_edges.m,
                        self.output_binning[bin_names[1]].bin_edges.m)
                )
                # Divide histogram by bin ExCZ "widths" to convert to aeff
                delta0 = self.output_binning[bin_names[0]].bin_sizes
                delta1 = self.output_binning[bin_names[1]].bin_sizes
                bin_areas = np.abs(delta0[:,  None] * delta1 * 2. * np.pi)
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

        self.nominal_transforms = TransformSet(transforms=nominal_transforms)
        print self.nominal_transforms

            
    def _compute_transforms(self):
        """Compute new oscillation transforms"""
        # Read parameters in in the units used for computation
        aeff_scale = self.params.aeff_scale.value.magnitude
        livetime_s = self.params.livetime.value.to('sec').magnitude
        logging.trace('livetime = %s --> %s sec'
                      %(self.params.livetime.value, livetime_s))

        new_transforms = []
        for xform in self.nominal_transforms.transforms:
            new_xform = copy.deepcopy(xform)
            new_xform.xform_array *= aeff_scale * livetime_s
            new_transforms.append(new_xform)
        return TransformSet(new_transforms)
