# authors: J.Lanfranchi/P.Eller
# date:   March 20, 2016

import numpy as np

from pisa.core.stage import Stage
from pisa.core.transform import BinnedTensorTransform, TransformSet
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging, set_verbosity

class dummy(Stage):
    """Example stage with maps as inputs and outputs, and no disk cache. E.g.,
    histogrammed oscillations stages will work like this.

    """
    def __init__(self, params, input_binning, output_binning, disk_cache=None,
                 transforms_cache_depth=20, results_cache_depth=20):
        # All of the following params (and no more) must be passed via the
        # `params` argument.
        expected_params = (
            'oversample_e', 'oversample_cz', 'earth_model',
            'YeI', 'YeM', 'YeO', 'deltacp', 'deltam21', 'deltam31',
            'detector_depth', 'prop_height', 'test', 'theta12', 'theta13',
            'theta23'
        )
        super(self.__class__, self).__init__(
            use_transforms=True,
            stage_name='osc',
            service_name='dummy',
            params=params,
            expected_params=expected_params,
            disk_cache=disk_cache,
            results_cache_depth=results_cache_depth,
            transforms_cache_depth=transforms_cache_depth,
            input_binning=input_binning,
            output_binning=output_binning
        )

    def _compute_transforms(self):
        """Compute new oscillation transforms"""
        seed = hash_obj(self.params.values, hash_to='int') % (2**32-1)
        np.random.seed(seed)

        transforms = []
        for flav in ['nue', 'numu', 'nutau', 'nuebar', 'numubar',
                     'nutaubar']:
            # Only particles oscillate to particles
            if 'bar' not in flav:
                input_names = ['nue', 'numu']
            # ... and antiparticles oscillate to antiparticles
            else:
                input_names = ['nuebar', 'numubar']

            # Dimensions are same as input binning but with added dim for
            # multiple inputs (concatenation of inputs is on last dimension --
            # see BinnedTensorTransform -- so this dimension goes last)
            dimensionality = list(self.input_binning.shape) + \
                    [len(input_names)]

            # Produce a random transform for demonstration only
            xform_array = np.random.rand(*dimensionality)

            # Construct the BinnedTensorTransform
            xform = BinnedTensorTransform(
                input_names=input_names,
                output_name=flav,
                input_binning=self.input_binning,
                output_binning=self.output_binning,
                xform_array=xform_array,
                #name='osc from %s to %s' %(', '.join(input_names), flav),
                params_hash=hash_obj((seed,)),
            )
            transforms.append(xform)

        # TODO: make TransformSet a mutable sequence (list-like), and so do
        # the append directly rather than create a list first and then pass
        # this to instantiation of a trnasform set
        return TransformSet(transforms=transforms)
