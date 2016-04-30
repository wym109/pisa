# authors: J.Lanfranchi/P.Eller
# date:   March 20, 2016

import numpy as np

from pisa.core.stage import InputStage
from pisa.core.transform import BinnedTensorTransform, TransformSet

class dummy(InputStage):
    """Example input stage, functionally close to osc"""

    def __init__(self, params, input_binning, output_binning, service,
                 disk_cache=None, transform_cache_depth=20,
                 result_cache_depth=20,
                 oversample_cz=1, oversample_e=1,
                 earth_model=None):
        # All of the following params (and no more) must be passed to the
        # `params` argument.
        expected_params = (
            'YeI', 'YeM', 'YeO', 'deltacp', 'deltam21', 'deltam31',
            'detector_depth', 'prop_height', 'test', 'theta12', 'theta13',
            'theta23'
        )
        super(self.__class__, self).__init__(
            stage_name='osc',
            service_name=service,
            expected_params=expected_params,
            params=params,
            disk_cache=disk_cache,
            transform_cache_depth=transform_cache_depth,
            result_cache_depth=result_cache_depth
        )
        self.input_binning = input_binning
        self.output_binning = output_binning
        self.oversample_e = oversample_e
        self.oversample_cz = oversample_cz
        self.earth_model = earth_model

    def _derive_transforms(self):
        """Compute new oscillation transforms"""
        print 'deriving transform'
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
                name='osc from %s to %s' %(', '.join(input_names), flav)
            )
            transforms.append(xform)

        # TODO: make TransformSet a mutable sequence (list-like), and so do
        # the append directly rather than create a list first and then pass
        # this to instantiation of a trnasform set
        return TransformSet(transforms=transforms, name='dummy osc')
