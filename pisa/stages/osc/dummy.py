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
                 transforms_cache_depth=20, outputs_cache_depth=20):
        # All of the following params (and no more) must be passed via the
        # `params` argument.
        expected_params = (
            'oversample_e', 'oversample_cz', 'earth_model',
            'YeI', 'YeM', 'YeO', 'deltacp', 'deltam21', 'deltam31',
            'detector_depth', 'prop_height', 'test', 'theta12', 'theta13',
            'theta23'
        )

        # Define the names of objects that are required by this stage (objects
        # will have the attribute "name": i.e., obj.name)
        input_names = (
            'nue', 'numu', 'nuebar', 'numubar'
        )

        # Define the names of objects that get produced by this stage
        output_names = (
            'nue', 'numu', 'nutau', 'nuebar', 'numubar', 'nutaubar'
        )

        # Invoke the init method from the parent class, which does a lot of
        # work for you.
        super(self.__class__, self).__init__(
            use_transforms=True,
            stage_name='osc',
            service_name='dummy',
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

        # There might be other things to do at init time than what Stage does,
        # but typically this is not much... and it's almost always a good idea
        # to have "real work" defined in another method besides init, which can
        # then get called from init (so that if anyone else wants to do the
        # same "real work" after object instantiation, (s)he can do so easily
        # by invoking that same method).

    def _compute_transforms(self):
        """Compute new oscillation transforms"""
        # This is done just to produce different set of transforms for
        # different set of parameters
        seed = hash_obj(self.params.values, hash_to='int') % (2**32-1)
        np.random.seed(seed)

        # Read parameters in in the units used for computation
        theta23 = self.params.theta23.value.to('rad').magnitude
        logging.trace('theta23 = %s --> %s rad'
                      %(self.params.theta23.value, theta23))

        transforms = []
        for flav in ['nue', 'numu', 'nutau', 'nuebar', 'numubar',
                     'nutaubar']:
            # Only particles oscillate to particles
            if 'bar' not in flav:
                xform_input_names = ['nue', 'numu']
            # and only antiparticles oscillate to antiparticles
            else:
                xform_input_names = ['nuebar', 'numubar']

            # Dimensions are same as input binning but with added dim for
            # multiple inputs (concatenation of inputs is on last dimension --
            # see BinnedTensorTransform -- so this dimension goes last)
            dimensionality = list(self.input_binning.shape) + \
                    [len(xform_input_names)]

            # Produce a random transform for demonstration only
            #xform_array = np.random.rand(*dimensionality)
            xform_array = np.ones(dimensionality)*1.1

            # Construct the BinnedTensorTransform
            xform = BinnedTensorTransform(
                input_names=xform_input_names,
                output_name=flav,
                input_binning=self.input_binning,
                output_binning=self.output_binning,
                xform_array=xform_array,
                params_hash=hash_obj((seed,)),
            )
            transforms.append(xform)

        # TODO: make TransformSet a mutable sequence (list-like), and so do
        # the append directly rather than create a list first and then pass
        # this to instantiation of a trnasform set
        return TransformSet(transforms=transforms)
