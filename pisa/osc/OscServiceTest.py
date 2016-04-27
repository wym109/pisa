
import collections
import inspect

from pisa.utils import utils
from pisa.utils import cache
from pisa.utils.utils import hash_obj


class OscServiceTest(InputStage):
    def __init__(self, params, disk_cache=None,
                 transform_cache_depth=20, result_cache_depth=20):
        super(OscServiceTest, self).__init__(
            params=params,
            disk_cache=disk_cache,
            transform_cache_depth=transform_cache_depth,
            result_cache_depth=result_cache_depth
        )

    def _derive_transform(self):
        """Compute a new oscillation transform"""
        utils.n_bad_seeds(self.params.values_hash % 2**32)
        inputs = ['nue', 'numu', 'nuebar', 'numubar']
        transforms = []
        for output in ['nue', 'numu', 'nutau', 'nuebar', 'numubar', 'nutaubar']:
            xform_array = np.random.rand(*self.input_binning.shape)
            transforms.append(LinearTransform(
                inputs=inputs, output=output, input_binning=self.binning,
                output_binning=self.binning, xform_array=xform_array,
                name='osc to ' + output
            ))
        return TransformSet(transforms=transforms, name='osc')
