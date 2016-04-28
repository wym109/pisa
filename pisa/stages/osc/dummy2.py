from pisa.stage import InputStage
from pisa.utils import utils
from pisa.utils import cache
from pisa.utils.utils import hash_obj
from pisa.utils.transform import LinearTransform, TransformSet
import numpy as np

class Osc(InputStage):
    def __init__(self,
                 params,
                 input_binning,
                 output_binning,
                 service,
                 disk_cache=None,
                 transform_cache_depth=20,
                 result_cache_depth=20,
                 oversample_cz =1,
                 oversample_e=1,
                 earth_model=None):
        expected_params = ['YeI', 'YeM', 'YeO', 'deltacp', 'deltam21',
        'deltam31', 'detector_depth', 'prop_height', 'theta12', 'theta13',
        'theta23', 'test']
        super(Osc, self).__init__(
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

    def _derive_transform(self):
        """Compute a new oscillation transform"""
        print 'deriving transforms...'
        #utils.n_bad_seeds(self.params.values_hash % 2**32)
        inputs = ['nue', 'numu', 'nuebar', 'numubar']
        outputs = None
        transforms = []
        for output in ['nue', 'numu', 'nutau', 'nuebar', 'numubar', 'nutaubar']:
            print output
            xform_array = np.random.rand(*self.input_binning.shape)
            transforms.append(LinearTransform(
                inputs=inputs, outputs=outputs, input_binning=self.input_binning,
                output_binning=self.output_binning, xform_array=xform_array,
                name='osc to %s'%output
            ))
        print 'transforms derived'
        return TransformSet(transforms=transforms, name='osc')

    def get_output_map_set(self, input_map_set):
        return input_map_set * self.params['theta13'].value.to('deg').magnitude
