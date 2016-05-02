from pisa.core.stage import Stage

class service_name(Stage):
    """
    Docstring here

    Parameters
    ----------
    params
    input_binning
    output_binning
    disk_cache
    transforms_cache_depth
    outputs_cache_depth

    """
    def __init__(self, params, input_binning, output_binning, disk_cache,
                 transforms_cache_depth=20, outputs_cache_depth=20):
        expected_params = (
            'oversample_e', 'oversample_cz', 'earth_model',
            'YeI', 'YeM', 'YeO', 'deltacp', 'deltam21', 'deltam31',
            'detector_depth', 'prop_height', 'test', 'theta12', 'theta13',
            'theta23'
        )
        super(self.__class__, self).__init__(
            use_transforms=True,
            stage_name='...',
            service_name='service_name',
            params=params,
            expected_params=expected_params,
            disk_cache=disk_cache,
            outputs_cache_depth=outputs_cache_depth,
            transforms_cache_depth=transforms_cache_depth,
            input_binning=input_binning,
            output_binning=output_binning
        )

    def _compute_transforms(self):
        """Docstring here"""
        pass

    def validate_params(self, params):
        """Docstring here"""
        pass
