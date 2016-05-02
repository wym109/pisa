from pisa.core.stage import Stage

class service_name(Stage):
    """
    Docstring here
    """
    def __init__(self, params, output_binning, disk_cache=None,
                 memcaching_enabled=True, propagate_errors=True,
                 outputs_cache_depth=20):
        # list expected parameters for this stage implementation
        expected_params = (
            'atm_delta_index', 'energy_scale', 'nu_nubar_ratio',
            'nue_numu_ratio', 'test', 'example_file', 'oversample_e',
            'oversample_cz'
        )
        # call parent constructor
        super(self.__class__, self).__init__(
            use_transforms=False,
            stage_name='flux',
            service_name='service_name',
            params=params,
            expected_params=expected_params,
            disk_cache=disk_cache,
            memcaching_enabled=memcaching_enabled,
            propagate_errors=propagate_errors,
            outputs_cache_depth=outputs_cache_depth,
            output_binning=output_binning
        )
    def _derive_output(self, **kwargs):
        pass

    def validate_params(self, params):
        pass

    @staticmethod
    def add_cmdline_args(parser):
        pass
