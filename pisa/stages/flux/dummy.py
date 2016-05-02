
import numpy as np
import pint
ureg = pint.UnitRegistry()

from pisa.core.stage import Stage
from pisa.core.map import Map, MapSet


class dummy(Stage):
    """
    This is a Flux Service just for testing purposes, generating a random map
    m1 and a map containing ones as m2; the parameter `test` is required.
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
            service_name='dummy',
            params=params,
            expected_params=expected_params,
            disk_cache=disk_cache,
            memcaching_enabled=memcaching_enabled,
            propagate_errors=propagate_errors,
            outputs_cache_depth=outputs_cache_depth,
            output_binning=output_binning
        )

    def _compute_outputs(self, inputs=None):
        outputs = ['nue', 'numu', 'nuebar', 'numubar']
        # create two histograms with the output shape
        height = self.params['test'].value.to('meter').magnitude
        output_maps = []
        for output in outputs:
            #hist = np.random.randint(height, size=self.output_binning.shape)
            hist = np.ones(self.output_binning.shape) * height
            # pack them into Map object, assign poisson errors
            m = Map(name=output, hist=hist, binning=self.output_binning)
            #m.set_poisson_errors()
            output_maps.append(m)
        mapset = MapSet(maps=output_maps, name='flux maps')
        return mapset

    def validate_params(self, params):
        # do some checks on the parameters
        assert (params['test'].value.dimensionality ==
                ureg.meter.dimensionality)
        assert params['test'].value.magnitude >= 0
