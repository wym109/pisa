from pisa.core.stage import NoInputStage
from pisa.core.map import Map, MapSet
import numpy as np
import pint
units = pint.UnitRegistry()

class Flux(NoInputStage):
    """
    This is a Flux Service just for testing purposes, generating a random map m1 and a map containing ones as m2
    a parameter test is required
    """
    def __init__(self, params, example_file, output_binning, 
                service, oversample_e=1, oversample_cz=1):
        # list expected parameters for this stage implementation
        expected_params = ['atm_delta_index', 'energy_scale', 'nu_nubar_ratio',
                            'nue_numu_ratio', 'test']
        # call parent constructor
        super(Flux, self).__init__(stage_name='flux', service_name=service,
                params=params, expected_params=expected_params)
        # asign other attributes
        self.filename = example_file
        self.output_binning = output_binning
        self.oversample_e = oversample_e
        self.oversample_cz = oversample_cz

    def _derive_output(self):
        outputs = ['nue', 'numu', 'nuebar', 'numubar']
        # create two histograms with the output shape
        height = self.params['test'].value.to('meter').magnitude
        output_maps = []
        for output in outputs:
            #hist = np.random.randint(height, size=self.output_binning.shape)
            hist = np.ones(self.output_binning.shape) * height
            # pack them into Map object, assign poisson errors
            m = Map(output, hist, self.output_binning)
            m.set_poisson_errors()
            output_maps.append(m)
        mapset = MapSet(maps=output_maps, name = 'flux maps')
        return mapset

    def validate_params(self, params):
        # do some checks on the parameters
        assert params['test'].value.dimensionality == units.meter.dimensionality
        assert params['test'].value.magnitude >= 0
