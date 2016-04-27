from pisa.stage import NoInputStage
from pisa.utils.map import Map, MapSet
import numpy as np
import pint
units = pint.UnitRegistry()

class Flux(NoInputStage):
    """
    This is a Flux Service just for testing purposes, generating a random map m1 and a map containing ones as m2
    a parameter test is required
    """
    def __init__(self, params, example_file, output_binning, service,
            oversample_e, oversample_cz):
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
        # create two histograms with the output shape
        height = self.params['test'].value.to('meter').magnitude
        hist1 = np.random.randint(height, size=self.output_binning.shape)
        hist2 = np.ones(self.output_binning.shape)
        # pack them into Map object, assign poisson errors to the first one and
        # a flat 5% error to the second one
        m1 = Map('m1', hist1, self.output_binning)
        m1.set_poisson_errors()
        m2 = Map('m2', hist2, self.output_binning)
        m2.set_errors(hist2/20.)
        # create a third map
        m3 = m1*m2
        # create a mapset
        mapset = MapSet(maps=(m1,m2,m3), name = 'mapset')
        return mapset

    def validate_params(self, params):
        # do some checks on the parameters
        assert params['test'].value.dimensionality == units.meter.dimensionality
        assert params['test'].value.magnitude >= 0
