from pisa.stage import NoInputStage
from pisa.utils.map import Map, MapSet
import numpy as np

class Flux(NoInputStage):
    """
    This is a Flux Service just for testing purposes, generating a random map m1 and a map containing ones as m2
    a parameter test is required
    """
    def __init__(self, params, example_file, output_binning):
        # call parent constructor
        super(Flux, self).__init__('flux', 'test', params)
        self.filename = example_file
        self.output_binning = output_binning

    def _derive_output(self):
        # create two histograms with the output shape
        hist1 = np.random.randint(self.params['test'].value, size=self.output_binning.shape)
        hist2 = np.ones(self.output_binning.shape)
        # pack them into Map object, assign poisson errors to the first one
        m1 = Map('m1', hist, self.output_binning)
        m1.set_poisson_errors()
        m2 = Map('m2', hist, self.output_binning)
        # create a mapset
        mapset = MapSet(maps=(m1,m2), name = 'mapset')
        return mapset

    def validate_params(self, params):
        # make sure we have a parameter called test, and that its value is greater or equal than zero
        assert('test' in params.names)
        assert(params['test'].value >= 0)

    @staticmethod
    def add_cmdline_args(parser):
        pass
