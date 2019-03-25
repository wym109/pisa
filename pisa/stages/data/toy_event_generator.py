"""
Stage to generate some random data
"""
from __future__ import absolute_import, print_function, division

import numpy as np

from pisa import FTYPE
from pisa.core.pi_stage import PiStage
from pisa.utils.log import logging
from pisa.utils import vectorizer
from pisa.core.container import Container


class toy_event_generator(PiStage):
    """
    random toy event generator PISA Pi class

    Paramaters
    ----------

    n_events : int
        Number of events to be generated per output name

    seed : int
        Seed to be used for random

    Notes
    -----

    """
    def __init__(self,
                 data=None,
                 params=None,
                 input_names=None,
                 output_names=None,
                 debug_mode=None,
                 input_specs=None,
                 calc_specs=None,
                 output_specs=None,
                ):

        expected_params = ('n_events',
                           'random',
                           'seed',
                           )
        # init base class
        super(toy_event_generator, self).__init__(data=data,
                                                  params=params,
                                                  expected_params=expected_params,
                                                  input_names=input_names,
                                                  output_names=output_names,
                                                  debug_mode=debug_mode,
                                                  input_specs=input_specs,
                                                  calc_specs=calc_specs,
                                                  output_specs=output_specs,
                                                 )

        # doesn't calculate anything
        assert self.calc_mode is None

    def setup_function(self):

        n_events = int(self.params.n_events.value.m)
        seed = int(self.params.seed.value.m)
        self.random_state = np.random.RandomState(seed)

        for name in self.output_names:

            container = Container(name)
            container.data_specs = self.input_specs
            nubar = -1 if 'bar' in name else 1
            if 'e' in name:
                flav = 0
            if 'mu' in name:
                flav = 1
            if 'tau' in name:
                flav = 2

            if self.input_mode == 'events':

                # generate events
                true_energy = np.power(10, self.random_state.rand(n_events).astype(FTYPE) * 3)
                true_coszen = self.random_state.rand(n_events).astype(FTYPE) * 2 - 1
                size = n_events
                container.add_array_data( 'true_energy', true_energy )
                container.add_array_data( 'true_coszen', true_coszen )

            elif self.input_mode == 'binned':

                # create variables using the grid
                size = self.input_specs.size
                mesh = self.input_specs.meshgrid(entity="midpoints",attach_units=False) #TODO How to enforce correct units? #TODO Use edges?
                for var_name,var_vals in zip(self.input_specs.names,mesh) :
                    container.add_array_data( var_name, var_vals.flatten().astype(FTYPE) )

            # choose initial weights
            if self.params.random.value:
                initial_weights = self.random_state.rand(size).astype(FTYPE)
            else:
                initial_weights = np.ones(size, dtype=FTYPE)
            weights = np.ones(size, dtype=FTYPE)

            # make container
            container.add_scalar_data('nubar', nubar)
            container.add_scalar_data('flav', flav)
            container.add_array_data('initial_weights',initial_weights.astype(FTYPE))
            container.add_array_data('weights',weights.astype(FTYPE))
            container.add_array_data('weighted_aeff',weights.astype(FTYPE))

            self.data.add_container(container)


    def apply_function(self):
        # reset weights
        for container in self.data:
            vectorizer.scale(1.,
                             container['initial_weights'],
                             out=container['weights'])
