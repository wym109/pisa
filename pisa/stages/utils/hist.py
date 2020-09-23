"""
Stage to transform arrays with weights into actual `histograms`
that represent event counts
"""

from __future__ import absolute_import, print_function, division

import numpy as np

from pisa import FTYPE
from pisa.core.stage import Stage
from pisa.utils.profiler import profile
from pisa.utils import vectorizer


class hist(Stage):  # pylint: disable=invalid-name
    """stage to histogram events"""
    def __init__(
        self,
        data=None,
        params=None,
        input_names=None,
        output_names=None,
        debug_mode=None,
        error_method=None,
        calc_mode=None,
        apply_mode=None,
        ):

        raise NotImplementedError('Needs some care, broken in pisa4')

        expected_params = ()
        input_names = ()
        output_names = ()

        # init base class
        super().__init__(
            data=data,
            params=params,
            expected_params=expected_params,
            input_names=input_names,
            output_names=output_names,
            debug_mode=debug_mode,
            error_method=error_method,
            calc_mode=calc_mode,
            apply_mode=apply_mode,
        )

        assert self.output_mode == 'binned'

    def setup_function(self):
        # create the variables to be filled in `apply`
        if self.error_method in ['sumw2']:
            for container in self.data:
                container['weights_squared'] = np.empty((container.size), dtype=FTYPE)
            self.data.representation = self.apply_mode
            for container in self.data:
                container['errors'] = np.empty((container.size), dtype=FTYPE)

    @profile
    def apply(self):
        # this is special, we want the actual event weights in the histo
        # therefor we're overwritting the apply function
        # normally in a stage you would implement the `apply_function` method
        # and not the `apply` method!



        if self.input_mode == 'binned':
            self.data.representation = self.apply_mode
            for container in self.data:
                # calcualte errors
                if self.error_method in ['sumw2']:
                    vectorizer.pow(
                        vals=container['weights'],
                        pwr=2,
                        out=container['weights_squared'],
                    )
                    vectorizer.sqrt(
                        vals=container['weights_squared'], out=container['errors']
                    )

        elif self.input_mode == 'events':
            for container in self.data:
                # calcualte errors
                if self.error_method in ['sumw2']:
                    vectorizer.pow(
                        vals=container['weights'],
                        pwr=2,
                        out=container['weights_squared'],
                    )
                self.data.representation = self.apply_mode
                if self.error_method in ['sumw2']:
                    pass
