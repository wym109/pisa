"""
Stage to take the initial errors of MC and keep them
for all minimization.

Needed to allow mod_chi2 to behave correctly
"""

from __future__ import absolute_import, print_function, division

import numpy as np

from pisa import FTYPE
from pisa.core.stage import Stage
from pisa.utils.profiler import profile
from pisa.utils import vectorizer
from pisa.utils.numba_tools import WHERE

class fix_error(Stage):  # pylint: disable=invalid-name
    """
    stage to fix the error returned by template_maker.
    """
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

        expected_params = ()
        input_names = ()
        output_names = ()

        # what are the keys used from the inputs during apply
        
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

        assert self.input_mode == "binned"
        assert self.calc_mode == "binned"
        assert self.output_mode == 'binned'

    def setup_function(self):
        for container in self.data:
            container['frozen_errors'] = np.empty((container.size), dtype=FTYPE)

    def apply_function(self):
        for container in self.data:
            vectorizer.assign(vals=container['frozen_errors'], out=container['errors'])
            container.mark_changed('errors')
 
    def compute_function(self):
        for container in self.data:
            vectorizer.assign(vals=container["errors"], out=container["frozen_errors"])
            container.mark_changed('frozen_errors')
