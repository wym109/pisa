"""
Stage class designed to be inherited by PISA Pi services, such that all basic
functionality is built-in.
"""


from __future__ import absolute_import, division

from collections import OrderedDict

from pisa.core.base_stage import BaseStage
from pisa.core.binning import MultiDimBinning
from pisa.core.container import ContainerSet
from pisa.utils.log import logging
from pisa.utils.format import arg_to_tuple
from pisa.utils.profiler import profile


__all__ = ["Stage"]
__author__ = "Philipp Eller (pde3@psu.edu)"


class Stage(BaseStage):
    """
    PISA stage base class. Should be used to implement PISA Pi stages

    Specialization should be done via subclasses.

    Parameters
    ----------
    data : ContainerSet or None
        object to be passed along

    params

    expected_params

    input_names : str, iterable thereof, or None

    output_names : str, iterable thereof, or None

    debug_mode : None, bool, or str
        If ``bool(debug_mode)`` is False, run normally. Otherwise, run in debug
        mode. See `pisa.core.base_stage.BaseStage` for more information

    error_method : None, bool, or str
        If ``bool(error_method)`` is False, run without computing errors.
        Otherwise, specifies a particular method for applying arrors.

    calc_mode : pisa.core.binning.MultiDimBinning, str, or None
        Specify in what to do the calculation

    apply_mode : pisa.core.binning.MultiDimBinning, str, or None
        Specify in what to do the application

    """

    def __init__(
        self,
        data=None,
        params=None,
        expected_params=None,
        input_names=None,
        output_names=None,
        debug_mode=None,
        error_method=None,
        calc_mode=None,
        apply_mode=None,
    ):
        super().__init__(
            params=params,
            expected_params=expected_params,
            input_names=input_names,
            output_names=output_names,
            debug_mode=debug_mode,
            error_method=error_method,
        )

        self.calc_mode = calc_mode
        self.apply_mode = apply_mode
        self.data = data

        self.param_hash = None
        # cake compatibility
        self.outputs = None

    @property
    def is_map(self):
        return self.data.is_map

    def setup(self):

        # check that data is a ContainerSet (downstream modules assume this)
        if self.data is not None:
            if not isinstance(self.data, ContainerSet):
                raise TypeError("`data` must be a `pisa.core.container.ContainerSet`")

        if self.calc_mode is not None:
            self.data.representation = self.calc_mode

        # call the user-defined setup function
        self.setup_function()

        # invalidate param hash:
        self.param_hash = -1

    def setup_function(self):
        """Implement in services (subclasses of Stage)"""
        pass

    @profile
    def compute(self):
        
        if len(self.params) == 0 and len(self.output_calc_keys) == 0:
            return

        # simplest caching algorithm: don't compute if params didn't change
        new_param_hash = self.params.values_hash
        if new_param_hash == self.param_hash:
            logging.trace("cached output")
            return

        if self.calc_mode is not None:
            self.data.representation = self.calc_mode

        self.compute_function()
        self.param_hash = new_param_hash

    def compute_function(self):
        """Implement in services (subclasses of Stage)"""
        pass

    @profile
    def apply(self):

        if self.apply_mode is not None:
            self.data.representation = self.apply_mode
        self.apply_function()


    def apply_function(self):
        """Implement in services (subclasses of Stage)"""
        pass

    def run(self, inputs=None):
        if not inputs is None:
            raise ValueError("PISA pi requires there not be any inputs.")
        self.compute()
        self.apply()
        return None

