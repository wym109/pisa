
# PISA authors: Lukas Schulte
#               schulte@physik.uni-bonn.de
#               Justin L. Lanfranchi
#               jll1062+pisa@phys.psu.edu
#
# CAKE author: Shivesh Mandalia
#              s.p.mandalia@qmul.ac.uk
#
# date:    2016-05-13
"""
The purpose of this stage is to simulate the event classification of PINGU,
sorting the reconstructed nue CC, numu CC, nutau CC, and NC events into the
track and cascade channels.

This service in particular takes in events from a PISA HDF5 file.

For each particle "signature", a 2D-histogram in energy and coszen is created,
which gives the PID probabilities in each bin.  The input maps are transformed
according to these probabilities to provide an output containing a map for
track-like events ('trck') and shower-like events ('cscd') which is then
returned.

"""

import numpy as np
import pint
ureg = pint.UnitRegistry()

from pisa.core.stage import Stage
from pisa.utils.log import logging

class mc(Stage):
    """PID based on input PISA events HDF5 file.

    Transforms a input map of the specified particle "signature" (aka ID) into
    a map of the track-like events ('trck') and a map of the shower-like events
    ('cscd').

    Parameters
    ----------
    params : ParamSet
        Parameters which set everything besides the binning.

    input_binning : MultiDimBinning
        The `inputs` must be a MapSet whose member maps (instances of Map)
        match the `input_binning` specified here.

    output_binning : MultiDimBinning
        The `outputs` produced by this service will be a MapSet whose member
        maps (instances of Map) will have binning `output_binning`.

    transforms_cache_depth : int >= 0
        Number of transforms (TransformSet) to store in the transforms cache.
        Setting this to 0 effectively disables transforms caching.

    outputs_cache_depth : int >= 0
        Number of outputs (MapSet) to store in the outputs cache. Setting this
        to 0 effectively disables outputs caching.

    Attributes
    ----------

    Methods
    ----------

    Notes
    ----------
    Blah blah blah ...

    """
    def __init__(self, pararms, input_binning, output_binning,
                 tranforms_cache_depth=20, outputs_cache_depth=20):
        # All of the following params (and no more) must be passed via
        # the `params` argument.
        # TODO(shivesh) how to set default values?
        expected_params = (
            'pid_events', 'pid_ver', 'pid_remove_true_downgoing', 'pid_spec',
            'pid_spec_source', 'compute_error', 'replace_invalid'
        )

        # Define the names of objects that are required by this stage (objects
        # will have the attribute "name": i.e., obj.name)
        input_names = (
            'nue_cc', 'numu_cc', 'nutau_cc', 'nuall_nc'
        )

        # Define the names of objects that get produced by this stage
        output_names = (
            'trck', 'cscd'
        )

        super(self.__class__, self).__init__(
            use_transforms=True,
            stage_name='pid',
            service_name='mc',
            params=params,
            expected_params=expected_params,
            input_names=input_names,
            output_names=output_names,
            disk_cache=disk_cache,
            outputs_cache_depth=outputs_cache_depth,
            transforms_cache_depth=transforms_cache_depth,
            input_binning=input_binning,
            output_binning=output_binning
        )
