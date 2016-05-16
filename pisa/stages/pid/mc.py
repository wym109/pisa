
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
from pisa.utils.events import Events
import pisa.utils.flavInt as flavInt
from pisa.utils.PIDSpec import PIDSpec
from pisa.utils.dataProcParams import DataProcParams
from pisa.utils.log import logging


class mc(Stage):
    """PID based on input PISA events HDF5 file.

    Transforms a input map of the specified particle "signature" (aka ID) into
    a map of the track-like events ('trck') and a map of the shower-like events
    ('cscd').

    Parameters
    ----------
    params : ParamSet or sequence with which to instantiate a ParamSet
        Parameters which set everything besides the binning.

        If str, interpret as resource location and load params from resource.
        If dict, set contained params. Format expected is
            {'<param_name>': <Param object or passable to Param()>}.

        Options:
            * pid_events : Events or filepath
                Events object or file path to HDF5 file containing events

            * pid_ver : string
                Version of PID to use (as defined for this
                detector/geometry/processing)

            * pid_remove_true_downgoing : Bool
                Remove MC-true-downgoing events

            * pid_spec : TODO(shivesh) figure out what this is

            * pid_spec_source : filepath
                Resource for loading PID specifications

            * compute_error : Bool
                Compute histogram errors

            * replace_invalid : Bool
                Replace invalid histogram entries with nearest neighbor's value

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
    def __init__(self, params, input_binning, output_binning, disk_cache=None,
                 transforms_cache_depth=20, outputs_cache_depth=20):
        # All of the following params (and no more) must be passed via
        # the `params` argument.
        # TODO(shivesh) hard-code replace_invalid?
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

    def _compute_transforms(self):
        """Compute new PID transforms."""
        logging.info('Updating PIDServiceMC PID histograms...')

        events = Events(self.params['pid_events'])
        data_proc_params = DataProcParams(
            detector=events.metadata['detector'],
            proc_ver=events.metadata['proc_ver']
        )

        if self.params['pid_remove_true_downgoing']:
            cut_events = dataProcParams.applyCuts(
                events, cuts='true_upgoing_coszen'
            )
        else:
            cut_events = events

        pid_spec = PIDSpec(
            detector=events.metadata['detector'],
            geom=events.metadata['geom'],
            proc_ver=events.metadata['proc_ver'],
            pid_specs=self.params['pid_spec_source']
        )
        # TODO(shivesh): Check to see if these are the same as output_names?
        signatures = pid_spec.get_signatures()

        # TODO: add importance weights, error computation

        logging.info("Separating events by PID...")
        separated_events = pid_spec.applyPID(
            events=cut_events,
            return_fields=['reco_energy', 'reco_coszen']
        )


    def validate_params(self, params):
        # do some checks on the parameters

        # Check type of pid_events
        assert isinstance(params['pid_events'], (basestring, Events))

        # Check type of compute_error, replace_invalid,
        # pid_remove_true_downgoing
        assert isinstance(compute_error, bool)
        assert isinstance(replace_invalid, bool)
        assert isinstance(pid_remove_true_downgoing, bool)

        # Check type of pid_ver, pid_spec_source
        assert isinstance(pid_ver, basestring)
        assert isinstance(pid_spec_source, basestring)

        # Check the groupings of the pid_events file
        # TODO(shivesh): check the events initialisation
        events = Events(pid_events)
        should_be_joined = sorted([
            flavInt.NuFlavIntGroup('nuecc+nuebarcc'),
            flavInt.NuFlavIntGroup('numucc+numubarcc'),
            flavInt.NuFlavIntGroup('nutaucc+nutaubarcc'),
            flavInt.NuFlavIntGroup('nuallnc+nuallbarnc'),
        ])
        are_joined = sorted([
            flavInt.NuFlavIntGroup(s)
            for s in events.metadata['flavints_joined']
        ])
        if are_joined != should_be_joined:
            raise ValueError('Events passed have %s joined groupings but'
                             ' it is required to have %s joined groupings.'
                             % (are_joined, should_be_joined))
