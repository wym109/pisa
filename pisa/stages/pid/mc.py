
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

from pisa.core.stage import Stage
from pisa.core.transform import BinnedTensorTransform, TransformSet
from pisa.utils.events import Events
from pisa.utils.flavInt import NuFlavInt
from pisa.utils.PIDSpec import PIDSpec
from pisa.utils.dataProcParams import DataProcParams
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging, set_verbosity
from pisa.utils.profiler import profile


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

            * pid_spec :
                TODO(shivesh): Why can a PIDSpec object be an input but not
                the dataProcParams object?
                TODO(shivesh): pid_events along with pid_spec_source predefines
                what pid_spec. Maybe there is a better way to implement this?

            * pid_spec_source : filepath
                TODO(shivesh): Why can the PIDSpec filepath be an input but not
                the dataProcParams filepath?
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
        # TODO(shivesh): hard-code replace_invalid?
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

        # Units must be the following for correctly converting a sum-of-
        # OneWeights-in-bin to an average effective area across the bin.
        # TODO(shivesh): azimuth dependence?
        comp_units = dict(energy='GeV', coszen=None)

        # Only works if energy and coszen is in input_binning
        if 'energy' not in self.input_binning:
            raise ValueError('Input binning must contain "energy" dimension,'
                             ' but does not.')
        if 'coszen' not in self.input_binning:
            raise ValueError('Input binning must contain "coszen" dimension,'
                             ' but does not.')

        # No further dimensions are allowed
        # TODO(shivesh): check this works
        excess_dims = set(self.input_binning.names).difference(comp_units.keys())
        if len(excess_dims) > 0:
            raise ValueError('Input binning has extra dimension(s): %s'
                             %sorted(excess_dims))

        # TODO: not handling rebinning in this stage or within Transform
        # objects; implement this! (and then this assert statement can go away)
        assert self.input_binning == self.output_binning

        # TODO(shivesh): check is error handled is managed by Transform
        #self.error_computed = False
        events = Events(self.params['pid_events'])

        # TODO(shivesh): figure out what this comment means
        # Select only the inits in the input/output binning for conversion
        # (can't pass more than what's actually there)
        in_units = {dim: unit for dim, unit in comp_units.items()
                    if dim in self.input_binning}
        out_units = {dim: unit for dim, unit in comp_units.items()
                     if dim in self.output_binning}

        # These will be in the computational units
        input_binning = self.input_binning.to(**in_units)
        output_binning = self.output_binning.to(**out_units)

        # TODO(shivesh): figure out if these todo's are relavant
        # TODO: take events object as an input instead of as a param that
        # specifies a file? Or handle both cases?

        # TODO: include here the logic from the make_events_file.py script so
        # we can go directly from a (reasonably populated) icetray-converted
        # HDF5 file (or files) to a nominal transform, rather than having to
        # rely on the intermediate step of converting that HDF5 file (or files)
        # to a PISA HDF5 file that has additional column(s) in it to account
        # for the combinations of flavors, interaction types, and/or simulation
        # runs. Parameters can include which groupings to use to formulate an
        # output.

        # TODO(shivesh): units for these
        # TODO(shivesh): allow for input of custom data_proc_params filepath?
        data_proc_params = DataProcParams(
            detector=events.metadata['detector'],
            proc_ver=events.metadata['proc_ver']
        )

        if self.params['pid_remove_true_downgoing']:
            # TODO(shivesh): more options for cuts?
            cut_events = data_proc_params.applyCuts(
                events, cuts='true_upgoing_coszen'
            )
        else:
            cut_events = events

        pid_spec = PIDSpec(
            detector=cut_events.metadata['detector'],
            geom=cut_events.metadata['geom'],
            proc_ver=cut_events.metadata['proc_ver'],
            pid_specs=self.params['pid_spec_source']
        )
        if self.output_names != pid_spec.get_signatures():
            msg = 'PID criteria from `pid_spec` ({0}) does not \
                    match the following specification {1}'
            raise ValueError(msg.format(pid_spec.get_signatures(),
                                        self.output_names))

        # TODO: add importance weights, error computation

        logging.info("Separating events by PID...")
        separated_events = pid_spec.applyPID(
            events=cut_events,
            return_fields=['reco_energy', 'reco_coszen']
        )

        # These get used in innermost loop, so produce it just once here
        all_bin_edges = [edges.magnitude for edges in output_binning.bin_edges]
        rep_flavint = flavInt.NuFlavIntGroup(self.input_names)

        transforms = []
        for signature in self.output_names:


    def validate_params(self, params):
        # do some checks on the parameters

        # Check type of pid_events
        assert isinstance(params['pid_events'], (basestring, Events))

        # Check type of compute_error, replace_invalid,
        # pid_remove_true_downgoing
        assert isinstance(params['compute_error'], bool)
        assert isinstance(params['replace_invalid'], bool)
        assert isinstance(params['pid_remove_true_downgoing'], bool)

        # Check type of pid_ver, pid_spec_source
        assert isinstance(params['pid_ver'], basestring)
        assert isinstance(params['pid_spec_source'], basestring)

        # Check the groupings of the pid_events file
        # TODO(shivesh): check the events initialisation
        events = Events(params['pid_events'])
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
