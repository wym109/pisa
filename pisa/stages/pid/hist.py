#
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
from pisa.utils.flavInt import NuFlavInt, NuFlavIntGroup
from pisa.utils.PIDSpec import PIDSpec
from pisa.utils.dataProcParams import DataProcParams
from pisa.utils.log import logging


class hist(Stage):
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

        Required fields:
            * pid_events : Events or filepath
                Events object or file path to HDF5 file containing events

            * pid_ver : string
                Version of PID to use (as defined for this
                detector/geometry/processing)

            * pid_remove_true_downgoing : Bool
                Remove MC-true-downgoing events

            TODO(shivesh): Either `pid_spec` or `pid_spec_source` can be used
            to define the PID specifications. Implement this behaviour and for
            the case when `pid_spec` is used, do a check to confirm that the
            pid_events object has the matching PID spec metadata
            * pid_spec : PIDSpec
                PIDSpec object which specifies the PID specifications

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
            service_name='hist',
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

        # TODO(shivesh): As of now, events do not have units as far as PISA
        # is concerned

        # Works only if either energy, coszen or azimuth is in input_binning
        bin_names = ('energy', 'coszen', 'azimuth')
        if set(self.input_binning.names).isdisjoint(bin_names):
            raise ValueError('Input binning must contain either one or a '
                             'combination of "energy", "coszen" or "azimuth" '
                             'dimensions.')

        # TODO: not handling rebinning in this stage or within Transform
        # objects; implement this! (and then this assert statement can go away)
        assert self.input_binning == self.output_binning

        events = Events(self.params['pid_events'].value)

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

        data_proc_params = DataProcParams(
            detector=events.metadata['detector'],
            proc_ver=events.metadata['proc_ver']
        )

        if self.params['pid_remove_true_downgoing'].value:
            # TODO(shivesh): more options for cuts?
            cut_events = data_proc_params.applyCuts(
                events, cuts='true_upgoing_coszen'
            )
        else:
            cut_events = events

        pid_spec = PIDSpec(
            detector=events.metadata['detector'],
            geom=events.metadata['geom'],
            proc_ver=events.metadata['proc_ver'],
            pid_specs=self.params['pid_spec_source'].value
        )
        u_out_names = map(unicode, self.output_names)
        if set(u_out_names) != set(pid_spec.get_signatures()):
            msg = 'PID criteria from `pid_spec` {0} does not match {1}'
            raise ValueError(msg.format(pid_spec.get_signatures(),
                                        u_out_names))

        # TODO: add importance weights, error computation

        logging.info("Separating events by PID...")
        var_names = ['reco_%s' %bin_name
                     for bin_name in self.output_binning.names]
        var_names += ['weighted_aeff']
        separated_events = pid_spec.applyPID(
            events=cut_events,
            return_fields=var_names
        )

        # These get used in innermost loop, so produce it just once here
        all_bin_edges = [edges.magnitude
                         for edges in self.output_binning.bin_edges]

        transforms = []
        for flavint in self.input_names:
            rep_flavint = NuFlavIntGroup(flavint)[0]
            raw_histo = {}
            # TODO(shivesh): errors
            # TODO(shivesh): total histo check?
            total_histo = np.zeros(self.output_binning.shape)

            for sig in self.output_names:
                raw_histo[sig] = {}
                flav_sigdata = separated_events[rep_flavint][sig]
                reco_params = [flav_sigdata[vn] for vn in var_names]
                raw_histo[sig], _ = np.histogramdd(
                    sample=reco_params[:-1],
                    weights=reco_params[-1],
                    bins=all_bin_edges
                )
                total_histo += raw_histo[sig]

            for sig in self.output_names:
                xform_array = raw_histo[sig]/ total_histo

                invalid_idx = total_histo == 0
                valid_idx = 1-invalid_idx
                invalid_idx = np.where(invalid_idx)[0]
                num_invalid = len(invalid_idx)

                if num_invalid > 0:
                    logging.warn(
                        'Group "%s", PID signature "%s" has %d bins with no'
                        ' events (and hence the ability to separate events'
                        ' by PID cannot be ascertained).'
                        %(flavint, sig, num_invalid)
                    )

                    if self.params['replace_invalid']:
                        logging.warn('Replacing the no-events bins with values'
                                     ' from closest bins that have events.')
                        replace_idx = []
                        logging.warn(message)
                        valid_idx = np.where(valid_idx)[0]
                        for idx in invalid_idx:
                            dist = np.abs(valid_idx-idx)
                            nearest_valid_idx = valid_idx[np.where(dist==np.min(dist))[0][0]]
                            replace_idx.append(nearest_valid_idx)
                            xform_array[idx] = xform_array[nearest_valid_idx]
                    else:
                        # TODO: raise ValueError(message) ?
                        pass

                xform = BinnedTensorTransform(
                    input_names=flavint,
                    output_name=sig,
                    input_binning=self.input_binning,
                    output_binning=self.output_binning,
                    xform_array=xform_array
                )
                transforms.append(xform)

        return TransformSet(transforms=transforms)

    def validate_params(self, params):
        # do some checks on the parameters

        # Check type of pid_events
        assert isinstance(params['pid_events'].value, (basestring, Events))

        # Check type of compute_error, replace_invalid,
        # pid_remove_true_downgoing
        assert isinstance(params['compute_error'].value, bool)
        assert isinstance(params['replace_invalid'].value, bool)
        assert isinstance(params['pid_remove_true_downgoing'].value, bool)

        # Check type of pid_ver, pid_spec_source
        assert isinstance(params['pid_ver'].value, basestring)
        assert isinstance(params['pid_spec_source'].value, basestring)

        # Check the groupings of the pid_events file
        events = Events(params['pid_events'].value)
        should_be_joined = sorted([
            NuFlavIntGroup('nuecc+nuebarcc'),
            NuFlavIntGroup('numucc+numubarcc'),
            NuFlavIntGroup('nutaucc+nutaubarcc'),
            NuFlavIntGroup('nuallnc+nuallbarnc'),
        ])
        are_joined = sorted([
            NuFlavIntGroup(s)
            for s in events.metadata['flavints_joined']
        ])
        if are_joined != should_be_joined:
            raise ValueError('Events passed have %s joined groupings but'
                             ' it is required to have %s joined groupings.'
                             % (are_joined, should_be_joined))
