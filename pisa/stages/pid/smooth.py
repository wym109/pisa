# authors: Shivesh Mandalia
#          s.p.mandalia@qmul.ac.uk
#          J.L. Lanfranchi
#          jll1062+pisa@phys.psu.edu
#
# date:    2016-09-18
"""
The purpose of this stage is to simulate the classification of events seen by a detector into different PID "channels."

For example, in PINGU, this separates reconstructed nue CC (+ nuebar CC),
numu CC (+ numubar CC), nutau CC (+ nutaubar CC), and all NC events into
separate 'trck' and 'cscd' distributions.

This service in particular takes in events from a PISA HDF5 file to transform
a set of input maps into a set of maps, one each per PID signature.

For each particle "signature," a histogram in the input binning dimensions is
created, which gives the PID probabilities in each bin. The input maps are
transformed according to these probabilities to provide an output containing a
map for track-like events ('trck') and shower-like events ('cscd'), which is
then returned.

"""


from itertools import product

import numpy as np

from pisa.core.stage import Stage
from pisa.core.transform import BinnedTensorTransform, TransformSet
from pisa.core.events import Events
from pisa.utils.dataProcParams import DataProcParams
from pisa.utils.flavInt import flavintGroupsFromString, NuFlavIntGroup, ALL_NUFLAVINTS
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging
from pisa.utils.PIDSpec import PIDSpec
from pisa.utils.profiler import profile
from pisa.utils.resources import find_resource


class smooth(Stage):
    """Parameterised and smoothed PID from Monte Carlo events.

    Transforms an input map of the specified particle "signature" (aka ID) into
    a map for each signature specified in `pid_specs`.

    Example signatures used in PINGU are track-like events ('trck') and
    shower-like events ('cscd').


    Parameters
    ----------
    params : ParamSet or instantiable thereto

        Parameters which set everything besides the binning.

        If str, interpret as resource location and load params from resource.
        If dict, set contained params. Format expected is
            {'<param_name>': <Param object or passable to Param()>}

        Parameters required by this service are
            * pid_events : Events or filepath
                Events object or file path to HDF5 file containing events

            * pid_specs : Mapping, preferably OrderedDict
                Mapping (OrderedDict) has following format:
                    {'<pid sig0 name>': '<criteria str>',
                     '<pid sig1 name>': '<criteria str>', ...}
                The order of signatures is important only for smoothing, where
                the first N-1 are smoothed but the final signature's value in a
                given bin is the difference between 1 and the sum of all other
                signatures' values in that bin.

            * pid_weights_name: str or NoneType
                Specify the name of the node whose data will be used as weights
                to create the reco and pid variables histogram. If NoneType is
                given then events will not be weighted.

            * pid_smooth_n_ebins : int >= 0
                Number of bins to use for subdividing the energy range for
                purposes of smoothing. The technique used for PINGU relied on
                300 bins for MC spanning [1, 80] GeV range. Set to 0 to disable
                smoothing in the `reco_energy` dimension.

            * pid_smooth_n_czbins : int >= 0
                Number of bins to use for subdividing the coszen range for
                purposes of smoothing. This was disabled for PINGU (except to
                verify that PID did not vary much as a function of
                `reco_coszen`). Set to 0 to disable smoothing in the
                `reco_coszen` dimension

            * transform_events_cuts : None, string, or sequence of strings
                Additional cuts that are applied to events prior to computing
                transforms with them. E.g., "true_coszen <= 0" removes all
                MC-true downgoing events. See `pisa.core.events.Events` class
                for details on cut specifications.

    particles

    input_names

    transform_groups

    TODO: sum_grouped_flavints

    input_binning : MultiDimBinning
        Arbitrary number of dimensions accepted. Contents of the input
        `pid_events` parameter defines the possible binning dimensions. Name(s)
        of given binning(s) must match to a reco variable in `pid_events`.

    output_binning : MultiDimBinning

    reco_input_binning : MultiDimBinning
        What binning is used at the input of the reconstructions stage? (I.e.,
        what are the limits of the MC-true variables that enter into reco?)
        This is necessary such that the same "cuts" (binning true variables
        effectively cuts out any events that do not fall within the binning)
        are applied to true variables so the same events are used to compute
        the PID transforms.

    error_method : None, bool, or string

    disk_cache : None, str, or DiskCache
        If None, no disk cache is available.
        If str, represents a path with which to instantiate a utils.DiskCache
        object. Must be concurrent-access-safe (across threads and processes).

    transforms_cache_depth : int >= 0

    outputs_cache_depth : int >= 0

    memcache_deepcopy : bool

    debug_mode : None, bool, or string
        Whether to store extra debug info for this service.


    Input Names
    ----------

    Output Names
    ----------

    Notes
    ----------
    If smoothing is enabled for multiple dimensions, smoothing is combined as
    an outer product.

    This service takes in events from a **joined** PISA HDF5 file. The current
    implementation of this service requires that the nodes on these file match
    a certain flavour/interaction combination or "particle signature", which is
    `nue_cc, numu_cc, nutau_cc, nuall_nc`. Thus, only the HDF5 files with the
    naming convention
    ```
    events__*__joined_G_nue_cc+nuebar_cc_G_numu_cc+numubar_cc_G_nutau_cc+nutaubar_cc_G_nuall_nc+nuallbar_nc.hdf5
    ```
    should be used as input. The structure of the datafile is
    ```
    flavour / int_type / value
    ```
    where
        *  `flavour` is one of `nue, nue_bar, numu, numu_bar, nutau, nutau_bar`
        *  `int_type` is one of `cc` or `nc`
        *  `values` is one of
            * `pid` : the pid score per event
            * `reco_*param*` : the reco *param* of the event
            * `weighted_aeff`: the effective area weight per event
              (see Stage 3, Effective Area)

    For the 'joined' event files, the charged current components for the
    particle and antiparticle of a specific neutrino flavour are summed so
    that, for example, the data in the nodes `nue/cc` and `nue_bar/cc` both
    contain their own and each others events. The combined neutral current
    interaction for all neutrino flavours is also summed in the same way, so
    that any `nc` node contains the data of all neutrino flavours.

    Once the file has been read in, for each particle signature, the input PID
    specification is used to set a limit on the pid score above which the
    events are distinguished as being track-like `trck` and below as
    shower-like `cscd`.

    Next, a histogram in the input binning dimensions is created for all
    combinations of particle signature and pid channel and then normalised to
    one with respect to the particle signature to give the probability of an
    event lying in a particular bin. The input maps for each signature can then
    be transformed according to these probabilities to provide an output which
    will separate each signature into `trck` and `cscd` maps, and this is then
    returned.

    """
    # TODO: add sum_grouped_flavints instantiation arg
    def __init__(self, params, particles, input_names, transform_groups,
                 input_binning, output_binning, reco_input_binning,
                 smoothing_method, memcache_deepcopy, transforms_cache_depth,
                 outputs_cache_depth,
                 #smooth_emin=1*ureg.GeV, smooth_emax=80*ureg.GeV,
                 #smooth_n_ebins=300, smooth_n_czbins=200,
                 disk_cache=None, error_method=None, debug_mode=None):
        self.events_hash = None
        """Hash of events file or Events object used"""

        assert particles in ['muons', 'neutrinos']
        self.particles = particles
        """Whether stage is instantiated to process neutrinos or muons"""

        self.transform_groups = flavintGroupsFromString(transform_groups)
        """Particle/interaction types to group for computing transforms"""

        # All of the following params (and no more) must be passed via
        # the `params` argument.
        expected_params = (
            'pid_events', 'pid_ver', 'pid_specs', 'pid_weights_name'
        )

        if isinstance(input_names, basestring):
            input_names = input_names.replace(' ', '').split(',')

        # Define the names of objects that get produced by this stage
        self.output_channels = ('trck', 'cscd')
        output_names = [self.suffix_channel(in_name, out_chan) for in_name,
                        out_chan in product(input_names, self.output_channels)]

        super(self.__class__, self).__init__(
            use_transforms=True,
            stage_name='pid',
            service_name='hist',
            params=params,
            expected_params=expected_params,
            input_names=input_names,
            output_names=output_names,
            error_method=error_method,
            disk_cache=disk_cache,
            outputs_cache_depth=outputs_cache_depth,
            transforms_cache_depth=transforms_cache_depth,
            memcache_deepcopy=memcache_deepcopy,
            input_binning=input_binning,
            output_binning=output_binning,
            debug_mode=debug_mode
        )

        # Can do these now that binning has been set up in call to Stage's init
        self.include_attrs_for_hashes('particles')
        self.include_attrs_for_hashes('transform_groups')

    def load_events(self):
        evts = self.params['pid_events'].value

        # "Normalize" the path by passing through find_resource (also makes
        # sure the file exists)
        if isinstance(evts, basestring):
            evts = find_resource(evts)

        this_hash = hash_obj(evts)
        if this_hash == self.events_hash:
            return
        logging.debug('Extracting events from Events obj or file: %s' %evts)

        # Load the events
        self.events = Events(evts)

        # Keep only events as used for reconstructions
        evts.keepInbounds(self.reco_input_binning)

        self.events_hash = this_hash
        self.data_proc_params = DataProcParams(
            detector=self.events.metadata['detector'],
            proc_ver=self.events.metadata['proc_ver']
        )

    @profile
    def _compute_nominal_transforms(self):
        """Compute new PID transforms."""
        logging.debug('Updating pid.hist PID histograms...')

        # TODO(shivesh): As of now, events do not have units as far as PISA
        # is concerned

        # Works only if either energy, coszen or azimuth is in input_binning
        dim_names = ('reco_energy', 'reco_coszen', 'reco_azimuth')
        if set(self.input_binning.names).isdisjoint(dim_names):
            raise ValueError(
                'Input binning must contain either one or a combination of'
                ' "reco_energy", "reco_coszen" or "reco_azimuth" dimensions.'
            )

        # TODO: not handling rebinning in this stage or within Transform
        # objects; implement this! (and then this assert statement can go away)
        assert self.input_binning == self.output_binning

        self.load_events()

        # TODO: in future, the events file will not have these combined
        # already, and it should be done here (or in a nominal transform,
        # etc.). See below about taking this step when we move to directly
        # using the I3-HDF5 files.
        events_file_combined_flavints = tuple([
            NuFlavIntGroup(s) for s in self.events.metadata['flavints_joined']
        ])

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

        if self.params['pid_remove_true_downgoing'].value:
            # TODO(shivesh): more options for cuts?
            cut_events = self.data_proc_params.applyCuts(
                self.events, cuts='true_upgoing_coszen'
            )
        else:
            cut_events = self.events

        pid_spec = PIDSpec(
            detector=self.events.metadata['detector'],
            geom=self.events.metadata['geom'],
            proc_ver=self.events.metadata['proc_ver'],
            pid_spec_ver=self.params['pid_ver'].value,
            pid_specs=self.params['pid_spec_source'].value
        )
        u_out_names = map(unicode, self.output_channels)
        if set(u_out_names) != set(pid_spec.get_signatures()):
            msg = 'PID criteria from `pid_spec` {0} does not match {1}'
            raise ValueError(msg.format(pid_spec.get_signatures(),
                                        u_out_names))

        # TODO: add importance weights, error computation

        logging.debug("Separating events by PID...")
        var_names = self.input_binning.names
        if self.params['pid_weights_name'].value is not None:
            var_names += [self.params['pid_weights_name'].value]
        separated_events = pid_spec.applyPID(
            events=cut_events,
            return_fields=var_names
        )

        # These get used in innermost loop, so produce it just once here
        all_bin_edges = [dim.bin_edges.magnitude
                         for dim in self.output_binning.dimensions]

        # Derive transforms by combining flavints that behave similarly, but
        # apply the derived transforms to the input flavints separately
        # (leaving combining these together to later)

        transforms = []
        for flav_int_group in self.transform_groups:
            logging.debug("Working on %s PID" %flav_int_group)

            repr_flav_int = flav_int_group[0]

            # TODO(shivesh): errors
            # TODO(shivesh): total histo check?
            raw_histo = {}
            total_histo = np.zeros(self.output_binning.shape)
            for sig in self.output_channels:
                raw_histo[sig] = {}
                flav_sigdata = separated_events[repr_flav_int][sig]
                reco_params = [flav_sigdata[n]
                               for n in self.output_binning.names]

                if self.params['pid_weights_name'].value is not None:
                    weights = flav_sigdata[self.params['pid_weights_name'].value]
                else:
                    weights = None

                raw_histo[sig], _ = np.histogramdd(
                    sample=reco_params,
                    weights=weights,
                    bins=all_bin_edges
                )
                total_histo += raw_histo[sig]

            for sig in self.output_channels:
                # Get fraction of each PID per bin
                with np.errstate(divide='ignore', invalid='ignore'):
                    xform_array = raw_histo[sig] / total_histo

                num_invalid = np.sum(~np.isfinite(xform_array))
                if num_invalid > 0:
                    logging.warn(
                        'Group "%s", PID signature "%s" has %d bins with no'
                        ' events (and hence the ability to separate events'
                        ' by PID cannot be ascertained). These are being'
                        ' masked off from any further computations.'
                        % (flav_int_group, sig, num_invalid)
                    )
                    # TODO: this caused buggy event propagation for some
                    # reason; check and re-introduced the masked array idea
                    # when this is fixed. For now, replicating the behavior
                    # from PISA 2.
                    #xform_array = np.ma.masked_invalid(xform_array)

                # Double check that no NaN remain
                #assert not np.any(np.isnan(xform_array))

                # Copy this transform to use for each input in the group
                for input_name in self.input_names:
                    if input_name not in flav_int_group:
                        continue
                    xform = BinnedTensorTransform(
                        input_names=input_name,
                        output_name=self.suffix_channel(input_name, sig),
                        input_binning=self.input_binning,
                        output_binning=self.output_binning,
                        xform_array=xform_array
                    )
                    transforms.append(xform)

        return TransformSet(transforms=transforms)

    def _compute_transforms(self):
        """There are no systematics in this stage, so the transforms are just
        the nominal transforms. Thus, this function just returns the nominal
        transforms, computed by `_compute_nominal_transforms`..

        """
        return self.nominal_transforms

    def suffix_channel(self, flavint, channel):
        return '%s_%s' % (flavint, channel)

    def validate_params(self, params):
        # do some checks on the parameters

        # Check type of pid_events
        assert isinstance(params['pid_events'].value, (basestring, Events))

        # Check type of pid_remove_true_downgoing
        assert isinstance(params['pid_remove_true_downgoing'].value, bool)

        # Check type of pid_ver, pid_spec_source
        assert isinstance(params['pid_ver'].value, basestring)
        assert isinstance(params['pid_spec_source'].value, basestring)

        # Check the groupings of the pid_events file
        events = Events(params['pid_events'].value)
        should_be_joined = sorted([
            NuFlavIntGroup('nue_cc + nuebar_cc'),
            NuFlavIntGroup('numu_cc + numubar_cc'),
            NuFlavIntGroup('nutau_cc + nutaubar_cc'),
            NuFlavIntGroup('nuall_nc + nuallbar_nc'),
        ])
        are_joined = sorted([
            NuFlavIntGroup(s)
            for s in events.metadata['flavints_joined']
        ])
        if are_joined != should_be_joined:
            raise ValueError('Events passed have %s joined groupings but'
                             ' it is required to have %s joined groupings.'
                             % (are_joined, should_be_joined))
