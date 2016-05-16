#
# author: J.L. Lanfranchi
#
# date:   March 2, 2016
"""
PID service using info directly from events.
"""

import collections
from itertools import izip

import numpy as np

from pisa.utils.log import logging
from pisa.pid.PIDServiceBase import PIDServiceBase
from pisa.utils.utils import get_bin_sizes
import pisa.utils.flavInt as flavInt
from pisa.utils.events import Events
from pisa.utils.PIDSpec import PIDSpec
from pisa.utils.dataProcParams import DataProcParams

# TODO: implement cuts via pertinent DataProcParams


class PIDServiceMC(PIDServiceBase):
    """
    Takes a PISA events HDF5 file and creates 2D-histogrammed PID in terms of
    energy and coszen, for each specified particle "signature" (aka ID).
    """
    def __init__(self, ebins, czbins, pid_events, pid_ver,
                 pid_remove_true_downgoing, pid_spec=None,
                 pid_spec_source=None, compute_error=False,
                 replace_invalid=False, **kwargs):
        super(PIDServiceBase, self).__init__(ebins, czbins)

        self.events_source = None
        self.events = None
        self.cut_events = None
        self.data_proc_params = None
        self.pid_remove_true_downgoing = None

        self.pid_ver = None
        self.pid_spec = None
        self.pid_spec_source = pid_spec_source

        self.compute_error = compute_error
        self.error_computed = False
        self.replace_invalid = replace_invalid

        self.get_pid_kernels(
            ebins=ebins, czbins=czbins, pid_events=pid_events, pid_ver=pid_ver,
            pid_remove_true_downgoing=pid_remove_true_downgoing,
            pid_spec=pid_spec, compute_error=compute_error,
            replace_invalid=replace_invalid,
        )

    def get_pid_kernels(self, ebins, czbins, pid_events, pid_ver,
                        pid_remove_true_downgoing=None, pid_spec=None,
                        compute_error=None, replace_invalid=None):
        """Compute and return PID maps"""
        # Default to values passed when class was instantiated
        if pid_remove_true_downgoing is None:
            pid_remove_true_downgoing = self.pid_remove_true_downgoing
        if replace_invalid is None:
            replace_invalid = self.replace_invalid
        if compute_error is None:
            compute_error = self.compute_error

        # TODO: add stateful return-early logic
        #if ebins == self.ebins and \
        #        czbins == self.czbins and \
        #        pid_events == self.events_source and \
        #        pid_ver == self.pid_ver and \
        #        pid_spec == self.pid_spec and \
        #        (not compute_error or (compute_error == self.compute_error)):
        #    return
        self.ebins = ebins
        self.czbins = czbins

        histo_binspec = (self.ebins, self.czbins)
        n_ebins = len(self.ebins) - 1
        n_czbins = len(self.czbins) - 1
        self.compute_error = compute_error
        logging.info('Updating PIDServiceMC PID histograms...')

        self.pid_remove_true_downgoing = pid_remove_true_downgoing

        new_events = False
        if self.events is None or pid_events != self.events_source:
            new_events = True
            if isinstance(pid_events, basestring):
                logging.info('Extracting events from file: %s' % (pid_events))
                self.events = Events(pid_events)
            elif isinstance(pid_events, Events):
                # Validate by (re)instantiating as an Events object
                self.events = pid_events
            else:
                raise TypeError('Unhandled `pid_events` type: "%s"' %
                                type(pid_events))
            should_be_joined = sorted([
                flavInt.NuFlavIntGroup('nuecc+nuebarcc'),
                flavInt.NuFlavIntGroup('numucc+numubarcc'),
                flavInt.NuFlavIntGroup('nutaucc+nutaubarcc'),
                flavInt.NuFlavIntGroup('nuallnc+nuallbarnc'),
            ])
            are_joined = sorted([
                flavInt.NuFlavIntGroup(s)
                for s in self.events.metadata['flavints_joined']
            ])
            if are_joined != should_be_joined:
                raise ValueError('Events passed have %s joined groupings but'
                                 ' it is required to have %s joined groupings.'
                                 % (are_joined, should_be_joined))
            self.events_source = pid_events
            self.data_proc_params = DataProcParams(
                detector=self.events.metadata['detector'],
                proc_ver=self.events.metadata['proc_ver']
            )

        if new_events or (self.cut_events is None) or \
                (pid_remove_true_downgoing != self.pid_remove_true_downgoing):
            if pid_remove_true_downgoing:
                self.cut_events = self.data_proc_params.applyCuts(
                    self.events, cuts='true_upgoing_coszen'
                )
            else:
                self.cut_events = self.events
            self.pid_remove_true_downgoing = pid_remove_true_downgoing

        if new_events or (self.pid_spec is None) or (pid_ver != self.pid_ver):
            self.pid_spec = PIDSpec(
                detector=self.events.metadata['detector'],
                geom=self.events.metadata['geom'],
                proc_ver=self.events.metadata['proc_ver'],
                pid_specs=self.pid_spec_source
            )
        self.signatures = self.pid_spec.get_signatures()

        # TODO: add importance weights, error computation

        logging.info("Separating events by PID...")
        self.separated_events = self.pid_spec.applyPID(
            events=self.cut_events,
            return_fields=['reco_energy', 'reco_coszen'],
        )

        self.pid_kernels = {'binning': {'ebins': self.ebins,
                                     'czbins': self.czbins}}
        self.pid_kernels_rel_error = {'binning': {'ebins': self.ebins,
                                               'czbins': self.czbins}}
        for label in ['nue_cc', 'numu_cc', 'nutau_cc', 'nuall_nc']:
            rep_flavint = flavInt.NuFlavIntGroup(label)[0]
            self.pid_kernels[label] = {}
            raw_histo = {}
            raw_histo_err = {}
            total_histo = np.zeros([n_ebins, n_czbins])
            total_histo_check = None
            if self.compute_error:
                total_err2 = np.zeros([n_ebins, n_czbins])

            for sig in self.signatures:
                flav_sigdata = self.separated_events[rep_flavint][sig]
                reco_e = flav_sigdata['reco_energy']
                reco_cz = flav_sigdata['reco_coszen']
                try:
                    #weights = flav_sigdata['importance_weight']
                    weights = flav_sigdata['weighted_aeff']
                    weights2 = weights * weights
                    #weights_check = self.cut_events[rep_flavint]['importance_weight']
                    weights_check = self.cut_events[rep_flavint]['weighted_aeff']
                except:
                    logging.warn('No importance weights found in events!')
                    weights = None
                    weights2 = None
                    weights_check = None
                raw_histo[sig], _, _ = np.histogram2d(
                    reco_e,
                    reco_cz,
                    weights=weights,
                    bins=histo_binspec,
                )
                total_histo += raw_histo[sig]

                if self.compute_error:
                    raw_histo_err[sig], _, _ = np.histogram2d(
                        reco_e,
                        reco_cz,
                        weights=weights2,
                        bins=histo_binspec,
                    )
                    total_err2 += raw_histo_err[sig] / \
                            (np.clip(raw_histo[sig], 1, np.inf)**2)
                    self.error_computed = True

            for sig in self.signatures:
                self.pid_kernels[label][sig] = raw_histo[sig] / total_histo

                invalid_idx = total_histo == 0
                valid_idx = 1-invalid_idx
                invalid_idx = np.where(invalid_idx)[0]
                num_invalid = len(invalid_idx)

                message = 'Group "%s", PID signature "%s" has %d invalid' \
                        ' entry(ies)!' % (label, sig, num_invalid)

                if num_invalid > 0 and not replace_invalid:
                    pass
                    #raise ValueError(message)

                replace_idx = []
                if num_invalid > 0 and replace_invalid:
                    logging.warn(message)
                    valid_idx = np.where(valid_idx)[0]
                    for idx in invalid_idx:
                        dist = np.abs(valid_idx-idx)
                        nearest_valid_idx = valid_idx[np.where(dist==np.min(dist))[0][0]]
                        replace_idx.append(nearest_valid_idx)
                        self.pid_kernels[label][sig][idx] = \
                                self.pid_kernels[label][sig][nearest_valid_idx]

            # Relative error is same for all signatures, since equations
            # implemented are
            #   pidhist_x / (pidhist_x + pidhist_y + ...)
            #   pidhist_y / (pidhist_x + pidhist_y + ...)
            #   ...
            if self.compute_error:
                if replace_invalid:
                    for orig_idx, repl_idx in izip(invalid_idx, replace_idx):
                        total_err2[orig_idx] = total_err2[repl_idx]
                #total_err2[total_err2 == 0] = \
                #        np.min(total_err2[total_err2 != 0])
                self.pid_kernels_rel_error[label] = np.sqrt(total_err2)

        return self.pid_kernels

    def get_pid(self, **kwargs):
        """Returns the PID maps"""
        return self.pid_kernels
    
    def get_rel_error(self):
        """Returns the PID maps' relative error"""
        assert self.error_computed
        return self.pid_kernels_rel_error

    @staticmethod
    def add_argparser_args(parser):
        parser.add_argument(
            '--pid-events', metavar='RESOURCE_NAME', type=str,
            default='events/pingu_v36/events__pingu__v36__runs_388-390__proc_v5__joined_G_nue_cc+nuebar_cc_G_numu_cc+numubar_cc_G_nutau_cc+nutaubar_cc_G_nuall_nc+nuallbar_nc.hdf5',
            help='''[ PID-MC ] PISA-standard events file'''
        )
        parser.add_argument(
            '--pid-ver', type=str,
            default='1',
            help='''[ PID-MC ] Version of PID to use (as defined for this
            detector/geometry/processing)'''
        )
        parser.add_argument(
            '--pid-remove-true-downgoing', action='store_true',
            help='''[ PID-MC ] Remove MC-true-downgoing events'''
        )
        parser.add_argument(
            '--pid-spec-source', default='pid/pid_specifications.json',
            help='''[ PID-MC ] Resource for loading PID specifications'''
        )
        parser.add_argument(
            '--compute-error', action='store_true',
            help='''[ PID-MC ] Compute histogram errors'''
        )
        parser.add_argument(
            '--replace-invalid', action='store_true',
            help='''[ PID-MC ] Replace invalid histogram entries with nearest
            neighbor's value'''
        )
        return parser
