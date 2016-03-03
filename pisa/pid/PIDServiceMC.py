#
# author: J.L. Lanfranchi
#
# date:   March 2, 2016
"""
PID histograms from a PISA events HDF5 file.
"""

import collections

import numpy as np
from scipy.interpolate import interp1d

from pisa.utils.log import logging
from pisa.utils.utils import get_bin_sizes
import pisa.utils.flavInt as flavInt
from pisa.utils.events import Events
from pisa.utils.PIDSpec import PIDSpec
from pisa.utils.dataProcParams import DataProcParams

# TODO: implement cuts via pertinent DataProcParams

class PIDServiceMC(object):
    """
    Takes a PISA events HDF5 file and creates 2D-histogrammed PID in terms of
    energy and coszen, for each specified particle "signature" (aka ID).
    """
    def __init__(self, ebins, czbins, events, pid_ver, remove_downgoing,
                 pid_spec=None, pid_spec_source=None, compute_error=False,
                 **kwargs):
        self.ebins = None
        self.czbins = None

        self.events_source = None
        self.events = None
        self.cut_events = None

        self.pid_ver = None
        self.remove_downgoing = None
        self.pid_spec_source = pid_spec_source
        self.data_proc_params = None
        self.pid_spec = None
        self.compute_error = False
        self.error_computed = False

        self.update(ebins=ebins, czbins=czbins, events=events, pid_ver=pid_ver,
                    remove_downgoing=remove_downgoing, pid_spec=pid_spec,
                    compute_error=compute_error)

    def update(self, ebins, czbins, events, pid_ver, remove_downgoing,
               pid_spec=None, compute_error=False):
        # TODO: stateful return-early
        #if ebins == self.ebins and \
        #        czbins == self.czbins and \
        #        events == self.events_source and \
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

        self.remove_downgoing = remove_downgoing

        new_events = False
        if self.events is None or events != self.events_source:
            new_events = True
            if isinstance(events, basestring):
                logging.info('Extracting events from file: %s' % (events))
                self.events = Events(events)
            elif isinstance(events, Events):
                # Validate by (re)instantiating as an Events object
                self.events = events
            else:
                raise TypeError('Unhandled `events` type: "%s"' % type(events))
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
            self.events_source = events
            self.data_proc_params = DataProcParams(
                detector=self.events.metadata['detector'],
                proc_ver=self.events.metadata['proc_ver']
            )

        if new_events or (self.cut_events is None) or \
                (remove_downgoing != self.remove_downgoing):
            if remove_downgoing:
                self.cut_events = self.data_proc_params.applyCuts(
                    self.events, cuts='true_upgoing_coszen'
                )
            else:
                self.cut_events = self.events
            self.remove_downgoing = remove_downgoing

        if new_events or (self.pid_spec is None) or (pid_ver != self.pid_ver):
            self.pid_spec = PIDSpec(
                detector=self.events.metadata['detector'],
                geom=self.events.metadata['geom'],
                proc_ver=self.events.metadata['proc_ver'],
                pid_specs=self.pid_spec_source
            )

        # TODO: add importance weights, error computation

        logging.info("Separating events by PID...")
        self.separated_events = self.pid_spec.applyPID(
            events=self.cut_events,
            return_fields=['reco_energy', 'reco_coszen'],
        )

        # PID maps are the fractional version of the above
        self.pid_maps = {'binning': {'ebins': self.ebins,
                                     'czbins': self.czbins}}
        self.agg_events = self.pid_spec.aggregate(self.separated_events)

        self.raw_histo = {}
        self.raw_histo_err = {}
        self.total_count = np.zeros([n_ebins, n_czbins])
        self.total_err2 = np.zeros([n_ebins, n_czbins])
        for sig in self.pid_spec.get_signatures():
            reco_e = self.agg_events[sig]['reco_energy']
            reco_cz = self.agg_events[sig]['reco_coszen']
            try:
                weights = self.agg_events[sig]['importance_weight']
                weights2 = weights * weights
            except:
                logging.warn('No importance weights found in events!')
                weights = None
                weights2 = None
            self.raw_histo[sig], _, _ = np.histogram2d(
                reco_e,
                reco_cz,
                weights=weights,
                bins=histo_binspec,
            )
            self.total_count += self.raw_histo[sig]
            if self.compute_error:
                self.raw_histo_err[sig], _, _ = np.histogram2d(
                    reco_e,
                    reco_cz,
                    weights=weights2,
                    bins=histo_binspec,
                )
                self.total_err2 += self.raw_histo_err[sig]
                self.error_computed = True

        self.pid_err = {}
        for sig in self.pid_spec.get_signatures():
            self.pid_maps[sig] = self.raw_histo[sig] / self.total_count
            #self.pid_err[sig] = 

    def get_pid(self, **kwargs):
        """Returns the PID maps"""
        return self.pid_maps
    
    #def get_pid_error(self, **kwargs):
    #    """Returns the effective areas FlavIntData object"""
    #    assert self.error_computed
    #    return self.pid_err
