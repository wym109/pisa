#
# Base class for all PID services, handles initialization and 
# classification. 
#
# author: Lukas Schulte
#         schulte@physik.uni-bonn.de
#
# date:   Oct 21, 2014
#

from copy import deepcopy

import numpy as np

from pisa.utils.log import logging, set_verbosity
from pisa.utils import fileio, proc
from pisa.utils.utils import is_equal_binning


class PIDServiceBase(object):
    """
    Base class for all PID services, provides initialization and tools 
    for the actual classification. The method 'get_pid_kernels' (from 
    functions, MC or whatever) has to be implemented in the derived PID 
    services.
    """
    def __init__(self, ebins, czbins):
        """Store state"""
        logging.debug('Instantiating %s' % self.__class__.__name__)
        self.signatures = ['trck', 'cscd']
        self.ebins = ebins
        self.czbins = czbins
        self.input_event_rate_hash = None
        self.F_recompute_output = True
        self.pid_kernels = None

    def get_binning(self):
        return self.ebins, self.czbins

    def get_signatures(self):
        return self.signatures

    def get_pid_kernels(self, **kwargs):
        """Calculate the PID kernels"""
        kernels = self.__get_pid_kernels(**kwargs)
        self.check_pid_kernels(self.ebins, self.czbins, kernels)
        return kernels

    # TODO: clean this up: make generic, faster, and more useful
    @staticmethod
    def check_pid_kernels(ebins, czbins, pid_kernels):
        """Check that PID maps have the correct shape and are not unphysical"""
        sane = True
        if not self.pid_kernels.has_key('binning'):
            sane = False
            raise KeyError('Binning of reco kernels not specified!')

        for key, val in self.pid_kernels.items():
            # check axes
            if key == 'binning':
                for (own_ax, ax) in [(ebins, 'ebins'), 
                                     (czbins, 'czbins')]:
                    if not is_equal_binning(val[ax], own_ax):
                        sane = False
                        raise ValueError("Binning of reconstruction kernel "
                                         "doesn't match the event maps!")
            # check actual kernels
            else:
                # negative probabilities?
                for chan in ['trck', 'cscd']:
                    if (val[chan]<0).any():
                        sane = False
                        logging.warn('Negative PID probability detected! '
                                     'Check PID kernels for %s to %s'
                                      %(key, chan))
                # total ID probability >1?
                if ((val['trck']+val['cscd'])>1).any():
                    sane = False
                    logging.warn('Total PID probability larger than '
                                 'one for %s events!'%key)
        if sane:
            logging.info('PID kernels are sane')
        else:
            raise ValueError('Problem in PID kernels detected! See logfile')
        return sane

    def store_pid_kernels(self, filename):
        """Store PID maps in JSON format"""
        fileio.to_file(self.pid_kernels, filename)

    def get_pid_maps(self, reco_events, return_unknown=False, **kwargs):
        """Primary function for this service, which returns the classified
        event rate maps (sorted after tracks and cascades) from the
        reconstructed ones (sorted after nu[e,mu,tau]_cc and nuall_nc).
        """
        # Be verbose on input
        params = proc.get_params()
        proc.report_params(params, units = [])

        # Initialize return dict
        empty_map = {
            'ebins': self.ebins,
            'czbins': self.czbins,
            'map': np.zeros_like(reco_events['nue_cc']['map']),
        }

        reco_events_pid = {'params': proc.add_params(params,
                                                     reco_events['params'])}
        for sig in self.signatures:
            reco_events_pid[sig] = deepcopy(empty_map)
        if return_unknown:
            reco_events_pid['unkn'] = deepcopy(empty_map)

        # Classify events
        for flav in reco_events:
            if flav == 'params':
                continue
            event_map = reco_events[flav]['map']
            if return_unknown:
                reco_events_pid['unkn']['map'] += event_map

            for sig in self.signatures:
                if return_unknown:
                    temp_sig_total = event_map * self.pid_kernels[flav][sig]
                    reco_events_pid['unkn']['map'] -= temp_sig_total
                    reco_events_pid[sig]['map'] += temp_sig_total
                else:
                    reco_events_pid[sig]['map'] += event_map * \
                            self.pid_kernels[flav][sig]

        return reco_events_pid

    @staticmethod
    def add_argparser_args(parser):
        return parser
