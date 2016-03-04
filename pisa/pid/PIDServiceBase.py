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

from pisa.utils.logging import logging, set_verbosity
from pisa.utils.jsons import to_json
from pisa.utils.utils import is_equal_binning


class PIDServiceBase:
    """
    Base class for all PID services, provides initialization and tools 
    for the actual classification. The method 'get_pid_maps' (from 
    functions, MC or whatever) has to be implemented in the derived PID 
    services.
    """
    def __init__(self, ebins, czbins, **kwargs):
        """
        Parameters needed to instantiate any reconstruction service:
        * ebins: Energy bin edges
        * czbins: cos(zenith) bin edges
        If further member variables are needed, override this method.
        """
        logging.debug('Instantiating %s'%self.__class__.__name__)

        # Set member variables
        self.ebins = ebins
        self.czbins = czbins

        # Calculate PID kernel
        self.get_pid_kernels(**kwargs)
        self.check_pid_kernels()


    def get_pid_kernels(self, **kwargs):
        """
        Calculates the actual PID kernels, implement this in derived 
        services!
        """
        raise NotImplementedError('Method not implemented for %s'
                                    %self.__class__.__name__)


    def check_pid_kernels(self):
        """
        Check that PID maps have the correct shape and are not unphysical
        """
        sane = True
        if not self.pid_kernels.has_key('binning'):
            sane = False
            raise KeyError('Binning of reco kernels not specified!')

        for key, val in self.pid_kernels.items():

            #check axes
            if key=='binning':
                for (own_ax, ax) in [(self.ebins, 'ebins'), 
                                     (self.czbins, 'czbins')]:
                    if not is_equal_binning(val[ax], own_ax):
                        sane = False
                        raise ValueError("Binning of reconstruction kernel "
                                         "doesn't match the event maps!")

            #check actual kernels
            else:
                #negative probabilities?
                for chan in ['trck', 'cscd']:
                    if (val[chan]<0).any():
                        sane = False
                        logging.warn('Negative PID probability detected! '
                                     'Check PID kernels for %s to %s'
                                      %(key, chan))
                #total ID probability >1?
                if ((val['trck']+val['cscd'])>1).any():
                    sane = False
                    logging.warn('Total PID probability larger than '
                                 'one for %s events!'%key)

        if sane:
            logging.info('PID kernels are sane')
        else:
            logging.warn('Problem in PID kernels detected! See logfile')
        return sane


    def recalculate_kernels(self, **kwargs):
        """
        Re-calculate PID maps and check for sanity. If new maps are 
        corrupted, stick with the old ones.
        """
        logging.info('Re-calculating PID maps')
        old_kernels = deepcopy(self.pid_kernels)
        self.get_pid_kernels(**kwargs)
        try:
            self.check_pid_kernels()
        except:
            logging.error('Failed to recalculate PID maps, '
                          'keeping old ones: ', exc_info=True)
            self.pid_kernels = old_kernels


    def store_pid_kernels(self, filename):
        """
        Store PID maps in JSON format
        """
        to_json(self.pid_kernels, filename)
