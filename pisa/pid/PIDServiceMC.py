#
# author: J.L. Lanfranchi
#
# date:   March 2, 2016
"""
PID histograms from a PISA events HDF5 file.
"""

import collections

import numpy as np

from pisa.utils.log import logging
from pisa.utils.utils import get_bin_sizes
import pisa.utils.flavInt as flavInt
from pisa.utils.events import Events


class PIDServiceMC:
    """
    Takes a PISA events HDF5 file and creates 2D-histogrammed PID in terms of
    energy and coszen, for each specified particle "signature".
    """
    def __init__(self, ebins, czbins, events, compute_error=False, **kwargs):
        self.__error_computed = False
        self.__ebins = None
        self.__czbins = None
        self.__events_source = None
        self.__events = None
        self.__compute_error = False
        self.update(ebins=ebins, czbins=czbins, events=events,
                    compute_error=compute_error)

    def update(self, ebins, czbins, events, compute_error):
        if ebins == self.__ebins and czbins == self.__czbins and \
                events == self.__events_source and \
                (not compute_error or (compute_error == self.__compute_error)):
            return
        self.__ebins = ebins
        self.__czbins = czbins
        self.__compute_error = compute_error
        logging.info('Updating PIDServiceMC PID histograms...')

        if self.__events is None or events != self.__events_source:
            if isinstance(events, basestring):
                logging.info('Extracting events from file: %s' % (events))
                self.__events = Events(events)
            elif isinstance(events, collections.Mapping):
                # Validate by (re)instantiating as an Events object
                self.__events = Events(events)
            else:
                raise TypeError('Unhandled `events` type: "%s"' % type(events))
            self.__events_source = events

        self.__pid = flavInt.FlavIntData()
        self.__pid_err = flavInt.FlavIntData()
        logging.info("Populating effective areas...")
        for flavint in flavInt.ALL_NUFLAVINTS:
            logging.debug("Computing %s effective areas" % flavint)
            bins = (self.__ebins, self.__czbins)
            true_e = self.__events[flavint]['true_energy']
            true_cz = self.__events[flavint]['true_coszen']
            try:
                weights = self.__events[flavint]['importance_weight']
            except:
                logging.warn('No importance weights found in events!')
                weights = None
            pid_hist, _, _ = np.histogram2d(
                true_e,
                true_cz,
                weights=weights,
                bins=bins
            )
            if self.__compute_error:
                bin_counts, _, _ = np.histogram2d(
                    true_e,
                    true_cz,
                    weights=None,
                    bins=bins
                )

            # Divide histogram by bin ExCZxAZ "widths" to convert to pid
            ebin_sizes = get_bin_sizes(self.__ebins)
            # Note that the following includes the azimuth angle bin size (the
            # 2pi factor since we use a single azimuth "bin")
            solidangle_bin_sizes = 2.0*np.pi*get_bin_sizes(self.__czbins)
            binsize_normfact = 1./np.outer(ebin_sizes, solidangle_bin_sizes)
            pid_hist *= binsize_normfact
            self.__pid[flavint] = pid_hist

            if self.__compute_error:
                pid_err = pid_hist / np.sqrt(bin_counts)
                self.__pid_err[flavint] = pid_err
                self.__error_computed = True

    def get_pid(self, **kwargs):
        """Returns the effective areas FlavIntData object"""
        return self.__pid
    
    def get_pid_with_error(self, **kwargs):
        """Returns the effective areas FlavIntData object"""
        assert self.__error_computed
        return self.__pid, self.__pid_err
