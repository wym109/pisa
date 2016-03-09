#
# author: Timothy C. Arlen
#
# date:   April 8, 2014
"""
Effective areas from a PISA events HDF5 file.
"""

import numpy as np

from pisa.utils.log import logging
from pisa.utils.utils import get_bin_sizes
import pisa.utils.flavInt as flavInt
import pisa.utils.events as events


class AeffServiceMC:
    """
    Takes a PISA events HDF5 file (which includes the 'weighted_aeff' field)
    and creates 2D-histogrammed effective areas in terms of energy and coszen,
    for each flavor (nue, nue_bar, numu, ...) and interaction type (CC, NC)
    """
    def __init__(self, ebins, czbins, aeff_weight_file, compute_error=False,
                 **kwargs):
        self.ebins = None
        self.czbins = None
        self.__error_computed = False
        self.__aeff_weight_file = None
        self.__compute_error = compute_error
        self.events = None
        self.update(ebins=ebins, czbins=czbins,
                    aeff_weight_file=aeff_weight_file,
                    compute_error=compute_error)

    def update(self, ebins, czbins, aeff_weight_file=None, compute_error=None):
        if aeff_weight_file is None:
            aeff_weight_file = self.__aeff_weight_file
        if compute_error is None:
            compute_error = self.__compute_error
        if np.all(ebins == self.ebins) and np.all(czbins == self.czbins) and \
                aeff_weight_file == self.__aeff_weight_file and \
                (not compute_error or (compute_error == self.__compute_error)):
            return
        self.ebins = ebins
        self.czbins = czbins
        self.__compute_error = compute_error
        logging.info('Updating AeffServiceMC aeffs...')

        if self.events is None or aeff_weight_file != self.__aeff_weight_file:
            logging.info('Extracting events from file: %s' % aeff_weight_file)
            self.events = events.Events(aeff_weight_file)
            self.__aeff_weight_file = aeff_weight_file

        self.__aeff = flavInt.FlavIntData()
        self.__aeff_err = flavInt.FlavIntData()
        logging.info("Populating effective areas...")
        for flavint in flavInt.ALL_NUFLAVINTS:
            logging.debug("Computing %s effective areas" % flavint)
            bins = (self.ebins, self.czbins)
            true_e = self.events[flavint]['true_energy']
            true_cz = self.events[flavint]['true_coszen']
            weights = self.events[flavint]['weighted_aeff']
            aeff_hist, _, _ = np.histogram2d(
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

            # Divide histogram by bin ExCZxAZ "widths" to convert to aeff
            ebin_sizes = get_bin_sizes(self.ebins)
            # Note that the following includes the azimuth angle bin size (the
            # 2pi factor since we use a single azimuth "bin")
            solidangle_bin_sizes = 2.0*np.pi*get_bin_sizes(self.czbins)
            binsize_normfact = 1./np.outer(ebin_sizes, solidangle_bin_sizes)
            aeff_hist *= binsize_normfact
            self.__aeff[flavint] = aeff_hist

            if self.__compute_error:
                aeff_err = aeff_hist / np.sqrt(bin_counts)
                self.__aeff_err[flavint] = aeff_err
                self.__error_computed = True

    def get_aeff(self, **kwargs):
        """Returns the effective areas FlavIntData object"""
        return self.__aeff
    
    def get_aeff_with_error(self, **kwargs):
        """Returns the effective areas FlavIntData object"""
        assert self.__error_computed
        return self.__aeff, self.__aeff_err

    @staticmethod
    def add_argparser_args(parser):
        parser.add_argument(
            '--aeff-weight-file', metavar='RESOURCE', type=str,
            default='events/pingu_v36/'
            'events__pingu__v36__runs_388-390__proc_v5__joined_G_nuall_nc_G_nuallbar_nc.hdf5',
            help='''PISA-standard events file, the events of which will be used
            to compute effective areas.'''
        )
