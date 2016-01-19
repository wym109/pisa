#
# Creates the effective areas from a simfile, then returns them
# re-scaled as desired.
#
# author: Timothy C. Arlen
#
# date:   April 8, 2014
#

import numpy as np
from pisa.utils.log import logging, set_verbosity
from pisa.utils.utils import get_bin_centers, get_bin_sizes
import pisa.utils.flavInt as flavInt
import pisa.utils.events as events


class AeffServiceMC:
    """
    Takes the weighted effective area files, and creates a dictionary
    of the 2D effective area in terms of energy and coszen, for each
    flavor (nue,nue_bar,numu,...) and interaction type (CC, NC)
    """

    def __init__(self, ebins, czbins, aeff_weight_file, **kwargs):
        self.ebins = ebins
        self.czbins = czbins
        logging.info('Initializing AeffServiceMC...')

        logging.info('Extracting events from file: %s' % (aeff_weight_file))
        evts = events.Events(aeff_weight_file)

        self.aeff_dict = flavInt.FlavIntData()
        logging.info("Creating effective area dict...")
        for flavint in flavInt.ALL_NUFLAVINTS:
            logging.debug("Working on %s effective areas" % flavint)
            bins = (self.ebins, self.czbins)
            aeff_hist,_,_ = np.histogram2d(
                evts.get(flavint, 'true_energy'),
                evts.get(flavint, 'true_coszen'),
                weights=evts.get(flavint, 'weighted_aeff'),
                bins=bins
            )
            # Divide histogram by bin ExCZ "widths" to convert to aeff
            ebin_sizes = get_bin_sizes(ebins)
            czbin_sizes = 2.0*np.pi*get_bin_sizes(czbins)
            bin_sizes = np.meshgrid(czbin_sizes, ebin_sizes)
            aeff_hist /= np.abs(bin_sizes[0]*bin_sizes[1])
            # Save the result to the FlavIntData object
            self.aeff_dict.set(flavint, aeff_hist)

    def get_aeff(self, **kwargs):
        """Returns the effective area dictionary"""
        return self.aeff_dict
