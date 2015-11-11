#
# Creates the effective areas from a simfile, then returns them
# re-scaled as desired.
#
# author: Timothy C. Arlen
# author: Justin L. Lanfranchi
#
# rev0:   2014-04-08
# rev1:   2015-11-06

import numpy as np
from pisa.utils.log import logging, set_verbosity
from pisa.utils.utils import get_bin_centers, get_bin_sizes
import pisa.utils.flavInt as flavInt
import pisa.utils.events as EVENTS


class AeffServiceMC:
    '''
    Takes the weighted effective area files, and creates a dictionary
    of the 2D effective area in terms of energy and coszen, for each
    flavor (nue,nue_bar,numu,...) and interaction type (CC, NC)
    '''

    def __init__(self, ebins, czbins, aeff_weight_file, **kwargs):
        self.ebins = ebins
        self.czbins = czbins
        logging.info('Initializing AeffServiceMC...')

        logging.info('Extracting events from file: %s' % (aeff_weight_file))
        events = EVENTS.Events(aeff_weight_file)

        self.aeff_dict = flavInt.FIData()
        logging.info("Creating effective area dict...")
        for kind in flavInt.ALL_KINDS:
            logging.debug("Working on %s effective areas" % kind)
            bins = (self.ebins, self.czbins)
            aeff_hist,_,_ = np.histogram2d(
                events.get(kind, 'true_energy'),
                events.get(kind, 'true_coszen'),
                weights=events.get(kind, 'weighted_aeff'),
                bins=bins
            )
            # Divide histogram by bin ExCZ "widths" to convert to aeff
            ebin_sizes = get_bin_sizes(ebins)
            czbin_sizes = 2.0*np.pi*get_bin_sizes(czbins)
            bin_sizes = np.meshgrid(czbin_sizes, ebin_sizes)
            aeff_hist /= np.abs(bin_sizes[0]*bin_sizes[1])
            # Save the result to the FIData object
            self.aeff_dict.set(kind, aeff_hist)

    def get_aeff(self, **kwargs):
        '''
        Returns the effective area dictionary
        '''
        return self.aeff_dict
