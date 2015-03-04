#
# Creates the pid functions from the pid input files, using
# parameterizations of the PID as a function of reconstructed energy.
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# date:   April 10, 2014
#

import sys
import numpy as np
import scipy.stats
from pisa.utils.log import logging
from pisa.utils.utils import get_bin_centers

class PIDServicePar:
    '''
    Create the PID maps for each flavor from the parametrized functions, and
    provide them for the PID stage.

    '''
    def __init__(self,ebins,czbins,particle_ID=None,**kwargs):

        logging.info('Initializing PIDServicePar...')

        #Evaluate the functions at the bin centers
        ecen = get_bin_centers(ebins)
        czcen = get_bin_centers(czbins)

        self.pid_maps = {}
        for signature in particle_ID.keys():
            logging.debug('Getting pid for type: %s'%signature)
            #Generate the functions
            to_trck_func = eval(particle_ID[signature]['trck'])
            to_cscd_func = eval(particle_ID[signature]['cscd'])

            #Make maps from the functions evaluated at the bin centers
            _,to_trck_map = np.meshgrid(czcen, to_trck_func(ecen))
            _,to_cscd_map = np.meshgrid(czcen, to_cscd_func(ecen))

            #for label,pidmap in [('Track',to_trck_map),('Cascade',to_cscd_map)]:
            #    if (pidmap < 0).any():
            #        raise ValueError('%s PID probabilites can not be negative!'
            #            ' Investigate parameterization'%label)

            self.pid_maps[signature] = {'trck':to_trck_map,
                                        'cscd':to_cscd_map}

    def get_maps(self):
        '''
        Return the pid functions.
        '''
        return self.pid_maps
