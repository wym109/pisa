#
# Creates the pid functions from the pid input files, using
# parameterizations of the PID as a function of reconstructed energy.
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# date:   April 10, 2014
#

import scipy
import numpy as np
from utils.utils import get_bin_centers

class PIDService:
    '''
    Creates the pid functions for each flavor and then performs the
    pid selection on each template.
    '''
    def __init__(self,pid_data,**kwargs):

        #self.ebins = ebins
        #self.czbins = czbins
        self.pid_func_dict = {}
        for signature in pid_data.keys():
            to_trck_func = eval(pid_data[signature]['trck'])
            to_cscd_func = eval(pid_data[signature]['cscd'])
            self.pid_func_dict[signature] = {'trck':to_trck_func,
                                             'cscd':to_cscd_func}
            
        return
    
    def get_pid_funcs(self):
        '''
        Return the pid functions.
        '''
        return self.pid_func_dict
