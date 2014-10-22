#
# PID service for parametrized PID functions. 
#
# author: Lukas Schulte
#         schulte@physik.uni-bonn.de
#
# date:   Oct 21, 2014
#

import logging

import numpy as np
import scipy.stats

from pisa.pid.PIDServiceBase import PIDServiceBase
from pisa.resources.resources import find_resource
from pisa.utils.jsons import from_json
from pisa.utils.utils import get_bin_centers


class ParamPIDService(PIDServiceBase):
    """
    Creates PID kernels from parametrization functions that are stored 
    in a JSON dict. numpy is accessible as np, and scipy.stats. 
    Systematic parameters 'PID_offset' and 'PID_scale' are supported.
    """
    
    def __init__(self, ebins, czbins, **kwargs):
        """
        Parameters needed to initialize a PID service with parametrizations:
        * ebins: Energy bin edges
        * czbins: cos(zenith) bin edges
        * pid_paramfile: JSON containing the parametrizations
        """
        PIDServiceBase.__init__(self, ebins, czbins, **kwargs)


    def get_pid_kernels(self, pid_paramfile=None, 
                        PID_offset=0., PID_scale=1., **kwargs):
        
        # load parametrization file
        logging.info('Opening PID parametrization file %s'%pid_paramfile)
        try:
            param_str = from_json(find_resource(pid_paramfile))
        except IOError, e:
            logging.error("Unable to open PID parametrization file %s"
                          %pid_paramfile)
            logging.error(e)
            sys.exit(1)
        
        ecen = get_bin_centers(self.ebins)
        czcen = get_bin_centers(self.czbins)
        
        self.pid_kernels = {'binning': {'ebins': self.ebins,
                                        'czbins': self.czbins}}
        for signature in param_str.keys():
            #Generate the functions
            to_trck_func = eval(param_str[signature]['trck'])
            to_cscd_func = eval(param_str[signature]['cscd'])

            #Make maps from the functions evaluated at the bin centers
            _,to_trck_map = np.meshgrid(czcen, PID_scale*to_trck_func(ecen-PID_offset))
            _,to_cscd_map = np.meshgrid(czcen, PID_scale*to_cscd_func(ecen-PID_offset))

            self.pid_kernels[signature] = {'trck':to_trck_map,
                                           'cscd':to_cscd_map}

        return self.pid_kernels
