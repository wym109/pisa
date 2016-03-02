#
# PID service for parametrized PID functions.
#
# author: Lukas Schulte
#         schulte@physik.uni-bonn.de
#
#         Timothy C. Arlen
#         tca3@psu.edu
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


class PIDServiceParam(PIDServiceBase):
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
        PIDServiceBase.__init__(self, ebins, czbins, pid_paramfile, **kwargs)

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
            raise

        ecen = get_bin_centers(self.ebins)
        czcen = get_bin_centers(self.czbins)

        self.pid_kernels = {'binning': {'ebins': self.ebins,
                                        'czbins': self.czbins}}
        for signature in param_str.keys():
            #Generate the functions
            to_trck_func = eval(param_str[signature]['trck'])
            to_cscd_func = eval(param_str[signature]['cscd'])

            # Make maps from the functions evaluated at the bin centers
            #
            # NOTE: np.where() is to catch the low energy nutau events
            # that are undefined. Often what happens is that the nutau
            # parameterization for trck events will drop below 0.0 at
            # low energies, but there are no nutau events at these
            # energies anyway, so we just set them to zero (and cscd =
            # 1.0) if the condition arises.
            to_trck = to_trck_func(ecen-PID_offset)
            to_cscd = to_cscd_func(ecen-PID_offset)
            to_trck = np.where(to_trck < 0.0,0.0,
                               np.where(to_trck > 1.0,1.0,to_trck))
            to_cscd = np.where(to_cscd < 0.0,0.0,
                               np.where(to_cscd > 1.0,1.0,to_cscd))
            _,to_trck_map = np.meshgrid(czcen, PID_scale*to_trck)
            _,to_cscd_map = np.meshgrid(czcen, PID_scale*to_cscd)

            self.pid_kernels[signature] = {'trck':to_trck_map,
                                           'cscd':to_cscd_map}

        return self.pid_kernels
