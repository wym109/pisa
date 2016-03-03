# author: J.L. Lanfranchi
#         jll1062+pisa@phys.psu.edu
#
# date:   March 2, 2016
#
"""
PID service for parametrized PID functions.
"""

import numpy as np
from scipy.iterpolate import interp1d

from pisa.utils import logging, set_verbosity
from pisa.pid.PIDServiceBase import PIDServiceBase
from pisa.resources.resources import find_resource
from pisa.utils.jsons import from_json
from pisa.utils.utils import get_bin_centers


class PIDServiceParam(PIDServiceBase):
    """
    Creates PID kernels by linear interpolation of sample points stored
    in a JSON file.

    Systematic parameter 'PID_offset' is supported, but 'PID_scale' has been
    temporarily removed until its implementation is corrected.

    Parameters
    ----------
    ebins, czbins
    pid_paramfile
    """
    def __init__(self, ebins, czbins, pid_paramfile, PID_offset=0, **kwargs):
        self.ebins = None
        self.czbins = None
        self.__pid_paramfile = None
        self.__param_data = None
        self.__PID_offset = None
        self.pid_kernels = None
        self.update(ebins=ebins, czbins=czbins, pid_paramfile=pid_paramfile,
                    PID_offset=PID_offset)

    def get_pid_kernels(self, pid_paramfile=None, PID_offset=None):
        return self.update(ebins=self.ebins, czbins=self.czbins,
                           pid_paramfile=pid_paramfile, PID_offset=PID_offset)


    def update(self, ebins=None, czbins=None, pid_paramfile=None,
               PID_offset=None):
        if pid_paramfile is None:
            pid_paramfile = self.__pid_paramfile
        if ebins is None:
            ebins = self.ebins
        if czbins is None:
            czbins = self.czbins

        if ebins == self.ebins and czbins == self.czbins and \
                pid_paramfile == self.__pid_paramfile and \
                PID_offset == self.__PID_offset:
            return self.pid_kernels

        self.ebins = ebins
        self.czbins = czbins

        if self.__param_data is None or pid_paramfile != self.__pid_paramfile:
            logging.info('Loading PID parametrization file %s' % pid_paramfile)
            self.__param_data = from_file(find_resource(pid_paramfile))

        ecen = self.ebins[
        n_czbins = len(self.czbins) - 1

        # Assume no variation in PID across coszen
        czvals = np.ones(n_czbins)

        pid_kernels = {'binning': {'ebins': self.ebins, 'czbins': self.czbins}}
        for signature in param_str.keys():
            # Generate the functions
            to_trck_func = eval(param_str[signature]['trck'])
            to_cscd_func = eval(param_str[signature]['cscd'])

            # Make maps from the functions evaluated at the bin centers

            # NOTE: np.clip() is to force unitarity on individual PID curves
            to_trck = np.clip(to_trck_func(ecen-PID_offset), 0, 1)
            to_cscd = np.clip(to_cscd_func(ecen-PID_offset), 0, 1)

        # Assume PID behaves same for all coszen
        to_trck_map = np.outer(to_trck, czvals)
        to_cscd_map = np.outer(to_cscd, czvals)

            pid_kernels[signature] = {'trck':to_trck_map,
                                      'cscd':to_cscd_map}

        return self.pid_kernels
