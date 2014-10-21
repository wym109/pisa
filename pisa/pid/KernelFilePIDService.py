#
# PID service for pre-calculated PID tables. 
#
# author: Lukas Schulte
#         schulte@physik.uni-bonn.de
#
# date:   Oct 21, 2014
#

import sys
import logging

from pisa.pid.PIDServiceBase import PIDServiceBase
from pisa.resources.resources import find_resource
from pisa.utils.jsons import from_json


class KernelFilePIDService(PIDServiceBase):
    """
    Loads a pre-calculated PID kernel (that has been saved via
    pid_service.store_kernels) from disk and uses that for classification.
    """
    def __init__(self, ebins, czbins, **kwargs):
        """
        Parameters needed to instantiate a PID service with
        pre-calculated kernels:
        * ebins: Energy bin edges
        * czbins: cos(zenith) bin edges
        * pid_kernelfile: JSON containing the kernel dict
        """
        RecoServiceBase.__init__(self, ebins, czbins, **kwargs)


    def get_reco_kernels(self, pid_kernelfile=None, **kwargs):
        logging.info('Opening file: %s'%(pid_kernelfile))
        try:
            self.pid_kernels = from_json(find_resource(pid_kernelfile))
        except IOError, e:
            logging.error("Unable to open kernel file %s"%pid_kernelfile)
            logging.error(e)
            sys.exit(1)
        return self.pid_kernels
