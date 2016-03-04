#
# PID service for pre-calculated PID tables. 
#
# author: Lukas Schulte
#         schulte@physik.uni-bonn.de
# author: J.L. Lanfranchi
#         jll1062+pisa@phys.psu.edu
#
# date:   Oct 21, 2014
#

from pisa.utils.log import logging

from pisa.pid.PIDServiceBase import PIDServiceBase
from pisa.utils import fileio


class PIDServiceKernelFile(PIDServiceBase):
    """Loads a pre-calculated PID kernel (that has been saved via
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
        super(PIDServiceKernelFile, self).__init__()
        self.__pid_kernelfile = None
        if 'pid_kernelfile' in kwargs:
            self.get_pid_kernels(**kwargs)

    def get_pid_kernels(self, pid_kernelfile, force_reload=False, **kwargs):
        if not force_reload and pid_kernelfile == self.__pid_kernelfile:
            return self.__pid_kernels
        self.__pid_kernels = fileio.from_file(pid_kernelfile)
        self.__pid_kernelfile = pid_kernelfile
        return self.__pid_kernels

    @staticmethod
    def add_argparser_args(parser):
        parser.add_argument(
            '--kernel_file', metavar='JSON', type=str, default=None,
            help='[ PID-Kernel ] JSON file containing pre-calculated PID kernels'
        )
        return parser
