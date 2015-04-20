# -*- coding: utf-8 -*-
#
#  RecoServiceKernelFile.py
#
# Loads a pre-calculated dict of reconstruction kernels stored in json format.
#
# author: Lukas Schulte
#         schulte@physik.uni-bonn.de
#
# date:   August 20, 2014
#


import sys
import logging

from pisa.reco.RecoServiceBase import RecoServiceBase
from pisa.resources.resources import find_resource
from pisa.utils import utils


class RecoServiceKernelFile(RecoServiceBase):
    """
    Loads a pre-calculated reconstruction kernel (that has been saved via
    reco_service.store_kernels) from disk and uses that for reconstruction.
    """
    def __init__(self, ebins, czbins, reco_kernel_file=None, **kwargs):
        """
        Parameters needed to instantiate a reconstruction service with
        pre-calculated kernels:
        * ebins: Energy bin edges
        * czbins: cos(zenith) bin edges
        * reco_kernel_file: JSON containing the kernel dict
        """
        self.kernelfile = reco_kernel_file
        RecoServiceBase.__init__(self, ebins, czbins,
                                 kernelfile=reco_kernel_file, **kwargs)


    def _get_reco_kernels(self, kernelfile=None, **kwargs):

        for reco_scale in ['e_reco_scale', 'cz_reco_scale']:
            if reco_scale in kwargs:
                if not kwargs[reco_scale]==1:
                    raise ValueError('%s = %.2f not valid for RecoServiceKernelFile!'
                                     %(reco_scale, kwargs[reco_scale]))

        if not kernelfile in [self.kernelfile, None]:
            logging.info('Reconstruction from non-default kernel file %s!'%kernelfile)
            return utils.from_file(find_resource(kernelfile))

        if not hasattr(self, 'kernels'):
            logging.info('Using file %s for default reconstruction'%(kernelfile))
            self.kernels = utils.from_file(find_resource(kernelfile))

        return self.kernels
