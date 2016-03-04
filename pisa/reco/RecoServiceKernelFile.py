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
from pisa.utils import fileio


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
        * reco_kernel_file: file containing the kernel dict
        """
        self.kernels = None
        self.kernelfile = None
        RecoServiceBase.__init__(self, ebins, czbins,
                                 kernelfile=reco_kernel_file, **kwargs)


    def _get_reco_kernels(self, kernelfile=None, e_reco_scale=1,
                          cz_reco_scale=1, **kwargs):
        assert e_reco_scale == 1, \
                'Only e_reco_scale == 1 allowd for RecoServiceKernelFile'
        assert cz_reco_scale == 1, \
                'Only cz_reco_scale == 1 allowd for RecoServiceKernelFile'

        if not kernelfile in [self.kernelfile, None]:
            logging.info('Reconstruction from non-default kernel file %s!' %
                         kernelfile)
            return fileio.from_file(kernelfile)

        if self.kernels is None:
            logging.info('Using file %s for default reconstruction' %
                         kernelfile)
            self.kernels = fileio.from_file(kernelfile)
            self.kernelfile = kernelfile

        return self.kernels
