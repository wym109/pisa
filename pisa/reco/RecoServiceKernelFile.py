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
from pisa.utils.utils import DictWithHash


class RecoServiceKernelFile(RecoServiceBase):
    """Loads a pre-calculated reconstruction kernel (that has been saved via
    reco_service.store_kernels) from disk and uses that for reconstruction.

    Parameters
    ----------
    ebins, czbins : array_like
        Energy and coszen bin edges
    reco_kernel_file : string
        file containing the kernel dict
    """
    def __init__(self, ebins, czbins, reco_kernel_file=None, **kwargs):
        super(RecoServiceKernelFile, self).__init__(ebins, czbins)
        self.kernels = None
        self.reco_kernel_file = None
        self.kernels = self.get_reco_kernels(reco_kernel_file=reco_kernel_file,
                                             **kwargs)

    def _get_reco_kernels(self, reco_kernel_file=None, e_reco_scale=1,
                          cz_reco_scale=1, **kwargs):
        assert e_reco_scale == 1, \
                'Only e_reco_scale == 1 allowd for RecoServiceKernelFile'
        assert cz_reco_scale == 1, \
                'Only cz_reco_scale == 1 allowd for RecoServiceKernelFile'

        if self.kernels is None or \
                not reco_kernel_file in [self.reco_kernel_file, None]:
            #logging.info('Using file %s for default reconstruction' %
            #             reco_kernel_file)
            self.kernels = DictWithHash(fileio.from_file(reco_kernel_file))
            #self.kernels.update_hash()
            self.reco_kernel_file = reco_kernel_file
        else:
            self.kernels.is_new = False

        return self.kernels
