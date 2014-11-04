# -*- coding: utf-8 -*-
#
#  KernelFileRecoService.py
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
from pisa.utils.jsons import from_json


class KernelFileRecoService(RecoServiceBase):
    """
    Loads a pre-calculated reconstruction kernel (that has been saved via 
    reco_service.store_kernels) from disk and uses that for reconstruction.
    """
    def __init__(self, ebins, czbins, **kwargs):
        """
        Parameters needed to instantiate a reconstruction service with 
        pre-calculated kernels:
        * ebins: Energy bin edges
        * czbins: cos(zenith) bin edges
        * kernelfile: JSON containing the kernel dict
        """
        RecoServiceBase.__init__(self, ebins, czbins, **kwargs)


    def get_reco_kernels(self, kernelfile=None, **kwargs):
        
        logging.info('Opening file: %s'%(kernelfile))
        try:
            self.kernels = from_json(find_resource(kernelfile))
        except IOError, e:
            logging.error("Unable to open kernel file %s"%kernelfile)
            logging.error(e)
            sys.exit(1)
        
        return self.kernels
