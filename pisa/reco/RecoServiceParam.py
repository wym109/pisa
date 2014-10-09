# -*- coding: utf-8 -*-
#
#  RecoServiceParam.py
# 
# Creates reconstruction kernels from a parametrization
#
# author: Lukas Schulte
#         schulte@physik.uni-bonn.de
#
# date:   October 9, 2014
#

import sys
import logging

from pisa.reco.RecoServiceBase import RecoServiceBase
from pisa.resources.resources import find_resource
from pisa.utils.jsons import from_json


def read_param_string(param_str):
    
    parametrization = dict.fromkeys(param_str)
    for channel in param_str:
        logging.debug('Parsing function strings for %s'%channel)
        for axis in ['energy', 'coszen']:
            #TODO: implement the actual parsing
            pass


class RecoServiceParam(RecoServiceBase):
    """
    Creates reconstruction kernels from a parametrization that is stored 
    as a json dict. The parametrizations are assumed to be double 
    Gaussians in both energy and cos(zenith), shape only depending on 
    the true energy (not the direction!). Systematic parameters 
    'energy_reco_scale' and 'coszen_reco_scale are supported.'
    """
    def __init__(self, ebins, czbins, **kwargs):
        """
        Parameters needed to instantiate a reconstruction service with 
        parametrizations:
        * ebins: Energy bin edges
        * czbins: cos(zenith) bin edges
        * paramfile: JSON containing the parametrizations
        """
        RecoServiceBase.__init__(self, ebins, czbins, **kwargs)
 
 
    def get_reco_kernels(self, paramfile=None, 
                         energy_reco_scale=1., coszen_reco_scale=1., 
                         **kwargs):

        logging.info('Opening reconstruction parametrization file %s'
                     %paramfile)
        try:
            param_str = from_json(find_resource(paramfile))
            self.parametrization = read_param_string(param_str)
        except IOError, e:
            logging.error("Unable to open parametrization file %s"
                          %paramfile)
            logging.error(e)
            sys.exit(1)
        
        logging.debug('''Scaling reconstruction resolutions
                         \n   energy: %.2f\n   coszen: %.2f'''
                         %(energy_reco_scale, coszen_reco_scale))
        self.apply_reco_scales(energy_reco_scale, coszen_reco_scale)
        
        logging.debug('Creating reconstruction kernels')
        self.calculate_kernels()
        
        return self.kernels
    
    
    def apply_reco_scales(self, e_scale, cz_scale):
        """
        Widen the gaussians used for reconstruction by the given factors
        """
        #TODO: implement
        #FIXME: actually possible here or does this have to happen while parsing?
        pass
    
    
    def calculate_kernels():
        """
        Use the parametrization functions to calculate the actual reco 
        kernels (i.e. 4D histograms).
        """
        #TODO: implement this
        self.kernels = None
