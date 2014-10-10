# -*- coding: utf-8 -*-
#
#  ParamRecoService.py
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

import numpy as np

from pisa.reco.RecoServiceBase import RecoServiceBase
from pisa.resources.resources import find_resource
from pisa.utils.jsons import from_json
from pisa.utils.utils import get_bin_centers


class ParamRecoService(RecoServiceBase):
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
            self.parametrization = self.read_param_string(param_str)
        except IOError, e:
            logging.error("Unable to open parametrization file %s"
                          %paramfile)
            logging.error(e)
            sys.exit(1)
        
        # Scale reconstruction widths
        self.apply_reco_scales(energy_reco_scale, coszen_reco_scale)
        
        logging.debug('Creating reconstruction kernels')
        self.calculate_kernels()
        
        return self.kernels
    
    
    def read_param_string(self, param_str):
        """
        Parse the dict with the parametrization strings and evaluate for 
        the bin energies needed.
        """
        
        evals = get_bin_centers(self.ebins)
        n_e = len(self.ebins)-1
        n_cz = len(self.czbins)-1
        
        parametrization = dict.fromkeys(param_str)
        for channel in param_str:
            logging.debug('Parsing function strings for %s'%channel)
            for axis in param_str[channel]:    #['energy', 'coszen']
                parameters = {}
                for par, funcstring in param_str[channel][axis].items():
                    # this should contain a lambda function:
                    function = eval(funcstring)
                    # evaluate the function at the given energies
                    vals = function(evals)
                    # repeat for all cos(zen) bins
                    parameters[par] = np.repeat(vals,n_cz).reshape((n_e,n_cz))
                parametrization[channel][axis] = parameters
        
        self.parametrization = parametrization
        return parametrization


    def apply_reco_scales(self, e_scale, cz_scale):
        """
        Widen the gaussians used for reconstruction by the given factors
        """
        
        for axis, scale in [('energy', e_scale), ('coszen', cz_scale)]:
            
            if scale==1.:
                continue
            
            logging.debug('Scaling %s reco precision by factor %.2f'
                          %(axis, scale))
            
            for channel in self.parametrization:
                for param in ['width1', 'width2']: #the widths of the gaussians
                    self.parametrization[channel][axis][param] *= scale
    
    
    def calculate_kernels():
        """
        Use the parametrization functions to calculate the actual reco 
        kernels (i.e. 4D histograms).
        """
        
        #TODO: implement this
        self.kernels = None
