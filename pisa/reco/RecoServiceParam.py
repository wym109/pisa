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
import itertools

import numpy as np
from scipy.stats import norm

from pisa.reco.RecoServiceBase import RecoServiceBase
from pisa.resources.resources import find_resource
from pisa.utils.jsons import from_json
from pisa.utils.utils import get_bin_centers, get_bin_sizes, is_linear



def double_gauss(x, loc1=0, width1=1.e-6, loc2=0., width2=1.e-6, fraction=0.):
    """Superposition of two gaussians"""
    
    return (1.-fraction)*norm(loc=loc1, scale=width1).pdf(x)\
             + (fraction)*norm(loc=loc2, scale=width2).pdf(x)



class RecoServiceParam(RecoServiceBase):
    """
    Creates reconstruction kernels from a parametrization that is stored 
    as a json dict. The parametrizations are assumed to be double 
    Gaussians in both energy and cos(zenith), shape only depending on 
    the true energy (not the direction!). Systematic parameters 
    'e_reco_scale' and 'cz_reco_scale are supported.'
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
                         e_reco_scale=1., cz_reco_scale=1., 
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
        self.apply_reco_scales(e_reco_scale, cz_reco_scale)
        
        logging.info('Creating reconstruction kernels')
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
        
        parametrization = dict.fromkeys(param_str, {'cc': {}, 'nc': {}})
        for flavour in param_str:
          for int_type in ['cc', 'nc']:
            logging.debug('Parsing function strings for %s %s'
                          %(flavour, int_type))
            for axis in param_str[flavour][int_type]:    #['energy', 'coszen']
                parameters = {}
                for par, funcstring in param_str[flavour][int_type][axis].items():
                    # this should contain a lambda function:
                    function = eval(funcstring)
                    # evaluate the function at the given energies
                    vals = function(evals)
                    # repeat for all cos(zen) bins
                    parameters[par] = np.repeat(vals,n_cz).reshape((n_e,n_cz))
                parametrization[flavour][int_type][axis] = parameters
        
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
            
            for flavour in self.parametrization:
              for int_type in ['cc', 'nc']:
                for param in ['width1', 'width2']: #the widths of the gaussians
                    self.parametrization[flavour][int_type][axis][param] *= scale
    
    
    def calculate_kernels(self, flipback=True):
        """
        Use the parametrization functions to calculate the actual reco 
        kernels (i.e. 4D histograms). If flipback==True, the zenith angle 
        part that goes below the zenith will be mirrored back in.
        """
        
        # get binning information
        evals, esizes = get_bin_centers(self.ebins), get_bin_sizes(self.ebins)
        czvals, czsizes = get_bin_centers(self.czbins), get_bin_sizes(self.czbins)
        n_e, n_cz = len(evals), len(czvals)
        
        # prepare for folding back at lower edge
        if flipback:
            if is_linear(self.czbins):
                czvals = np.append(czvals-(self.czbins[-1] \
                                            - self.czbins[0]), 
                                   czvals)
                czsizes = np.append(czsizes, czsizes)
            else:
                logging.warn("cos(zenith) bins have different "
                             "sizes! Unable to fold around edge "
                             "of histogram, will not do that.")
                flipback = False
        
        kernel_dict = dict.fromkeys(self.parametrization, 
                                    {'cc': None, 'nc': None})
        
        for flavour in self.parametrization:
          for int_type in ['cc', 'nc']:
            logging.debug('Calculating parametrized reconstruction kernel for %s %s'
                          %(flavour, int_type))
            
            # create empty kernel
            kernel = np.zeros((n_e, n_cz, n_e, n_cz))
            
            # quick handle to parametrization
            e_pars = self.parametrization[flavour][int_type]['energy']
            cz_pars = self.parametrization[flavour][int_type]['coszen']
            
            # loop over every bin in true (energy, coszen)
            for (i, j) in itertools.product(range(n_e), range(n_cz)):
                
                e_kern = double_gauss(evals, 
                                     loc1=e_pars['loc1'][i,j]+evals[i], 
                                     width1=e_pars['width1'][i,j], 
                                     loc2=e_pars['loc2'][i,j]+evals[i], 
                                     width2=e_pars['width2'][i,j], 
                                     fraction=e_pars['fraction'][i,j])
                
                offset = n_cz if flipback else 0
                cz_kern = double_gauss(czvals, 
                                     loc1=cz_pars['loc1'][i,j]+czvals[j+offset], 
                                     width1=cz_pars['width1'][i,j], 
                                     loc2=cz_pars['loc2'][i,j]+czvals[j+offset], 
                                     width2=cz_pars['width2'][i,j], 
                                     fraction=cz_pars['fraction'][i,j])
                
                if flipback:
                    # fold back
                    cz_kern = cz_kern[:len(czvals)/2][::-1] \
                                + cz_kern[len(czvals)/2:]
                
                kernel[i,j] = np.outer(e_kern, cz_kern)
            
            kernel_dict[flavour][int_type] = kernel
        
        kernel_dict['ebins'] = self.ebins
        kernel_dict['czbins'] = self.czbins
        
        self.kernels = kernel_dict
        return self.kernels


