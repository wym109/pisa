#
# Base class for reconstruction services, handles the actual smearing 
# of events from the reco kernels. Kernel generation has to be implemented
# in the derived classes.
# 
#
# author: Lukas Schulte
#         schulte@physik.uni-bonn.de
#
# date:   August 15, 2014
#

import sys, os
import logging

import numpy as np
from itertools import product

from pisa.utils.utils import is_equal_binning, get_binning
from pisa.utils.jsons import to_json
from pisa.utils.proc import report_params, get_params, add_params


class RecoServiceBase:
    """
    Base class for reconstruction services, handles the actual smearing 
    of events from the reco kernels. Kernel generation has to be implemented
    in the derived classes.
    """
    
    def __init__(self, ebins, czbins, **kwargs):
        """
        Parameters needed to instantiate any reconstruction service:
        * ebins: Energy bin edges
        * czbins: cos(zenith) bin edges
        If further member variables are needed, override this method.
        """
        logging.debug('Instantiating %s'%self.__class__.__name__)
        self.ebins = ebins
        self.czbins = czbins
        for ax in [self.ebins, self.czbins]:
            if (len(np.shape(ax)) != 1):
                raise IndexError('Axes must be 1d! '+str(np.shape(ax)))
        
        #Get kernels already now. Can be recalculated later, if needed.
        self.kernels = self.get_reco_kernels(**kwargs)
        self.check_kernels()
        self.normalize_kernels()
    
    
    def get_reco_kernels(self, **kwargs):
        """
        This method is called to construct the reco kernels, i.e. a 4D 
        histogram of true (1st and 2nd axis) vs. reconstructed (3rd and 
        4th axis) energy (1st and 3rd axis) and cos(zenith) (2nd and 4th 
        axis). It has to be implemented in the derived classes individually, 
        since the way the reco kernels are generated is the depends on
        the reco method. Normalization of the kernels is taken care of 
        elsewhere.
        """
        raise NotImplementedError('Method not implemented for %s'
                                    %self.__class__.__name__)
    
    
    def check_kernels(self):
        """
        Test whether the reco kernels have the correct shape and normalize them
        """
        # check axes
        logging.debug('Checking binning of reconstruction kernels')
        for kernel_axis, own_axis in [(self.kernels['ebins'], self.ebins),
                                       (self.kernels['czbins'], self.czbins)]:
            if not is_equal_binning(kernel_axis, own_axis):
                raise ValueError("Binning of reconstruction kernel doesn't "
                                  "match the event maps!")
            else:
                pass
        # check shape of kernels
        logging.debug('Checking shape of reconstruction kernels')
        shape = (len(self.ebins)-1, len(self.czbins)-1, 
                 len(self.ebins)-1, len(self.czbins)-1)
        for flavour in self.kernels:
            if flavour in ['ebins', 'czbins']: continue
            for interaction in self.kernels[flavour]:
                if not np.shape(self.kernels[flavour][interaction])==shape:
                    raise IndexError('Reconstruction kernel for %s/%s has wrong shape: '
                                      '%s, %s' %(flavour, interaction, str(shape),
                                      str(np.shape(self.kernels[flavour][interaction]))) )
                else:
                    pass
        logging.info('Reconstruction kernels are sane')
        return True
    
    
    def normalize_kernels(self):
        """
        Ensure that all reco kernels are normalized.
        """
        logging.debug('Normalizing reconstruction kernels')
        for flavour in self.kernels:
            if flavour in ['ebins', 'czbins']: continue
            for interaction in self.kernels[flavour]:
                k_shape = np.shape(self.kernels[flavour][interaction])
                for true_bin in product(range(k_shape[0]), range(k_shape[1])):
                    #TODO: here might be NaNs appearing, gotta catch 'em all!
                    try:
                        self.kernels[flavour][interaction][true_bin] \
                            /= np.sum(self.kernels[flavour][interaction][true_bin])
                    except Warning:
                        print self.kernels[flavour][interaction][true_bin]
    
    
    def get_reco_maps(self, true_event_maps, recalculate=False, **kwargs):
        """
        Primary function for this service, which returns the reconstructed
        event rate maps from the true event rate maps. The returned maps will
        be in the form of a dictionary with parameters:
        {'nue_cc':{'ebins':ebins,'czbins':czbins,'map':map},
         'numu_cc':{...},
         'nutau_cc':{...},
         'nuall_nc':{...}
        }
        Note that in this function, the nu<x> is now combined with nu_bar<x>.
        """
        if recalculate: 
            self.recalculate_kernels(**kwargs)
        
        #Be verbose on input
        params = get_params()
        report_params(params, units = ['',''])
        
        #Initialize return dict
        reco_maps = {'params': add_params(params,true_event_maps['params'])}

        #Check binning
        ebins, czbins = get_binning(true_event_maps)
        for map_axis, own_axis in [(ebins, self.ebins),
                                    (czbins, self.czbins)]:
            if not is_equal_binning(map_axis, own_axis):
                raise ValueError("Binning of reconstruction kernel doesn't "
                                  "match the event maps!")
            else:
                pass
        
        flavours = ['nue','numu','nutau']
        int_types = ['cc','nc']
        
        for int_type in int_types:
            for flavor in flavours:
                logging.info("Getting reco event rates for %s %s"%(flavor,int_type))
                reco_evt_rate = np.zeros((len(ebins)-1,len(czbins)-1),
                                         dtype=np.float32)
                for mID in ['','_bar']:
                    flav = flavor+mID
                    true_evt_rate = true_event_maps[flav][int_type]['map']
                    
                    kernels = self.kernels[flav][int_type]
                        
                    for ie,egy in enumerate(ebins[:-1]):
                        for icz,cz in enumerate(czbins[:-1]):
                            # Get kernel at these true parameters from 4D hist
                            reco_evt_rate += true_evt_rate[ie,icz]*kernels[ie,icz]
                
                reco_maps[flavor+'_'+int_type] = {'map':reco_evt_rate,
                                                  'ebins':ebins,
                                                  'czbins':czbins}
                logging.info("  Total counts: %.2f"%np.sum(reco_evt_rate))

        #Finally sum up all the NC contributions
        logging.info("Summing up rates for %s %s"%('all',int_type))
        reco_evt_rate = np.sum([reco_maps.pop(key)['map'] for key in reco_maps.keys()
                                if key.endswith('_nc')], axis = 0)
        reco_maps['nuall_nc'] = {'map':reco_evt_rate,
                                 'ebins':ebins,
                                 'czbins':czbins}
        logging.info("  Total counts: %.2f"%np.sum(reco_evt_rate))

        return reco_maps
    
    
    def store_kernels(self, filename):
        """
        Store reconstruction kernels in json format
        """
        to_json(self.kernels, filename)


    def recalculate_kernels(self, **kwargs):
        """
        Re-calculate reconstruction kernels and do all necessary checks.
        If new kernels are corrupted, stick with the old ones.
        """
        logging.info('Re-calculating reconstruction kernels')
        old_kernels = self.kernels.copy()
        self.recalculate_kernels(**kwargs)
        try:
            self.check_kernels()
            self.normalize_kernels()
        except:
            logging.error('Failed to recalculate reconstruction kernels, '
                          'keeping old ones: ', exc_info=True)
            self.kernels = old_kernels
