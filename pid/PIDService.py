#
# Creates the pid functions from the pid input files, using
# parameterizations of the PID as a function of reconstructed energy.
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# date:   April 10, 2014
#

import scipy
import numpy as np
from utils.utils import get_bin_centers

class PIDService:
    '''
    Creates the pid functions for each flavor and then performs the
    pid selection on each template.
    '''
    def __init__(self,pid_data,ebins,czbins,**kwargs):

        self.ebins = ebins
        self.czbins = czbins
        self.pid_func_dict = {}
        for signature in pid_data.keys():
            to_trck_func = eval(pid_data[signature]['trck'])
            to_cscd_func = eval(pid_data[signature]['cscd'])
            self.pid_func_dict[signature] = {'trck':to_trck_func,
                                             'cscd':to_cscd_func}
            
        return
    
    def get_pid_maps(self,reco_events, **kwargs):
        '''
        Uses the parameterized pid functions to return maps of
        reconstructed tracks ('trck') and cascades ('cscd').
        '''
        
        flavours = ['nue_cc','numu_cc','nutau_cc','nuall_nc']
        ecen = get_bin_centers(self.ebins)
        czcen = get_bin_centers(self.czbins)
        
        #self.pid_func_dict = pid_service.get_pid_funcs()

        reco_events_pid = { 'trck': {'map':np.zeros((len(ecen),len(czcen))),
                                     'czbins':self.czbins,
                                     'ebins':self.ebins},
                            'cscd': {'map':np.zeros((len(ecen),len(czcen))),
                                     'czbins':self.czbins,
                                     'ebins':self.ebins}
                            }
        
        for flav in flavours:
            event_map = reco_events[flav]['map']
            
            to_trck_func = self.pid_func_dict[flav]['trck']
            to_cscd_func = self.pid_func_dict[flav]['cscd']
            
            to_trck = to_trck_func(ecen)
            to_trck_map = np.reshape(np.repeat(to_trck, len(czcen)), 
                                     (len(ecen), len(czcen)))*event_map
            to_cscd = to_cscd_func(ecen)
            to_cscd_map = np.reshape(np.repeat(to_cscd, len(czcen)), 
                                     (len(ecen), len(czcen)))*event_map
        
            reco_events_pid['trck']['map'] += to_trck_map
            reco_events_pid['cscd']['map'] += to_cscd_map
        

        return reco_events_pid

