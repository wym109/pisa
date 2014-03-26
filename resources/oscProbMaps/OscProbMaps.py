#! /usr/bin/env python
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# date:   18 March 2014
#

import numpy as np
import logging
from utils.utils import is_linear

class OscProbMaps:
    '''
    This class makes an oscillation probability lookup table (LT) of
    finely spaced energy and coszenith. The exact value of the
    oscillation probability is taken at the CENTER of the bin.
    
    These lookup tables become the smoothed oscillation maps which are
    the foundation of the LLR optimizer-based analysis.
    '''
    def __init__(self, czbins, ebins, deltam31, theta23, **kwargs):
        self.czbins = czbins
        self.ebins = ebins       # [GeV] or log10(E/GeV)

        self.deltam31 = deltam31 # [eV^2]
        self.theta23 = theta23   # [deg]
        
        self.deltam21 = kwargs.pop('deltam21') if 'deltam21' in kwargs else 7.54e-05
        self.theta12 = kwargs.pop('theta12') if 'theta12' in kwargs else 33.647
        self.theta13 = kwargs.pop('theta13') if 'theta13' in kwargs else 8.931
        self.deltacp = kwargs.pop('deltacp') if 'deltacp' in kwargs else 0.0
        self.earth_model=kwargs.pop('earth_model') if 'earth_model' in kwargs else "prem"
        
        self.param_dict = {'deltam21':self.deltam21,
                           'deltam31':self.deltam31,
                           'theta12':self.theta12,
                           'theta13':self.theta13,
                           'theta23':self.theta23,
                           'deltacp':self.deltacp,
                           'earth_model':self.earth_model}
        
        logging.debug("Initializing OscProbMaps, with values...")
        logging.debug("  deltam31:    %s eV^2"%self.deltam31)
        logging.debug("  deltam21:    %s eV^2"%self.deltam21)
        logging.debug("  theta12:     %s deg"%self.theta12)
        logging.debug("  theta13:     %s deg"%self.theta13)
        logging.debug("  theta23:     %s deg"%self.theta23)
        logging.debug("  deltacp:     %s rad"%self.deltacp)
        logging.debug("  earth_model: %s"%self.earth_model)
        
        return
        
    def GetOscProbLT(self):
        '''
        This will create the oscillation probability maps
        corresponding to atmospheric neutrinos oscillating through the
        earth, and will return a dictionary of maps:
          {'nue_maps': [to_nue_map,to_numu_map,to_nutau_map],
           'numu_maps: [...],
           'nue_bar_maps': [...],
           'numu_bar_maps': [...],
           'czbins': czbins,
           'ebins': ebins}
        to be saved in an hdf5 file.
        '''

        osc_service = MakeNuCraft(self.deltam31,self.deltam21,self.theta12,self.theta13,
                                  self.theta23,self.deltacp,self.earth_model)
        
        ebin_width = self.ebins[1] - self.ebins[0]
        czbin_width = self.czbins[1] - self.czbins[0]
        
        nue = 12; nue_bar = -12
        numu = 14; numu_bar = -14
        nuTypeDict = {'nue': nue,'nue_bar': nue_bar, 
                      'numu': numu, 'numu_bar': numu_bar}

        linear_egy = is_linear(self.ebins)
        oscprob_dict = {'ebins':self.ebins,'czbins':self.czbins}
        shape = (np.shape(self.ebins)[0]-1,np.shape(self.czbins)[0]-1)
        for nuKey,nuVal in nuTypeDict.iteritems():
            logging.debug("getting %s oscprob maps..."%nuKey)
            logging.debug("  val: %i"%nuVal)
            to_nue_map = np.zeros(shape,dtype=np.float32)
            to_numu_map = np.zeros(shape,dtype=np.float32)
            to_nutau_map = np.zeros(shape,dtype=np.float32)
            for ie,egy in enumerate(self.ebins[:-1]):
                egy += (ebin_width/2.0) if linear_egy else np.sqrt(egy*self.ebins[ie+1])
                for icz, cosZen in enumerate(self.czbins[:-1]):
                    cosZen += (czbin_width/2.0)
                    zenith = np.arccos(cosZen)
                    
                    # returns prob [to_nue, to_numu, to_nutau]
                    probNuTypetoX = osc_service.CalcWeights([(nuVal,egy,zenith)])[0]
                    to_nue_map[ie][icz] = probNuTypetoX[0]
                    to_numu_map[ie][icz] = probNuTypetoX[1]
                    to_nutau_map[ie][icz] = probNuTypetoX[2]
            keyNames = ['nue_bar','numu_bar','nutau_bar'] if 'bar' in nuKey else ['nue','numu','nutau']
            oscprob_dict[nuKey+'_maps'] = {keyNames[0]: to_nue_map,
                                           keyNames[1]: to_numu_map,
                                           keyNames[2]: to_nutau_map}
            
        
        return oscprob_dict

    def SaveHDF5(self,filename,oscprob_dict):
        
        import h5py
        fh = h5py.File(filename,'w')
        logging.info("Saving file: %s",filename)
        
        edata = fh.create_dataset('ebins',data=oscprob_dict['ebins'],dtype=np.float32)
        czdata = fh.create_dataset('czbins',data=oscprob_dict['czbins'],dtype=np.float32)
        
        for key in oscprob_dict.keys():
            if 'maps' in key:
                logging.info("  key %s",key)
                group_base = fh.create_group(key)
                for subkey in oscprob_dict[key].keys():
                    logging.info("    subkey %s",subkey)
                    dset = group_base.create_dataset(subkey,data=oscprob_dict[key][subkey],dtype=np.float32)
                    dset.attrs['ebins'] = edata.ref
                    dset.attrs['czbins'] = czdata.ref
            
        param_group = fh.create_group("params")
        logging.info("  saving param dict...")
        for key in self.param_dict.keys():
            param_group.create_dataset(key,data=self.param_dict[key])
            
        fh.close()
        return
                
def MakeNuCraft(deltam31,deltam21,theta12,theta13,theta23,deltacp,earth_model):
    earth_model_dict = {"prem":(0.5,0.5,0.5)}
    if earth_model not in earth_model_dict.keys():
        logging.error("WARNING: earth_model %s not in earth_model_dict, so setting it to default model."%earth_model)
        earth_model = "prem"
        
    baseline_neutrino_mass = 1.0
    mass_tuple = (baseline_neutrino_mass, deltam21, deltam31)
    
    theta_12_tuple = (1, 2, theta12)
    theta_13_tuple = (1,3,theta13) if deltacp is None else (1,3,theta13,deltacp)
    theta_23_tuple = (2, 3, theta23)
    angle_list = [theta_12_tuple, theta_23_tuple, theta_13_tuple]
    
    from nuCraft import NuCraft
    from nuCraft.NuCraft import EarthModel
    model = EarthModel(earth_model,earth_model_dict[earth_model])
    oscillation_service = NuCraft(mass_tuple, angle_list,
                                  earthModel=model)
    
    return oscillation_service

