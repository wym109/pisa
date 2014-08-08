#
# This is a service which will return an oscillation probability map
# corresponding to the desired binning. Code is based on the Nucraft
# oscillation package.
#
# author: Lukas Schulte
#         lschulte@physik.uni-bonn.de
#

import os, sys
import logging
import numpy as np
from pisa.oscillations.OscillationServiceBase import OscillationServiceBase
#TODO: Find out how to ship Nucraft along with pisa
try:
    from pisa.oscillations.nuCraft.NuCraft import NuCraft, EarthModel
except ImportError, e:
    logging.info("Can't load default oscillation code nuCraft: %s",e)
    sys.exit(1)
from pisa.resources.resources import find_resource


class NucraftOscillationService(OscillationServiceBase):
    """
    This class handles all tasks related to the oscillation
    probability calculations using the prob3 oscillation code
    """
    def __init__(self, ebins, czbins,
                 earth_model='oscillations/PREM_60layer.dat',
                 detector_depth=2.0, prop_height=20.0, **kwargs):
        """
        Parameters needed to instantiate a NucraftOscillationService:
        * ebins: Energy bin edges
        * czbins: cos(zenith) bin edges
        * earth_model: Earth density model used for matter oscillations.
                       Default: 60-layer PREM model shipped with pisa.
        * detector_depth: Detector depth in km. Default: 2.0
        * prop_height: Height in the atmosphere to begin in km. 
                       Default: 20.0
        """
        OscillationServiceBase.__init__(self, ebins, czbins)
        
        self.prop_height = prop_height
        self.detector_depth = detector_depth
        #TODO: find a way to convert between prob3 and NuCraft earth model files
        self.earth_model = find_resource(earth_model)
        
        
    
    def fill_osc_prob(self, osc_prob_dict, ecen, czcen,
                  theta12=None, theta13=None, theta23=None,
                  deltam21=None, deltam31=None, deltacp=None, **kwargs):
        '''
        Loops over ecen,czcen and fills the osc_prob_dict maps, with
        probabilities calculated according to NuCraft
        '''
        #TODO: implement
