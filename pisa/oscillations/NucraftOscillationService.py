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
    logging.error("Can't load default oscillation code nuCraft: %s",e)
    sys.exit(1)
from pisa.resources.resources import find_resource


def GetIceCubePID(name):
    """Return the IceCube/Corsika Particle ID for a given particle"""
    ptcl_dict = {'nue': 66, 'nue_bar': 67, 'numu': 68, 'numu_bar': 69}
    return ptcl_dict[name]


class NucraftOscillationService(OscillationServiceBase):
    """
    This class handles all tasks related to the oscillation
    probability calculations using the prob3 oscillation code
    """
    def __init__(self, ebins, czbins,
                 earth_model='oscillations/PREM_60layer.dat',
                 detector_depth=2.0, prop_height=None, **kwargs):
        """
        Parameters needed to instantiate a NucraftOscillationService:
        * ebins: Energy bin edges
        * czbins: cos(zenith) bin edges
        * earth_model: Earth density model used for matter oscillations.
                       Default: 60-layer PREM model shipped with pisa.
        * detector_depth: Detector depth in km. Default: 2.0
        * prop_height: Height in the atmosphere to begin in km. 
                       Default: None, samples from a parametrization to 
                       the atmospheric interaction model presented in 
                       "Path length distributions of atmospheric neutrinos",
                       Gaisser and Stanev, PhysRevD.57.1977
        """
        OscillationServiceBase.__init__(self, ebins, czbins)
        
        self.prop_height = prop_height # km above spherical Earth surface
        self.height_mode = 3 if self.prop_height is None else 1
        self.detector_depth = detector_depth # km below spherical Earth surface
        #TODO: find a way to convert between prob3 and NuCraft earth model files
        self.earth_model = find_resource(earth_model)
        
        
    
    def fill_osc_prob(self, osc_prob_dict, ecen, czcen,
                  theta12=None, theta13=None, theta23=None,
                  deltam21=None, deltam31=None, deltacp=None, **kwargs):
        '''
        Loops over ecen,czcen and fills the osc_prob_dict maps, with
        probabilities calculated according to NuCraft
        '''
        
        #Setup NuCraft for the given oscillation parameters
        mass_splitting = (1., deltam21, deltam31)
        mixing_angles = [(1,2,theta12),
                         (1,3,theta13,deltacp),
                         (2,3,theta23)]
        engine = NuCraft(mass_splitting, mixing_angles,
                         earthModel = EarthModel(self.earthModel))
        engine.detectorDepth = self.detector_depth
        
        if self.prop_height is not None:
            # Fix neutrino production height and detector depth for 
            # simulating reactor experiments. 
            # In this case, there should be only one zenith angle corresponding 
            # to the baseline B. It can be calculated according to:
            #   cos(zen) = ( (r_E - detectorDepth)**2 + B**2 (r_E + atmHeight)**2 ) \
            #               / ( 2 * (r_E + detectorDepth) * B)
            # with r_E = 6371. km
            engine.atmHeight = self.prop_height
        
        #TODO: file through osc_prob_dict properly
        
        #Convert the particle into a list of IceCube particle IDs
        ps = n.ones_like(ecen)*GetIceCubePID(prim)
        
        # run it
        logging.debug("Calculating oscillation probabilites for %s at %u points..."
                        %(prim,len(ps)))
        probs = engine.CalcWeights((ps, ecen, np.arccos(czcen)), 
                                   mode=self.height_mode)
        logging.debug("...done")

        #Return probabilites
        return n.array(probs)

