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
from tempfile import NamedTemporaryFile

import numpy as np

from pisa.oscillations.OscillationServiceBase import OscillationServiceBase
from pisa.oscillations.nuCraft.NuCraft import NuCraft, EarthModel
from pisa.resources.resources import find_resource
from pisa.utils.physics import get_PDG_ID


class NucraftOscillationService(OscillationServiceBase):
    """
    This class handles all tasks related to the oscillation
    probability calculations using the prob3 oscillation code
    """
    def __init__(self, ebins, czbins,
                 earth_model='oscillations/PREM_60layer.dat',
                 detector_depth=2.0, prop_height=None, osc_precision=5e-4,
                 **kwargs):
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
        * osc_precision: Numerical precision for oscillation probabilities
        """
        OscillationServiceBase.__init__(self, ebins, czbins)
        
        self.prop_height = prop_height # km above spherical Earth surface
        self.height_mode = 3 if self.prop_height is None else 1
        self.detector_depth = detector_depth # km below spherical Earth surface
        self.num_prec = osc_precision
        self.get_earth_model(earth_model)
        
    
    def fill_osc_prob(self, osc_prob_dict, ecen, czcen,
                  theta12=None, theta13=None, theta23=None,
                  deltam21=None, deltam31=None, deltacp=None, **kwargs):
        """
        Loops over ecen,czcen and fills the osc_prob_dict maps, with
        probabilities calculated according to NuCraft
        """
        
        #Setup NuCraft for the given oscillation parameters
        #TODO: compatible with new NuCraft version?
        mass_splitting = (1., deltam21, deltam31)
        mixing_angles = [(1,2,np.rad2deg(theta12)),
                         (1,3,np.rad2deg(theta13),np.rad2deg(deltacp)),
                         (2,3,np.rad2deg(theta23))]
        engine = NuCraft(mass_splitting, mixing_angles,
                         earthModel = self.earth_model)
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
        
        #Make input arrays in correct format
        es, zs = np.meshgrid(ecen, czcen)
        shape = es.shape
        es, zs = es.flatten(), zs.flatten()
        
        for prim in osc_prob_dict:
            
            if 'bins' in prim: continue
            
            #Convert the particle into a list of IceCube particle IDs
            ps = np.ones_like(es)*get_PDG_ID(prim.rsplit('_', 1)[0])
            
            # run it
            logging.debug("Calculating oscillation probabilites for %s at %u points..."
                            %(prim.rsplit('_', 1)[0], len(ps)))
            probs = engine.CalcWeights((ps, es, np.arccos(zs)), 
                                       atmMode=self.height_mode,
                                       numPrec=self.num_prec)
            logging.debug("...done")
            
            #Bring into correct shape
            probs = np.array([ x.reshape(shape).T for x in np.array(probs).T ])
            
            #Fill probabilities into dict
            for i, sec in enumerate(['nue', 'numu', 'nutau']):
                sec_key = sec+'_bar' if 'bar' in prim else sec
                osc_prob_dict[prim][sec_key] = probs[i]

        return


    def get_earth_model(self, model):
        """
        Check whether the specified Earth density profile has a correct 
        NuCraft preface. If not, create a temporary file that does.
        """
        logging.debug('Trying to construct Earth model from "%s"'%model)
        try:
            self.earth_model = EarthModel(model)
            if os.path.isfile(model):
                logging.info('Loaded Earth model from %s'%model)
            else:
                logging.info('Using NuCraft built-in Earth model "%s"'%model)
        except NotImplementedError:
            #Probably we have to find the correct path to the file
            self.get_earth_model(find_resource(model))
        except SyntaxError:
            #Probably the file is lacking the correct preamble
            logging.warn('Failed to construct NuCraft Earth model from '
                         '%s! Adding default preamble...'%model)
            #Generate tempfile with preamble
            with open(model, 'r') as infile:
                profile_lines = infile.readlines()
            preamble = ['# nuCraft Earth model with PREM density '
                         'values for use as template; keep structure '
                         'of the first six lines unmodified!\n',
                        '(0.5, 0.5, 0.5)   # tuple of (relative) '
                         'electron numbers for mantle, outer core, '
                         'and inner core\n',
                        '6371.    # radius of the Earth\n',
                        '3480.    # radius of the outer core\n',
                        '1121.5   # radius of the inner core\n',
                        '# two-columned list of radii and corresponding '
                         'matter density values in km and kg/dm^3; '
                         'add, remove or modify lines as necessary\n']
            tfile = NamedTemporaryFile()
            tfile.writelines(preamble+profile_lines)
            tfile.flush()
            try:
                self.earth_model = EarthModel(tfile.name)
            except:
                logging.error('Could not construct Earth model from %s: %s'
                              %(model, sys.exc_info()[1]))
                sys.exit(1)
            logging.info('Successfully constructed Earth model')
            tfile.close()
