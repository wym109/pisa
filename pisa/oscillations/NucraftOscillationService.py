#
# This is a service which will return an oscillation probability map
# corresponding to the desired binning. Code is based on the Nucraft
# oscillation package.
#
# author: Lukas Schulte
#         lschulte@physik.uni-bonn.de
#

import os, sys
from tempfile import NamedTemporaryFile

import numpy as np

from pisa.utils.log import logging
from pisa.oscillations.OscillationServiceBase import OscillationServiceBase
from pisa.oscillations.nuCraft.NuCraft import NuCraft, EarthModel
from pisa.resources.resources import find_resource
from pisa.utils.physics import get_PDG_ID
from pisa.utils.proc import get_params, report_params


class NucraftOscillationService(OscillationServiceBase):
    """
    This class handles all tasks related to the oscillation
    probability calculations using the NuCraft oscillation code
    """
    def __init__(self, ebins, czbins, detector_depth=None, earth_model=None,
                 prop_height=None, osc_precision=5e-4,
                 **kwargs):
        """
        Parameters needed to instantiate a NucraftOscillationService:
        * ebins: Energy bin edges
        * czbins: cos(zenith) bin edges
        * earth_model: Earth density model used for matter oscillations.
        * detector_depth: Detector depth in km.
        * prop_height: Height in the atmosphere to begin in km.
                       Default: 'sample', samples from a parametrization to
                       the atmospheric interaction model presented in
                       "Path length distributions of atmospheric neutrinos",
                       Gaisser and Stanev, PhysRevD.57.1977
        * osc_precision: Numerical precision for oscillation probabilities
        """
        OscillationServiceBase.__init__(self, ebins, czbins)

        print get_params()
        report_params(get_params(),['km','','','km'])

        self.prop_height = prop_height # km above spherical Earth surface
 	''' height_mode = 0 ensures that interaction takes place at chosen height '''
	''' whereas height_mode = 1 samples single altitude from distribution '''
        self.height_mode = 3 if self.prop_height is 'sample' else 0
        self.detector_depth = detector_depth # km below spherical Earth surface
        self.num_prec = osc_precision
        self.get_earth_model(earth_model)


    def fill_osc_prob(self, osc_prob_dict, ecen, czcen,
                      theta12=None, theta13=None, theta23=None,
                      deltam21=None, deltam31=None, deltacp=None,
                      energy_scale=None,**kwargs):
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

        if self.prop_height is not 'sample':
            # Fix neutrino production height and detector depth for
            # simulating reactor experiments.
            # In this case, there should be only one zenith angle corresponding
            # to the baseline B. It can be calculated according to:
            #   cos(zen) = ( (r_E - detectorDepth)**2 + B**2 (r_E + atmHeight)**2 ) \
            #               / ( 2 * (r_E + detectorDepth) * B)
            # with r_E = 6371. km
            engine.atmHeight = self.prop_height

        # Make input arrays in correct format (nucraft input type 1)
	zs, es = np.meshgrid(czcen, ecen)
        zs, es = zs.flatten(), es.flatten()
	# we need flat lists with probabilities for further processing
	shape = int(len(ecen)*len(czcen))

        # Apply Energy scaling factor:
	if energy_scale is not None:
	    es *= energy_scale

        for prim in osc_prob_dict:

            if 'bins' in prim: continue

            #Convert the particle into a list of IceCube particle IDs
            ps = np.ones_like(es)*get_PDG_ID(prim.rsplit('_', 1)[0])

            # run it
            logging.debug("Calculating oscillation probabilities for %s at %u points..."
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

        evals,czvals = np.meshgrid(ecen,czcen,indexing='ij')
        return evals.flatten(),czvals.flatten()


    def get_earth_model(self, model):
        """
        Check whether the specified Earth density profile has a correct
        NuCraft preface. If not, create a temporary file that does.
        """
        logging.debug('Trying to construct Earth model from "%s"'%model)
        try:
            resource_path = find_resource(model)
            self.earth_model = EarthModel(resource_path)
            logging.info('Loaded Earth model from %s'%model)
        except SyntaxError:
            #Probably the file is lacking the correct preamble
            logging.info('Failed to construct NuCraft Earth model directly from'
                         ' %s! Adding default preamble...'%resource_path)
            #Generate tempfile with preamble
            with open(resource_path, 'r') as infile:
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
        except IOError:
            logging.info('Using NuCraft built-in Earth model "%s"'%model)
            self.earth_model = EarthModel(model)
