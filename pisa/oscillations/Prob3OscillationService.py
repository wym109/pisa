#
# This is a service which will return an oscillation probability map
# corresponding to the desired binning. Code is based on the prob3 oscillation
# package.
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# author: Sebastian Boeser
#         sboeser@physik.uni-bonn.de
#

import sys
import itertools

import numpy as np

from pisa.utils.log import logging, tprofile
from pisa.oscillations import Oscillation
from pisa.oscillations.OscillationServiceBase import OscillationServiceBase
from pisa.oscillations.prob3.BargerPropagator import BargerPropagator
from pisa.resources.resources import find_resource
from pisa.utils.proc import get_params, report_params
from pisa.utils.utils import hash_obj


class Prob3OscillationService(OscillationServiceBase):
    """
    This class handles all tasks related to the oscillation
    probability calculations using the prob3 oscillation code
    """
    def __init__(self, ebins, czbins, detector_depth, earth_model, prop_height,
                 **kwargs):
        """
        Parameters needed to instantiate a Prob3OscillationService:
        * ebins: Energy bin edges
        * czbins: cos(zenith) bin edges
        * earth_model: Earth density model used for matter oscillations.
        * detector_depth: Detector depth in km.
        * prop_height: Height in the atmosphere to begin in km.
        """
        super(Prob3OscillationService, self).__init__(ebins, czbins)
        logging.info('Initializing %s...' % self.__class__.__name__)

        #report_params(get_params(), ['km', '', 'km'])

        self.__osc_prob_dict = None
        self.prop_height = prop_height
        earth_model = find_resource(earth_model)
        self.barger_prop = BargerPropagator(earth_model, detector_depth)
        self.barger_prop.UseMassEigenstates(False)

    def fill_osc_prob(self, ecen, czcen, theta12, theta13, theta23, deltam21,
                      deltam31, deltacp, energy_scale, YeI, YeO, YeM,
                      **kwargs):
        """Loops over ecen, czcen and fills the osc_prob_dict maps with
        probabilities calculated according to prob3.
        """
        cache_key = hash_obj((ecen, czcen, theta12, theta13, theta23, deltam21,
                              deltam31, deltacp, energy_scale, YeI, YeO,
                              YeM))
        try:
            return self.transform_cache.get(cache_key)
        except KeyError:
            pass

        neutrinos = ['nue', 'numu', 'nutau']
        anti_neutrinos = ['nue_bar', 'numu_bar', 'nutau_bar']
        mID = ['', '_bar']

        nu_barger = {'nue':1, 'numu':2, 'nutau':3,
                     'nue_bar':1, 'numu_bar':2, 'nutau_bar':3}
        barger_nu = {1: 'nue', 2: 'numu', 3: 'nutau'}
        barger_nubar = {1: 'nue_bar', 2: 'numu_bar', 3: 'nutau_bar'}

        barger_numap = {1: 'nue_maps', 2: 'numu_maps', 3: 'nutau_maps'}
        barger_nubarmap = {1: 'nue_bar_maps', 2: 'numu_bar_maps', 3:
                           'nutau_bar_maps'}

        logging.info("Defining osc_prob_dict from BargerPropagator...")
        tprofile.info("start oscillation calculation")
        # Set to true, since we are using sin^2(theta) variables
        kSquared = True
        sin2th12Sq = np.sin(theta12)**2
        sin2th13Sq = np.sin(theta13)**2
        sin2th23Sq = np.sin(theta23)**2

        n_ecen = len(ecen)
        n_czcen = len(czcen)
        evals = np.zeros(n_ecen * n_czcen)
        czvals = np.zeros(n_ecen * n_czcen)

        self.__osc_prob_dict = Oscillation.newOscProbDict(ecen, czcen)

        total_bins = len(ecen)*len(czcen)
        mod = total_bins//20
        loglevel = logging.root.getEffectiveLevel()
        for ie, energy in enumerate(ecen):
            for icz, coszen in enumerate(czcen):
                N = ie*n_czcen + icz
                evals[N] = energy #.append(energy)
                czvals[N] = coszen #.append(coszen)
                scaled_energy = energy*energy_scale

                if loglevel <= logging.INFO:
                    if (N % mod) == 0:
                        #sys.stdout.write(str(N) + ' ')
                        sys.stdout.write(".")
                        sys.stdout.flush()

                # In BargerPropagator code, it takes the "atmospheric
                # mass difference"-the nearest two mass differences, so
                # that it takes as input deltam31 for IMH and deltam32
                # for NMH
                mAtm = deltam31 if deltam31 < 0.0 else (deltam31 - deltam21)

                ########### FIRST FOR NEUTRINOS ##########
                kNuBar = 1 # +1 for nu -1 for nubar
                self.barger_prop.SetMNS(sin2th12Sq, sin2th13Sq, sin2th23Sq,
                                        deltam21, mAtm, deltacp, scaled_energy,
                                        kSquared, kNuBar)

                self.barger_prop.DefinePath(coszen, self.prop_height, YeI,
                                            YeO, YeM)
                self.barger_prop.propagate(kNuBar)

                [self.__osc_prob_dict[barger_numap[nucode_initial]][barger_nu[nucode_final]].__setitem__(N, self.barger_prop.GetProb(nucode_initial, nucode_final)) for nucode_initial, nucode_final in itertools.product([1,2], [1,2,3])]

                ########### SECOND FOR ANTINEUTRINOS ##########
                kNuBar = -1
                self.barger_prop.SetMNS(sin2th12Sq, sin2th13Sq, sin2th23Sq,
                                        deltam21, mAtm, deltacp, scaled_energy,
                                        kSquared, kNuBar)
                self.barger_prop.DefinePath(coszen, self.prop_height, YeI,
                                            YeO, YeM)
                self.barger_prop.propagate(kNuBar)

                [self.__osc_prob_dict[barger_nubarmap[nucode_initial]][barger_nubar[nucode_final]].__setitem__(N, self.barger_prop.GetProb(nucode_initial, nucode_final)) for nucode_initial, nucode_final in itertools.product([1,2], [1,2,3])]

        if loglevel <= logging.INFO:
            sys.stdout.write("\n")

        self.__osc_prob_dict['evals'] = evals
        self.__osc_prob_dict['czvals'] = czvals
        self.__osc_prob_dict.update_hash(cache_key)

        self.transform_cache.set(cache_key, self.__osc_prob_dict)

        tprofile.info("stop oscillation calculation")

        return self.__osc_prob_dict
