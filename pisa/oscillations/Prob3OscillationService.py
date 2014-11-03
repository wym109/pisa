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
import numpy as np
from pisa.utils.log import logging, profile
from pisa.oscillations.OscillationServiceBase import OscillationServiceBase
from pisa.oscillations.prob3.BargerPropagator import BargerPropagator
from pisa.resources.resources import find_resource
from pisa.utils.jsons import to_json

class Prob3OscillationService(OscillationServiceBase):
    """
    This class handles all tasks related to the oscillation
    probability calculations using the prob3 oscillation code
    """
    def __init__(self, ebins, czbins, earth_model='oscillations/PREM_60layer.dat',
                 detector_depth=2.0, prop_height=20.0, **kwargs):
        """
        Parameters needed to instantiate a Prob3OscillationService:
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
        earth_model = find_resource(earth_model)
        self.barger_prop = BargerPropagator(earth_model, detector_depth)
        self.barger_prop.UseMassEigenstates(False)

    def fill_osc_prob(self, osc_prob_dict, ecen, czcen,
                      theta12=None, theta13=None, theta23=None,
                      deltam21=None, deltam31=None, deltacp=None,
                      energy_scale=None,**kwargs):
        '''
        Loops over ecen,czcen and fills the osc_prob_dict maps, with
        probabilities calculated according to prob3
        '''

        neutrinos = ['nue','numu','nutau']
        anti_neutrinos = ['nue_bar','numu_bar','nutau_bar']
        mID = ['','_bar']

        nu_barger = {'nue':1,'numu':2,'nutau':3,
                     'nue_bar':1,'numu_bar':2,'nutau_bar':3}

        logging.info("Defining osc_prob_dict from BargerPropagator...")
        profile.info("start oscillation calculation")
        # Set to false, since we are using sin^2(2 theta) variables
        kSquared = False
        sin2th12Sq = np.sin(2.0*theta12)**2
        sin2th13Sq = np.sin(2.0*theta13)**2
        sin2th23Sq = np.sin(2.0*theta23)**2

        total_bins = int(len(ecen)*len(czcen))
        mod = total_bins/50
        ibin = 0
        loglevel = logging.root.getEffectiveLevel()
        for icz, coszen in enumerate(czcen):

            for ie,energy in enumerate(ecen):
                if energy_scale is not None: energy*=energy_scale
                ibin+=1
                if loglevel <= logging.INFO:
                    if (ibin%mod) == 0:
                        sys.stdout.write(".")
                        sys.stdout.flush()

                # In BargerPropagator code, it takes the "atmospheric
                # mass difference"-the nearest two mass differences, so
                # that it takes as input deltam31 for IMH and deltam32
                # for NMH
                mAtm = deltam31 if deltam31 < 0.0 else (deltam31 - deltam21)

                ########### FIRST FOR NEUTRINOS ##########
                kNuBar = 1 # +1 for nu -1 for nubar
                self.barger_prop.SetMNS(sin2th12Sq,sin2th13Sq,sin2th23Sq,deltam21,mAtm,
                                        deltacp,energy,kSquared,kNuBar)
                self.barger_prop.DefinePath(coszen, self.prop_height)
                self.barger_prop.propagate(kNuBar)

                for nu in ['nue','numu']:
                    nu_i = nu_barger[nu]
                    nu = nu+'_maps'
                    for to_nu in neutrinos:
                        nu_f = nu_barger[to_nu]
                        osc_prob_dict[nu][to_nu][ie][icz]=self.barger_prop.GetProb(nu_i,nu_f)

                ########### SECOND FOR ANTINEUTRINOS ##########
                kNuBar = -1
                self.barger_prop.SetMNS(sin2th12Sq,sin2th13Sq,sin2th23Sq,deltam21,
                                        mAtm,deltacp,energy,kSquared,kNuBar)
                self.barger_prop.DefinePath(coszen, self.prop_height)
                self.barger_prop.propagate(kNuBar)

                for nu in ['nue_bar','numu_bar']:
                    nu_i = nu_barger[nu]
                    nu+='_maps'
                    for to_nu in anti_neutrinos:
                        nu_f = nu_barger[to_nu]
                        osc_prob_dict[nu][to_nu][ie][icz] = self.barger_prop.GetProb(nu_i,nu_f)

        if loglevel <= logging.INFO:
            sys.stdout.write("\n")

        profile.info("stop oscillation calculation")

        return
