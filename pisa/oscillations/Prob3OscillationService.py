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
from pisa.utils.proc import get_params, report_params


class Prob3OscillationService(OscillationServiceBase):
    """
    This class handles all tasks related to the oscillation
    probability calculations using the prob3 oscillation code
    """
    def __init__(self, ebins, czbins, detector_depth=None, earth_model=None,
                 prop_height=None, **kwargs):
        """
        Parameters needed to instantiate a Prob3OscillationService:
        * ebins: Energy bin edges
        * czbins: cos(zenith) bin edges
        * earth_model: Earth density model used for matter oscillations.
        * detector_depth: Detector depth in km.
        * prop_height: Height in the atmosphere to begin in km.
        """
        OscillationServiceBase.__init__(self, ebins, czbins)
        logging.info('Initializing %s...'%self.__class__.__name__)

        report_params(get_params(),['km','','km'])

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
        # Set to true, since we are using sin^2(theta) variables
        kSquared = True
        sin2th12Sq = np.sin(theta12)**2
        sin2th13Sq = np.sin(theta13)**2
        sin2th23Sq = np.sin(theta23)**2
        evals = []
        czvals = []
        total_bins = int(len(ecen)*len(czcen))
        mod = total_bins/20
        loglevel = logging.root.getEffectiveLevel()
        for ie,energy in enumerate(ecen):
            for icz, coszen in enumerate(czcen):
                index = int((ie+1)*(icz+1) - 1)
                evals.append(energy)
                czvals.append(coszen)
                if energy_scale is not None:
                    if icz == 0: energy*=energy_scale

                if loglevel <= logging.INFO:
                    if( (ie+1)*(icz+1) % mod == 0):
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
                        osc_prob_dict[nu][to_nu].append(self.barger_prop.GetProb(nu_i,
                                                                                 nu_f))


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
                        osc_prob_dict[nu][to_nu].append(self.barger_prop.GetProb(nu_i,
                                                                                 nu_f))

        if loglevel <= logging.INFO:
            sys.stdout.write("\n")

        profile.info("stop oscillation calculation")

        # Saving fine maps: Testing purposes!
        #for from_nu in ['nue','numu','nue_bar','numu_bar']:
        #    from_nu += '_maps'
        #    for to_nu in ['nue','numu','nutau']:
        #        if 'bar' in from_nu: to_nu+='_bar'
        #        filename = (from_nu+'_'+to_nu+'.dat').replace('_maps','').replace('_bar','bar')
        #        print "Saving to file: ",filename
        #        np.savetxt(filename,(evals,czvals,osc_prob_dict[from_nu][to_nu]))

        return evals,czvals
