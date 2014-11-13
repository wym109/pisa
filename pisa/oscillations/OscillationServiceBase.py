#
# This is the base class all other oscillation services should be derived from
#
# author: Lukas Schulte <lschulte@physik.uni-bonn.de>
#
# date:   July 31, 2014
#

import sys
import numpy as np
from pisa.utils.log import logging, profile
from pisa.utils.utils import get_smoothed_map
from pisa.utils.utils import get_bin_centers, is_coarser_binning, is_linear, is_logarithmic


def check_fine_binning(fine_bins, coarse_bins):
    """
    This function checks whether the specified fine binning exists and
    is actually finer than the coarse one.
    """

    if fine_bins is not None:
        if is_coarser_binning(coarse_bins, fine_bins):
            logging.info('Using requested binning for oversampling.')
            #everything is fine
            return True
        else:
            logging.error('Requested oversampled binning is coarser '
                          'than output binning. Aborting.')
            sys.exit(1)

    return False


def oversample_binning(coarse_bins, factor):
    """
    Oversample bins by the given factor
    """

    if is_linear(coarse_bins):
        logging.info('Oversampling linear output binning by factor %i.'
                %factor)
        fine_bins = np.linspace(coarse_bins[0], coarse_bins[-1],
                                factor*(len(coarse_bins)-1)+1)
    elif is_logarithmic(coarse_bins):
        logging.info('Oversampling logarithmic output binning by factor %i.'
                %factor)
        fine_bins = np.logspace(np.log10(coarse_bins[0]),
                                np.log10(coarse_bins[-1]),
                                factor*(len(coarse_bins)-1)+1)
    else:
        logging.warn('Irregular binning detected! Evenly oversampling '
                     'by factor %i'%factor)
        fine_bins = np.array([])
        for i, upper_edge in enumerate(coarse_bins[1:]):
            fine_bins = np.append(fine_bins,
                                  np.linspace(coarse_bins[i], upper_edge,
                                              factor, endpoint=False))

    return fine_bins


class OscillationServiceBase:
    """
    Base class for all oscillation services.
    """

    def __init__(self, ebins, czbins):
        """
        Parameters needed to instantiate any oscillation service:
        * ebins: Energy bin edges
        * czbins: cos(zenith) bin edges
        If further member variables are needed, extend this method.
        """
        logging.trace('Instantiating %s'%self.__class__.__name__)
        self.ebins = np.array(ebins)
        self.czbins = np.array(czbins)
        for ax in [self.ebins, self.czbins]:
            if (len(np.shape(ax)) != 1):
                raise IndexError('Axes must be 1d! '+str(np.shape(ax)))


    def get_osc_prob_maps(self, **kwargs):
        """
        Returns an oscillation probability map dictionary calculated
        at the values of the input parameters:
          deltam21,deltam31,theta12,theta13,theta23,deltacp
        for flavor_from to flavor_to, with the binning of ebins,czbins.
        The dictionary is formatted as:
          'nue_maps': {'nue':map,'numu':map,'nutau':map},
          'numu_maps': {...}
          'nue_bar_maps': {...}
          'numu_bar_maps': {...}
        NOTES: * expects all angles in [rad]
               * this method doesn't calculate the oscillation probabi-
                 lities itself, but calls get_osc_probLT_dict internally
        """
        #Get the finely binned maps as implemented in the derived class
        logging.info('Retrieving finely binned maps')
        fine_maps = self.get_osc_probLT_dict(**kwargs)

        logging.info("Smoothing fine maps...")
        profile.info("start smoothing maps")
        smoothed_maps = {}
        smoothed_maps['ebins'] = self.ebins
        smoothed_maps['czbins'] = self.czbins

        for from_nu, tomap_dict in fine_maps.items():
            if 'bins' in from_nu: continue
            new_tomaps = {}
            for to_nu, tomap in tomap_dict.items():
                logging.debug("Getting smoothed map %s/%s"%(from_nu,to_nu))
                new_tomaps[to_nu] = get_smoothed_map(tomap,
                                         fine_maps['ebins'],
                                         fine_maps['czbins'],
                                         self.ebins, self.czbins)
            smoothed_maps[from_nu] = new_tomaps

        profile.info("stop smoothing maps")

        return smoothed_maps


    def get_osc_probLT_dict(self, ebins=None, czbins=None,
                            oversample_e=None,oversample_cz=None, **kwargs):
        """
        This will create the oscillation probability map lookup tables
        (LT) corresponding to atmospheric neutrinos oscillation
        through the earth, and will return a dictionary of maps:
        {'nue_maps':[to_nue_map, to_numu_map, to_nutau_map],
         'numu_maps: [...],
         'nue_bar_maps': [...],
         'numu_bar_maps': [...],
         'czbins':czbins,
         'ebins': ebins}
        Will call fill_osc_prob to calculate the individual
        probabilities on the fly.
        By default, the standard binning is oversampled by a factor 10.
        Alternatively, the oversampling factor can be changed or a fine
        binning specified explicitly. In the latter case, the oversampling
        factor is ignored.
        """
        #First initialize the fine binning if not explicitly given
        if not check_fine_binning(ebins, self.ebins):
            ebins = oversample_binning(self.ebins, oversample_e)
        if not check_fine_binning(czbins, self.czbins):
            czbins = oversample_binning(self.czbins, oversample_cz)
        ecen = get_bin_centers(ebins)
        czcen = get_bin_centers(czbins)

        osc_prob_dict = {'ebins':ebins, 'czbins':czbins}
        shape = (len(ecen),len(czcen))
        for nu in ['nue_maps','numu_maps','nue_bar_maps','numu_bar_maps']:
            isbar = '_bar' if 'bar' in nu else ''
            osc_prob_dict[nu] = {'nue'+isbar: np.zeros(shape,dtype=np.float32),
                                 'numu'+isbar: np.zeros(shape,dtype=np.float32),
                                 'nutau'+isbar: np.zeros(shape,dtype=np.float32)}

        self.fill_osc_prob(osc_prob_dict, ecen, czcen, **kwargs)

        return osc_prob_dict


    def fill_osc_prob(self, osc_prob_dict, ecen, czcen,
                  theta12=None, theta13=None, theta23=None,
                  deltam21=None, deltam31=None, deltacp=None, **kwargs):
        """
        This method is called by get_osc_probLT_dict and should be
        implemented in any derived class individually as here the actual
        oscillation code should be run.
        NOTE: Expects all angles to be in [rad], and all deltam to be in [eV^2]
        """
        raise NotImplementedError('Method not implemented for %s'
                                    %self.__class__.__name__)
