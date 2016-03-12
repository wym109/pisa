#
# This is the base class all other oscillation services should be derived from
#
# author: Lukas Schulte <lschulte@physik.uni-bonn.de>
#         Timothy C. Arlen tca3@psu.edu
#
# date:   July 31, 2014
#


import numpy as np

from pisa.utils.log import logging, tprofile
from pisa.utils.utils import get_smoothed_map, get_bin_centers, check_fine_binning, oversample_binning, Timer, get_binning
from pisa.utils.proc import report_params, get_params, add_params


class OscillationServiceBase(object):
    """Base class for all oscillation services.

    Parameters
    ----------
    ebins, czbins : array_like
        Energy and coszen bin edges
    """
    def __init__(self, ebins, czbins):
        logging.trace('Instantiating %s' % self.__class__.__name__)
        self.ebins = np.array(ebins)
        self.czbins = np.array(czbins)
        for ax in [self.ebins, self.czbins]:
            if (len(np.shape(ax)) != 1):
                raise IndexError('Axes must be 1d! '+str(np.shape(ax)))

        # Instantiate the LRU cache
        #self.result_cache = 
        #self.

    def get_osc_prob_maps(self, **kwargs):
        """
        Returns an oscillation probability map dictionary calculated
        at the values of the input parameters:
          deltam21, deltam31, theta12, theta13, theta23, deltacp
        for flavor_from to flavor_to, with the binning of ebins, czbins.
        The dictionary is formatted as:
          'nue_maps': {'nue':map, 'numu':map, 'nutau':map},
          'numu_maps': {...}
          'nue_bar_maps': {...}
          'numu_bar_maps': {...}

        Notes
        -----
          * expects all angles in [rad]
          * this method doesn't calculate the oscillation probabilities
            itself, but calls get_osc_probLT_dict internally, to get a
            high resolution map of the oscillation probs,
        """
        # Get the finely binned maps as implemented in the derived class
        logging.info('Retrieving finely binned maps')
        with Timer(verbose=False) as t:
            fine_maps = self.get_osc_probLT_dict(**kwargs)
        logging.debug("       ==> elapsed time to get all fine maps: %s sec" % t.secs)

        logging.info("Smoothing fine maps...")
        smoothed_maps = {}
        smoothed_maps['ebins'] = self.ebins
        smoothed_maps['czbins'] = self.czbins

        with Timer(verbose=False) as t:
            for from_nu, tomap_dict in fine_maps.items():
                if 'vals' in from_nu:
                    continue
                new_tomaps = {}
                for to_nu, pvals in tomap_dict.items():
                    logging.debug("Getting smoothed map %s/%s" % (from_nu, to_nu))

                    new_tomaps[to_nu] = get_smoothed_map(
                        pvals, fine_maps['evals'], fine_maps['czvals'],
                        self.ebins, self.czbins
                    )

                smoothed_maps[from_nu] = new_tomaps

        tprofile.debug("       ==> elapsed time to smooth maps: %s sec" % t.secs)

        return smoothed_maps

    def get_osc_probLT_dict(self, ebins=None, czbins=None,
                            oversample_e=None, oversample_cz=None, **kwargs):
        """Create the oscillation probability map lookup tables (LT)
        corresponding to atmospheric neutrino oscillation through the earth.

        Parameters
        ----------
        ebins, czbins : array_like
            Energy and coszen bin edges; if not specified, use binning
            specified at instantiation
        oversample_e, oversample_cz
            Factor by which to over (up) sample self.ebins and self.czbins,
            respectively (if parameters `ebins` and/or `czbins` are not
            specified)
        **kwargs
            Sent on to fill_osc_prob

        Returns
        -------
        Dictionary of maps:
            {'nue_maps':[to_nue_map, to_numu_map, to_nutau_map],
             'numu_maps: [...],
             'nue_bar_maps': [...],
             'numu_bar_maps': [...],
             'czbins':czbins,
             'ebins': ebins}

        Notes
        -----
        Will call fill_osc_prob to calculate the individual
        probabilities on the fly.
        """
        # First initialize the fine binning if not explicitly given
        if not check_fine_binning(ebins, self.ebins):
            ebins = oversample_binning(self.ebins, oversample_e)
        if not check_fine_binning(czbins, self.czbins):
            czbins = oversample_binning(self.czbins, oversample_cz)
        ecen = get_bin_centers(ebins)
        czcen = get_bin_centers(czbins)

        osc_prob_dict = {}
        for nu in ['nue_maps', 'numu_maps', 'nue_bar_maps', 'numu_bar_maps']:
            isbar = '_bar' if 'bar' in nu else ''
            osc_prob_dict[nu] = {'nue'+isbar: [],
                                 'numu'+isbar: [],
                                 'nutau'+isbar: []}

        evals, czvals = self.fill_osc_prob(osc_prob_dict, ecen, czcen, **kwargs)
        osc_prob_dict['evals'] = evals
        osc_prob_dict['czvals'] = czvals

        return osc_prob_dict

    def fill_osc_prob(self, osc_prob_dict, ecen, czcen, theta12, theta13,
                      theta23, deltam21, deltam31, deltacp, **kwargs):
        """This method is called by get_osc_probLT_dict and should be
        implemented in any derived class individually as here the actual
        oscillation code should be run.

        NOTE: Expects all angles to be in [rad], and all deltam to be in [eV^2]
        """
        raise NotImplementedError('Method not implemented for %s'
                                  % self.__class__.__name__)

    def get_osc_flux(self, flux_maps, deltam21, deltam31, energy_scale,
                     theta12, theta13, theta23, deltacp, YeI, YeO, YeM,
                     **kwargs):
        """Obtain a map in energy and cos(zenith) of the oscillation probabilities
        from the OscillationService and compute the oscillated flux.

        Parameters
        ----------
        flux_maps
            Dictionary of atmospheric flux ['nue', 'numu', 'nue_bar', 'numu_bar']
        **kwargs
            Oscillation parameters to compute oscillation probability maps from.
        """
        # Be verbose on input
        params = get_params()

        report_params(
            params, units=['', '', '', 'rad', 'eV^2', 'eV^2', '', 'rad', 'rad', 'rad']
        )

        ebins, czbins = get_binning(flux_maps)

        # Initialize return dict
        osc_flux_maps = {'params': add_params(params, flux_maps['params'])}

        # Get oscillation probability map from service
        osc_prob_maps = self.get_osc_prob_maps(
            deltam21=deltam21, deltam31=deltam31,
            theta12=theta12, theta13=theta13, theta23=theta23,
            deltacp=deltacp,
            energy_scale=energy_scale,
            YeI=YeI, YeO=YeO, YeM=YeM,
            **kwargs
        )

        for to_flav in ['nue', 'numu', 'nutau']:
            for mID in ['', '_bar']: # 'matter' ID
                nue_flux = flux_maps['nue'+mID]['map']
                numu_flux = flux_maps['numu'+mID]['map']
                oscflux = {
                    'ebins':ebins, 'czbins':czbins,
                    'map':(nue_flux
                           * osc_prob_maps['nue'+mID+'_maps'][to_flav+mID]
                           + numu_flux*osc_prob_maps['numu'+mID+'_maps'][to_flav+mID])
                }
                osc_flux_maps[to_flav+mID] = oscflux

        return osc_flux_maps

