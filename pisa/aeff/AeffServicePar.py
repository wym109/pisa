#
# Creates effective areas by parameterizing the detector
# response. Effective areas are always 2D in coszen and energy
#
# author: Timothy C. Arlen
#
# date:   June 9, 2014
#

import os

import numpy as np
from scipy.interpolate import interp1d

from pisa.utils.log import logging
from pisa.utils.utils import get_bin_centers, get_bin_sizes
from pisa.utils.jsons import from_json
from pisa.resources.resources import find_resource, open_resource
from pisa.utils import flavInt


class AeffServicePar:
    '''
    Takes a .json file with the names of .dat files, and
    creates a dictionary of the 2D effective area in terms of energy
    and coszen, for each flavor (nue, nue_bar, numu,...) and interaction
    type (CC, NC)

    The final aeff dict for each flavor is in units of [m^2] in each
    energy/coszen bin.
    '''
    def __init__(self, ebins, czbins, aeff_egy_par, aeff_coszen_par, **kwargs):
        '''
        Parameters:
        * aeff_egy_par - effective area vs. Energy 1D parameterizations for
        each flavor,
        in a text file (.dat)
        * aeff_coszen_par - json file containing 1D coszen parameterization for
        each flavor 
        '''
        logging.info('Initializing AeffServicePar...')
        self.aeff_egy_par = aeff_egy_par
        self.aeff_coszen_par = aeff_coszen_par
        self.energy_interpolants = {}
        self.coszen_lambdas = {}
        self.ebins = ebins
        self.czbins = czbins
        # Because naming things consistently is really hard
        self.label_flavintgroup_mapping = {
            'energy': {
                'nue': flavInt.NuFlavIntGroup('nue_cc'),
                'numu': flavInt.NuFlavIntGroup('numu_cc'),
                'nutau': flavInt.NuFlavIntGroup('nutau_cc'),
                'nue_bar': flavInt.NuFlavIntGroup('nuebar_cc'),
                'numu_bar': flavInt.NuFlavIntGroup('numubar_cc'),
                'nutau_bar': flavInt.NuFlavIntGroup('nutaubar_cc'),
                'NC': flavInt.NuFlavIntGroup('nuall_nc'),
                'NC_bar': flavInt.NuFlavIntGroup('nuallbar_nc'),
            },
            'coszen': {
                'nue': flavInt.NuFlavIntGroup('nue_cc + nuebar_cc'),
                'numu': flavInt.NuFlavIntGroup('numu_cc + numubar_cc'),
                'nutau': flavInt.NuFlavIntGroup('nutau_cc + nutaubar_cc'),
                'NC': flavInt.NuFlavIntGroup('nuall_nc + nuallbar_nc'),
            }
        }

        # Load the info from .dat files into a dict...
        # Parametric approach treats all NC events the same
        self.aeff_coszen_par_str = from_json(find_resource(aeff_coszen_par))
        aeff2d_nc = self.get_aeff_flavor('NC', self.aeff_egy_par,
                                         self.aeff_coszen_par_str)
        aeff2d_nc_bar = self.get_aeff_flavor('NC_bar', self.aeff_egy_par,
                                             self.aeff_coszen_par_str)

        self.aeff_fidata = flavInt.FlavIntData()
        logging.info("Creating effective area parametric dict...")
        for flavor in ['nue', 'nue_bar', 'numu', 'numu_bar', 'nutau',
                       'nutau_bar']:
            logging.debug("Working on %s effective areas"%flavor)

            aeff2d = self.get_aeff_flavor(flavor, self.aeff_egy_par,
                                          self.aeff_coszen_par_str)

            self.aeff_fidata[flavor + 'cc'] = aeff2d
            if 'bar' in flavor:
                self.aeff_fidata[flavor + 'nc'] = aeff2d_nc_bar
            else:
                self.aeff_fidata[flavor + 'nc'] = aeff2d_nc

    def get_aeff_flavor(self, flavor, aeff_egy_par, aeff_coszen_par):
        """Creates the 2d aeff file from the parameterized aeff
        vs. energy .dat file, an input to the parametric settings file.
        """
        energy_group = self.label_flavintgroup_mapping['energy'][flavor]
        coszen_group = self.label_flavintgroup_mapping['coszen'][flavor.strip('_bar')]

        aeff_file = aeff_egy_par[flavor]
        aeff_arr = np.loadtxt(open_resource(aeff_file)).T
        # interpolate
        aeff_func = interp1d(aeff_arr[0], aeff_arr[1], kind='linear',
                             bounds_error=False, fill_value=0)

        czcen = get_bin_centers(self.czbins)
        ecen = get_bin_centers(self.ebins)

        # Get 1D array interpolated values at bin centers, assume no cz dep
        aeff1d = aeff_func(ecen)

        # Correct for final energy bin, since interpolation does not
        # extend to JUST right outside the final bin:
        if aeff1d[-1] == 0.0: aeff1d[-1] = aeff1d[-2]

        # Make this into a 2D array:
        aeff2d = np.reshape(np.repeat(aeff1d, len(czcen)), (len(ecen),
                                                            len(czcen)))

        # Now add cz-dependence, assuming nu and nu_bar has same dependence:
        cz_dep = eval(aeff_coszen_par[flavor.strip('_bar')])(czcen)
        # Normalize:
        # NB: All absolute value info about Aeff is carried in the energy
        # parameterization. Combining bins' Aeffs requires a flux-weighted
        # average of each bin's Aeff; assuming the same flux in each bin
        # (as we do when we marginalize to 1D for parameterization), this
        # is a simple average. Therefore, the *average* of the coszen
        # parameterization must be 1 (*not* the sum).
        cz_dep *= len(cz_dep)/np.sum(cz_dep)

        # Store interplant/lambda function for later examination
        self.energy_interpolants[energy_group] = aeff_func
        self.coszen_lambdas[coszen_group] = \
                eval(aeff_coszen_par[flavor.strip('_bar')])

        return (aeff2d*cz_dep)

    def get_aeff(self,*kwargs):
        """Returns the effective area dictionary"""
        return self.aeff_fidata

    def sample_cz_curve(self, flavint, czbin_midpoints, normalize=False):
        coszen_group = [g for g in self.coszen_lambdas.keys()
                        if flavint in g][0]
        coszen_lambda = self.coszen_lambdas[coszen_group]
        samples = coszen_lambda(czbin_midpoints)
        if normalize:
            samples /= (np.sum(samples)/len(samples))
        return samples

    def sample_egy_curve(self, flavint, ebin_midpoints):
        energy_group = [g for g in self.energy_interpolants.keys()
                        if flavint in g][0]
        aeff_func = self.energy_interpolants[energy_group]
        return aeff_func(ebin_midpoints)

    @staticmethod
    def add_argparser_args(parser):
        parser.add_argument(
            '--aeff-egy-par', metavar='RESOURCE', type=eval,
            default='''{
                "NC": "aeff/V36/cuts_V5/a_eff_nuall_nc.dat",
                "NC_bar": "aeff/V36/cuts_V5/a_eff_nuallbar_nc.dat",
                "nue": "aeff/V36/cuts_V5/a_eff_nue.dat",
                "nue_bar": "aeff/V36/cuts_V5/a_eff_nuebar.dat",
                "numu": "aeff/V36/cuts_V5/a_eff_numu.dat",
                "numu_bar": "aeff/V36/cuts_V5/a_eff_numubar.dat",
                "nutau": "aeff/V36/cuts_V5/a_eff_nutau.dat",
                "nutau_bar": "aeff/V36/cuts_V5/a_eff_nutaubar.dat",
            }''',
            help='''Stringified dictionary containing locations of
            energy-dependent parameterization of effective areas.'''
        )
        parser.add_argument(
            '--aeff-coszen-par', metavar='RESOURCE', type=str,
            default='aeff/V36/V36_aeff_cz.json',
            help='''Resource containing coszen-dependent parameterizations of
            effective areas.'''
        )
