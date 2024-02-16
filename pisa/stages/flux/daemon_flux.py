"""
Implementation of DEAMON flux (https://arxiv.org/abs/2303.00022) 
by Juan Pablo Yañez and Anatoli Fedynitch for use in PISA.

Maria Liubarska, J.P. Yanez 2023
"""

import numpy as np
from daemonflux import Flux

from pisa import FTYPE, TARGET
from pisa.core.stage import Stage
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from numba import jit
from scipy import interpolate


class daemon_flux(Stage):
    """
    DEAMON fulx stage
    
    Parameters
    ----------

    params: ParamSet
        Must have parameters: .. ::

            K_158G : quantity (dimensionless)

            K_2P : quantity (dimensionless)

            K_31G : quantity (dimensionless)

            antiK_158G : quantity (dimensionless)

            antiK_2P : quantity (dimensionless)

            antiK_31G : quantity (dimensionless)

            n_158G : quantity (dimensionless)

            n_2P : quantity (dimensionless)

            p_158G : quantity (dimensionless)

            p_2P : quantity (dimensionless)

            pi_158G : quantity (dimensionless)

            pi_20T : quantity (dimensionless)

            pi_2P : quantity (dimensionless)

            pi_31G : quantity (dimensionless)

            antipi_158G : quantity (dimensionless)

            antipi_20T : quantity (dimensionless)

            antipi_2P : quantity (dimensionless)

            antipi_31G : quantity (dimensionless)

            GSF_1 : quantity (dimensionless)

            GSF_2 : quantity (dimensionless)

            GSF_3 : quantity (dimensionless)

            GSF_4 : quantity (dimensionless)

            GSF_5 : quantity (dimensionless)

            GSF_6 : quantity (dimensionless)

    """

    def __init__(
        self,
        **std_kwargs,
    ):

        self.deamon_params = ['K_158G',
                              'K_2P',
                              'K_31G',
                              'antiK_158G',
                              'antiK_2P',
                              'antiK_31G',
                              'n_158G',
                              'n_2P',
                              'p_158G',
                              'p_2P',
                              'pi_158G',
                              'pi_20T',
                              'pi_2P',
                              'pi_31G',
                              'antipi_158G',
                              'antipi_20T',
                              'antipi_2P',
                              'antipi_31G',
                              'GSF_1',
                              'GSF_2',
                              'GSF_3',
                              'GSF_4',
                              'GSF_5',
                              'GSF_6',
                             ]

        self.deamon_names =  ['K+_158G',
                              'K+_2P',
                              'K+_31G',
                              'K-_158G',
                              'K-_2P',
                              'K-_31G',
                              'n_158G',
                              'n_2P',
                              'p_158G',
                              'p_2P',
                              'pi+_158G',
                              'pi+_20T',
                              'pi+_2P',
                              'pi+_31G',
                              'pi-_158G',
                              'pi-_20T',
                              'pi-_2P',
                              'pi-_31G',
                              'GSF_1',
                              'GSF_2',
                              'GSF_3',
                              'GSF_4',
                              'GSF_5',
                              'GSF_6',
                             ]

        # init base class
        super(daemon_flux, self).__init__(
            expected_params=tuple(self.deamon_params),
            **std_kwargs,
        )

    def setup_function(self):

        self.data.representation = self.calc_mode

        self.flux_obj = Flux(location='IceCube')

        for container in self.data:
            container['nu_flux'] = np.empty((container.size, 2), dtype=FTYPE)

    @profile
    def compute_function(self):

        self.data.representation = self.calc_mode

        # get modified parameters (in units of sigma)
        modif_param_dict = {}
        for i,k in enumerate(self.deamon_params):
            modif_param_dict[self.deamon_names[i]] = getattr(self.params, k).value.m_as("dimensionless")

        flux_map_numu    = make_2d_flux_map(self.flux_obj,
                                            particle = 'numu',
                                            params = modif_param_dict)
        flux_map_numubar = make_2d_flux_map(self.flux_obj,
                                            particle = 'antinumu',
                                            params = modif_param_dict)
        flux_map_nue     = make_2d_flux_map(self.flux_obj,
                                            particle = 'nue',
                                            params = modif_param_dict)
        flux_map_nuebar  = make_2d_flux_map(self.flux_obj,
                                            particle = 'antinue',
                                            params = modif_param_dict)
        

        # calc modified flux using provided parameters
        for container in self.data:
            nubar = container['nubar']

            nue_flux   = evaluate_flux_map(flux_map_nuebar if nubar>0 else flux_map_nue, 
                                           container['true_energy'],
                                           container['true_coszen'])

            numu_flux  = evaluate_flux_map(flux_map_numubar if nubar>0 else flux_map_numu, 
                                           container['true_energy'],
                                           container['true_coszen'])


            container['nu_flux'][:,0] = nue_flux
            container['nu_flux'][:,1] = numu_flux

            container.mark_changed("nu_flux")

@jit(forceobj=True)
def make_2d_flux_map(flux_obj,
                     particle = 'numuflux',
                     egrid = np.logspace(-1,5,500),
                     params = {},
                     ):

    icangles = list(flux_obj.zenith_angles)
    icangles_array = np.array(icangles, dtype=float)
    mysort = icangles_array.argsort()
    icangles = np.array(icangles)[mysort][::-1]

    flux_ref = np.zeros([len(egrid), len(icangles)])
    costheta_angles = np.zeros(len(icangles))

    for index in range(len(icangles)):
        costheta_angles[index] = np.cos(np.deg2rad(float(icangles[index])))
        flux_ref[:,index] = flux_obj.flux(egrid, icangles[index], particle, params)

    fcn = interpolate.RectBivariateSpline(egrid,
                                          costheta_angles,
                                          flux_ref)
    return fcn

@jit(forceobj=True)
def evaluate_flux_map(flux_map, true_energy, true_coszen):

    uconv = true_energy**-3 * 1e4
    return flux_map.ev(true_energy, true_coszen) * uconv
            