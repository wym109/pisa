"""
The purpose of this stage is to reweight an event sample to include effects of
oscillation and various systematics.

This service in particular is intended to follow a `data` service which takes
advantage of the Events object being passed as a sideband in the Stage.
"""
from copy import deepcopy

import numpy as np
import pint
from uncertainties import unumpy as unp

import pycuda.driver as cuda
import pycuda.autoinit

from pisa import ureg, Q_
from pisa.core.stage import Stage
from pisa.core.map import Map, MapSet
from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.core.param import ParamSet
from pisa.utils.flavInt import ALL_NUFLAVINTS
from pisa.utils.flavInt import NuFlavInt, NuFlavIntGroup, FlavIntDataGroup
from pisa.utils.comparisons import normQuant
from pisa.utils.const import FTYPE
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging
from pisa.utils.profiler import profile

from pisa.stages.flux.honda import honda
from pisa.stages.osc.prob3gpu import prob3gpu


class weight(Stage):
    """
            * livetime : ureg.Quantity
                Desired lifetime.

    TODO(shivesh): docstring."""
    def __init__(self, params, output_binning, input_names, output_names,
                 error_method=None, debug_mode=None, disk_cache=None,
                 memcache_deepcopy=True, transforms_cache_depth=20,
                 outputs_cache_depth=20):

        self.weight_params = (
            'livetime',
            'oscillate',
        )

        self.flux_params = (
            'atm_delta_index',
            'energy_scale',
            'nu_nubar_ratio',
            'nue_numu_ratio',
            'oversample_e',
            'oversample_cz',
            'flux_file',
            'flux_mode'
        )

        self.osc_params = (
            'detector_depth',
            'earth_model',
            'prop_height',
            'YeI',
            'YeO',
            'YeM',
            'theta12',
            'theta13',
            'theta23',
            'deltam21',
            'deltam31',
            'deltacp',
            'no_nc_osc',
        )

        expected_params = self.flux_params + self.osc_params + self.weight_params

        self.neutrino = False
        self.muongun = False
        self.noise = False

        self.flux_hash = None
        self.osc_hash = None

        input_names = input_names.replace(' ','').split(',')
        clean_innames = []
        for name in input_names:
            if 'muongun' in name:
                raise NotImplementedError
                clean_innames.append(name)
            elif 'noise' in name:
                clean_innames.append(name)
            elif 'all_nu' in name:
                clean_innames = [str(NuFlavIntGroup(f)) for f in ALL_NUFLAVINTS]
            else:
                clean_innames.append(str(NuFlavIntGroup(name)))

        output_names = output_names.replace(' ','').split(',')
        clean_outnames = []
        self._output_nu_groups = []
        for name in output_names:
            if 'muongun' in name:
                raise NotImplementedError
                self.muongun = True
                clean_outnames.append(name)
            elif 'noise' in name:
                self.noise = True
                clean_outnames.append(name)
            elif 'all_nu' in name:
                self.neutrino = True
                self._output_nu_groups = \
                        [NuFlavIntGroup(f) for f in ALL_NUFLAVINTS]
            else:
                self.neutrino = True
                self._output_nu_groups.append(NuFlavIntGroup(name))

        if self.neutrino:
            clean_outnames += [str(f) for f in self._output_nu_groups]

        super(self.__class__, self).__init__(
            use_transforms=False,
            params=params,
            expected_params=expected_params,
            input_names=clean_innames,
            output_names=clean_outnames,
            error_method=error_method,
            debug_mode=debug_mode,
            disk_cache=disk_cache,
            memcache_deepcopy=memcache_deepcopy,
            outputs_cache_depth=outputs_cache_depth,
            transforms_cache_depth=transforms_cache_depth,
            output_binning=output_binning
        )

    @profile
    def _compute_outputs(self, inputs=None):
        """Compute nominal histograms for output channels."""
        self.nu_data = inputs

        if self.neutrino:
            flux_weights = self.compute_flux_weights()
            if not self.params['oscillate'].value:
                # no oscillations
                for fig in self.nu_data:
                    flav_pdg = abs(NuFlavInt(fig).flavCode())
                    if flav_pdg == 12:
                        self.nu_data[fig]['pisa_weight'] *= \
                                flux_weights[fig]['nue_flux']
                    elif flav_pdg == 14:
                        self.nu_data[fig]['pisa_weight'] *= \
                                flux_weights[fig]['numu_flux']
                    elif flav_pdg == 16:
                        self.nu_data[fig]['pisa_weight'] *= 0.
            else:
                # oscillations
                osc_weights = self.compute_osc_weights(flux_weights)
                for fig in self.nu_data:
                    self.nu_data[fig]['pisa_weight'] *= osc_weights[fig]

        outputs = []
        if self.neutrino:
            trans_nu_data = self.nu_data.transform_groups(
                self._output_nu_groups
            )
            for fig in trans_nu_data.iterkeys():
                outputs.append(trans_nu_data.histogram(
                    kinds       = fig,
                    binning     = self.output_binning,
                    weights_col = 'pisa_weight',
                    errors      = True,
                    name        = str(NuFlavIntGroup(fig)),
                ))

        return MapSet(maps=outputs, name=self.nu_data.metadata['name'])

    def compute_flux_weights(self):
        """Neutrino fluxes via `honda` service."""
        this_hash = normQuant([self.params[name].value
                               for name in self.flux_params])
        if self.flux_hash == this_hash:
            return self._flux_weights
        data_contains_flux = all(['nue_flux' in fig and 'numu_flux' in fig
                                  for fig in self.nu_data.itervalues()])
        if data_contains_flux:
            flux_weights = {}
            for fig in self.nu_data.iterkeys():
                flux_weights[fig] = {}
                flux_weights[fig]['nue_flux'] = self.nu_data[fig]['nue_flux']
                flux_weights[fig]['numu_flux'] = self.nu_data[fig]['numu_flux']
        try:
            flux_weights = self._flux_weights
        except AttributeError:
            flux_params = []
            for param in self.params:
                if param.name in self.flux_params:
                    flux_params.append(param)
            flux_weights = self._compute_flux_weights(
                self.nu_data, ParamSet(flux_params)
            )
        # TODO(shivesh): flux systematics

        self.flux_hash = this_hash
        self._flux_weights = flux_weights
        return flux_weights

    def compute_osc_weights(self, flux_weights):
        """Neutrino oscillations calculation via Prob3."""
        this_hash = normQuant([self.params[name].value
                               for name in self.osc_params])
        if self.flux_hash == this_hash:
            return self._osc_weights
        osc_params = []
        for param in self.params:
            if param.name in self.osc_params:
                osc_params.append(param)
        osc_weights = self._compute_osc_weights(
            self.nu_data, ParamSet(osc_params), flux_weights
        )
        self.osc_hash = this_hash
        self._osc_weights = osc_weights
        return self._osc_weights

    @staticmethod
    def _compute_flux_weights(nu_data, params):
        """Neutrino fluxes via integral preserving spline."""
        logging.debug('Computing flux values')
        fake_binning = MultiDimBinning((
            OneDimBinning(name='true_energy', num_bins=2, is_log=True,
                          domain=[1, 300]*ureg.GeV),
            OneDimBinning(name='true_coszen', num_bins=2, is_lin=True,
                          domain=[-1, 1])
        ))

        flux = honda(
            params = params,
            output_binning = fake_binning,
            error_method = None,
            outputs_cache_depth = 0,
            memcache_deepcopy = False,
        )

        flux_weights = {}
        for fig in nu_data.iterkeys():
            flux_weights[fig] = {}
            particle = NuFlavInt(fig).isParticle()
            if particle: prefix = ''
            else : prefix = 'bar'

            logging.debug('Computing flux values for flavour {0}'.format(fig))
            # TODO(shivesh): make faster
            flux_weights[fig]['nue_flux'] = flux.calculate_flux_weights(
                'nue'+prefix, nu_data[fig]['energy'], nu_data[fig]['coszen']
            )
            flux_weights[fig]['numu_flux'] = flux.calculate_flux_weights(
                'numu'+prefix, nu_data[fig]['energy'], nu_data[fig]['coszen']
            )

        return flux_weights

    @staticmethod
    def _compute_osc_weights(nu_data, params, flux_weights):
        """Neutrino oscillations calculation via Prob3."""
        logging.debug('Computing oscillation weights')
        # Read parameters in, convert to the units used internally for
        # computation, and then strip the units off. Note that this also
        # enforces compatible units (but does not sanity-check the numbers).
        theta12 = params['theta12'].m_as('rad')
        theta13 = params['theta13'].m_as('rad')
        theta23 = params['theta23'].m_as('rad')
        deltam21 = params['deltam21'].m_as('eV**2')
        deltam31 = params['deltam31'].m_as('eV**2')
        deltacp = params['deltacp'].m_as('rad')
        YeI = params['YeI'].m_as('dimensionless')
        YeO = params['YeO'].m_as('dimensionless')
        YeM = params['YeM'].m_as('dimensionless')
        prop_height = params['prop_height'].m_as('km')

        osc = prob3gpu(
            params = params,
            input_binning = None,
            output_binning = None,
            error_method = None,
            memcache_deepcopy = False,
            transforms_cache_depth = 0,
            outputs_cache_depth = 0
        )

        osc_data = {}
        for fig in nu_data.iterkeys():
            if 'nc' in fig and params['no_nc_osc'].value:
                continue
            osc_data[fig] = {}
            energy_array = nu_data[fig]['energy'].astype(FTYPE)
            coszen_array = nu_data[fig]['coszen'].astype(FTYPE)
            pisa_weights = nu_data[fig]['pisa_weight'].astype(FTYPE)
            n_evts = np.uint32(len(energy_array))
            osc_data[fig]['n_evts'] = n_evts

            device = {}
            device['true_energy'] = energy_array
            device['prob_e'] = np.zeros(n_evts, dtype=FTYPE)
            device['prob_mu'] = np.zeros(n_evts, dtype=FTYPE)
            out_layers_n = ('numLayers', 'densityInLayer', 'distanceInLayer')
            out_layers = osc.calc_Layers(coszen_array)
            device.update(dict(zip(out_layers_n, out_layers)))

            osc_data[fig]['device'] = {}
            for key in device.iterkeys():
                osc_data[fig]['device'][key] = \
                        cuda.mem_alloc(device[key].nbytes)
                cuda.memcpy_htod(osc_data[fig]['device'][key], device[key])

        osc.update_MNS(theta12, theta13, theta23, deltam21, deltam31, deltacp)

        osc_weights = {}
        for fig in nu_data.iterkeys():
            flavint = NuFlavInt(fig)
            pdg = abs(flavint.flavCode())
            kNuBar = 1 if flavint.isParticle() else -1
            if pdg == 12: kFlav = 0
            elif pdg == 14: kFlav = 1
            elif pdg == 16: kFlav = 2
            if 'nc' in fig and params['no_nc_osc'].value:
                if kFlav == 0: osc_weights[fig] = flux_weights[fig]['nue_flux']
                elif kFlav == 1:
                    osc_weights[fig] = flux_weights[fig]['numu_flux']
                elif kFlav == 2: osc_weights[fig] = 0.
                continue

            osc.calc_probs(
                kNuBar, kFlav, osc_data[fig]['n_evts'], **osc_data[fig]['device']
            )

            prob_e = np.zeros(osc_data[fig]['n_evts'], dtype=FTYPE)
            prob_mu = np.zeros(osc_data[fig]['n_evts'], dtype=FTYPE)
            cuda.memcpy_dtoh(prob_e, osc_data[fig]['device']['prob_e'])
            cuda.memcpy_dtoh(prob_mu, osc_data[fig]['device']['prob_mu'])

            osc_weights[fig] = flux_weights[fig]['nue_flux']*prob_e + \
                    flux_weights[fig]['numu_flux']*prob_mu

        return osc_weights

    def validate_params(self, params):
        assert isinstance(params['livetime'].value, pint.quantity._Quantity)
        assert isinstance(params['oscillate'].value, bool)
        assert isinstance(params['earth_model'].value, basestring)
        assert isinstance(params['detector_depth'].value, pint.quantity._Quantity)
        assert isinstance(params['theta12'].value, pint.quantity._Quantity)
        assert isinstance(params['theta13'].value, pint.quantity._Quantity)
        assert isinstance(params['theta23'].value, pint.quantity._Quantity)
        assert isinstance(params['deltam21'].value, pint.quantity._Quantity)
        assert isinstance(params['deltam31'].value, pint.quantity._Quantity)
        assert isinstance(params['deltacp'].value, pint.quantity._Quantity)
        assert isinstance(params['YeI'].value, pint.quantity._Quantity)
        assert isinstance(params['YeO'].value, pint.quantity._Quantity)
        assert isinstance(params['YeM'].value, pint.quantity._Quantity)
        assert isinstance(params['prop_height'].value, pint.quantity._Quantity)
        assert isinstance(params['no_nc_osc'].value, bool)
