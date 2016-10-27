"""
The purpose of this stage is to reweight an event sample to include effects of
oscillation and various systematics.

This service in particular is intended to follow a `data` service which takes
advantage of the Data object being passed as a sideband in the Stage.
"""
import numpy as np
import pint

import pycuda.driver as cuda
import pycuda.autoinit

from pisa import ureg, Q_
from pisa.core.stage import Stage
from pisa.core.events import Data
from pisa.core.map import MapSet
from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.core.param import ParamSet
from pisa.utils.flavInt import ALL_NUFLAVINTS
from pisa.utils.flavInt import NuFlavInt, NuFlavIntGroup
from pisa.utils.comparisons import normQuant
from pisa.utils.const import FTYPE
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging
from pisa.utils.profiler import profile

from pisa.stages.flux.honda import honda
from pisa.stages.osc.prob3gpu import prob3gpu


class weight(Stage):
    """mc service to reweight an event sample taking into account atmospheric
    fluxes, neutrino oscillations and various other systematics.

    Parameters
    ----------
    params: ParamSet of sequence with which to instantiate a ParamSet
        Parameters which set everything besides the binning

        Parameters required by this service are
            * output_events : bool
                Flag to specify whether the service output returns a
                MapSet or the Data

            * livetime : ureg.Quantity
                Desired lifetime.

            * Flux related parameters:
                For more information see `$PISA/pisa/stages/flux/honda.py`
                - flux_file
                - flux_mode
                - atm_delta_index
                - energy_scale
                - nue_numu_ratio
                - nu_nubar_ratio
                - oversample_e
                - oversample_cz
                - cache_flux : bool
                    Flag to specifiy whether to cache the flux values if
                    calculated inside this service to a file specified
                    by `disk_cache`.

            * Oscillation related parameters:
                For more information see `$PISA/pisa/stage/osc/prob3gpu.py`
                - oscillate : bool
                    Flag to specifiy whether to include the effects of neutrino
                    oscillation.
                - earth_model
                - YeI
                - YeM
                - YeO
                - detector_depth
                - prop_height
                - no_nc_osc : bool
                    Flag to turn off oscillations for the neutral current
                    interactions.
                - deltacp
                - deltam21
                - deltam31
                - theta12
                - theta13
                - theta23

    input_names : string
        Specifies the string representation of the NuFlavIntGroup(s) that
        belong in the Data object passed to this service.

    output_binning : MultiDimBinning or convertible thereto
        The binning desired for the output maps.

    output_names : string
        Specifies the string representation of the NuFlavIntGroup(s) which will
        be produced as an output.

    error_method : None, bool, or string
        If None, False, or empty string, the stage does not compute errors for
        the transforms and does not apply any (additional) error to produce its
        outputs. (If the inputs already have errors, these are propagated.)

    debug_mode : None, bool, or string
        If None, False, or empty string, the stage runs normally.
        Otherwise, the stage runs in debug mode. This disables caching (forcing
        recomputation of any nominal transforms, transforms, and outputs).

    transforms_cache_depth
    outputs_cache_depth : int >= 0

    """
    def __init__(self, params, output_binning, input_names, output_names,
                 error_method=None, debug_mode=None, disk_cache=None,
                 memcache_deepcopy=True, outputs_cache_depth=20):

        self.weight_params = (
            'output_events',
            'livetime',
            'oscillate',
            'cache_flux'
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
            'no_nc_osc'
        )

        expected_params = self.flux_params + self.osc_params + self.weight_params

        self.neutrino = False
        self.muons = False
        self.noise = False

        self.flux_hash = None
        self.osc_hash = None

        input_names = input_names.replace(' ','').split(',')
        clean_innames = []
        for name in input_names:
            if 'muons' in name:
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
            if 'muons' in name:
                self.muons = True
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
            output_binning=output_binning
        )

        if disk_cache is not None and self.params['cache_flux'].value:
            self.instantiate_disk_cache()

    @profile
    def _compute_outputs(self, inputs=None):
        """Compute histograms for output channels."""
        if not isinstance(inputs, Data):
            raise AssertionError('inputs is not a Data object, instead is '
                                 'type {0}'.format(type(inputs)))
        self._data = inputs
        hash_params = ['livetime', 'oscillate'] + \
                list(self.flux_params) + list(self.osc_params)
        self._data.metadata['params_hash'] = self.params.values_hash
        self._data.update_hash()

        if self.neutrino:
            flux_weights = self.compute_flux_weights()
            if not self.params['oscillate'].value:
                # no oscillations
                for fig in self._data:
                    flav_pdg = NuFlavInt(fig).flavCode()
                    if flav_pdg == 12:
                        self._data[fig]['pisa_weight'] *= \
                                flux_weights[fig]['nue_flux']
                    elif flav_pdg == 14:
                        self._data[fig]['pisa_weight'] *= \
                                flux_weights[fig]['numu_flux']
                    elif flav_pdg == -12:
                        self._data[fig]['pisa_weight'] *= \
                                flux_weights[fig]['nuebar_flux']
                    elif flav_pdg == -14:
                        self._data[fig]['pisa_weight'] *= \
                                flux_weights[fig]['numubar_flux']
                    elif abs(flav_pdg) == 16:
                        self._data[fig]['pisa_weight'] *= 0.
            else:
                # oscillations
                osc_weights = self.compute_osc_weights(flux_weights)
                for fig in self._data:
                    self._data[fig]['pisa_weight'] *= osc_weights[fig]

        if self.params['output_events'].value:
            return self._data

        outputs = []
        if self.neutrino:
            trans_nu_data = self._data.transform_groups(
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

        if self.muons:
            outputs.append(self._data.histogram(
                kinds       = 'muons',
                binning     = self.output_binning,
                weights_col = 'pisa_weight',
                errors      = True,
                name        = 'muons',
                tex         = r'\rm{muons}'
            ))

        return MapSet(maps=outputs, name=self._data.metadata['name'])

    def compute_flux_weights(self):
        """Neutrino fluxes via `honda` service."""
        this_hash = normQuant([self.params[name].value
                               for name in self.flux_params])
        if self.flux_hash == this_hash:
            return self._flux_weights

        data_contains_flux = all(
            ['nue_flux' in fig and 'numu_flux' in fig and 'nuebar_flux' in fig
             and 'numubar_flux' in fig for fig in self._data.itervalues()]
        )
        if data_contains_flux:
            logging.info('Loading flux values from data.')
            flux_weights = {}
            for fig in self._data.iterkeys():
                flux_weights[fig] = {}
                flux_weights[fig]['nue_flux'] = self._data[fig]['nue_flux']
                flux_weights[fig]['numu_flux'] = self._data[fig]['numu_flux']
                flux_weights[fig]['nuebar_flux'] = self._data[fig]['nuebar_flux']
                flux_weights[fig]['numubar_flux'] = self._data[fig]['numubar_flux']
        elif self.params['cache_flux'].value:
            cache_flux_params = (
                'energy_scale',
                'flux_file',
                'flux_mode'
            )
            this_cache_hash = [self.params[name].value
                               for name in cache_flux_params]
            this_cache_hash = normQuant([self._data.metadata['name'],
                                         self._data.metadata['sample'],
                                         this_cache_hash])
            this_cache_hash = hash_obj(this_cache_hash)

            if self.disk_cache.has_key(this_cache_hash):
                logging.info('Loading flux values from cache.')
                flux_weights = self.disk_cache[this_cache_hash]
            else:
                flux_params = []
                for param in self.params:
                    if param.name in self.flux_params:
                        flux_params.append(param)
                flux_weights = self._compute_flux_weights(
                    self._data, ParamSet(flux_params)
                )
        else:
            flux_params = []
            for param in self.params:
                if param.name in self.flux_params:
                    flux_params.append(param)
            flux_weights = self._compute_flux_weights(
                self._data, ParamSet(flux_params)
            )

        if self.params['cache_flux'].value:
            if not self.disk_cache.has_key(this_cache_hash):
                logging.info('Caching flux values to disk.')
                self.disk_cache[this_cache_hash] = flux_weights

        # TODO(shivesh): more flux systematics
        for fig in flux_weights:
            nue_flux = flux_weights[fig]['nue_flux']
            numu_flux = flux_weights[fig]['numu_flux']
            nuebar_flux = flux_weights[fig]['nuebar_flux']
            numubar_flux = flux_weights[fig]['numubar_flux']

            nue_flux, nuebar_flux = self.apply_ratio_scale(
                nue_flux, nuebar_flux, self.params['nu_nubar_ratio'].value
            )
            numu_flux, numubar_flux = self.apply_ratio_scale(
                numu_flux, numubar_flux, self.params['nu_nubar_ratio'].value
            )
            nue_flux, numu_flux = self.apply_ratio_scale(
                nue_flux, numu_flux, self.params['nue_numu_ratio'].value
            )
            nuebar_flux, numubar_flux = self.apply_ratio_scale(
                nuebar_flux, numubar_flux, self.params['nue_numu_ratio'].value
            )

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
            self._data, ParamSet(osc_params), flux_weights
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

            logging.debug('Computing flux values for flavour {0}'.format(fig))
            flux_weights[fig]['nue_flux'] = flux.calculate_flux_weights(
                'nue', nu_data[fig]['energy'], nu_data[fig]['coszen']
            )
            flux_weights[fig]['numu_flux'] = flux.calculate_flux_weights(
                'numu', nu_data[fig]['energy'], nu_data[fig]['coszen']
            )
            flux_weights[fig]['nuebar_flux'] = flux.calculate_flux_weights(
                'nuebar', nu_data[fig]['energy'], nu_data[fig]['coszen']
            )
            flux_weights[fig]['numubar_flux'] = flux.calculate_flux_weights(
                'numubar', nu_data[fig]['energy'], nu_data[fig]['coszen']
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
            p = '' if flavint.isParticle() else 'bar'
            if pdg == 12: kFlav = 0
            elif pdg == 14: kFlav = 1
            elif pdg == 16: kFlav = 2

            if 'nc' in fig and params['no_nc_osc'].value:
                if kFlav == 0:
                    osc_weights[fig] = flux_weights[fig]['nue'+p+'_flux']
                elif kFlav == 1:
                    osc_weights[fig] = flux_weights[fig]['numu'+p+'_flux']
                elif kFlav == 2: osc_weights[fig] = 0.
                continue

            osc.calc_probs(
                kNuBar, kFlav, osc_data[fig]['n_evts'], **osc_data[fig]['device']
            )

            prob_e = np.zeros(osc_data[fig]['n_evts'], dtype=FTYPE)
            prob_mu = np.zeros(osc_data[fig]['n_evts'], dtype=FTYPE)
            cuda.memcpy_dtoh(prob_e, osc_data[fig]['device']['prob_e'])
            cuda.memcpy_dtoh(prob_mu, osc_data[fig]['device']['prob_mu'])

            for key in osc_data[fig]['device']:
                osc_data[fig]['device'][key].free()

            osc_weights[fig] = flux_weights[fig]['nue'+p+'_flux']*prob_e + \
                    flux_weights[fig]['numu'+p+'_flux']*prob_mu

        return osc_weights

    @staticmethod
    def apply_ratio_scale(flux_a, flux_b, ratio_scale):
        """Apply a ratio systematic to the flux weights."""
        orig_ratio = flux_a / flux_b
        orig_sum = flux_a + flux_b

        scaled_a = orig_sum / (1 + ratio_scale*orig_ratio)
        scaled_b = ratio_scale*orig_ratio * scaled_a
        return scaled_a, scaled_b

    def validate_params(self, params):
        assert isinstance(params['output_events'].value, bool)
        assert isinstance(params['livetime'].value, pint.quantity._Quantity)
        assert isinstance(params['oscillate'].value, bool)
        assert isinstance(params['cache_flux'].value, bool)
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
