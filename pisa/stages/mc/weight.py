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
from pisa.utils.flavInt import NuFlavInt, NuFlavIntGroup, FlavIntDataGroup
from pisa.utils.comparisons import normQuant
from pisa.utils.const import FTYPE
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging
from pisa.utils.profiler import profile

from pisa.stages.data.sample import sideband
from pisa.stages.osc.prob3gpu import prob3gpu


class weight(Stage):
    """
            * livetime : ureg.Quantity
                Desired lifetime.

    TODO(shivesh): docstring."""
    def __init__(self, params, output_binning, output_names, error_method=None,
                 debug_mode=None, disk_cache=None, memcache_deepcopy=True,
                 transforms_cache_depth=20, outputs_cache_depth=20):

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

        self.weight_params = (
            'livetime',
        )

        expected_params = (self.osc_params + self.weight_params)

        self.osc_hash = None

        output_names = output_names.replace(' ','').split(',')
        self._clean_outnames = []
        self._output_nu_groups = []
        for name in output_names:
            if 'muongun' in name:
                self.muongun = True
                self._clean_outnames.append(name)
            elif 'noise' in name:
                self.noise = True
                self._clean_outnames.append(name)
            else:
                self.neutrino = True
                self._output_nu_groups.append(NuFlavIntGroup(name))

        if self.neutrino:
            self._clean_outnames += [str(f) for f in self._output_nu_groups]

        super(self.__class__, self).__init__(
            use_transforms=False,
            params=params,
            expected_params=expected_params,
            input_names=['events'],
            output_names=output_names,
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
        if not isinstance(inputs, sideband):
            raise ValueError('inputs unrecognised type '
                             '{0}'.format(type(inputs)))
        if self.neutrino:
            nu_fidg = inputs.side_obj['neutrino']
            osc_weights = self.compute_osc_weights(nu_fidg)
            for fig in nu_fidg:
                if 'nc' in fig and params['no_nc_osc'].value:
                    continue
                # TODO(shivesh): osc_nue = nue*pee + numu*pme
                nu_fidg[fig]['pisa_weight'] *= osc_weights[fig]
        if self.muongun:
            muongun_events = inputs.sideband['muongun']

        outputs = []
        if self.neutrino:
            trans_nu_fidg = nu_fidg(
                self._output_nu_groups
            )
            for fig in trans_nu_fidg.iterkeys():
                outputs.append(self._histogram(
                    events  = trans_nu_fidg[fig],
                    binning = self.output_binning,
                    weights = trans_nu_fidg[fig]['pisa_weight'],
                    errors  = True,
                    name    = str(NuFlavIntGroup(fig)),
                ))

        if self.muongun:
            outputs.append(self._histogram(
                events  = muongun_events,
                binning = self.output_binning,
                weights = self._muongun_events['pisa_weight'],
                errors  = True,
                name    = 'muongun',
                tex     = r'\rm{muongun}'
            ))

        name = sideband.name
        return MapSet(maps=outputs, name=name)


    def compute_osc_weights(self, events):
        """Neutrino oscillations calculation via Prob3."""
        this_hash = normQuant([self.params[name].value
                               for name in self.osc_params])
        if not events.has_key('neutrino') or self.osc_hash == this_hash:
            return
        osc_params = []
        for param in self.params:
            if param.name in self.osc_params:
                osc_params.append(param)
        osc_weights = self._compute_osc_weights(
            events['neutrino'], ParamSet(osc_params)
        )
        self.osc_hash = this_hash
        return osc_weights

    @staticmethod
    def _compute_osc_weights(nu_fidg, params):
        """Neutrino oscillations calculation via Prob3."""
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
        for fig in nu_fidg.iterkeys():
            if 'nc' in fig and params['no_nc_osc'].value:
                continue
            osc_data[fig] = {}
            energy_array = nu_fidg[fig]['energy'].astype(FTYPE)
            coszen_array = np.cos(nu_fidg[fig]['zenith']).astype(FTYPE)
            pisa_weights = nu_fidg[fig]['pisa_weight'].astype(FTYPE)
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

        prob_tables = {}
        for fig in nu_fidg.iterkeys():
            if 'nc' in fig and params['no_nc_osc'].value:
                continue
            flavint = NuFlavInt(fig)
            pdg = abs(flavint.flavCode())
            kNuBar = 1 if flavint.isParticle() else -1
            if pdg == 12: kFlav = 0
            elif pdg == 14: kFlav = 1
            elif pdg == 16: kFlav = 2

            osc.calc_probs(
                kNuBar, kFlav, osc_data[fig]['n_evts'], **osc_data[fig]['device']
            )

            prob_tables[fig] = {}
            prob_tables[fig]['prob_e'] = \
                    np.zeros(osc_data[fig]['n_evts'], dtype=FTYPE)
            prob_tables[fig]['prob_mu'] = \
                    np.zeros(osc_data[fig]['n_evts'], dtype=FTYPE)
            cuda.memcpy_dtoh(prob_tables[fig]['prob_e'],
                             osc_data[fig]['device']['prob_e'])
            cuda.memcpy_dtoh(prob_tables[fig]['prob_mu'],
                             osc_data[fig]['device']['prob_mu'])

        return prob_tables

    @staticmethod
    def _histogram(events, binning, weights=None, errors=False, **kwargs):
        """Histogram the events given the input binning."""
        if isinstance(binning, OneDimBinning):
            binning = MultiDimBinning([binning])
        elif not isinstance(binning, MultiDimBinning):
            raise TypeError('Unhandled type %s for `binning`.' %type(binning))
        if not isinstance(events, dict):
            raise TypeError('Unhandled type %s for `events`.' %type(events))

        bin_names = binning.names
        bin_edges = [edges.m for edges in binning.bin_edges]
        for name in bin_names:
            if not events.has_key(name):
                if 'coszen' in name and events.has_key('zenith'):
                    events[name] = np.cos(events['zenith'])
                else:
                    raise AssertionError('Input events object does not have '
                                         'key {0}'.format(name))

        sample = [events[colname] for colname in bin_names]
        hist, edges = np.histogramdd(
            sample=sample, weights=weights, bins=bin_edges
        )
        if errors:
            hist2, edges = np.histogramdd(
                sample=sample, weights=np.square(weights), bins=bin_edges
            )
            hist = unp.uarray(hist, np.sqrt(hist2))

        return Map(hist=hist, binning=binning, **kwargs)

    def validate_params(self, params):
        assert isinstance(params['livetime'].value, pint.quantity._Quantity)
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
