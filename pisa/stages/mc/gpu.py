import sys, os
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import time

from pisa.core.stage import Stage
from pisa.utils.resources import find_resource
from pisa.utils.log import logging
from pisa.stages.osc.prob3gpu import Prob3GPU
from pisa.utils.GPUhist import GPUhist
from pisa.stages.mc.GPUweight import GPUweight
from pisa.utils.const import FTYPE
from pisa.utils.events import Events
from pisa import ureg, Q_
from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.core.map import Map, MapSet
from pisa.utils.log import logging
from pisa.utils.comparisons import normQuant
from pisa.utils.hash import hash_obj


def copy_dict_to_d(events):
    d_events = {}
    for key, val in events.items():
        d_events[key] = cuda.mem_alloc(val.nbytes)
        cuda.memcpy_htod(d_events[key], val)
    return d_events

class gpu(Stage):

    def __init__(self, params, output_binning, disk_cache=None,
                memcaching_enabled=True, error_method=None,
                outputs_cache_depth=20, debug_mode=None):

        expected_params = (
            'detector_depth',
            'earth_model',
            'prop_height', 
            'YeI',
            'YeO',
            'YeM',
            'events_file',
            'nu_nubar_ratio',
            'nue_numu_ratio',
            'theta12',
            'theta13',
            'theta23',
            'deltam21',
            'deltam31',
            'deltacp',
            'livetime',
            'aeff_scale',
            'pid_bound',
            'pid_remove',
            'nu_nc_norm',
            'nutau_cc_norm'
        )


        #output_names = ( 'nue_cc_trck','nue_cc_cscd',
        #                'nuebar_cc_trck','nuebar_cc_cscd',
        #                'numu_cc_trck','numu_cc_cscd',
        #                'numubar_cc_trck','numubar_cc_cscd',
        #                'nutau_cc_trck','nutau_cc_cscd',
        #                'nutaubar_cc_trck','nutaubar_cc_cscd',
        #                'nue_nc_trck','nue_nc_cscd',
        #                'nuebar_nc_trck','nuebar_nc_cscd',
        #                'numu_nc_trck','numu_nc_cscd',
        #                'numubar_nc_trck','numubar_nc_cscd',
        #                'nutau_nc_trck','nutau_nc_cscd',
        #                'nutaubar_nc_trck','nutaubar_nc_cscd',
        #                )
        output_names = ('trck','cscd')

        super(self.__class__, self).__init__(
            use_transforms=False,
            stage_name='mc',
            service_name='gpu',
            params=params,
            expected_params=expected_params,
            output_names=output_names,
            error_method=error_method,
            disk_cache=disk_cache,
            memcaching_enabled=memcaching_enabled,
            outputs_cache_depth=outputs_cache_depth,
            output_binning=output_binning,
            debug_mode=debug_mode
        )

        self.osc_hash = None

        # initialize classes
        earth_model = find_resource(self.params.earth_model.value)
        YeI = self.params.YeI.value.m_as('dimensionless')
        YeO = self.params.YeO.value.m_as('dimensionless')
        YeM = self.params.YeM.value.m_as('dimensionless')
        prop_height = self.params.prop_height.value.m_as('km')
        detector_depth = self.params.detector_depth.value.m_as('km')

        self.osc = Prob3GPU(detector_depth,
                            earth_model,
                            prop_height,
                            YeI,
                            YeO,
                            YeM)

        self.weight = GPUweight()

        self.e_dim_num = self.output_binning.names.index('reco_energy')
        self.cz_dim_num = self.output_binning.names.index('reco_coszen')
        self.bin_names = self.output_binning.names
        self.bin_edges = []
        for name in self.bin_names:
            if 'energy' in  name:
                bin_edges = self.output_binning[name].bin_edges.to('GeV').magnitude.astype(FTYPE)
            else:
                bin_edges = self.output_binning[name].bin_edges.magnitude.astype(FTYPE)
            self.bin_edges.append(bin_edges)
            

        self.histogrammer = GPUhist(*self.bin_edges)

        # --- Load events
        # open Events file
        evts = Events(self.params.events_file.value)

        # Load and copy events
        variables = ['true_energy', 'true_coszen', 'reco_energy', 'reco_coszen',
                    'neutrino_nue_flux', 'neutrino_numu_flux', 'neutrino_oppo_nue_flux',
                    'neutrino_oppo_numu_flux', 'weighted_aeff', 'pid']
        empty = ['prob_e', 'prob_mu', 'weight_trck', 'weight_cscd']
        self.flavs = ['nue_cc', 'numu_cc', 'nutau_cc', 'nue_nc', 'numu_nc', 'nutau_nc',
                'nuebar_cc', 'numubar_cc', 'nutaubar_cc', 'nuebar_nc', 'numubar_nc', 'nutaubar_nc']
        kFlavs = [0, 1, 2] * 4
        kNuBars = [1] *6 + [-1] * 6

        logging.info('read in events and copy to GPU')
        start_t = time.time()
        self.events_dict = {}
        for flav, kFlav, kNuBar in zip(self.flavs, kFlavs, kNuBars):
            self.events_dict[flav] = {}
            # neutrinos: 1, anti-neutrinos: -1 
            self.events_dict[flav]['kNuBar'] = kNuBar
            # electron: 0, muon: 1, tau: 2
            self.events_dict[flav]['kFlav'] = kFlav
            # host arrays
            self.events_dict[flav]['host'] = {}
            for var in variables:
                self.events_dict[flav]['host'][var] = evts[flav][var].astype(FTYPE)
            self.events_dict[flav]['n_evts'] = np.uint32(len(self.events_dict[flav]['host'][variables[0]]))
            for var in empty:
                self.events_dict[flav]['host'][var] = np.zeros(self.events_dict[flav]['n_evts'], dtype=FTYPE)
            # calulate layers
            self.events_dict[flav]['host']['numLayers'], self.events_dict[flav]['host']['densityInLayer'], self.events_dict[flav]['host']['distanceInLayer'] = self.osc.calc_Layers(self.events_dict[flav]['host']['true_coszen'])
            # copy to device arrays
            self.events_dict[flav]['device'] = copy_dict_to_d(self.events_dict[flav]['host'])
        end_t = time.time()
        logging.debug('copy done in %.4f ms'%((end_t - start_t) * 1000))

    def _compute_outputs(self, inputs=None):
        logging.info('retreive weighted histo')
        start_t = time.time()
        osc_params = ['theta12','theta13','theta23','deltam21','deltam31','deltacp']
        theta12 = self.params.theta12.value.m_as('rad')
        theta13 = self.params.theta13.value.m_as('rad')
        theta23 = self.params.theta23.value.m_as('rad')
        deltam21 = self.params.deltam21.value.m_as('eV**2')
        deltam31 = self.params.deltam31.value.m_as('eV**2')
        deltacp = self.params.deltacp.value.m_as('rad')
        
        # get hash
        osc_hash = hash_obj(normQuant([self.params[name].value for name in osc_params]))
        recalc_osc = not (osc_hash == self.osc_hash)

        livetime = self.params.livetime.value.m_as('seconds')
        pid_bound = self.params.pid_bound.value.m_as('dimensionless')
        pid_remove = self.params.pid_remove.value.m_as('dimensionless')
        aeff_scale = self.params.aeff_scale.value.m_as('dimensionless')
        nue_numu_ratio = self.params.nue_numu_ratio.value.m_as('dimensionless')
        nu_nubar_ratio = self.params.nu_nubar_ratio.value.m_as('dimensionless')

        if recalc_osc:
            self.osc.update_MNS(theta12, theta13, theta23, deltam21, deltam31, deltacp)
        tot = 0
        for flav in self.flavs:
            if recalc_osc:
                self.osc.calc_probs(self.events_dict[flav]['kNuBar'], self.events_dict[flav]['kFlav'],
                                self.events_dict[flav]['n_evts'], **self.events_dict[flav]['device'])

            self.weight.calc_weight(self.events_dict[flav]['n_evts'], livetime=livetime,
                                pid_bound=pid_bound, pid_remove=pid_remove,
                                aeff_scale=aeff_scale, nue_numu_ratio=nue_numu_ratio, 
                                nu_nubar_ratio=nu_nubar_ratio, kNuBar=self.events_dict[flav]['kNuBar'],
                                **self.events_dict[flav]['device'])

            self.events_dict[flav]['hist_cscd'] = self.histogrammer.get_hist(self.events_dict[flav]['n_evts'],
                                                                    self.events_dict[flav]['device'][self.bin_names[0]],
                                                                    self.events_dict[flav]['device'][self.bin_names[1]],
                                                                    self.events_dict[flav]['device']['weight_cscd'])

            self.events_dict[flav]['hist_trck'] = self.histogrammer.get_hist(self.events_dict[flav]['n_evts'],
                                                                    self.events_dict[flav]['device'][self.bin_names[0]],
                                                                    self.events_dict[flav]['device'][self.bin_names[1]],
                                                                    self.events_dict[flav]['device']['weight_trck'])

            tot += self.events_dict[flav]['n_evts']
        end_t = time.time()
        logging.debug('GPU done in %.4f ms for %s events'%(((end_t - start_t) * 1000),tot))
        
        # set new hash
        self.osc_hash = osc_hash

        maps = []
        for i,flav in enumerate(self.flavs):
            if flav in ['nutau_cc','nutaubar_cc']:
                f = self.params.nutau_cc_norm.value.m_as('dimensionless')
            elif '_nc' in flav:
                f = self.params.nu_nc_norm.value.m_as('dimensionless')
            else:
                f = 1.0
            if i == 0:
                hist_cscd = self.events_dict[flav]['hist_cscd'] * f
                hist_trck = self.events_dict[flav]['hist_trck'] * f
            else:
                hist_cscd += self.events_dict[flav]['hist_cscd'] * f
                hist_trck += self.events_dict[flav]['hist_trck'] * f

            
        maps.append(Map(name='cscd', hist=hist_cscd, binning=self.output_binning))
        maps.append(Map(name='trck', hist=hist_trck, binning=self.output_binning))

        logging.info('total number of cscd events: %s'%np.sum(hist_cscd))
        logging.info('total number of trck events: %s'%np.sum(hist_trck))

        template = MapSet(maps,name='gpu_mc')

        return template
