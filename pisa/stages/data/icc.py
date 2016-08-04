import sys, os
import numpy as np

from pisa.core.stage import Stage
from pisa.utils.resources import find_resource
from pisa.utils.log import logging
from pisa import ureg, Q_
from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.core.map import Map, MapSet
from pisa.utils.log import logging
from pisa.utils.comparisons import normQuant
import h5py


class icc(Stage):

    def __init__(self, params, output_binning, disk_cache=None,
                memcaching_enabled=True, error_method=None,
                outputs_cache_depth=20, debug_mode=None):

        expected_params = (
            'atm_muon_scale',
            'icc_bg_file',
            'pid_bound',
            'pid_remove',
            'use_def1',
            'sim_ver',
            'livetime'
        )

        output_names = ('trck','cscd')

        super(self.__class__, self).__init__(
            use_transforms=False,
            stage_name='data',
            service_name='icc',
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

        # get params
        icc_bg_file = self.params.icc_bg_file.value
        sim_ver = self.params.sim_ver.value
        pid_bound = self.params.pid_bound.value.m_as('dimensionless')
        pid_remove = self.params.pid_remove.value.m_as('dimensionless')
        use_def1 = self.params.use_def1.value

	self.bin_names = self.output_binning.names
        self.bin_edges = []
        for name in self.bin_names:
            if 'energy' in  name:
                bin_edges = self.output_binning[name].bin_edges.to('GeV').magnitude
            else:
                bin_edges = self.output_binning[name].bin_edges.magnitude
            self.bin_edges.append(bin_edges)

        # the rest of this function is PISA v2 legacy code...
	logging.info('Initializing BackgroundServiceICC...')
        logging.info('Opening file: %s'%(icc_bg_file))

        try:
            bg_file = h5py.File(find_resource(icc_bg_file),'r')
        except IOError,e:
            logging.error("Unable to open icc_bg_file %s"%icc_bg_file)
            logging.error(e)
            sys.exit(1)

        self.icc_bg_dict = {}
        logging.info("Creating a ICC background dict...")

        # sanity check 
        santa_doms = bg_file['IC86_Dunkman_L6_SANTA_DirectDOMs']['value']
        l3 = bg_file['IC86_Dunkman_L3']['value']
        l4 = bg_file['IC86_Dunkman_L4']['result']
        l5 = bg_file['IC86_Dunkman_L5']['bdt_score']
        if use_def1 == True:
            l4_pass = np.all(l4==1)
        else:
            if sim_ver == 'dima' or sim_ver =='5digit':
                l4_invVICH = bg_file['IC86_Dunkman_L4']['result_invertedVICH']
                l4_pass = np.all(np.logical_or(l4==1, l4_invVICH==1))
            else:
                print "For the old simulation, def.2 background not done yet, so still use def1 for it."
                l4_pass = np.all(l4==1)
        assert(np.all(santa_doms>=3) and np.all(l3 == 1) and l4_pass and np.all(l5 >= 0.1))
        l6 = bg_file['IC86_Dunkman_L6']
        corridor_doms_over_threshold = l6['corridor_doms_over_threshold']
        inverted_corridor_cut = corridor_doms_over_threshold > 1
        assert(np.all(inverted_corridor_cut) and np.all(l6['santa_direct_doms'] >= 3) and np.all(l6['mn_start_contained'] == 1.) and np.all(l6['mn_stop_contained'] == 1.))

        dLLH = np.array(bg_file['IC86_Dunkman_L6']['delta_LLH'])
        if sim_ver == '4digit':
            reco_energy_all = np.array(bg_file['IC86_Dunkman_L6_MultiNest8D_PDG_Neutrino']['energy'])
            reco_coszen_all = np.array(np.cos(bg_file['IC86_Dunkman_L6_MultiNest8D_PDG_Neutrino']['zenith']))
        elif sim_ver == '5digit' or 'dima':
            reco_energy_all = np.array(bg_file['IC86_Dunkman_L6_PegLeg_MultiNest8D_NumuCC']['energy'])
            reco_coszen_all = np.array(np.cos(bg_file['IC86_Dunkman_L6_PegLeg_MultiNest8D_NumuCC']['zenith']))
        else:
            raise ValueError('Only allow sim_ver  4digit, 5 digit or dima!')

        # throw away delta LLH < pid_remove:
        reco_energy_all = reco_energy_all[dLLH>=pid_remove]
        reco_coszen_all = reco_coszen_all[dLLH>=pid_remove]
        dLLH = dLLH[dLLH>=pid_remove]
        pid_cut = pid_bound 
        #print "pid_remove = ", pid_remove
        #print "pid_bound = ", pid_bound

        # split in half for testing:
        #the commented out section was just a test for using subsets of the MC files
        #reco_energy_all = reco_energy_all[len(reco_energy_all)/2:] 
        #reco_coszen_all = reco_coszen_all[len(reco_coszen_all)/2:]
        #dLLH = dLLH[len(dLLH)/2:]
        #reco_energy_all = reco_energy_all[1::2]
        #reco_coszen_all = reco_coszen_all[1::2]
        #dLLH = dLLH[::2]

        # write to dictionary
        for flavor in ['cscd','trck']:
            final_events= {}
            if flavor == 'cscd':
                cut = dLLH < pid_cut 
            if flavor == 'trck':
                cut = dLLH >= pid_cut 
            final_events['reco_energy'] = reco_energy_all[cut]
            final_events['reco_coszen'] = reco_coszen_all[cut]

            logging.debug("Working on %s background"%flavor)

            icc_bg_hist,_,_ = np.histogram2d(final_events[self.bin_names[0]], final_events[self.bin_names[1]], bins=self.bin_edges)

            self.icc_bg_dict[flavor] = icc_bg_hist


    def _compute_outputs(self, inputs=None):

        scale = self.params.atm_muon_scale.value.m_as('dimensionless')
        scale *= self.params.livetime.value.m_as('common_year')

        maps = []
        for flavor in ['cscd','trck']:
            if self.error_method == 'sumw2':
                maps.append(Map(name=flavor, hist=(self.icc_bg_dict[flavor] * scale), error_hist=(np.sqrt(self.icc_bg_dict[flavor]) * scale) ,binning=self.output_binning))
            else:
                maps.append(Map(name=flavor, hist=(self.icc_bg_dict[flavor] * scale), binning=self.output_binning))
                
        template = MapSet(maps,name='icc')

        return template
