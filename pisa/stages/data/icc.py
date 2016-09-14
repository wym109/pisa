import os
import sys

import h5py
import numpy as np

from pisa import ureg, Q_
from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.core.map import Map, MapSet
from pisa.core.stage import Stage
from pisa.utils.comparisons import normQuant
from pisa.utils.log import logging
from pisa.utils.resources import find_resource


# TODO: use logging in lieu of print!
class icc(Stage):
    """TODO: document me, Philipp!"""

    def __init__(self, params, output_binning, disk_cache=None,
                memcache_deepcopy=True, error_method=None,
                outputs_cache_depth=20, debug_mode=None):

        expected_params = (
            'atm_muon_scale',
            'icc_bg_file',
            'pid_bound',
            'pid_remove',
            'use_def1',
            'sim_ver',
            'livetime',
            'bdt_cut',
            'alt_icc_bg_file'
        )

        output_names = ('trck', 'cscd')

        super(self.__class__, self).__init__(
            use_transforms=False,
            stage_name='data',
            service_name='icc',
            params=params,
            expected_params=expected_params,
            output_names=output_names,
            error_method=error_method,
            disk_cache=disk_cache,
            memcache_deepcopy=memcache_deepcopy,
            outputs_cache_depth=outputs_cache_depth,
            output_binning=output_binning,
            debug_mode=debug_mode
        )

        # get params
        icc_bg_file = self.params.icc_bg_file.value
        if self.error_method == 'sumw2+shape':
            alt_icc_bg_file = self.params.alt_icc_bg_file.value
        else:
            alt_icc_bg_file = None
        sim_ver = self.params.sim_ver.value
        pid_bound = self.params.pid_bound.m_as('dimensionless')
        pid_remove = self.params.pid_remove.m_as('dimensionless')
        use_def1 = self.params.use_def1.value
        bdt_cut = self.params.bdt_cut.value.m_as('dimensionless')

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
            if alt_icc_bg_file is not None:
                alt_bg_file = h5py.File(find_resource(alt_icc_bg_file),'r')
        except IOError,e:
            logging.error("Unable to open icc_bg_file %s"%icc_bg_file)
            logging.error(e)
            sys.exit(1)

        # sanity check 
        santa_doms = bg_file['IC86_Dunkman_L6_SANTA_DirectDOMs']['value']
        l3 = bg_file['IC86_Dunkman_L3']['value']
        l4 = bg_file['IC86_Dunkman_L4']['result']
        l5 = bg_file['IC86_Dunkman_L5']['bdt_score']
        l6 = bg_file['IC86_Dunkman_L6']
        if use_def1:
            l4_pass = np.all(l4==1)
        else:
            if sim_ver in ['5digit', 'dima']:
                l4_invVICH = bg_file['IC86_Dunkman_L4']['result_invertedVICH']
                l4_pass = np.all(np.logical_or(l4==1, l4_invVICH==1))
            else:
                print ('For the old simulation, def.2 background not done yet,'
                       ' so still use def1 for it.')
                l4_pass = np.all(l4==1)
        assert (np.all(santa_doms>=3) and np.all(l3 == 1) and l4_pass and
                np.all(l5 >= 0.1))
        l6 = bg_file['IC86_Dunkman_L6']
        corridor_doms_over_threshold = l6['corridor_doms_over_threshold']

        inverted_corridor_cut = corridor_doms_over_threshold > 1
        assert (np.all(inverted_corridor_cut) and
                np.all(l6['santa_direct_doms'] >= 3) and
                np.all(l6['mn_start_contained'] == 1.) and
                np.all(l6['mn_stop_contained'] == 1.))

        #load events
        if sim_ver == '4digit':
            reco_energy_all = np.array(
                bg_file['IC86_Dunkman_L6_MultiNest8D_PDG_Neutrino']['energy']
            )
            reco_coszen_all = np.array(np.cos(
                bg_file['IC86_Dunkman_L6_MultiNest8D_PDG_Neutrino']['zenith']
            ))
        elif sim_ver == '5digit' or 'dima':
            reco_energy_all = np.array(
                bg_file['IC86_Dunkman_L6_PegLeg_MultiNest8D_NumuCC']['energy']
            )
            reco_coszen_all = np.array(np.cos(
                bg_file['IC86_Dunkman_L6_PegLeg_MultiNest8D_NumuCC']['zenith']
            ))
        else:
            raise ValueError('Only allow sim_ver  4digit, 5 digit or dima!')
        reco_energy_all = np.array(bg_file[variable]['energy'])
        reco_coszen_all = np.array(np.cos(bg_file[variable]['zenith']))
        dLLH = np.array(bg_file['IC86_Dunkman_L6']['delta_LLH'])
        if alt_icc_bg_file is not None:
            alt_reco_energy_all = np.array(alt_bg_file[variable]['energy'])
            alt_reco_coszen_all = np.array(np.cos(alt_bg_file[variable]['zenith']))
            alt_dLLH = np.array(alt_bg_file['IC86_Dunkman_L6']['delta_LLH'])
            alt_l5 = alt_bg_file['IC86_Dunkman_L5']['bdt_score']
        
	# Cut1: throw away delta LLH < pid_remove:
        cut1 = dLLH>=pid_remove
        reco_energy_cut1 = reco_energy_all[cut1]
        reco_coszen_cut1 = reco_coszen_all[cut1]
        dLLH_cut1 = dLLH[cut1]
        l5_cut1 = l5[cut1]

        # Cut2: Only keep bdt score >= 0.2 (from MSU latest result, make data/MC agree much better)
        pid_cut = pid_bound 
        #print "pid_remove = ", pid_remove
        #print "pid_bound = ", pid_bound

        cut2 = l5_cut1>=bdt_cut
        reco_energy_cut2 = reco_energy_cut1[cut2]
        reco_coszen_cut2 = reco_coszen_cut1[cut2]
        dLLH_cut2 = dLLH_cut1[cut2]

        if alt_icc_bg_file is not None:
            # Cut1: throw away delta LLH < pid_remove:
            alt_cut1 = alt_dLLH>=pid_remove
            alt_reco_energy_cut1 = alt_reco_energy_all[alt_cut1]
            alt_reco_coszen_cut1 = alt_reco_coszen_all[alt_cut1]
            alt_dLLH_cut1 = alt_dLLH[alt_cut1]
            alt_l5_cut1 = alt_l5[alt_cut1]

            # Cut2: Only keep bdt score >= 0.2 (from MSU latest result, make data/MC agree much better)
            alt_cut2 = alt_l5_cut1>=bdt_cut
            alt_reco_energy_cut2 = alt_reco_energy_cut1[alt_cut2]
            alt_reco_coszen_cut2 = alt_reco_coszen_cut1[alt_cut2]
            alt_dLLH_cut2 = alt_dLLH_cut1[alt_cut2]

        self.icc_bg_dict = {}
        logging.info("Creating a ICC background dict...")
        # write to dictionary
        for flavor in ['cscd', 'trck']:
            final_events= {}
            if flavor == 'cscd':
                cut = dLLH_cut2 < pid_bound
            if flavor == 'trck':
                cut = dLLH_cut2 >= pid_bound
            final_events['reco_energy'] = reco_energy_cut2[cut]
            final_events['reco_coszen'] = reco_coszen_cut2[cut]
            logging.debug("Working on %s background"%flavor)
            icc_bg_hist,_,_ = np.histogram2d(final_events[self.bin_names[0]], final_events[self.bin_names[1]], bins=self.bin_edges)
            self.icc_bg_dict[flavor] = icc_bg_hist

        if alt_icc_bg_file is not None:
            self.alt_icc_bg_dict = {}
            # write to dictionary
            for flavor in ['cscd','trck']:
                alt_events= {}
                if flavor == 'cscd':
                    cut = alt_dLLH_cut2 < pid_bound
                if flavor == 'trck':
                    cut = alt_dLLH_cut2 >= pid_bound
                alt_events['reco_energy'] = alt_reco_energy_cut2[cut]
                alt_events['reco_coszen'] = alt_reco_coszen_cut2[cut]
                alt_icc_bg_hist,_,_ = np.histogram2d(alt_events[self.bin_names[0]], alt_events[self.bin_names[1]], bins=self.bin_edges)
                scale = self.icc_bg_dict[flavor].sum()/alt_icc_bg_hist.sum()
                self.alt_icc_bg_dict[flavor] = alt_icc_bg_hist * scale

    def _compute_outputs(self, inputs=None):
        """TODO: document me, Philipp!"""

        scale = self.params.atm_muon_scale.m_as('dimensionless')
        scale *= self.params.livetime.m_as('common_year')

        maps = []
        for flavor in ['cscd', 'trck']:
            #print '%s %.4f'%(flavor, np.sum(self.icc_bg_dict[flavor] * scale))
            if self.error_method == 'sumw2':
                maps.append(Map(name=flavor, hist=(self.icc_bg_dict[flavor] * scale), error_hist=(np.sqrt(self.icc_bg_dict[flavor]) * scale) ,binning=self.output_binning))
            elif self.error_method == 'sumw2+shape':
                error = scale * np.sqrt(self.icc_bg_dict[flavor] + (self.icc_bg_dict[flavor] - self.alt_icc_bg_dict[flavor])**2 ) 
                maps.append(Map(name=flavor, hist=(self.icc_bg_dict[flavor] * scale), error_hist=error ,binning=self.output_binning))
            else:
                maps.append(Map(
                    name=flavor, hist=(self.icc_bg_dict[flavor] * scale),
                    binning=self.output_binning
                ))
                
        template = MapSet(maps, name='icc')

        return template
