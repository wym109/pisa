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


class data(Stage):

    def __init__(self, params, output_binning, disk_cache=None,
                memcaching_enabled=True, error_method=None,
                outputs_cache_depth=20, debug_mode=None):

        expected_params = (
            'data_file',
            'pid_bound',
            'pid_remove',
            'sim_ver',
        )

        output_names = ('trck','cscd')

        super(self.__class__, self).__init__(
            use_transforms=False,
            stage_name='data',
            service_name='data',
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
        data_file_name = self.params.data_file.value
        sim_version = self.params.sim_ver.value
        pid_bound = self.params.pid_bound.value.m_as('dimensionless')
        pid_remove = self.params.pid_remove.value.m_as('dimensionless')

	self.bin_names = self.output_binning.names
        self.bin_edges = []
        for name in self.bin_names:
            if 'energy' in  name:
                bin_edges = self.output_binning[name].bin_edges.to('GeV').magnitude
            else:
                bin_edges = self.output_binning[name].bin_edges.magnitude
            self.bin_edges.append(bin_edges)

        # the rest of this function is PISA v2 legacy code...
	# right now only use burn sample with sim_version = '4digit'
	print "sim_version == ", sim_version
	if sim_version == "4digit":
	    Reco_Neutrino_Name = 'IC86_Dunkman_L6_MultiNest8D_PDG_Neutrino'
	    Reco_Track_Name = 'IC86_Dunkman_L6_MultiNest8D_PDG_Track'
	elif sim_version == "5digit" or sim_version=="dima":
	    Reco_Neutrino_Name = 'IC86_Dunkman_L6_PegLeg_MultiNest8D_NumuCC'
	    Reco_Track_Name = 'IC86_Dunkman_L6_PegLeg_MultiNest8D_Track'
	else:
	    raise ValueError('only allow 4digit, 5digit(H2 model for hole ice) or dima (dima p1 and p2 for hole ice)!') 

	data_file = h5py.File(find_resource(data_file_name),'r')
	L6_result = np.array(data_file['IC86_Dunkman_L6']['result'])
	dLLH = np.array(data_file['IC86_Dunkman_L6']['delta_LLH'])
	reco_energy_all = np.array(data_file[Reco_Neutrino_Name]['energy'])
	reco_coszen_all = np.array(np.cos(data_file[Reco_Neutrino_Name]['zenith']))
	reco_trck_len_all = np.array(data_file[Reco_Track_Name]['length'])
	data_file.close()
	#print "before L6 cut, no. of burn sample = ", len(reco_coszen_all)

	dLLH_L6 = dLLH[L6_result==1]
	reco_energy_L6 = reco_energy_all[L6_result==1]
	reco_coszen_L6 = reco_coszen_all[L6_result==1]

	#print "after L6 cut, no. of burn sample = ", len(reco_coszen_L6)
       
	# throw away dLLH < -3
	reco_energy_L6_cut1 = reco_energy_L6[dLLH_L6>=pid_remove]
	reco_coszen_L6_cut1 = reco_coszen_L6[dLLH_L6>=pid_remove]
	dLLH_L6_cut1 = dLLH_L6[dLLH_L6>=pid_remove]

	# don't throw away dLLH < -3, only use this when using param service for PID in PISA
	#reco_energy_L6_cut1 = reco_energy_L6
	#reco_coszen_L6_cut1 = reco_coszen_L6
	#dLLH_L6_cut1 = dLLH_L6

	# write burn sample data to dictionary
	self.data_dict = {}
	for flav in ['cscd','trck']:
            final_events = {}
	    if flav == 'cscd':
		cut_pid = dLLH_L6_cut1 < pid_bound 
	    if flav == 'trck':
		cut_pid = dLLH_L6_cut1 >= pid_bound 

            final_events['reco_energy'] = reco_energy_L6_cut1[cut_pid]
            final_events['reco_coszen'] = reco_coszen_L6_cut1[cut_pid]

            data_hist,_,_ = np.histogram2d(final_events[self.bin_names[0]], final_events[self.bin_names[1]], bins=self.bin_edges)

            self.data_dict[flav] = data_hist

        maps = []
        for flavor in ['cscd','trck']:
            maps.append(Map(name=flavor, hist=self.data_dict[flavor], binning=self.output_binning))

        self.template = MapSet(maps,name='data')

    def _compute_outputs(self, inputs=None):
        return self.template
