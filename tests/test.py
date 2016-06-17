#! /usr/bin/env python
# author: S.Wren
# date:   March 20, 2016
"""
Runs the pipeline multiple times to test everything still agrees with PISA 2.
Test data for comparing against should be in the tests/data directory.
A set of plots will be output in the tests/output directory for you to check.
Agreement is expected to order 10^{-14} in the far right plots.
"""

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from copy import deepcopy

from pisa.core.pipeline import Pipeline
from pisa.utils.log import logging
from pisa.utils.parse_config import parse_config
from pisa.utils.jsons import from_json
from pisa.utils.plot import show_map, delta_map, ratio_map

def do_comparison(config=None, stagenum=None, stagename=None,
                  servicename=None, pisa2file=None, systname=None):

    if systname is not None:
        try:
            config[stagename.lower()]['params'][systname] = config[stagename.lower()]['params'][systname].value+config[stagename.lower()]['params'][systname].prior.stddev
        except:
            config[stagename.lower()]['params'][systname] = 1.25*config[stagename.lower()]['params'][systname].value

    pipeline = Pipeline(config)
    stage = pipeline.stages[stagenum]
    outputs = stage.get_outputs(inputs=None)
    pisa2_comparisons = from_json(pisa2file)

    for nukey in pisa2_comparisons.keys():
        if 'nu' in nukey:

            pisa_map_to_plot = pisa2_comparisons[nukey]
        
            if nukey == 'numu_bar':
                nukey = 'numubar'
            if nukey == 'nue_bar':
                nukey = 'nuebar'
            
            cake_map = outputs[nukey]
            cake_map_to_plot = {}
            cake_map_to_plot['ebins'] = cake_map.binning['true_energy'].bin_edges.magnitude
            cake_map_to_plot['czbins'] = cake_map.binning['true_coszen'].bin_edges.magnitude
            cake_map_to_plot['map'] = cake_map.hist.T

            RatioMapObj = ratio_map(cake_map_to_plot, pisa_map_to_plot)
            DiffMapObj = delta_map(pisa_map_to_plot, cake_map_to_plot)
            DiffRatioMapObj = ratio_map(DiffMapObj, pisa_map_to_plot)

            plt.figure(figsize = (20,5))

            plt.subplot(1,5,1)
            show_map(pisa_map_to_plot)
            plt.xlabel(r'$\cos\theta_Z$')
            plt.ylabel(r'Energy [GeV]')
            plt.title('$%s$ %s PISA V2'%(outputs[nukey].tex,stagename))

            plt.subplot(1,5,2)
            show_map(cake_map_to_plot)
            plt.xlabel(r'$\cos\theta_Z$')
            plt.ylabel(r'Energy [GeV]')
            plt.title('$%s$ %s PISA V3'%(outputs[nukey].tex,stagename))

            plt.subplot(1,5,3)
            show_map(RatioMapObj)
            plt.xlabel(r'$\cos\theta_Z$')
            plt.ylabel(r'Energy [GeV]')
            plt.title('$%s$ %s PISA V3/V2'%(outputs[nukey].tex,stagename))

            plt.subplot(1,5,4)
            show_map(DiffMapObj)
            plt.xlabel(r'$\cos\theta_Z$')
            plt.ylabel(r'Energy [GeV]')
            plt.title('$%s$ %s PISA V2-V3'%(outputs[nukey].tex,stagename))

            plt.subplot(1,5,5)
            show_map(DiffRatioMapObj)
            plt.xlabel(r'$\cos\theta_Z$')
            plt.ylabel(r'Energy [GeV]')
            plt.title('$%s$ %s PISA (V2-V3)/V2'%(outputs[nukey].tex,stagename))

            plt.tight_layout()

            plt.savefig('output/flux/%s_PISAV2-V3_Comparisons_%s_Stage_%s_Service.png'%(nukey,stagename,servicename))

base_config = parse_config('settings/pipeline_test.ini')

logging.debug('Performing Tests of Flux Stage')
logging.trace('>>> BEGIN: honda.py, integral-preserving, honda-2015-spl-solmax-aa.d')

flux_config = deepcopy(base_config)

flux_config['flux']['params']['flux_file'] = 'flux/honda-2015-spl-solmax-aa.d'
flux_config['flux']['params']['flux_mode'] = 'integral-preserving'

logging.trace('>>> >>> Systematics at Nominal')

do_comparison(config=deepcopy(flux_config),
              stagenum=0,
              stagename='Flux',
              servicename='IP_Honda',
              pisa2file='data/flux/PISAV2IPHonda2015SPLSolMaxFlux.json',
              systname=None)

logging.trace('>>> >>> Atm Delta Index +1 Sigma (0.05)')

do_comparison(config=deepcopy(flux_config),
              stagenum=0,
              stagename='Flux',
              servicename='IP_Honda_Atm_Delta_Index_0.05',
              pisa2file='data/flux/PISAV2IPHonda2015SPLSolMaxFlux-DeltaIndex0.05.json',
              systname='atm_delta_index')

logging.trace('>>> >>> NuE / NuMu Ratio +1 Sigma (1.03)')

do_comparison(config=deepcopy(flux_config),
              stagenum=0,
              stagename='Flux',
              servicename='IP_Honda_NuE_NuMu_Ratio_1.03',
              pisa2file='data/flux/PISAV2IPHonda2015SPLSolMaxFlux-NuENuMuRatio1.03.json',
              systname='nue_numu_ratio')

logging.trace('>>> >>> Nu / NuBar Ratio +1 Sigma (1.10)')

do_comparison(config=deepcopy(flux_config),
              stagenum=0,
              stagename='Flux',
              servicename='IP_Honda_Nu_NuBar_Ratio_1.10',
              pisa2file='data/flux/PISAV2IPHonda2015SPLSolMaxFlux-NuNuBarRatio1.10.json',
              systname='nu_nubar_ratio')

logging.trace('>>> >>> Energy Scale +1 Sigma (1.10)')

do_comparison(config=deepcopy(flux_config),
              stagenum=0,
              stagename='Flux',
              servicename='IP_Honda_Energy_Scale_1.10',
              pisa2file='data/flux/PISAV2IPHonda2015SPLSolMaxFlux-EnergyScale1.10.json',
              systname='energy_scale')

logging.trace('>>> END: honda.py, integral-preserving, honda-2015-spl-solmax-aa.d')

logging.trace('>>> BEGIN: honda.py, bisplrep, honda-2015-spl-solmax-aa.d')

flux_config['flux']['params']['flux_mode'] = 'bisplrep'

logging.trace('>>> >>> Systematics at Nominal')

do_comparison(config=deepcopy(flux_config),
              stagenum=0,
              stagename='Flux',
              servicename='bisplrep_Honda',
              pisa2file='data/flux/PISAV2bisplrepHonda2015SPLSolMaxFlux.json',
              systname=None)

logging.trace('>>> END: honda.py, bisplrep, honda-2015-spl-solmax-aa.d')
