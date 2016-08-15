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
import numpy as np
import os

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from copy import deepcopy

from pisa.core.map import Map, MapSet
from pisa.core.pipeline import Pipeline
from pisa.utils.log import logging
from pisa.utils.parse_config import parse_config
from pisa.utils.jsons import from_json
from pisa.utils.plot import show_map, delta_map, ratio_map

def do_plotting(pisa_map=None, cake_map=None,
                stagename=None, servicename=None,
                nutexname=None, nukey=None):

    RatioMapObj = ratio_map(cake_map, pisa_map)
    DiffMapObj = delta_map(pisa_map, cake_map)
    DiffRatioMapObj = ratio_map(DiffMapObj, pisa_map)

    plt.figure(figsize = (20,5))
    
    plt.subplot(1,5,1)
    show_map(pisa_map)
    plt.xlabel(r'$\cos\theta_Z$')
    plt.ylabel(r'Energy [GeV]')
    plt.title('$%s$ %s PISA V2'%(nutexname,stagename))

    plt.subplot(1,5,2)
    show_map(cake_map)
    plt.xlabel(r'$\cos\theta_Z$')
    plt.ylabel(r'Energy [GeV]')
    plt.title('$%s$ %s PISA V3'%(nutexname,stagename))

    plt.subplot(1,5,3)
    show_map(RatioMapObj)
    plt.xlabel(r'$\cos\theta_Z$')
    plt.ylabel(r'Energy [GeV]')
    plt.title('$%s$ %s PISA V3/V2'%(nutexname,stagename))

    plt.subplot(1,5,4)
    show_map(DiffMapObj)
    plt.xlabel(r'$\cos\theta_Z$')
    plt.ylabel(r'Energy [GeV]')
    plt.title('$%s$ %s PISA V2-V3'%(nutexname,stagename))

    plt.subplot(1,5,5)
    show_map(DiffRatioMapObj)
    plt.xlabel(r'$\cos\theta_Z$')
    plt.ylabel(r'Energy [GeV]')
    plt.title('$%s$ %s PISA (V2-V3)/V2'%(nutexname,stagename))

    plt.tight_layout()

    plt.savefig(args.outdir+'/%s/%s_PISAV2-V3_Comparisons_%s_Stage_%s_Service.png'%(stagename.lower(),nukey,stagename,servicename))

    plt.close()

def do_flux_comparison(config=None, servicename=None,
                       pisa2file=None, systname=None,
                       outdir=None):

    if systname is not None:
        try:
            config['flux']['params'][systname] = config['flux']['params'][systname].value+config['flux']['params'][systname].prior.stddev
        except:
            config['flux']['params'][systname] = 1.25*config['flux']['params'][systname].value

        pisa2file = pisa2file.split('.json')[0]+'-%s%.2f.json'%(systname,config['flux']['params'][systname].value)
        servicename += '-%s%.2f.json'%(systname,config['flux']['params'][systname].value)

    pipeline = Pipeline(config)
    stage = pipeline.stages[0]
    outputs = stage.get_outputs(inputs=None)
    pisa2_comparisons = from_json(pisa2file)
            
    for nukey in pisa2_comparisons.keys():
        if 'nu' in nukey:

            pisa_map_to_plot = pisa2_comparisons[nukey]

            if '_' in nukey:
                if nukey.split('_')[1] == 'bar':
                    new_nukey = ""
                    for substr in nukey.split('_'):
                        new_nukey += substr
                    nukey = new_nukey
            
            cake_map = outputs[nukey]
            cake_map_to_plot = {}
            cake_map_to_plot['ebins'] = cake_map.binning['true_energy'].bin_edges.magnitude
            cake_map_to_plot['czbins'] = cake_map.binning['true_coszen'].bin_edges.magnitude
            cake_map_to_plot['map'] = cake_map.hist.T

            do_plotting(pisa_map=pisa_map_to_plot,
                        cake_map=cake_map_to_plot,
                        stagename='Flux',
                        servicename=servicename,
                        nutexname=outputs[nukey].tex,
                        nukey=nukey)

def do_osc_comparison(config=None, servicename=None,
                      pisa2file=None, systname=None,
                      outdir=None):

    if systname is not None:
        try:
            config['osc']['params'][systname] = config['osc']['params'][systname].value+config['osc']['params'][systname].prior.stddev
        except:
            config['osc']['params'][systname] = 1.25*config['osc']['params'][systname].value

        if config['osc']['params'][systname].value.magnitude < 0.01:
            systval = '%e'%config['osc']['params'][systname].value.magnitude
            systval = systval[0:4]
        else:
            systval = '%.2f'%config['osc']['params'][systname].value.magnitude

        pisa2file = pisa2file.split('.json')[0]+'-%s%s.json'%(systname,systval)
        servicename += '-%s%s.json'%(systname,systval)

    pipeline = Pipeline(config)
    stage = pipeline.stages[0]
    input_maps = []
    for name in stage.input_names:
        hist = np.ones(stage.input_binning.shape)
        input_maps.append(
            Map(name=name, hist=hist, binning=stage.input_binning)
        )
    outputs = stage.get_outputs(inputs=MapSet(maps=input_maps, name='ones', hash=1))
    pisa2_comparisons = from_json(pisa2file)
            
    for nukey in pisa2_comparisons.keys():
        if 'nu' in nukey:

            pisa_map_to_plot = pisa2_comparisons[nukey]

            if '_' in nukey:
                if nukey.split('_')[1] == 'bar':
                    new_nukey = ""
                    for substr in nukey.split('_'):
                        new_nukey += substr
                    nukey = new_nukey
            
            cake_map = outputs[nukey]
            cake_map_to_plot = {}
            cake_map_to_plot['ebins'] = cake_map.binning['true_energy'].bin_edges.magnitude
            cake_map_to_plot['czbins'] = cake_map.binning['true_coszen'].bin_edges.magnitude
            cake_map_to_plot['map'] = cake_map.hist.T

            do_plotting(pisa_map=pisa_map_to_plot,
                        cake_map=cake_map_to_plot,
                        stagename='Osc',
                        servicename=servicename,
                        nutexname=outputs[nukey].tex,
                        nukey=nukey)

def do_aeff_comparison(config=None, servicename=None,
                       pisa2file=None, systname=None,
                       outdir=None):

    if systname is not None:
        try:
            config['aeff']['params'][systname] = config['aeff']['params'][systname].value+config['aeff']['params'][systname].prior.stddev
        except:
            config['aeff']['params'][systname] = 1.25*config['aeff']['params'][systname].value

        pisa2file = pisa2file.split('.json')[0]+'-%s%.2f.json'%(systname,config['aeff']['params'][systname].value)
        servicename += '-%s%.2f'%(systname,config['aeff']['params'][systname].value)

    pipeline = Pipeline(config)
    stage = pipeline.stages[0]
    input_maps = []
    for name in stage.input_names:
        hist = np.ones(stage.input_binning.shape)
        input_maps.append(
            Map(name=name, hist=hist, binning=stage.input_binning)
        )
    outputs = stage.get_outputs(inputs=MapSet(maps=input_maps, name='ones', hash=1))
    pisa2_comparisons = from_json(pisa2file)
    
    for nukey in pisa2_comparisons.keys():
        if 'nu' in nukey:
            for intkey in pisa2_comparisons[nukey].keys():
                if '_' in nukey:
                    if nukey.split('_')[1] == 'bar':
                        new_nukey = ""
                        for substr in nukey.split('_'):
                            new_nukey += substr
                else:
                    new_nukey = nukey 
                cakekey = new_nukey + '_' + intkey
                pisa_map_to_plot = pisa2_comparisons[nukey][intkey]

                cake_map = outputs[cakekey]
                cake_map_to_plot = {}
                cake_map_to_plot['ebins'] = cake_map.binning['true_energy'].bin_edges.magnitude
                cake_map_to_plot['czbins'] = cake_map.binning['true_coszen'].bin_edges.magnitude
                cake_map_to_plot['map'] = cake_map.hist.T

                do_plotting(pisa_map=pisa_map_to_plot,
                            cake_map=cake_map_to_plot,
                            stagename='Aeff',
                            servicename=servicename,
                            nutexname=outputs[cakekey].tex,
                            nukey=cakekey)

def do_reco_comparison(config=None, servicename=None,
                       pisa2file=None, outdir=None):

    pipeline = Pipeline(config)
    stage = pipeline.stages[0]
    input_maps = []
    for name in stage.input_names:
        hist = np.ones(stage.input_binning.shape)
        input_maps.append(
            Map(name=name, hist=hist, binning=stage.input_binning)
        )
    outputs = stage.get_outputs(inputs=MapSet(maps=input_maps, name='ones', hash=1))

    modified_cake_outputs = {}

    for name in outputs.names:
        if name in ['nue_cc','nuebar_cc']:
            if 'nue_cc' in modified_cake_outputs.keys():
                modified_cake_outputs['nue_cc']['map'] += outputs[name].hist.T
            else:
                modified_cake_outputs['nue_cc'] = {}
                modified_cake_outputs['nue_cc']['ebins'] = outputs[name].binning['reco_energy'].bin_edges.magnitude
                modified_cake_outputs['nue_cc']['czbins'] = outputs[name].binning['reco_coszen'].bin_edges.magnitude
                modified_cake_outputs['nue_cc']['map'] = outputs[name].hist.T
        elif name in ['numu_cc','numubar_cc']:
            if 'numu_cc' in modified_cake_outputs.keys():
                modified_cake_outputs['numu_cc']['map'] += outputs[name].hist.T
            else:
                modified_cake_outputs['numu_cc'] = {}
                modified_cake_outputs['numu_cc']['ebins'] = outputs[name].binning['reco_energy'].bin_edges.magnitude
                modified_cake_outputs['numu_cc']['czbins'] = outputs[name].binning['reco_coszen'].bin_edges.magnitude
                modified_cake_outputs['numu_cc']['map'] = outputs[name].hist.T
        elif name in ['nutau_cc','nutaubar_cc']:
            if 'nutau_cc' in modified_cake_outputs.keys():
                modified_cake_outputs['nutau_cc']['map'] += outputs[name].hist.T
            else:
                modified_cake_outputs['nutau_cc'] = {}
                modified_cake_outputs['nutau_cc']['ebins'] = outputs[name].binning['reco_energy'].bin_edges.magnitude
                modified_cake_outputs['nutau_cc']['czbins'] = outputs[name].binning['reco_coszen'].bin_edges.magnitude
                modified_cake_outputs['nutau_cc']['map'] = outputs[name].hist.T
        elif 'nc' in name:
            if 'nuall_nc' in modified_cake_outputs.keys():
                modified_cake_outputs['nuall_nc']['map'] += outputs[name].hist.T
            else:
                modified_cake_outputs['nuall_nc'] = {}
                modified_cake_outputs['nuall_nc']['ebins'] = outputs[name].binning['reco_energy'].bin_edges.magnitude
                modified_cake_outputs['nuall_nc']['czbins'] = outputs[name].binning['reco_coszen'].bin_edges.magnitude
                modified_cake_outputs['nuall_nc']['map'] = outputs[name].hist.T
    
    pisa2_comparisons = from_json(pisa2file)
    for nukey in pisa2_comparisons.keys():
        if 'nu' in nukey:

            pisa_map_to_plot = pisa2_comparisons[nukey]

            if '_' in nukey:
                if nukey.split('_')[1] == 'bar':
                    new_nukey = ""
                    for substr in nukey.split('_'):
                        new_nukey += substr
                    nukey = new_nukey
            
            cake_map_to_plot = modified_cake_outputs[nukey]

            do_plotting(pisa_map=pisa_map_to_plot,
                        cake_map=cake_map_to_plot,
                        stagename='Reco',
                        servicename=servicename,
                        nutexname=outputs[nukey].tex,
                        nukey=nukey)

def do_pid_comparison(config=None, servicename=None,
                      pisa2file=None, outdir=None):
    
    pipeline = Pipeline(config)
    stage = pipeline.stages[0]
    input_maps = []
    for name in stage.input_names:
        hist = np.ones(stage.input_binning.shape)
        input_maps.append(
            Map(name=name, hist=hist, binning=stage.input_binning)
        )
    outputs = stage.get_outputs(inputs=MapSet(maps=input_maps, name='ones', hash=1))

    total_cake_trck_dict = {}
    total_cake_cscd_dict = {}

    for cake_key in outputs.names:
        if 'trck' in cake_key:
            if len(total_cake_trck_dict.keys()) == 0:
                total_cake_trck_dict['ebins'] = outputs[cake_key].binning['reco_energy'].bin_edges.magnitude
                total_cake_trck_dict['czbins'] = outputs[cake_key].binning['reco_coszen'].bin_edges.magnitude
                total_cake_trck_dict['map'] = outputs[cake_key].hist.T
            else:
                total_cake_trck_dict['map'] += outputs[cake_key].hist.T
        elif 'cscd' in cake_key:
            if len(total_cake_cscd_dict.keys()) == 0:
                total_cake_cscd_dict['ebins'] = outputs[cake_key].binning['reco_energy'].bin_edges.magnitude
                total_cake_cscd_dict['czbins'] = outputs[cake_key].binning['reco_coszen'].bin_edges.magnitude
                total_cake_cscd_dict['map'] = outputs[cake_key].hist.T
            else:
                total_cake_cscd_dict['map'] += outputs[cake_key].hist.T
        
    pisa2_comparisons = from_json(pisa2file)

    total_pisa_trck_dict = pisa2_comparisons['trck']
    total_pisa_cscd_dict = pisa2_comparisons['cscd']

    RatioMapObj = ratio_map(total_cake_trck_dict, total_pisa_trck_dict)
    DiffMapObj = delta_map(total_pisa_trck_dict, total_cake_trck_dict)
    DiffRatioMapObj = ratio_map(DiffMapObj, total_pisa_trck_dict)

    plt.figure(figsize = (20,5))

    plt.subplot(1,5,1)
    show_map(total_pisa_trck_dict)
    plt.xlabel(r'$\cos\theta_Z$')
    plt.ylabel(r'Energy [GeV]')
    plt.title('Track-Like Events PISA V2')

    plt.subplot(1,5,2)
    show_map(total_cake_trck_dict)
    plt.xlabel(r'$\cos\theta_Z$')
    plt.ylabel(r'Energy [GeV]')
    plt.title('Track-Like Events PISA V3')

    plt.subplot(1,5,3)
    show_map(RatioMapObj)
    plt.xlabel(r'$\cos\theta_Z$')
    plt.ylabel(r'Energy [GeV]')
    plt.title('Track-Like Events PISA V3/V2')

    plt.subplot(1,5,4)
    show_map(DiffMapObj)
    plt.xlabel(r'$\cos\theta_Z$')
    plt.ylabel(r'Energy [GeV]')
    plt.title('Track-Like Events PISA V2-V3')

    plt.subplot(1,5,5)
    show_map(DiffRatioMapObj)
    plt.xlabel(r'$\cos\theta_Z$')
    plt.ylabel(r'Energy [GeV]')
    plt.title('Track-Like Events PISA (V2-V3)/V2')

    plt.tight_layout()

    plt.savefig(args.outdir+'/pid/DeepCore_PISAV2-V3_Comparisons_PID_%s_Track-Like.png'%servicename)

    plt.close()

    RatioMapObj = ratio_map(total_cake_cscd_dict, total_pisa_cscd_dict)
    DiffMapObj = delta_map(total_pisa_cscd_dict, total_cake_cscd_dict)
    DiffRatioMapObj = ratio_map(DiffMapObj, total_pisa_cscd_dict)

    plt.figure(figsize = (20,5))

    plt.subplot(1,5,1)
    show_map(total_pisa_cscd_dict)
    plt.xlabel(r'$\cos\theta_Z$')
    plt.ylabel(r'Energy [GeV]')
    plt.title('Cascade-Like Events PISA V2')

    plt.subplot(1,5,2)
    show_map(total_cake_cscd_dict)
    plt.xlabel(r'$\cos\theta_Z$')
    plt.ylabel(r'Energy [GeV]')
    plt.title('Cascade-Like Events PISA V3')

    plt.subplot(1,5,3)
    show_map(RatioMapObj)
    plt.xlabel(r'$\cos\theta_Z$')
    plt.ylabel(r'Energy [GeV]')
    plt.title('Cascade-Like Events PISA V3/V2')

    plt.subplot(1,5,4)
    show_map(DiffMapObj)
    plt.xlabel(r'$\cos\theta_Z$')
    plt.ylabel(r'Energy [GeV]')
    plt.title('Cascade-Like Events PISA V2-V3')

    plt.subplot(1,5,5)
    show_map(DiffRatioMapObj)
    plt.xlabel(r'$\cos\theta_Z$')
    plt.ylabel(r'Energy [GeV]')
    plt.title('Cascade-Like Events PISA (V2-V3)/V2')

    plt.tight_layout()

    plt.savefig(args.outdir+'/pid/DeepCore_PISAV2-V3_Comparisons_PID_%s_Cascade-Like.png'%servicename)

    plt.close()

def do_pipeline_comparison(config=None, pisa2file=None, outdir=None):
    
    pipeline = Pipeline(config)
    pisa2_comparisons = from_json(pisa2file)

    # Up to flux stage comparisons
    pisa2_comparison = pisa2_comparisons[0]
    outputs = pipeline.get_outputs(idx=1)

    for nukey in pisa2_comparison.keys():
        if 'nu' in nukey:

            pisa_map_to_plot = pisa2_comparison[nukey]

            if '_' in nukey:
                if nukey.split('_')[1] == 'bar':
                    new_nukey = ""
                    for substr in nukey.split('_'):
                        new_nukey += substr
                    nukey = new_nukey
            
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
            plt.title('$%s$ Flux PISA V2'%(outputs[nukey].tex))

            plt.subplot(1,5,2)
            show_map(cake_map_to_plot)
            plt.xlabel(r'$\cos\theta_Z$')
            plt.ylabel(r'Energy [GeV]')
            plt.title('$%s$ Flux PISA V3'%(outputs[nukey].tex))

            plt.subplot(1,5,3)
            show_map(RatioMapObj)
            plt.xlabel(r'$\cos\theta_Z$')
            plt.ylabel(r'Energy [GeV]')
            plt.title('$%s$ Flux PISA V3/V2'%(outputs[nukey].tex))

            plt.subplot(1,5,4)
            show_map(DiffMapObj)
            plt.xlabel(r'$\cos\theta_Z$')
            plt.ylabel(r'Energy [GeV]')
            plt.title('$%s$ Flux PISA V2-V3'%(outputs[nukey].tex))

            plt.subplot(1,5,5)
            show_map(DiffRatioMapObj)
            plt.xlabel(r'$\cos\theta_Z$')
            plt.ylabel(r'Energy [GeV]')
            plt.title('$%s$ Flux PISA (V2-V3)/V2'%(outputs[nukey].tex))

            plt.tight_layout()

            plt.savefig(args.outdir+'/full/%s_PISAV2-V3_Comparisons_Flux_Stage_IP_Honda_Service.png'%(nukey))

            plt.close()

    # Up to osc stage comparisons
    pisa2_comparison = pisa2_comparisons[1]
    outputs = pipeline.get_outputs(idx=2)

    for nukey in pisa2_comparison.keys():
        if 'nu' in nukey:

            pisa_map_to_plot = pisa2_comparison[nukey]

            if '_' in nukey:
                if nukey.split('_')[1] == 'bar':
                    new_nukey = ""
                    for substr in nukey.split('_'):
                        new_nukey += substr
                    nukey = new_nukey
            
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
            plt.title('$%s$ Osc PISA V2'%(outputs[nukey].tex))

            plt.subplot(1,5,2)
            show_map(cake_map_to_plot)
            plt.xlabel(r'$\cos\theta_Z$')
            plt.ylabel(r'Energy [GeV]')
            plt.title('$%s$ Osc PISA V3'%(outputs[nukey].tex))

            plt.subplot(1,5,3)
            show_map(RatioMapObj)
            plt.xlabel(r'$\cos\theta_Z$')
            plt.ylabel(r'Energy [GeV]')
            plt.title('$%s$ Osc PISA V3/V2'%(outputs[nukey].tex))

            plt.subplot(1,5,4)
            show_map(DiffMapObj)
            plt.xlabel(r'$\cos\theta_Z$')
            plt.ylabel(r'Energy [GeV]')
            plt.title('$%s$ Osc PISA V2-V3'%(outputs[nukey].tex))

            plt.subplot(1,5,5)
            show_map(DiffRatioMapObj)
            plt.xlabel(r'$\cos\theta_Z$')
            plt.ylabel(r'Energy [GeV]')
            plt.title('$%s$ Osc PISA (V2-V3)/V2'%(outputs[nukey].tex))

            plt.tight_layout()

            plt.savefig(args.outdir+'/full/%s_PISAV2-V3_Comparisons_Osc_Stage_prob3_Service.png'%(nukey))

            plt.close()

    # Up to aeff stage comparisons
    pisa2_comparison = pisa2_comparisons[2]
    outputs = pipeline.get_outputs(idx=3)

    total_pisa_nuall_nc_dict = {}
    total_pisa_nuallbar_nc_dict = {}

    i = 0

    for nukey in pisa2_comparison.keys():
        if 'nu' in nukey:
            for intkey in pisa2_comparison[nukey].keys():
                if intkey == 'cc':

                    if '_' in nukey:
                        if nukey.split('_')[1] == 'bar':
                            new_nukey = ""
                            for substr in nukey.split('_'):
                                new_nukey += substr
                    else:
                        new_nukey = nukey
                        
                    cakekey = new_nukey + '_' + intkey
                    pisa_map_to_plot = pisa2_comparison[nukey][intkey]

                    cake_map = outputs[cakekey]
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
                    plt.title('$%s$ Aeff PISA V2'%(outputs[cakekey].tex))

                    plt.subplot(1,5,2)
                    show_map(cake_map_to_plot)
                    plt.xlabel(r'$\cos\theta_Z$')
                    plt.ylabel(r'Energy [GeV]')
                    plt.title('$%s$ Aeff PISA V3'%(outputs[cakekey].tex))

                    plt.subplot(1,5,3)
                    show_map(RatioMapObj)
                    plt.xlabel(r'$\cos\theta_Z$')
                    plt.ylabel(r'Energy [GeV]')
                    plt.title('$%s$ Aeff PISA V3/V2'%(outputs[cakekey].tex))

                    plt.subplot(1,5,4)
                    show_map(DiffMapObj)
                    plt.xlabel(r'$\cos\theta_Z$')
                    plt.ylabel(r'Energy [GeV]')
                    plt.title('$%s$ Aeff PISA V2-V3'%(outputs[cakekey].tex))

                    plt.subplot(1,5,5)
                    show_map(DiffRatioMapObj)
                    plt.xlabel(r'$\cos\theta_Z$')
                    plt.ylabel(r'Energy [GeV]')
                    plt.title('$%s$ Aeff PISA (V2-V3)/V2'%(outputs[cakekey].tex))

                    plt.tight_layout()

                    plt.savefig(args.outdir+'/full/%s_PISAV2-V3_Comparisons_Aeff_Stage_hist1X585_Service.png'%(cakekey))

                    plt.close()

                else:

                    if 'bar' in nukey:
                        if len(total_pisa_nuallbar_nc_dict.keys()) == 0:
                            total_pisa_nuallbar_nc_dict = pisa2_comparison[nukey][intkey]
                            total_pisa_nuallbar_nc_dict['map'] = np.array(total_pisa_nuallbar_nc_dict['map'])
                        else:
                            total_pisa_nuallbar_nc_dict['map'] += np.array(pisa2_comparison[nukey][intkey]['map'])
                    else:
                        if len(total_pisa_nuall_nc_dict.keys()) == 0:
                            total_pisa_nuall_nc_dict = pisa2_comparison[nukey][intkey]
                            total_pisa_nuall_nc_dict['map'] = np.array(total_pisa_nuall_nc_dict['map'])
                        else:
                            total_pisa_nuall_nc_dict['map'] += np.array(pisa2_comparison[nukey][intkey]['map'])

    cakekey = 'nuall_nc'
    cake_map_to_plot = {}
    for cake_map in outputs:
        if cake_map.name == cakekey:
            if len(cake_map_to_plot.keys()) == 0:
                cake_map_to_plot['ebins'] = cake_map.binning['true_energy'].bin_edges.magnitude
                cake_map_to_plot['czbins'] = cake_map.binning['true_coszen'].bin_edges.magnitude
                cake_map_to_plot['map'] = cake_map.hist.T
            else:
                cake_map_to_plot['map'] += cake_map.hist.T

    pisa_map_to_plot = total_pisa_nuall_nc_dict

    RatioMapObj = ratio_map(cake_map_to_plot, pisa_map_to_plot)
    DiffMapObj = delta_map(pisa_map_to_plot, cake_map_to_plot)
    DiffRatioMapObj = ratio_map(DiffMapObj, pisa_map_to_plot)

    plt.figure(figsize = (20,5))
    
    plt.subplot(1,5,1)
    show_map(pisa_map_to_plot)
    plt.xlabel(r'$\cos\theta_Z$')
    plt.ylabel(r'Energy [GeV]')
    plt.title('$%s$ Aeff PISA V2'%(outputs[cakekey].tex))

    plt.subplot(1,5,2)
    show_map(cake_map_to_plot)
    plt.xlabel(r'$\cos\theta_Z$')
    plt.ylabel(r'Energy [GeV]')
    plt.title('$%s$ Aeff PISA V3'%(outputs[cakekey].tex))

    plt.subplot(1,5,3)
    show_map(RatioMapObj)
    plt.xlabel(r'$\cos\theta_Z$')
    plt.ylabel(r'Energy [GeV]')
    plt.title('$%s$ Aeff PISA V3/V2'%(outputs[cakekey].tex))

    plt.subplot(1,5,4)
    show_map(DiffMapObj)
    plt.xlabel(r'$\cos\theta_Z$')
    plt.ylabel(r'Energy [GeV]')
    plt.title('$%s$ Aeff PISA V2-V3'%(outputs[cakekey].tex))

    plt.subplot(1,5,5)
    show_map(DiffRatioMapObj)
    plt.xlabel(r'$\cos\theta_Z$')
    plt.ylabel(r'Energy [GeV]')
    plt.title('$%s$ Aeff PISA (V2-V3)/V2'%(outputs[cakekey].tex))

    plt.tight_layout()

    plt.savefig(args.outdir+'/full/%s_PISAV2-V3_Comparisons_Aeff_Stage_hist1X585_Service.png'%(cakekey))

    plt.close()

    cakekey = 'nuallbar_nc'
    cake_map_to_plot = {}
    for cake_map in outputs:
        if cake_map.name == cakekey:
            if len(cake_map_to_plot.keys()) == 0:
                cake_map_to_plot['ebins'] = cake_map.binning['true_energy'].bin_edges.magnitude
                cake_map_to_plot['czbins'] = cake_map.binning['true_coszen'].bin_edges.magnitude
                cake_map_to_plot['map'] = cake_map.hist.T
            else:
                cake_map_to_plot['map'] += cake_map.hist.T

    pisa_map_to_plot = total_pisa_nuallbar_nc_dict

    RatioMapObj = ratio_map(cake_map_to_plot, pisa_map_to_plot)
    DiffMapObj = delta_map(pisa_map_to_plot, cake_map_to_plot)
    DiffRatioMapObj = ratio_map(DiffMapObj, pisa_map_to_plot)

    plt.figure(figsize = (20,5))
    
    plt.subplot(1,5,1)
    show_map(pisa_map_to_plot)
    plt.xlabel(r'$\cos\theta_Z$')
    plt.ylabel(r'Energy [GeV]')
    plt.title('$%s$ Aeff PISA V2'%(outputs[cakekey].tex))

    plt.subplot(1,5,2)
    show_map(cake_map_to_plot)
    plt.xlabel(r'$\cos\theta_Z$')
    plt.ylabel(r'Energy [GeV]')
    plt.title('$%s$ Aeff PISA V3'%(outputs[cakekey].tex))

    plt.subplot(1,5,3)
    show_map(RatioMapObj)
    plt.xlabel(r'$\cos\theta_Z$')
    plt.ylabel(r'Energy [GeV]')
    plt.title('$%s$ Aeff PISA V3/V2'%(outputs[cakekey].tex))

    plt.subplot(1,5,4)
    show_map(DiffMapObj)
    plt.xlabel(r'$\cos\theta_Z$')
    plt.ylabel(r'Energy [GeV]')
    plt.title('$%s$ Aeff PISA V2-V3'%(outputs[cakekey].tex))

    plt.subplot(1,5,5)
    show_map(DiffRatioMapObj)
    plt.xlabel(r'$\cos\theta_Z$')
    plt.ylabel(r'Energy [GeV]')
    plt.title('$%s$ Aeff PISA (V2-V3)/V2'%(outputs[cakekey].tex))

    plt.tight_layout()

    plt.savefig(args.outdir+'/full/%s_PISAV2-V3_Comparisons_Aeff_Stage_hist1X585_Service.png'%(cakekey))

    plt.close()

    # Up to PID stage comparisons
    pisa2_comparison = pisa2_comparisons[4]
    outputs = pipeline.get_outputs(idx=5)

    total_cake_trck_dict = {}
    total_cake_cscd_dict = {}
    
    for cake_key in outputs.names:
        if 'trck' in cake_key:
            if len(total_cake_trck_dict.keys()) == 0:
                total_cake_trck_dict['ebins'] = outputs[cake_key].binning['reco_energy'].bin_edges.magnitude
                total_cake_trck_dict['czbins'] = outputs[cake_key].binning['reco_coszen'].bin_edges.magnitude
                total_cake_trck_dict['map'] = outputs[cake_key].hist.T
            else:
                total_cake_trck_dict['map'] += outputs[cake_key].hist.T
        elif 'cscd' in cake_key:
            if len(total_cake_cscd_dict.keys()) == 0:
                total_cake_cscd_dict['ebins'] = outputs[cake_key].binning['reco_energy'].bin_edges.magnitude
                total_cake_cscd_dict['czbins'] = outputs[cake_key].binning['reco_coszen'].bin_edges.magnitude
                total_cake_cscd_dict['map'] = outputs[cake_key].hist.T
            else:
                total_cake_cscd_dict['map'] += outputs[cake_key].hist.T


    total_pisa_trck_dict = pisa2_comparison['trck']
    total_pisa_cscd_dict = pisa2_comparison['cscd']

    RatioMapObj = ratio_map(total_cake_trck_dict, total_pisa_trck_dict)
    DiffMapObj = delta_map(total_pisa_trck_dict, total_cake_trck_dict)
    DiffRatioMapObj = ratio_map(DiffMapObj, total_pisa_trck_dict)

    plt.figure(figsize = (20,5))

    plt.subplot(1,5,1)
    show_map(total_pisa_trck_dict)
    plt.xlabel(r'$\cos\theta_Z$')
    plt.ylabel(r'Energy [GeV]')
    plt.title('Track-Like Events PISA V2')

    plt.subplot(1,5,2)
    show_map(total_cake_trck_dict)
    plt.xlabel(r'$\cos\theta_Z$')
    plt.ylabel(r'Energy [GeV]')
    plt.title('Track-Like Events PISA V3')

    plt.subplot(1,5,3)
    show_map(RatioMapObj)
    plt.xlabel(r'$\cos\theta_Z$')
    plt.ylabel(r'Energy [GeV]')
    plt.title('Track-Like Events PISA V3/V2')

    plt.subplot(1,5,4)
    show_map(DiffMapObj)
    plt.xlabel(r'$\cos\theta_Z$')
    plt.ylabel(r'Energy [GeV]')
    plt.title('Track-Like Events PISA V2-V3')

    plt.subplot(1,5,5)
    show_map(DiffRatioMapObj)
    plt.xlabel(r'$\cos\theta_Z$')
    plt.ylabel(r'Energy [GeV]')
    plt.title('Track-Like Events PISA (V2-V3)/V2')

    plt.tight_layout()

    plt.savefig(args.outdir+'/full/DeepCore_PISAV2-V3_Comparisons_Full_Pipeline_Track-Like.png')

    plt.close()

    RatioMapObj = ratio_map(total_cake_cscd_dict, total_pisa_cscd_dict)
    DiffMapObj = delta_map(total_pisa_cscd_dict, total_cake_cscd_dict)
    DiffRatioMapObj = ratio_map(DiffMapObj, total_pisa_cscd_dict)

    plt.figure(figsize = (20,5))

    plt.subplot(1,5,1)
    show_map(total_pisa_cscd_dict)
    plt.xlabel(r'$\cos\theta_Z$')
    plt.ylabel(r'Energy [GeV]')
    plt.title('Cascade-Like Events PISA V2')

    plt.subplot(1,5,2)
    show_map(total_cake_cscd_dict)
    plt.xlabel(r'$\cos\theta_Z$')
    plt.ylabel(r'Energy [GeV]')
    plt.title('Cascade-Like Events PISA V3')

    plt.subplot(1,5,3)
    show_map(RatioMapObj)
    plt.xlabel(r'$\cos\theta_Z$')
    plt.ylabel(r'Energy [GeV]')
    plt.title('Cascade-Like Events PISA V3/V2')

    plt.subplot(1,5,4)
    show_map(DiffMapObj)
    plt.xlabel(r'$\cos\theta_Z$')
    plt.ylabel(r'Energy [GeV]')
    plt.title('Cascade-Like Events PISA V2-V3')

    plt.subplot(1,5,5)
    show_map(DiffRatioMapObj)
    plt.xlabel(r'$\cos\theta_Z$')
    plt.ylabel(r'Energy [GeV]')
    plt.title('Cascade-Like Events PISA (V2-V3)/V2')

    plt.tight_layout()

    plt.savefig(args.outdir+'/full/DeepCore_PISAV2-V3_Comparisons_Full_Pipeline_Cascade-Like.png')

    plt.close()

parser = ArgumentParser(
    description='''
    Runs a set of tests on the PISA 3 pipeline against benchmark PISA 2 data. 
    The plots will be deleted and re-made every time you run this script so you 
    can always be sure that the ones you have represent your PISA 3 in its 
    current state. This script should always be run when you make any major 
    modifications to be sure nothing has broken. If you find this script does 
    not work, please either fix it or report it! In general, this will signify 
    you have "changed" something, somehow in the basic functionality which you 
    should understand!
    ''',
    formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--all', action='store_true', default=True,
                    help='''Run all tests. This is recommended and is the 
                    default behaviour!''')
parser.add_argument('--flux', action='store_true', default=False,
                    help='''Run flux tests i.e. the interpolation methods and 
                    the flux systematics.''')
parser.add_argument('--osc', action='store_true', default=False,
                    help='''Run osc tests i.e. the oscillograms with one sigma 
                    deviations in the parameters.''')
parser.add_argument('--aeff', action='store_true', default=False,
                    help='''Run effective area tests i.e. the different 
                    transforms with the aeff systematics.''')
parser.add_argument('--reco', action='store_true', default=False,
                    help='''Run reco tests i.e. the different reco kernels and 
                    their systematics.''')
parser.add_argument('--pid', action='store_true', default=False,
                    help='''Run PID tests i.e. the different pid kernels 
                    methods and their systematics.''')
parser.add_argument('--full', action='store_true', default=False,
                    help='''Run full pipeline tests for the baseline i.e. all 
                    stages simultaneously rather than indiviually.''')
parser.add_argument('--outdir', metavar='DIR', type=str, required=True,
                    help='''Store all output plots to this directory. If they 
                    don't exist, the script will make them, including all 
                    subdirectories.''')
args = parser.parse_args()

###############################################################################
#                                                                             #
# First check if output directories exist and are empty. If not, create them. #
#                                                                             #
###############################################################################

if not os.path.isdir(args.outdir):
    os.makedirs(args.outdir)

outdirs = ['flux', 'osc', 'aeff', 'reco', 'pid', 'full']

for outdir in outdirs:
    if not os.path.isdir(args.outdir+'/'+outdir):
        os.makedirs(args.outdir+'/'+outdir)
    for outimg in os.listdir(args.outdir+'/'+outdir):
        os.unlink(os.path.join(args.outdir+'/'+outdir,outimg))

##############################################################
#                                                            #
# Then set which tests to be done. This is likely to be all, #
# but has been made more flexible for debugging purposes.    #
#                                                            #
##############################################################

if args.flux or args.osc or args.aeff or args.reco or args.pid or args.full:
    test_all = False
else:
    test_all = args.all

if test_all:
    args.flux = True
    args.osc = True
    args.aeff = True
    args.reco = True
    args.pid = True
    args.full = True

###############################################################
#                                                             #
# Perform Flux Tests.                                         #
#                                                             #
###############################################################

if args.flux:
    flux_config = parse_config('settings/flux_test.ini')
    flux_config['flux']['params']['flux_file'] = 'flux/honda-2015-spl-solmax-aa.d'
    flux_config['flux']['params']['flux_mode'] = 'integral-preserving'
    
    for syst in [None, 'atm_delta_index', 'nue_numu_ratio', 'nu_nubar_ratio', 'energy_scale']:
        do_flux_comparison(config=deepcopy(flux_config),
                           servicename='IP_Honda',
                           pisa2file='data/flux/PISAV2IPHonda2015SPLSolMaxFlux.json',
                           systname=syst,
                           outdir=args.outdir)

    flux_config['flux']['params']['flux_mode'] = 'bisplrep'
    do_flux_comparison(config=deepcopy(flux_config),
                       servicename='bisplrep_Honda',
                       pisa2file='data/flux/PISAV2bisplrepHonda2015SPLSolMaxFlux.json',
                       systname=None,
                       outdir=args.outdir)

###############################################################
#                                                             #
# Perform Oscillations Tests.                                 #
#                                                             #
###############################################################

if args.osc:
    osc_config = parse_config('settings/osc_test.ini')

    for syst in [None, 'theta12', 'theta13', 'theta23', 'deltam21', 'deltam31']:
        do_osc_comparison(config=deepcopy(osc_config),
                          servicename='prob3',
                          pisa2file='data/osc/PISAV2OscStageProb3Service.json',
                          systname=syst,
                          outdir=args.outdir)

###############################################################
#                                                             #
# Perform Effective Area Tests.                               #
#                                                             #
###############################################################

if args.aeff:
    aeff_config = parse_config('settings/aeff_test.ini')
    aeff_config['aeff']['params']['aeff_weight_file'] = 'events/DC/2015/mdunkman/1XXXX/UnJoined/DC_MSU_1X585_unjoined_events_mc.hdf5'

    for syst in [None, 'aeff_scale']:
        do_aeff_comparison(config=deepcopy(aeff_config),
                           servicename='hist1X585',
                           pisa2file='data/aeff/PISAV2AeffStageHist1X585Service.json',
                           systname=syst,
                           outdir=args.outdir)

###############################################################
#                                                             #
# Perform Reconstruction Tests.                               #
#                                                             #
###############################################################

if args.reco:
    reco_config = parse_config('settings/reco_test.ini')
    reco_config['reco']['params']['reco_weights_name'] = None
    reco_config['reco']['params']['reco_weight_file'] = 'events/DC/2015/mdunkman/1XXXX/Joined/DC_MSU_1X585_joined_nu_nubar_events_mc.hdf5'
    do_reco_comparison(config=deepcopy(reco_config),
                       servicename='hist1X585',
                       pisa2file='data/reco/PISAV2RecoStageHist1X585Service.json',
                       outdir=args.outdir)

    reco_config['reco']['params']['reco_weight_file'] = 'events/DC/2015/mdunkman/1XXX/Joined/DC_MSU_1X60_joined_nu_nubar_events_mc.hdf5'
    do_reco_comparison(config=deepcopy(reco_config),
                       servicename='hist1X60',
                       pisa2file='data/reco/PISAV2RecoStageHist1X60Service.json',
                       outdir=args.outdir)

###############################################################
#                                                             #
# Perform PID Tests.                                          #
#                                                             #
###############################################################

if args.pid:
    pid_config = parse_config('settings/pid_test.ini')
    do_pid_comparison(config=deepcopy(pid_config),
                      servicename='pidV39',
                      pisa2file='data/pid/PISAV2PIDStageHistV39Service.json',
                      outdir=args.outdir)
    pid_config['pid']['params']['pid_events'] = 'events/DC/2015/mdunkman/1XXXX/Joined/DC_MSU_1X585_joined_nu_nubar_events_mc.hdf5'
    pid_config['pid']['params']['pid_weights_name'] = 'weighted_aeff'
    pid_config['pid']['params']['pid_ver'] = 'msu_mn8d-mn7d'
    do_pid_comparison(config=deepcopy(pid_config),
                      servicename='pid1X585',
                      pisa2file='data/pid/PISAV2PIDStageHist1X585Service.json',
                      outdir=args.outdir)

###############################################################
#                                                             #
# Perform Full Pipeline Tests.                                #
#                                                             #
###############################################################

if args.full:
    full_config = parse_config('settings/full_pipeline_test.ini')
    do_pipeline_comparison(config=deepcopy(full_config),
                           pisa2file='data/full/PISAV2FullDeepCorePipeline-IPSPL2015SolMax-Prob3CPUNuFit2014-AeffHist1X585-RecoHist1X585-PIDHist1X585.json',
                           outdir=args.outdir)
