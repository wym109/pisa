#! /usr/bin/env python
# author: S.Wren
# date:   March 20, 2016
"""
Runs the pipeline multiple times to test everything still agrees with PISA 2.
Test data for comparing against should be in the tests/data directory.
A set of plots will be output in the tests/output directory for you to check.
Agreement is expected to order 10^{-14} in the far right plots.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import Sequence
from copy import deepcopy
import os
import shutil

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
import numpy as np

from pisa.core.map import Map, MapSet
from pisa.core.pipeline import Pipeline
from pisa.utils.log import logging, set_verbosity
from pisa.utils.fileio import mkdir
from pisa.utils.parse_config import parse_config
from pisa.utils.jsons import from_json
from pisa.utils.plot import show_map, delta_map, ratio_map

# TODO:
# * Currently must be run from within tests dir. Fix this.
# * Data files should live somewhere relatively reference-able via $PISA dir
#   (or find-able by resources module)


def clean_dir(path):
    if isinstance(path, Sequence):
        path = os.path.join(*path)
    assert isinstance(path, basestring)

    if os.path.exists(path):
        # Remove if (possibly non-empty) directory
        if os.path.isdir(path):
            shutil.rmtree(path)
        # Remove if file
        else:
            os.remove(path)
    # Create the new directory at the path
    mkdir(path)


def baseplot(m, title):
    show_map(m)
    plt.xlabel(r'$\cos\theta_Z$')
    plt.ylabel(r'Energy (GeV)')
    plt.title(title)


def plot_comparisons(ref_map, new_map, ref_abv, new_abv, outdir, subdir, name,
                     texname, stagename, servicename, ftype='png'):
    path = [outdir]

    if subdir is None:
        subdir = stagename.lower()
    path.append(subdir)

    mkdir(os.path.join(*path))

    fname = ['pisa_%s_%s_comparisons' %(ref_abv.lower(), new_abv.lower()),
             'stage_'+stagename]
    if servicename is not None:
        fname.append('service_'+servicename)
    if name is not None:
        fname.append(name.lower())
    fname = '__'.join(fname) + '.' + ftype

    path.append(fname)

    basetitle = []
    if texname is not None:
        basetitle.append(r'$%s$' % texname)
    if stagename is not None:
        basetitle.append('%s' % stagename)
    basetitle.append('PISA')
    basetitle = ' '.join(basetitle)

    RatioMapObj = ratio_map(new_map, ref_map)
    DiffMapObj = delta_map(new_map, ref_map)
    DiffRatioMapObj = ratio_map(DiffMapObj, ref_map)

    plt.figure(figsize = (20,5))
    plt.subplot(1,5,1)
    baseplot(m=ref_map, title=basetitle+' '+ref_abv)

    plt.subplot(1,5,2)
    baseplot(m=new_map, title=basetitle+' '+new_abv)

    plt.subplot(1,5,3)
    baseplot(m=RatioMapObj, title=basetitle+' %s/%s' %(new_abv, ref_abv))

    plt.subplot(1,5,4)
    baseplot(m=DiffMapObj, title=basetitle+' %s-%s' %(new_abv, ref_abv))

    plt.subplot(1,5,5)
    baseplot(m=DiffRatioMapObj, title=basetitle+' (%s-%s)/%s'
             %(new_abv, ref_abv, ref_abv))

    plt.tight_layout()
    plt.savefig(os.path.join(*path))
    plt.close()


def compare_flux(config, servicename, pisa2file, systname, outdir):
    if systname is not None:
        try:
            config['flux']['params'][systname] = config['flux']['params'][systname].value+config['flux']['params'][systname].prior.stddev
        except:
            config['flux']['params'][systname] = 1.25*config['flux']['params'][systname].value

        pisa2file = pisa2file.split('.json')[0]+'-%s%.2f.json'%(systname,config['flux']['params'][systname].value)
        servicename += '-%s%.2f.json'%(systname,config['flux']['params'][systname].value)

    pipeline = Pipeline(config)
    stage = pipeline.stages[0]
    outputs = stage.get_outputs()
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

            plot_comparisons(
                ref_map=pisa_map_to_plot,
                new_map=cake_map_to_plot,
                ref_abv='V2', new_abv='V3',
                outdir=outdir,
                subdir='flux',
                stagename='flux',
                servicename=servicename,
                name=nukey,
                texname=outputs[nukey].tex
            )
    
    return pipeline


def compare_osc(config, servicename, pisa2file, systname, outdir):
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

            plot_comparisons(
                ref_map=pisa_map_to_plot,
                new_map=cake_map_to_plot,
                ref_abv='V2', new_abv='V3',
                outdir=outdir,
                subdir='osc',
                stagename='osc',
                servicename=servicename,
                name=nukey,
                texname=outputs[nukey].tex
            )
    
    return pipeline


def compare_aeff(config, servicename, pisa2file, systname, outdir):
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

                plot_comparisons(
                    ref_map=pisa_map_to_plot,
                    new_map=cake_map_to_plot,
                    ref_abv='V2', new_abv='V3',
                    outdir=outdir,
                    subdir='aeff',
                    stagename='aeff',
                    servicename=servicename,
                    name=cakekey,
                    texname=outputs[cakekey].tex,
                )
    
    return pipeline


def compare_reco(config, servicename, pisa2file, outdir):
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

            plot_comparisons(
                ref_map=pisa_map_to_plot,
                new_map=cake_map_to_plot,
                ref_abv='V2', new_abv='V3',
                outdir=outdir,
                subdir='reco',
                stagename='reco',
                servicename=servicename,
                name=nukey,
                texname=outputs[nukey].tex
            )
    
    return pipeline


def compare_pid(config, servicename, pisa2file, outdir):
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

    plot_comparisons(
        ref_map=total_pisa_cscd_dict,
        new_map=total_cake_cscd_dict,
        ref_abv='V2', new_abv='V3',
        outdir=outdir,
        subdir='pid',
        stagename='pid',
        servicename=servicename,
        name='cscd',
        texname=r'{\rm cscd}'
    )
    plot_comparisons(
        ref_map=total_pisa_trck_dict,
        new_map=total_cake_trck_dict,
        ref_abv='V2', new_abv='V3',
        outdir=outdir,
        subdir='pid',
        stagename='pid',
        servicename=servicename,
        name='trck',
        texname=r'{\rm trck}'
    )
    
    return pipeline


def compare_pipeline(config, pisa2file, outdir):
    pipeline = Pipeline(config)
    outputs = pipeline.get_outputs()

    total_cake_trck_dict = {}
    total_cake_cscd_dict = {}

    for cake_key in outputs.names:
        if 'trck' in cake_key:
            if len(total_cake_trck_dict.keys()) == 0:
                total_cake_trck_dict['ebins'] = outputs[cake_key].binning['reco_energy'].bin_edges.magnitude
                total_cake_trck_dict['czbins'] = outputs[cake_key].binning['reco_coszen'].bin_edges.magnitude
                total_cake_trck_dict['map'] = outputs[cake_key].hist.T
            else:
                total_cake_trck_dict['ebins'] = outputs[cake_key].binning['reco_energy'].bin_edges.magnitude
                total_cake_trck_dict['czbins'] = outputs[cake_key].binning['reco_coszen'].bin_edges.magnitude
                total_cake_trck_dict['map'] += outputs[cake_key].hist.T
        elif 'cscd' in cake_key:
            if len(total_cake_cscd_dict.keys()) == 0:
                total_cake_cscd_dict['map'] = outputs[cake_key].hist.T
                total_cake_cscd_dict['ebins'] = outputs[cake_key].binning['reco_energy'].bin_edges.magnitude
                total_cake_cscd_dict['czbins'] = outputs[cake_key].binning['reco_coszen'].bin_edges.magnitude
            else:
                total_cake_cscd_dict['map'] += outputs[cake_key].hist.T
                total_cake_cscd_dict['ebins'] = outputs[cake_key].binning['reco_energy'].bin_edges.magnitude
                total_cake_cscd_dict['czbins'] = outputs[cake_key].binning['reco_coszen'].bin_edges.magnitude

    pisa2_comparisons = from_json(pisa2file)

    total_pisa_trck_dict = pisa2_comparisons['trck']
    total_pisa_cscd_dict = pisa2_comparisons['cscd']

    plot_comparisons(
        ref_map=total_pisa_cscd_dict,
        new_map=total_cake_cscd_dict,
        ref_abv='V2', new_abv='V3',
        outdir=outdir,
        subdir='fullpipeline',
        stagename='fullpipeline',
        servicename=None,
        name='cscd',
        texname=r'{\rm cscd}'
    )
    plot_comparisons(
        ref_map=total_pisa_trck_dict,
        new_map=total_cake_trck_dict,
        ref_abv='V2', new_abv='V3',
        outdir=outdir,
        subdir='fullpipeline',
        stagename='fullpipeline',
        servicename=None,
        name='trck',
        texname=r'{\rm trck}'
    )
    
    return pipeline


if __name__ == '__main__':
    parser = ArgumentParser(
        description='''Runs a set of tests on the PISA 3 pipeline against
        benchmark PISA 2 data. The plots will be deleted and re-made every time
        you run this script so you can always be sure that the ones you have
        represent your PISA 3 in its current state. This script should always
        be run when you make any major modifications to be sure nothing has
        broken. If you find this script does not work, please either fix it or
        report it! In general, this will signify you have "changed" something,
        somehow in the basic functionality which you should understand!''',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
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
                        stages simultaneously rather than each in isolation.''')
    parser.add_argument('--outdir', metavar='DIR', type=str, required=True,
                        help='''Store all output plots to this directory. If they
                        don't exist, the script will make them, including all
                        subdirectories.''')
    parser.add_argument('-v', action='count', default=None,
                        help='set verbosity level')
    args = parser.parse_args()
    set_verbosity(args.v)

    # Figure out which tests to do
    test_all = True
    if args.flux or args.osc or args.aeff or args.reco or args.pid or args.full:
        test_all = False

    # Perform Flux Tests.
    if args.flux or test_all:
        flux_config = parse_config('settings/flux_test.ini')
        flux_config['flux']['params']['flux_file'] = 'flux/honda-2015-spl-solmax-aa.d'
        flux_config['flux']['params']['flux_mode'] = 'integral-preserving'

        for syst in [None, 'atm_delta_index', 'nue_numu_ratio',
                     'nu_nubar_ratio', 'energy_scale']:
            flux_pipeline = compare_flux(
                config=deepcopy(flux_config),
                servicename='IP_Honda',
                pisa2file='data/flux/PISAV2IPHonda2015SPLSolMaxFlux.json',
                systname=syst,
                outdir=args.outdir
            )

        flux_config['flux']['params']['flux_mode'] = 'bisplrep'
        flux_pipeline = compare_flux(
            config=deepcopy(flux_config),
            servicename='bisplrep_Honda',
            pisa2file='data/flux/PISAV2bisplrepHonda2015SPLSolMaxFlux.json',
            systname=None,
            outdir=args.outdir
        )

    # Perform Oscillations Tests.
    if args.osc or test_all:
        osc_config = parse_config('settings/osc_test.ini')

        for syst in [None, 'theta12', 'theta13', 'theta23', 'deltam21', 'deltam31']:
            osc_pipeline = compare_osc(
                config=deepcopy(osc_config),
                servicename='prob3',
                pisa2file='data/osc/PISAV2OscStageProb3Service.json',
                systname=syst,
                outdir=args.outdir
            )

    # Perform Effective Area Tests.
    if args.aeff or test_all:
        aeff_config = parse_config('settings/aeff_test.ini')
        aeff_config['aeff']['params']['aeff_weight_file'] = 'events/DC/2015/mdunkman/1XXXX/UnJoined/DC_MSU_1X585_unjoined_events_mc.hdf5'

        for syst in [None, 'aeff_scale']:
            aeff_pipeline = compare_aeff(
                config=deepcopy(aeff_config),
                servicename='hist1X585',
                pisa2file='data/aeff/PISAV2AeffStageHist1X585Service.json',
                systname=syst,
                outdir=args.outdir
            )

    # Perform Reconstruction Tests.
    if args.reco or test_all:
        reco_config = parse_config('settings/reco_test.ini')
        reco_config['reco']['params']['reco_weights_name'] = None
        reco_config['reco']['params']['reco_weight_file'] = 'events/DC/2015/mdunkman/1XXXX/Joined/DC_MSU_1X585_joined_nu_nubar_events_mc.hdf5'
        reco_pipeline = compare_reco(
            config=deepcopy(reco_config),
            servicename='hist1X585',
            pisa2file='data/reco/PISAV2RecoStageHist1X585Service.json',
            outdir=args.outdir
        )

        reco_config['reco']['params']['reco_weight_file'] = 'events/DC/2015/mdunkman/1XXX/Joined/DC_MSU_1X60_joined_nu_nubar_events_mc.hdf5'
        reco_pipeline = compare_reco(
            config=deepcopy(reco_config),
            servicename='hist1X60',
            pisa2file='data/reco/PISAV2RecoStageHist1X60Service.json',
            outdir=args.outdir
        )

    # Perform PID Tests.
    if args.pid or test_all:
        pid_config = parse_config('settings/pid_test.ini')
        pid_pipeline = compare_pid(
            config=deepcopy(pid_config),
            servicename='pidV39',
            pisa2file='data/pid/PISAV2PIDStageHistV39Service.json',
            outdir=args.outdir
        )
        pid_config['pid']['params']['pid_events'] = 'events/DC/2015/mdunkman/1XXXX/Joined/DC_MSU_1X585_joined_nu_nubar_events_mc.hdf5'
        pid_config['pid']['params']['pid_weights_name'] = 'weighted_aeff'
        pid_config['pid']['params']['pid_ver'] = 'msu_mn8d-mn7d'
        pid_pipeline = compare_pid(
            config=deepcopy(pid_config),
            servicename='pid1X585',
            pisa2file='data/pid/PISAV2PIDStageHist1X585Service.json',
            outdir=args.outdir
        )

    # Perform Full Pipeline Tests.
    if args.full or test_all:
        full_config = parse_config('settings/full_pipeline_test.ini')
        full_pipeline = compare_pipeline(
            config=deepcopy(full_config),
            pisa2file='data/full/PISAV2FullDeepCorePipeline-IPSPL2015SolMax-Prob3CPUNuFit2014-AeffHist1X585-RecoHist1X585-PIDHist1X585.json',
            outdir=args.outdir
        )
