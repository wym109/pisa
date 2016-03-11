#! /usr/bin/env python
#
# Aeff.py
#
# This module is the implementation of the stage2 analysis. The main
# purpose of stage2 is to combine the "oscillated Flux maps" with the
# effective areas to create oscillated event rate maps, using the true
# information. This signifies what the "true" event rate would be for
# a detector with our effective areas, but with perfect PID and
# resolutions.
#
# If desired, this will create an output file with the results of
# the current stage of processing.
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# date:   April 8, 2014
#


import sys

import numpy as np
from scipy.constants import Julian_year
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from pisa.utils import flavInt
from pisa.utils.log import logging, set_verbosity
from pisa.utils.fileio import from_file, to_file
from pisa.utils.proc import report_params, get_params, add_params
from pisa.utils.utils import check_binning, get_binning, prefilled_map


def get_event_rates(osc_flux_maps, aeff_service, livetime, aeff_scale,
                    **kwargs):
    '''
    Main function for this module, which returns the event rate maps
    for each flavor and interaction type, using true energy and zenith
    information. The content of each bin will be the weighted aeff
    multiplied by the oscillated flux, so that the returned dictionary
    will be of the form:
    {'nue': {'cc':map, 'nc':map},
     'nue_bar': {'cc':map, 'nc':map}, ...
     'nutau_bar': {'cc':map, 'nc':map} }
    \params:
      * osc_flux_maps - maps containing oscillated fluxes
      * aeff_service - the effective area service to use
      * livetime - detector livetime for which to calculate event counts
      * aeff_scale - systematic to be a proxy for the realistic effective area
    '''

    # Get parameters used here
    params = get_params()
    report_params(params, units=['', 'yrs', ''])

    # Initialize return dict
    event_rate_maps = {'params': add_params(params, osc_flux_maps['params'])}

    # Get effective area
    aeff_dict = aeff_service.get_aeff()

    ebins, czbins = get_binning(osc_flux_maps)

    # apply the scaling for nu_xsec_scale and nubar_xsec_scale...
    flavours = ['nue', 'numu', 'nutau', 'nue_bar', 'numu_bar', 'nutau_bar']
    for flavour in flavours:
        osc_flux_map = osc_flux_maps[flavour]['map']
        int_type_dict = {}
        for int_type in ['cc', 'nc']:
            event_rate = osc_flux_map * aeff_dict[flavour][int_type] \
                    * (livetime * Julian_year * aeff_scale)

            int_type_dict[int_type] = {'map':event_rate,
                                       'ebins':ebins,
                                       'czbins':czbins}
            logging.debug("  Event Rate before reco for %s/%s: %.2f"
                          % (flavour, int_type, np.sum(event_rate)))
        event_rate_maps[flavour] = int_type_dict

    return event_rate_maps


def aeff_service_factory(aeff_mode, **kwargs):
    """Construct and return a AeffService class based on `mode`
    
    Parameters
    ----------
    aeff_mode : str
        Identifier for which AeffService class to instantiate. Currently
        understood are 'param' and 'mc'.
    **kwargs
        All subsequent kwargs are passed (as **kwargs), to the class being
        instantiated.
    """
    aeff_mode = aeff_mode.lower()
    if aeff_mode == 'mc':
        from pisa.aeff.AeffServiceMC import AeffServiceMC
        return AeffServiceMC(**kwargs)

    if aeff_mode == 'param':
        from pisa.aeff.AeffServicePar import AeffServicePar
        return AeffServicePar(**kwargs)

    if aeff_mode == 'smooth':
        from pisa.aeff.AeffServiceSmooth import AeffServiceSmooth
        return AeffServiceSmooth(**kwargs)

    if aeff_mode == 'slice_smooth':
        from pisa.aeff.AeffServiceSliceSmooth import AeffServiceSliceSmooth
        return AeffServiceSliceSmooth(**kwargs)

    raise ValueError('Unrecognized Aeff `aeff_mode`: "%s"' % aeff_mode)


def add_argparser_args(parser):
    from pisa.aeff.AeffServiceMC import AeffServiceMC
    from pisa.aeff.AeffServicePar import AeffServicePar
    from pisa.aeff.AeffServiceSmooth import AeffServiceSmooth

    parser.add_argument(
        '--aeff-mode', type=str, required=True,
        choices=['mc', 'param', 'smooth', 'slice_smooth'],
        help='Aeff service to use'
    )

    # Add args specific to the known classes
    AeffServiceMC.add_argparser_args(parser)
    AeffServicePar.add_argparser_args(parser)
    AeffServiceSmooth.add_argparser_args(parser)
    AeffServiceSliceSmooth.add_argparser_args(parser)

    return parser


def plot_2d_comparisons(ebins=np.logspace(0, np.log10(80), 40),
                        czbins=np.linspace(-1, 0, 21)):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    def updateExtrema(_min, _max, a):
        return np.min([_min, np.min(a)]), np.max([_max, np.max(a)])

    c1 = (0.0, 0.6, 0.8)
    c2 = (0.7, 0.2, 0.5)
    c3 = (0.4, 0.3, 0.0)
    diff_cmap = mpl.cm.coolwarm
    abs_cmap = mpl.cm.Paired
    diff_cmap.set_bad((1,1,1), 1)
    abs_cmap.set_bad((1,1,1), 1)

    ebin_midpoints = (ebins[:-1] + ebins[1:])/2.
    czbin_midpoints = (czbins[:-1] + czbins[1:])/2.
    e_oversamp = np.logspace(np.log10(ebins[0]), np.log10(ebins[-1]), 1001)
    cz_oversamp = np.linspace(czbins[0], czbins[-1], 1001)

    mc_service = aeff_service_factory(
        aeff_mode='mc', ebins=ebins, czbins=czbins, compute_error=True,
        aeff_weight_file='events/pingu_v36/'
        'events__pingu__v36__runs_388-390__proc_v5__joined_G_nuall_nc_G_nuallbar_nc.hdf5'
    )
    aeff_mc, aeff_mc_err = mc_service.get_aeff_with_error()

    # The rest of the aeffs will be stored in dicts to allow iterating
    aeff = {}
    aeff_oversamp = {}

    aeff_egy_par = {
        'NC': 'aeff/V36/cuts_V5/a_eff_nuall_nc.dat',
        'NC_bar': 'aeff/V36/cuts_V5/a_eff_nuallbar_nc.dat',
        'nue': 'aeff/V36/cuts_V5/a_eff_nue.dat',
        'nue_bar': 'aeff/V36/cuts_V5/a_eff_nuebar.dat',
        'numu': 'aeff/V36/cuts_V5/a_eff_numu.dat',
        'numu_bar': 'aeff/V36/cuts_V5/a_eff_numubar.dat',
        'nutau': 'aeff/V36/cuts_V5/a_eff_nutau.dat',
        'nutau_bar': 'aeff/V36/cuts_V5/a_eff_nutaubar.dat'
    }
    param_service = aeff_service_factory(
        aeff_mode='param', ebins=ebins, czbins=czbins,
        aeff_egy_par=aeff_egy_par,
        aeff_coszen_par='aeff/V36/V36_aeff_cz.json'
    )
    aeff['param'] = param_service.get_aeff()

    param_service = aeff_service_factory(
        aeff_mode='param', ebins=e_oversamp, czbins=cz_oversamp,
        aeff_egy_par=aeff_egy_par,
        aeff_coszen_par='aeff/V36/V36_aeff_cz.json'
    )
    aeff_oversamp['param'] = param_service.get_aeff()

    smooth_service = aeff_service_factory(
        aeff_mode='smooth', ebins=ebins, czbins=czbins,
        aeff_energy_smooth='aeff/pingu_v36/'
            'aeff_energy_smooth__pingu_v36__runs_388-390__proc_5.json',
        aeff_coszen_smooth='aeff/pingu_v36/'
            'aeff_coszen_smooth__pingu_v36__runs_388-390__proc_5.json'
    )
    aeff['smooth'] = smooth_service.get_aeff()
    smooth_service.update(ebins=e_oversamp, czbins=cz_oversamp)
    aeff_oversamp['smooth'] = smooth_service.get_aeff()

    slice_service = aeff_service_factory(
        aeff_mode='slice_smooth', ebins=ebins, czbins=czbins,
        aeff_slice_smooth='aeff/pingu_v36/'
        'aeff_slice_smooth__pingu_v36__runs_388-390__proc_5.hdf5',
    )
    aeff['slice'] = slice_service.get_aeff()
    slice_service.update(ebins=e_oversamp, czbins=cz_oversamp)
    aeff_oversamp['slice'] = slice_service.get_aeff()

    n_services = len(aeff)
    services = ['param', 'slice']

    grouped = sorted([flavInt.NuFlavIntGroup(fi)
                      for fi in mc_service.events.metadata['flavints_joined']])
    should_be_grouped = sorted([flavInt.NuFlavIntGroup('nuall_nc'),
                                flavInt.NuFlavIntGroup('nuallbar_nc')])
    if grouped != should_be_grouped:
        if len(grouped) == 0:
            grouped = None
        raise ValueError('Only works with groupings (%s) but instead got'
                         ' groupings (%s).' % (should_be_grouped, grouped))

    # Get *un*joined flavints
    individual_flavints = flavInt.NuFlavIntGroup(flavInt.ALL_NUFLAVINTS)
    for group in grouped:
        individual_flavints -= group
    ungrouped = sorted([flavInt.NuFlavIntGroup(fi) for fi in individual_flavints])

    # Sample points for plotting (cz comes first, as it's on the x-axis for
    # plotting purposes)
    x, y = np.meshgrid(czbin_midpoints, np.log10(ebin_midpoints))
    x, y = np.meshgrid(czbins, np.log10(ebins))
    x_oversamp, y_oversamp = np.meshgrid(cz_oversamp,
                                         np.log10(e_oversamp), indexing='xy')

    abs_plt_kwargs = dict(cmap=abs_cmap)
    fractdiff_plt_kwargs = dict(cmap=diff_cmap)
    for group in ungrouped + grouped:
        rep_flavint = group.flavints()[0]

        mc = np.ma.masked_invalid(aeff_mc[rep_flavint])
        mc_err = np.ma.masked_invalid(aeff_mc_err[rep_flavint])
        log_mc = np.ma.masked_invalid(np.log10(mc))

        vmin_abs, vmax_abs = np.inf, -np.inf
        vmin_fractdiff, vmax_fractdiff = np.inf, -np.inf

        all_abs_qm = []
        all_fractdiff_qm = []

        fig, axgrp = plt.subplots(3, len(services)+1, figsize=(15, 10))
        fig.suptitle('$' + group.tex() + '$', fontsize=18)
        axiter = iter(axgrp.flatten())

        # Turn off unused axes
        axgrp[2,0].axis('off')

        # Plot MC as reference
        ax = axgrp[1, 0]
        qm = ax.pcolormesh(x, y, log_mc, **abs_plt_kwargs)
        all_abs_qm.append(qm)
        vmin_abs, vmax_abs = updateExtrema(vmin_abs, vmax_abs, log_mc)
        ax.set_xlabel(r'$\cos\,\theta_{\rm z}$')
        ax.set_ylabel(r'$\log_{10}(E/{\rm GeV})$')
        ax.set_title(r'$\log_{10}({\rm MC-hist})$, "standard" binning', fontsize=14)
        cbar = plt.colorbar(qm, ax=ax)

        other_svc_os = None
        for svc_num, svc_key in enumerate(services):
            this = aeff[svc_key][rep_flavint]
            this_oversamp = aeff_oversamp[svc_key][rep_flavint]

            log_this = np.ma.masked_invalid(np.log10(this))
            log_this_oversamp = np.ma.masked_invalid(np.log10(this_oversamp))
            fractdiff = np.ma.masked_invalid((this - mc)/mc)
            fractdiff_normed = np.ma.masked_invalid((this - mc)/ mc_err)

            # Plot the oversampled map
            ax = axgrp[0, svc_num + 1]
            qm = ax.pcolormesh(x_oversamp, y_oversamp, log_this_oversamp,
                               **abs_plt_kwargs)
            ax.set_xlabel(r'$\cos\,\theta_{\rm z}$')
            ax.set_ylabel(r'$\log_{10}(E/{\rm GeV})$')
            ax.set_title(r'$\log_{10}({\rm aeff_' + svc_key + '})$, dense binning',
                         fontsize=14)
            all_abs_qm.append(qm)
            vmin_abs, vmax_abs = updateExtrema(vmin_abs, vmax_abs, log_this_oversamp)
            cbar = plt.colorbar(qm, ax=ax)

            # Plot the standard-binning map
            ax = axgrp[1, svc_num + 1]
            qm = ax.pcolormesh(x, y, log_this, **abs_plt_kwargs)
            all_abs_qm.append(qm)
            vmin_abs, vmax_abs = updateExtrema(vmin_abs, vmax_abs, log_this)
            ax.set_xlabel(r'$\cos\,\theta_{\rm z}$')
            ax.set_ylabel(r'$\log_{10}(E/{\rm GeV})$')
            ax.set_title(r'$\log_{10}({\rm aeff_' + svc_key + '})$, "standard" binning',
                         fontsize=14)
            cbar = plt.colorbar(qm, ax=ax)

            # Plot the fractional-difference map
            ax = axgrp[2, svc_num + 1]
            qm = ax.pcolormesh(x, y, fractdiff, **fractdiff_plt_kwargs)
            all_fractdiff_qm.append(qm)
            vmin_abs, vmax_abs = updateExtrema(vmin_abs, vmax_abs, fractdiff)
            ax.set_xlabel(r'$\cos\,\theta_{\rm z}$')
            ax.set_ylabel(r'$\log_{10}(E/{\rm GeV})$')
            ax.set_title(svc_key + '/MC - 1; $\mu=%0.3f,\;\sigma=%0.03f$' %
                         (np.mean(fractdiff), np.std(fractdiff)), fontsize=14)
            cbar = plt.colorbar(qm, ax=ax)

            if other_svc_os is None:
                other_svc_os = this_oversamp
            else:
                # Plot fractional difference between maps
                ax = axgrp[0,0] #fig2.add_subplot(1, 1, svc_num+1)
                fd = np.ma.masked_invalid((other_svc_os - this_oversamp)/this_oversamp)
                qm = ax.pcolormesh(x_oversamp, y_oversamp, fd,
                                   **fractdiff_plt_kwargs)
                #all_fractdiff_qm.append(qm)
                qm.set_clim(-.25, 0.25)
                ax.set_xlabel(r'$\cos\,\theta_{\rm z}$')
                ax.set_ylabel(r'$\log_{10}(E/{\rm GeV})$')
                ax.set_title(r'%s/%s - 1' % (services[0], services[1]), fontsize=14)
                cbar = plt.colorbar(qm, ax=ax)

            print svc_key, group, 'fractional difference with histogram'
            print '  mean:', np.mean(fractdiff)
            print '   RMS:', np.sqrt(np.mean(fractdiff*fractdiff))
            print '   std:', np.std(fractdiff)
            print '   MAD:', np.median(np.abs(fractdiff))
            print 'nrmstd:', np.std(fractdiff_normed)
            print '   max:', np.max(fractdiff)
            print '   min:', np.min(fractdiff)

        print ''

        [qm.set_clim((-7,-3.8)) for qm in all_abs_qm]
        [qm.set_clim((-1,1)) for qm in all_fractdiff_qm]
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig('/tmp/aeff_slice_vs_param_vs_mc_' + str(group) + '.png')


def plot_1d_comparisons(ebins=np.logspace(0, np.log10(80), 21),
                        czbins=np.linspace(-1, 1, 21)):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from pisa.utils.plot import stepHist

    c1 = (0.0, 0.6, 0.8)
    c2 = (0.7, 0.2, 0.5)
    c3 = (0.4, 0.3, 0.0)
    single_ebin = (np.min(ebins), np.max(ebins))
    single_czbin = (np.min(czbins), np.max(czbins))
    ebin_midpoints = (ebins[:-1] + ebins[1:])/2.
    czbin_midpoints = (czbins[:-1] + czbins[1:])/2.
    e_oversamp = np.logspace(np.log10(ebins[0]), np.log10(ebins[-1]), 1001)
    cz_oversamp = np.linspace(czbins[0], czbins[-1], 1001)

    mc_service = aeff_service_factory(
        aeff_mode='mc', ebins=ebins, czbins=single_czbin, compute_error=True,
        aeff_weight_file='events/pingu_v36/'
        'events__pingu__v36__runs_388-390__proc_v5__joined_G_nuall_nc_G_nuallbar_nc.hdf5'
    )
    aeff_e_mc, aeff_e_mc_err = mc_service.get_aeff_with_error()
    #aeff_e_mc = mc_service.get_aeff()
    mc_service.update(ebins=single_ebin, czbins=czbins)
    aeff_cz_mc, aeff_cz_mc_err = mc_service.get_aeff_with_error()

    param_service = aeff_service_factory(
        aeff_mode='param', ebins=ebins, czbins=czbins,
        aeff_egy_par={
            'NC': 'aeff/V36/cuts_V5/a_eff_nuall_nc.dat',
            'NC_bar': 'aeff/V36/cuts_V5/a_eff_nuallbar_nc.dat',
            'nue': 'aeff/V36/cuts_V5/a_eff_nue.dat',
            'nue_bar': 'aeff/V36/cuts_V5/a_eff_nuebar.dat',
            'numu': 'aeff/V36/cuts_V5/a_eff_numu.dat',
            'numu_bar': 'aeff/V36/cuts_V5/a_eff_numubar.dat',
            'nutau': 'aeff/V36/cuts_V5/a_eff_nutau.dat',
            'nutau_bar': 'aeff/V36/cuts_V5/a_eff_nutaubar.dat'
        },
        aeff_coszen_par='aeff/V36/V36_aeff_cz.json'
    )

    smooth_service = aeff_service_factory(
        aeff_mode='smooth', ebins=ebins, czbins=czbins,
        aeff_energy_smooth='aeff/pingu_v36/'
            'aeff_energy_smooth__pingu_v36__runs_388-390__proc_5.json',
        aeff_coszen_smooth='aeff/pingu_v36/'
            'aeff_coszen_smooth__pingu_v36__runs_388-390__proc_5.json'
    )

    # Plot aeff vs energy
    #plt.close(1); plt.close(2); plt.close(3); plt.close(4)
    [plt.figure(n).clf() for n in range(1,9)]
    cc_fig, cc_axgrp = plt.subplots(2, 3, num=1, figsize=(12, 7))
    nc_fig, nc_axgrp = plt.subplots(2, 3, num=2, figsize=(12, 7))
    cc_axiter = iter((cc_axgrp.T).flatten())
    nc_axiter = iter((nc_axgrp.T).flatten())
    cc_fig2, cc_axgrp2 = plt.subplots(2, 3, num=3, figsize=(12, 7))
    nc_fig2, nc_axgrp2 = plt.subplots(2, 3, num=4, figsize=(12, 7))
    cc_axiter2 = iter((cc_axgrp2.T).flatten())
    nc_axiter2 = iter((nc_axgrp2.T).flatten())
    for flavint in (list(flavInt.ALL_NUCC.flavints()) +
                    list(flavInt.ALL_NUNC.flavints())):
        if flavint.isCC():
            ax = cc_axiter.next()
            ax2 = cc_axiter2.next()
            clip_to = dict(a_min=1e-7, a_max=2e-4)
        else:
            ax = nc_axiter.next()
            ax2 = nc_axiter2.next()
            clip_to = dict(a_min=1e-9, a_max=1e-4)

        e_mc = np.squeeze(aeff_e_mc[flavint])
        e_mc_err = np.squeeze(aeff_e_mc_err[flavint])
        e_param_bin = param_service.sample_egy_curve(flavint, ebin_midpoints)
        e_smooth_bin = smooth_service.sample_egy_curve(flavint, ebin_midpoints)
        e_param_oversamp = param_service.sample_egy_curve(flavint, e_oversamp)
        e_smooth_oversamp = smooth_service.sample_egy_curve(flavint,
                                                            e_oversamp)
        stepHist(ebins, y=e_mc, yerr=e_mc_err, ax=ax, color='k',
                 label='MC')
        ax.plot(e_oversamp, e_param_oversamp, '-', color=c1, label='param')
        ax.plot(e_oversamp, e_smooth_oversamp, '-', color=c2, label='smooth')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(np.clip(ax.get_ylim(), **clip_to))
        leg = ax.legend(loc='lower right', frameon=False,
                        title='$' + flavint.tex() + '$')
        leg.get_title().set_fontsize(18)

        ax2.plot([1,100], [0,0], 'k-')
        ax2.plot([1,100], np.ones(2) * np.mean((e_param_bin-e_mc)/e_mc),
                 '--', color=c1)
        ax2.plot([1,100], np.ones(2) * np.mean((e_smooth_bin-e_mc)/e_mc),
                 '--', color=c2)
        stepHist(ebins, y=(e_param_bin-e_mc)/e_mc, yerr=e_mc_err,
                 ax=ax2, color=c1, lw=1.5,
                 label='(param-mc)/mc')
        stepHist(ebins, y=(e_smooth_bin-e_mc)/e_mc, yerr=e_mc_err,
                 ax=ax2, color=c2, lw=1.5,
                 label='(smooth-mc)/mc')
        ax2.plot(e_oversamp,
                 (e_smooth_oversamp-e_param_oversamp)/e_param_oversamp,
                 '-', color=c3,
                 label='(smooth-param)/param')

        ax2.set_xscale('log')
        ax2.set_ylim((-0.4,0.4))
        ax2.grid(b=False, which='both')
        leg = ax2.legend(loc='lower right', frameon=False,
                         title='$' + flavint.tex() + '$')
        leg.get_title().set_fontsize(18)

    [f.tight_layout() for f in [cc_fig, cc_fig2, nc_fig, nc_fig2]]

    #plt.close(5); plt.close(6); plt.close(7); plt.close(8)
    cc_fig, cc_axgrp = plt.subplots(2, 3, num=5, figsize=(12, 7))
    nc_fig, nc_axgrp = plt.subplots(2, 3, num=6, figsize=(12, 7))
    cc_fig2, cc_axgrp2 = plt.subplots(2, 3, num=7, figsize=(12, 7))
    nc_fig2, nc_axgrp2 = plt.subplots(2, 3, num=8, figsize=(12, 7))
    cc_axiter = iter((cc_axgrp.T).flatten())
    nc_axiter = iter((nc_axgrp.T).flatten())
    cc_axiter2 = iter((cc_axgrp2.T).flatten())
    nc_axiter2 = iter((nc_axgrp2.T).flatten())
    for flavint in (list(flavInt.ALL_NUCC.flavints()) +
                    list(flavInt.ALL_NUNC.flavints())):
        if flavint.isCC():
            ax = cc_axiter.next()
            ax2 = cc_axiter2.next()
            clip_to = dict(a_min=1e-7, a_max=2e-4)
        else:
            ax = nc_axiter.next()
            ax2 = nc_axiter2.next()
            clip_to = dict(a_min=1e-9, a_max=1e-4)

        cz_mc = np.squeeze(aeff_cz_mc[flavint])
        cz_mc_normfactor = 1/(np.sum(cz_mc)/len(cz_mc))
        cz_mc *= cz_mc_normfactor
        cz_mc_err = np.squeeze(aeff_cz_mc_err[flavint]) * cz_mc_normfactor
        cz_param_bin = param_service.sample_cz_curve(
            flavint, czbin_midpoints, normalize=True
        )
        cz_smooth_bin = smooth_service.sample_cz_curve(
            flavint, czbin_midpoints, normalize=True
        )
        cz_param_oversamp = param_service.sample_cz_curve(
            flavint, cz_oversamp, normalize=True
        )
        cz_smooth_oversamp = smooth_service.sample_cz_curve(
            flavint, cz_oversamp, normalize=True
        )

        stepHist(czbins, y=cz_mc, yerr=cz_mc_err, ax=ax, color='k',
                 label='MC')
        ax.plot(cz_oversamp, cz_param_oversamp, '-', color=c1, label='param')
        ax.plot(cz_oversamp, cz_smooth_oversamp, '-', color=c2, label='smooth')

        #ax.set_xscale('log')
        #ax.set_yscale('log')
        #ax.set_ylim(np.clip(ax.get_ylim(), **clip_to))
        leg = ax.legend(loc='best', frameon=False,
                        title='$' + flavint.tex() + '$')
        leg.get_title().set_fontsize(18)

        ax2.plot([-1,1], [0,0], 'k-')
        ax2.plot([-1,1], np.ones(2) * np.mean((cz_param_bin-cz_mc)/cz_mc),
                 '--', color=c1)
        ax2.plot([-1,1], np.ones(2) * np.mean((cz_smooth_bin-cz_mc)/cz_mc),
                 '--', color=c2)
        stepHist(czbins, y=(cz_param_bin-cz_mc)/cz_mc, yerr=cz_mc_err,
                 ax=ax2, color=c1, lw=1.5,
                 label='(param-mc)/mc')
        stepHist(czbins, y=(cz_smooth_bin-cz_mc)/cz_mc, yerr=cz_mc_err,
                 ax=ax2, color=c2, lw=1.5,
                 label='(smooth-mc)/mc')
        ax2.plot(cz_oversamp,
                 (cz_smooth_oversamp-cz_param_oversamp)/cz_param_oversamp,
                 '-', color=c3,
                 label='(smooth-param)/param')

        #ax2.set_xscale('log')
        #ax2.set_ylim((-0.4,0.4))
        ax2.grid(b=False, which='both')
        leg = ax2.legend(loc='best', frameon=False,
                         title='$' + flavint.tex() + '$')
        leg.get_title().set_fontsize(18)

    [f.tight_layout() for f in [cc_fig, cc_fig2, nc_fig, nc_fig2]]

    return mc_service, param_service, smooth_service


if __name__ == '__main__':
    parser = ArgumentParser(
        description='''Take an oscillated flux file as input & write out a set
        of oscillated event counts.''',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--osc-flux-maps', metavar='FLUX', type=from_file,
        help='''Osc flux input file with the following parameters:
        {"nue": {'czbins':[], 'ebins':[], 'map':[]},
         "numu": {...},
         "nutau": {...},
         "nue_bar": {...},
         "numu_bar": {...},
         "nutau_bar": {...} }'''
    )
    parser.add_argument(
        '--livetime', type=float, default=1.0,
        help='''livetime in years to re-scale by.'''
    )
    parser.add_argument(
        '--aeff-scale', type=float, default=1.0,
        help='''Overall scale on aeff'''
    )

    # Add AeffService-specific args
    add_argparser_args(parser)

    # Back to generic args
    parser.add_argument(
        '--outfile', dest='outfile', metavar='FILE', type=str,
        default="aeff_output.json",
        help='''file to store the output'''
    )
    parser.add_argument(
        '--plot', action='store_true',
        help='plot resulting maps'
    )
    parser.add_argument(
        '-v', '--verbose', action='count', default=None,
        help='set verbosity level'
    )

    # Parse args & convert them to a dict
    args = vars(parser.parse_args())

    # Set verbosity level
    set_verbosity(args.pop('verbose'))

    # Output file
    outfile = args.pop('outfile')

    livetime = args.pop('livetime')
    aeff_scale = args.pop('aeff_scale')

    osc_flux_maps = args.pop('osc_flux_maps')
    if osc_flux_maps is not None:
        # Load event maps (expected to be something like the output from a reco
        # stage)
        osc_flux_maps = fileio.from_file(args.pop('osc_flux_maps'))
        flavs = [fg for fg in sorted(osc_flux_maps)
                    if fg not in ['params', 'ebins', 'czbins']]
    else:
        # Otherwise, generate maps with all 1's to send through the PID stage
        flavs = ['nue', 'nue_bar', 'numu', 'numu_bar', 'nutau', 'nutau_bar']
        n_ebins = 39
        n_czbins = 40
        ebins = np.logspace(0, np.log10(80), n_ebins+1)
        czbins = np.linspace(-1, 1, n_czbins+1)
        osc_flux_maps = {f: prefilled_map(ebins, czbins,
                                          1/(livetime*Julian_year*aeff_scale))
                         for f in flavs} 
        osc_flux_maps['params'] = {}

    # Check, return binning
    args['ebins'], args['czbins'] = check_binning(osc_flux_maps)

    # Initialize the PID service
    aeff_service = aeff_service_factory(aeff_mode=args.pop('aeff_mode'),
                                        **args)

    # Calculate event rates after Aeff
    event_rate_aeff = get_event_rates(
        osc_flux_maps=osc_flux_maps, aeff_service=aeff_service,
        livetime=livetime, aeff_scale=aeff_scale
    )

    # Save the results to disk
    to_file(event_rate_aeff, outfile)

    # Produce plots useful for debugging
    if args['plot']:
        import os
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from pisa.utils import flavInt
        from pisa.utils import plot
        n_flavs = len(flavs)
        flavintgrp = flavInt.NuFlavIntGroup(flavs)

        n_rows = 2
        n_cols = n_flavs #int(np.ceil(n_flavs/2.0))*2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(26,14),
                                 dpi=50, sharex=True, sharey=True)

        # Plot fractional effect of effective areas
        osc_flux_maps = {f: prefilled_map(ebins, czbins, 1) for f in flavs}
        osc_flux_maps['params'] = {}
        fract_aeff = get_event_rates(
            osc_flux_maps=osc_flux_maps,
            aeff_service=aeff_service,
            livetime=1./Julian_year, aeff_scale=1, #aeff_scale
        )

        # TODO: Make aggregate maps: flav+antiflav CC, nuall NC, nuallbar NC,
        # nuall+nuallbar NC
        #cc_agg_maps = {prefilled_map(ebins, czbins, 0)}
        for flav_num, flav in enumerate(flavintgrp.flavs()):
            with flavInt.BarSep('_'):
                flav_key = str(flav)
            flav_tex = flavInt.tex(flav, d=1)

            for int_num, int_type in enumerate(['cc', 'nc']):
                flavint = flavInt.NuFlavInt(flav_key+int_type)
                flavint_tex = flavInt.tex(flavint, d=1)
                with flavInt.BarSep('_'):
                    flavint_key = str(flavint)

                #agg_map['map'] += fract_aeff[flav_key][int_type]['map']

                ax = axes[flav_num % 2, flav_num//2 + int_num*n_flavs//2]
                plt.sca(ax)
                plot.show_map(fract_aeff[flav_key][int_type], log=True,
                              cmap=mpl.cm.Paired)
                              #cmap=mpl.cm.Accent_r)
                #ax.get_children()[0].autoscale()
                ax.set_title('Fract. of ' + flavint_tex + ' in bin',
                             fontsize=14)

            #ax = axes[2, flav_num]
            #plt.sca(ax)
            #plot.show_map(agg_map, log=True, cmap=mpl.cm.Accent_r)
            ##ax.get_children()[0].autoscale()
            #ax.set_title('Total, CC+NC ' + flav_tex, fontsize=14)

        fig.tight_layout()
        fig2.tight_layout()
        base, ext = os.path.splitext(outfile)
        fig.savefig(base + '.pdf')
        fig.savefig(base + '.png')
        plt.draw()
        plt.show()
