#! /usr/bin/env python
#
# author: J.L. Lanfranchi
# date:   2016-03-08
#
"""
One-dimensional (in energy) effective areas are smoothed and sampled, one for
each slide in cosine-zenith.

Interpolation can be achieved by...?

Events of a given flavor/interaction type (or all events from grouped
flavor/interaction types) from a PISA events file have their effective areas
computed. This is smoothed with a spline, and the spline fit is sampled at the
specified energy bins' midpoints (on a linear scale).
"""


import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from itertools import izip

import numpy as np
from scipy.interpolate import splrep, splev

from pisa.utils.log import logging, set_verbosity
from pisa.utils.events import Events
from pisa.utils import flavInt
from pisa.utils import jsons
from pisa.utils import hdf
from pisa.utils import utils
from pisa.aeff.AeffServiceMC import AeffServiceMC


parser = ArgumentParser(
    '''Generate smoothed effective areas at energy bin centers. NOTE: at
    present, uses *ONLY* the MC-true-upgoing events.''',
    formatter_class=ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--events', metavar='RESOURCE_LOC', type=str,
    default='events/pingu_v36/'
    'events__pingu__v36__runs_388-390__proc_v5__joined_G_nuall_nc_G_nuallbar_nc.hdf5',
    help='''PISA events file used for computing smooth. it is expected that
    nuall_nc and nuallbar_nc are joined in this file, while other
    flavor/interaction types are unjoined.'''
)
parser.add_argument(
    '--outdir', metavar='dir', type=str,
    required=True,
    help='Directory into which to place the produced files.'
)
parser.add_argument(
    '--emin', metavar='emin_gev', type=float,
    required=True,
    help='Energy bins\' left-most edge, in gev'
)
parser.add_argument(
    '--emax', metavar='emax_gev', type=float,
    required=True,
    help='Energy bins\' righ-most edge, in gev'
)
parser.add_argument(
    '--n-ebins', type=int,
    default=50,
    help='Number of energy bins (logarithmically-spaced)'
)
parser.add_argument(
    '--czmin', metavar='COSZEN', type=float,
    default=-1,
    help='Cosine-zenith bins\' lowest edge'
)
parser.add_argument(
    '--czmax', metavar='COSZEN', type=float,
    default=1,
    help='Cosine-zenith bins\' highest edge'
)
parser.add_argument(
    '--n-czbins', type=int,
    required=True,
    help='Number of cosine-zenith bins (linearly-spaced)'
)
parser.add_argument(
    '--no-plots', action='store_true',
    help='Do not make debug plots (which are made by default)'
)
parser.add_argument(
    '-v', '--verbose',
    action='count', default=0,
    help='Set verbosity level'
)
args = parser.parse_args()

set_verbosity(args.verbose)

events_rsrc_path = args.events
outdir = os.path.expandvars(os.path.expanduser(args.outdir))
make_plots = not args.no_plots

# Load the events
events = Events(events_rsrc_path)

# Define binning for 1D A_eff smooth data. Note that a single CZ bin is
# employed to collapse that dimension of the histogram for characterizing
# energy smooth, and likewise a single E bin is employed to collapse that
# dimension for characterizing CZ smooth.
emin, emax, n_ebins = args.emin, args.emax, args.n_ebins
czmin, czmax, n_czbins = -1, +1, args.n_czbins

ebins = np.logspace(np.log10(emin), np.log10(emax), n_ebins+1)
czbins = np.linspace(czmin, czmax, n_czbins+1)

ebin_midpoints = (ebins[:-1] + ebins[1:])/2.0
czbin_midpoints = (czbins[:-1] + czbins[1:])/2.0

# Verify flavints joined in the file are those expected
grouped = sorted([flavInt.NuFlavIntGroup(fi)
                  for fi in events.metadata['flavints_joined']])
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

logging.debug("Groupings: %s" % grouped)
logging.debug("Ungrouped: %s" % ungrouped)

if make_plots:
    import matplotlib as mpl
    mpl.use('pdf')
    import matplotlib.pyplot as plt
    from pisa.utils.plot import stepHist

    FIGSIZE = (11,7)
    #plt.close(1)
    #plt.close(2)
    #plt.figure(1).clf()
    #plt.figure(2).clf()

smooth_store = {'ebins': ebins,
                'czbins': czbins}
slice_store = flavInt.FlavIntData()
for flavint in slice_store.flavints():
    slice_store[flavint] = []

aeff_service = AeffServiceMC(ebins, czbins[0:2], events_rsrc_path,
                             compute_error=True)

#def czslice_smoothing(hist, hist_err, xwindow=None):
#
#    zero_and_nan_indices = np.squeeze(
#        (s_aeff == 0) | (s_aeff != s_aeff) |
#        (s_aeff_err == 0) | (s_aeff_err != s_aeff_err)
#    )
#    min_err = np.min(s_aeff_err[s_aeff_err > 0])
#    s_aeff_err[zero_and_nan_indices] = min_err
#
#    # Smooth histogrammed A_eff(E) using a spline
#    spline = spline_fit[rep_flavint] = splrep(
#        #ebin_midpoints, s_aeff, w=1./np.array(s_aeff_err), k=3, s=100
#        ebin_midpoints, s_aeff, w=1./np.array(s_aeff_err),
#        k=3, s=100
#    )
#
#    # Sample the spline at the bin midpoints
#    smoothed_aeff = splev(ebin_midpoints, spline)
#
#    # Force bins that were previously NaN or 0 back to 0
#    smoothed_aeff[zero_and_nan_indices] = 0

for czbin in izip(czbins[:-1], czbins[1:]):
    aeff_service.update(ebins, czbin)
    aeff_data, aeff_err = aeff_service.get_aeff_with_error()

    #if make_plots:
    #    fig_part, ax_part = plt.subplots(2, 2, sharex=True, figsize=FIGSIZE)
    #    fig_anti, ax_anti = plt.subplots(2, 2, sharex=True, figsize=FIGSIZE)
    #    ax_part = ax_part.flatten()
    #    ax_anti = ax_anti.flatten()
    #    #basetitle = (
    #    #    r'cz $\in$ [' + '%s, %s' + r'] $A_{\rm eff}\;{\rm[m^2]}$,'
    #    #    ' %s geometry %s, MC runs %s with ver %s processing' %
    #    #    (czbin[0],
    #    #     czbin[1],
    #    #     events.metadata['detector'],
    #    #     events.metadata['geom'],
    #    #     utils.list2hrlist(events.metadata['runs']),
    #    #     events.metadata['proc_ver'])
    #    #)
    #    #fig_part.suptitle('Particle ' + basetitle, fontsize=12)
    #    #fig_anti.suptitle('Antiparticle ' + basetitle, fontsize=12)
    #    fig_part.suptitle('Particle')
    #    fig_anti.suptitle('Antiparticle')

    spline_fit = flavInt.FlavIntData()
    #smoothed_aeff = flavInt.FlavIntData()
    for group in ungrouped + grouped:
        # Only need to do computations for a single flavint from the group,
        # since all data in other flavints is just duplicated
        rep_flavint = group.flavints()[0]

        s_aeff = np.squeeze(aeff_data[rep_flavint])
        s_aeff_err = np.squeeze(aeff_err[rep_flavint])

        # Eliminate ~0 and nan in errors to avoid infinite weights
        # Smallest non-zero error
        zero_and_nan_indices = np.squeeze(
            (s_aeff == 0) | (s_aeff != s_aeff) |
            (s_aeff_err == 0) | (s_aeff_err != s_aeff_err)
        )
        min_err = np.min(s_aeff_err[s_aeff_err > 0])
        s_aeff_err[zero_and_nan_indices] = min_err

        # Smooth histogrammed A_eff(E) using a spline
        spline = spline_fit[rep_flavint] = splrep(
            #ebin_midpoints, s_aeff, w=1./np.array(s_aeff_err), k=3, s=100
            ebin_midpoints, s_aeff, w=1./np.array(s_aeff_err),
            k=3, s=50
        )

        # Sample the spline at the bin midpoints
        smoothed_aeff = splev(ebin_midpoints, spline)

        # Force bins that were previously NaN or 0 back to 0
        #smoothed_aeff[zero_and_nan_indices] = 0

        # Populate datastructure to be written to disk
        if not repr(group) in smooth_store:
            smooth_store[repr(group)] = {
                'hist': [],
                'hist_err': [],
                'smooth_cz_slices': [],
                'smooth': [],
            }
        smooth_store[repr(group)]['hist'].append(s_aeff)
        smooth_store[repr(group)]['hist_err'].append(s_aeff_err)
        smooth_store[repr(group)]['smooth_cz_slices'].append(smoothed_aeff)

        # Append to the slice list for this flavint/group
        slice_store[rep_flavint].append(smoothed_aeff)

        #if make_plots:
        #    # Figure out whether particle or anti-particle axis group
        #    if rep_flavint.isParticle():
        #        axgrp = ax_part
        #    else:
        #        axgrp = ax_anti

        #    # Determine axis by flavor
        #    axnum = 3
        #    if rep_flavint in flavInt.NuFlavIntGroup('nuecc+nuebarcc'):
        #        axnum = 0
        #    elif rep_flavint in flavInt.NuFlavIntGroup('numucc+numubarcc'):
        #        axnum = 1
        #    elif rep_flavint in flavInt.NuFlavIntGroup('nutaucc+nutaubarcc'):
        #        axnum = 2
        #    ax = axgrp[axnum]

        #    # Plot the histogram with errors
        #    stepHist(ebins, y=s_aeff, yerr=s_aeff_err,
        #             ax=ax, label='Histogramed', color=(0.8,0.2,0.6))

        #    # Plot the smoothed curve
        #    ax.plot(ebin_midpoints, smoothed_aeff, 'k-o',
        #            lw=0.25, ms=2,
        #            label='Spline-smoothed',)

        #    ax.set_xscale('log')
        #    ax.set_yscale('log')
        #    if axnum in [2,3]:
        #        ax.set_xlabel('Energy (GeV)')

        #    if axnum in [0, 1]:
        #        ax.set_ylim(1e-7, 2e-4)
        #    else:
        #        ax.set_ylim(1e-8, 2e-4)

        #    ax.grid(b=True, which='both', ls='-', alpha=0.5, color=[0.7]*3,
        #            lw=0.5)

        #    leg = ax.legend(loc='lower right', title=flavInt.tex(group, d=True),
        #                    frameon=False)
        #    leg.get_title().set_fontsize(18)

    #if make_plots:
    #    fig_part.tight_layout(rect=(0, 0, 1, 0.96))
    #    fig_anti.tight_layout(rect=(0, 0, 1, 0.96))

# Convert lists-of-arrays to 2D arrays
for group in ungrouped + grouped:
    branch = smooth_store[repr(group)]
    for k, v in branch.iteritems():
        # Transpose so that energy is first index (low->high) and coszen is
        # second index
        branch[k] = np.array(v).T

# Similar process, but slice in E (using already-existing binning + smoothed
# samples therefrom) and smooth-spline CZ dependence
for group in ungrouped + grouped:
    branch = smooth_store[repr(group)]
    hist = branch['hist']
    hist_err = branch['hist_err']
    smooth_cz_slices = branch['smooth_cz_slices']
    smooth = []
    for e_slice_n in xrange(smooth_cz_slices.shape[0]):
        e_slice = smooth_cz_slices[e_slice_n,:]
        # TODO: propagate errors!
        spline = spline_fit[rep_flavint] = splrep(
            czbin_midpoints, e_slice, #w=1./np.array(s_aeff_err),
            k=3, s=50
        )

        # Sample the spline at the bin midpoints
        smoothed_aeff = splev(czbin_midpoints, spline)

        smooth.append(smoothed_aeff)

    smooth = np.array(smooth)
    branch['smooth'] = smooth

## Back to CZ slices and E-dep splining especially to clean up empty bins
#for group in ungrouped + grouped:
#    branch = smooth_store[repr(group)]
#    hist = branch['hist']
#    hist_err = branch['hist_err']
#    smooth = branch['smooth']
#    smooth = []
#    for cz_slice_n in xrange(smooth.shape[1]):
#        cz_slice = smooth[:,cz_slice_n]
#        # TODO: propagate errors!
#        spline = splrep(
#            ebin_midpoints, cz_slice, #w=1./np.array(s_aeff_err),
#            k=3, s=10
#        )
#
#        # Sample the spline at the bin midpoints
#        smoothed_aeff = splev(ebin_midpoints, spline)
#
#        smooth.append(smoothed_aeff)
#
#    smooth = np.array(smooth).T
#    branch['smooth'] = smooth

 
# Derive output filename
outfname = (
    'aeff_slice_smooth__%s_%s__runs_%s__proc_%s.hdf5' % (
        events.metadata['detector'],
        events.metadata['geom'],
        utils.list2hrlist(events.metadata['runs']),
        events.metadata['proc_ver']
    )
)
outfpath = os.path.join(outdir, outfname)
logging.info('Saving Aeff slice smooth info to file "%s"' % outfpath)
hdf.to_hdf(smooth_store, outfpath)

if make_plots:
    X, Y = np.meshgrid(smooth_store['czbins'],
                       np.log10(smooth_store['ebins']))
    for group in ungrouped + grouped:
        fig, axgrp = plt.subplots(1, 4, figsize=(20,5))
        fig.suptitle('$'+group.tex()+'$', fontsize=12)
        axiter = iter(axgrp.flatten())

        branch = smooth_store[repr(group)]
        hist = branch['hist']
        hist_err = branch['hist_err']
        smooth_cz_slices = branch['smooth_cz_slices']
        smooth = branch['smooth']

        hist = np.ma.masked_invalid(hist)
        hist_err = np.ma.masked_invalid(hist_err)
        smooth_cz_slices = np.ma.masked_invalid(smooth_cz_slices)
        smooth = np.ma.masked_invalid(smooth)

        vmin, vmax = np.clip((np.min(np.log10(hist[hist != 0])),
                              np.max(np.log10(hist))),
                             a_min=-8, a_max=-3)

        vmin, vmax = None, None

        log_h = np.log10(hist)
        log_s0 = np.log10(smooth_cz_slices)
        log_s1 = np.log10(smooth)
        fract_err = (smooth-hist)/hist

        vmin = np.inf
        vmax = -np.inf
        for p in [log_h, log_s0, log_s1]:
            vmin = np.min([vmin, np.min(p)])
            vmax = np.max([vmax, np.max(p)])

        plt.sca(axiter.next())
        plt.pcolormesh(X, Y, log_h,
                       vmin=vmin, vmax=vmax, cmap=mpl.cm.Paired) #, **kwargs)
        plt.colorbar()

        plt.sca(axiter.next())
        plt.pcolormesh(X, Y, log_s0,
                       vmin=vmin, vmax=vmax, cmap=mpl.cm.Paired) #, **kwargs)
        plt.colorbar()

        plt.sca(axiter.next())
        plt.pcolormesh(X, Y, np.log10(smooth),
                       vmin=vmin, vmax=vmax, cmap=mpl.cm.Paired) #, **kwargs)
        plt.colorbar()

        #plt.sca(axiter.next())
        #plt.pcolormesh(X, Y, np.log10(smooth),
        #               vmin=vmin, vmax=vmax, cmap=mpl.cm.Paired) #, **kwargs)
        #plt.colorbar()

        plt.sca(axiter.next())
        maxdev = np.max(np.abs(fract_err))
        #vmin, vmax = -maxdev, maxdev
        vmin, vmax = -1.0, 1.0
        plt.pcolormesh(X, Y, fract_err,
                       vmin=vmin, vmax=vmax, cmap=mpl.cm.coolwarm) #, **kwargs)
        plt.colorbar()

        print group, 'fractional error compared with histogram'
        print '  mean:', np.mean(fract_err)
        print '   RMS:', np.sqrt(np.mean(fract_err*fract_err))
        print '   max:', np.max(fract_err)
        print '   min:', np.min(fract_err)


#    if make_plots:
#        fig_part.tight_layout(rect=(0,0,1,0.96))
#        fig_anti.tight_layout(rect=(0,0,1,0.96))
#        basefname = (
#            'aeff_energy_smooth__%s_%s__runs_%s__proc_%s__'
#            % (events.metadata['detector'], events.metadata['geom'],
#               utils.list2hrlist(events.metadata['runs']),
#               events.metadata['proc_ver'])
#        )
#        fig_part.savefig(os.path.join(outdir, basefname + 'particles.pdf'))
#        fig_part.savefig(os.path.join(outdir, basefname + 'particles.png'))
#        fig_anti.savefig(os.path.join(outdir, basefname + 'antiparticles.pdf'))
#        fig_anti.savefig(os.path.join(outdir, basefname + 'antiparticles.png'))


