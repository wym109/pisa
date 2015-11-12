#! /usr/bin/env python
#
# Generate a PISA-standard events file from HDF5 file(s) which in turn were
# generated from I3 files by the icecube.hdfwriter.I3HDFTableService
#
# author: Justin L. Lanfranchi
#         jll1062+pisa@phys.psu.edu
#
# date:   October 24, 2015
#

import os
import numpy as np

from pisa.utils.log import logging, set_verbosity
from pisa.utils import hdf
import pisa.utils.utils as utils
import pisa.utils.flavInt as FI
import pisa.utils.events as events
import pisa.utils.mcSimRunSettings as MCSRS
import pisa.utils.dataProcParams as DPP
from pisa.resources import resources as RES

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


CMSQ_TO_MSQ = 1.0e-4

keep_fields = (
    'true_energy',
    'true_coszen',
    'reco_energy',
    'reco_coszen',
    'one_weight',
)

output_fields = (
    'true_energy',
    'true_coszen',
    'reco_energy',
    'reco_coszen',
    'weighted_aeff',
)


def makeEventsFile(data_files, outdir, run_settings, proc_settings, cut,
                   join=None, cust_cuts=None, compute_aeff=True):
    '''
        cust_cuts
            dict with a single procSettings cutspec or list of same
    '''
    # Create Events object to store data
    evts = events.Events()
    evts.metadata.update({
        'detector': run_settings.detector,
        'proc_ver': proc_settings.proc_ver,
    })
    cuts = []
    if isinstance(cust_cuts, dict):
        cust_cuts = [cust_cuts]
    if cut is not None:
        evts.metadata['cuts'].append(cut)
        cuts.append(cut)
    if cust_cuts is not None:
        for ccut in cust_cuts:
            evts.metadata['cuts'].append('custom: ' + ccut['pass_if'])
            cuts.append(ccut)

    orig_outdir = outdir
    outdir = utils.expandPath(outdir)
    logging.info('Output dir spec\'d: %s', orig_outdir)
    if outdir != orig_outdir:
        logging.info('Output dir expands to: %s', outdir)
    utils.mkdir(outdir)

    detector_label = str(proc_settings.detector)
    proc_label = 'proc_v' + str(proc_settings.proc_ver)

    # Set "bar" (vs. "_bar") as naming convention (output file uses the latter
    # convention so this will be changed prior to writing the data for that file)
    FI.set_bar_ssep('')

    # What kinds to group together
    if join is None or join == '':
        kind_groupings = []
        groups_label = 'unjoined'
        logging.info('Events in the following groups will be joined together:'
                     ' (none)')
    else:
        kind_groupings = [FI.NuKindGroup(s) for s in join.split(';')]
        evts.metadata['kinds_joined'] = [str(g) for g in kind_groupings]
        groups_label = 'joined_G_' + '_G_'.join([
            g.simpleStr(flavsep='_', flavintsep='_', kindsep='_', addsep='')
            for g in kind_groupings
        ])
        logging.info('Events in the following groups will be joined together: '
                     + '; '.join([str(g) for g in kind_groupings]))
    # Find any kinds not included in the above groupings
    all_kinds = set(FI.ALL_KINDS)
    all_grouped_kinds = set(FI.NuKindGroup(kind_groupings))
    ungrouped_kinds = [FI.NuKindGroup(k) for k in
                       sorted(all_kinds.difference(all_grouped_kinds))]
    kind_groupings.extend(ungrouped_kinds)
    if len(ungrouped_kinds) == 0:
        ungrouped_kinds = ['(none)']
    logging.info('Events of the following kinds will NOT be joined together: '
                 + '; '.join([str(k) for k in ungrouped_kinds]))

    # Enforce that kinds composing groups are mutually exclusive
    for n, kg0 in enumerate(kind_groupings[:-1]):
        for m, kg1 in enumerate(kind_groupings[n+1:]):
            assert len(set(kg0).intersection(set(kg1))) == 0

    kg_names = [str(nkg) for nkg in kind_groupings]

    # Instantiate storage for all intermediate destination fields
    #   keep_data[group number][interaction type][keep field name] = list of data
    keep_data = [
        {it: {f:[] for f in keep_fields} for it in FI.ALL_INT_TYPES}
        for name in kg_names
    ]

    # Instantiate generated-event counts for destination fields; count
    # nc separately from nc because aeff's for cc & nc add, whereas
    # aeffs intra-CC should be weighted-averaged (as for intra-NC)
    ngen = [{it:{} for it in FI.ALL_INT_TYPES} for name in kg_names]

    # Loop through all of the files, retrieving the events, filtering,
    # and recording the number of generated events pertinent to
    # calculating aeff
    filecount = {}
    all_runs = set()
    detector_geom = None
    for userspec_baseflav, fnames in data_files.iteritems():
        userspec_baseflav = FI.NuFlav(userspec_baseflav)
        for fname in fnames:
            # Retrieve data from all nodes specified in the processing
            # settings file
            data = proc_settings.getData(fname, run_settings=run_settings)

            # Check to make sure only one run is present in the data
            runs_in_data = set(data['run'])
            assert len(runs_in_data) == 1, 'Must be just one run present in data'

            run = int(data['run'][0])
            all_runs.add(run)
            if not run in filecount:
                filecount[run] = 0
            filecount[run] += 1
            rs_run = run_settings[run]
          
            # Record geom; check that geom is consistent with other runs
            if detector_geom is None:
                detector_geom = rs_run['geom']
            assert rs_run['geom'] == detector_geom, 'All runs\' geometries must match!'

            # Loop through all kinds (flav/int-type comb) spec'd for run
            for run_kind in rs_run['kinds']:
                flav = run_kind.flav()
                barnobar = run_kind.barNoBar()
                int_type = run_kind.intType()

                # Retrieve this-interaction-type- & this-barnobar-only events
                # that also pass cuts. (note that cut names are strings)
                intonly_cut_data = proc_settings.applyCuts(
                    data,
                    cuts=cuts+[str(int_type), str(barnobar)],
                    return_fields=keep_fields
                )
                
                # Record the generated count and data for this run/flavor for
                # each group to which it's applicable
                for n, kind_group in enumerate(kind_groupings):
                    if not run_kind in kind_group:
                        continue

                    # Instantiate a field for particles and anti-particles,
                    # keyed by the output of the barNoBar() method for each
                    if not run in ngen[n][int_type]:
                        ngen[n][int_type][run] = {
                            FI.NuFlav(12).barNoBar(): 0,
                            FI.NuFlav(-12).barNoBar(): 0,
                        }

                    # Record count only if it hasn't already been recorded
                    if ngen[n][int_type][run][barnobar] == 0:
                        # Note that one_weight includes cc/nc:total fraction, so DO
                        # NOT specify the full kind here, only flav (since
                        # one_weight does NOT take bar/nobar fraction, it must
                        # be included here in the ngen computation)
                        flav_ngen = run_settings.totGen(run=run, barnobar=barnobar)
                        ngen[n][int_type][run][barnobar] = flav_ngen

                    # Append the data. Note that keep_data is:
                    # keep_data[group n][int_type][keep field name] = list
                    [keep_data[n][int_type][f].extend(intonly_cut_data[f])
                     for f in keep_fields]

    # Compute "weighted_aeff" field:
    #
    # Within each int type (CC or NC), ngen should be added together;
    # events recorded of that int type then get their one_weight divided by the
    # total **for that int type only** to obtain the "weighted_aeff" for that
    # event (even if int types are being grouped/joined together).
    #
    # This has the effect that within a group, ...
    #   ... and within an interaction type, effective area is a weighted
    #   average of that of the flavors being combined. E.g. for CC,
    #
    #                  \sum_{run x}\sum_{flav y} (Aeff_{x,y} * ngen_{x,y})
    #       Aeff_CC = ----------------------------------------------------- ,
    #                       \sum_{run x}\sum_{flav y} (ngen_{x,y})
    #
    #   ... and then across interaction types, the results of the above for
    #   each int type need to be summed together, i.e.:
    # 
    #       Aeff_total = Aeff_CC + Aeff_NC
    #
    # Note that each grouping of flavors is calculated with the above math completely
    # independently from other flavor groupings specified.
    #
    # See Justin Lanfranchi's presentation on the PINGU Analysis call,
    # 2015-10-21, for more details.

    fmtfields = (' '*12+'kind_group', 'int type', '     run', 'part/anti',
                 'part/anti count', 'aggregate count')
    fmt_n = [len(f) for f in fmtfields]
    fmt = '  '.join([r'%'+str(n)+r's' for n in fmt_n])
    lines = '  '.join(['-'*n for n in fmt_n])
    logging.info(fmt % fmtfields)
    logging.info(lines)
    for n, kind_group in enumerate(kind_groupings):
        for int_type in FI.ALL_INT_TYPES:
            ngen_it_tot = 0
            for run, run_counts in ngen[n][int_type].iteritems():
                for barnobar, barnobar_counts in run_counts.iteritems():
                    ngen_it_tot += barnobar_counts
                    logging.info(fmt %
                        (kind_group.simpleStr(), int_type, str(run), barnobar,
                         int(barnobar_counts), int(ngen_it_tot)))
            # Convert data to numpy array
            for field in keep_fields:
                keep_data[n][int_type][field] = np.array(keep_data[n][int_type][field])
            
            # Generate weighted_aeff field for this group / int type's data
            keep_data[n][int_type]['weighted_aeff'] = \
                    keep_data[n][int_type]['one_weight'] / ngen_it_tot * CMSQ_TO_MSQ
  
    # Report file count per run
    for run, count in filecount.items():
        logging.info('Files read, run %s: %d' % (run, count))
        ref_num_i3_files = run_settings[run]['num_i3_files']
        if count != ref_num_i3_files:
            logging.warn('Run %d, Number of files read (%d) != number of '
                         'source I3 files (%d), which may indicate an error.' %
                         (run, count, ref_num_i3_files))

    # Generate output data...
    for kind in FI.ALL_KINDS:
        flav = kind.flav()
        int_type = kind.intType()
        for n, kind_group in enumerate(kind_groupings):
            if not kind in kind_group:
                logging.trace('kind %s not in kind_group %s, passing.' %
                              (kind, kind_group))
                continue
            else:
                logging.trace('kind %s **IS** in kind_group %s, storing.' %
                              (kind, kind_group))
            evts.set(kind, {f: keep_data[n][int_type][f] for f in output_fields})

    # Update metadata
    evts.metadata['geom'] = detector_geom
    evts.metadata['runs'] = sorted(all_runs)

    # Generate file name
    run_label = 'runs_' + utils.list2hrlist(all_runs)
    geom_label = '' + detector_geom
    fname = 'events__' + '__'.join([
        detector_label,
        geom_label,
        run_label,
        proc_label,
        groups_label,
    ]) + '.hdf5'

    outfpath = os.path.join(outdir, fname)
    logging.info('Writing events to ' + outfpath)

    # Save data to output file
    evts.save(outfpath)


def main():
    parser = ArgumentParser(
        description= \
    '''Takes the simulated (and reconstructed) HDF5 file(s) (as converted from I3 by
    icecube.hdfwriter.I3HDFTableService) as input and writes out a simplified HDF5
    file for use in the aeff and reco stages of the template maker.''',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--det',
        metavar='DETECTOR',
        type=str,
        required=True,
        help='Detector, e.g. "PINGU" or "DeepCore". This'
             ' is used as the top-most key in run_settings.json and'
             ' proc_settings.json files.'
    )
    parser.add_argument(
        '--proc',
        metavar='PROC_VER',
        type=str,
        required=True,
        help='Processing version applied to simulation; processing versions are'
        ' defined with respect to each geometry version. See proc_settings.json'
        ' file for definitions (or edit that file to add more).'
    )
    
    parser.add_argument(
        '--nue',
        metavar='H5_FILE',
        type=str,
        nargs="+",
        required=True,
        help='nue HDF5 file(s)'
    )
    parser.add_argument(
        '--numu',
        metavar='H5_FILE',
        type=str,
        nargs="+",
        required=True,
        help='numu HDF5 file(s)'
    )
    parser.add_argument(
        '--nutau',
        metavar='H5_FILE',
        type=str,
        nargs="+",
        required=True,
        help='nutau HDF5 file(s)'
    )
    
    parser.add_argument(
        '--outdir',
        metavar='DIR',
        type=str,
        default='$PISA/pisa/resources/events',
        help='directory into which to store resulting HDF5 file'
    )
    
    parser.add_argument(
        '--run-settings',
        metavar='JSON_FILE',
        type=str,
        default='events/mc_sim_run_settings.json',
        help='JSON file with reference run settings'
    )
    
    parser.add_argument(
        '--proc-settings',
        metavar='JSON_FILE',
        type=str,
        default='events/data_proc_params.json',
        help='JSON file with reference processing settings'
    )
   
    # Removed this in favor of forcing standard events groupings to be output
    # all at once, to ensure all files get generated all the time. Also
    # implementing validation for consistent events file usage in PISA for
    # template settings file
    
    #parser.add_argument(
    #    '--join',
    #    const='nuecc,nuebarcc;numucc,numubarcc;nutaucc,nutaubarcc;nuallnc,nuallbarnc',
    #    default='',
    #    action='store',
    #    nargs='?',
    #    type=str,
    #    help= \
    #    '''Optionally join flavors together to increase statistics for Aeff
    #    and/or resolutions (aeff and reco stages, respectively). Specifying the
    #    --join option without an argument joins together: nu_x &
    #    nu_x_bar CC events together (one set for each of x=e, x=mu, and x=tau),
    #    and joins nuall NC & nuallbar NC events tegether. If a string
    #    argument is supplied, this specifies custom groups to join together
    #    instead. The string must be a semicolon-separated list each field of
    #    which itself a comma-separated list of event "kinds" (flavor and
    #    interaction type) to grup together. Any event kind not included in that
    #    string will be found individually, i.e., not joined together with any
    #    other flavors'''
    #)
    
    parser.add_argument(
        '--cut',
        metavar='CUT_NAME',
        type=str,
        default='analysis',
        help='Name of cut to apply. See proc_settings.json for definitions (note'
        ' that these vary by geometry and processing version)'
    )
    
    parser.add_argument(
        '--ccut-pass-if',
        metavar='CRITERIA',
        type=str,
        default='',
        help= \
        '''Custom cut: String containing criteria for passing a cut, using
        field names specified by the --ccut-fields argument. Standard Python-
        and numpy-namespace expressions are allowed as well, since this string
        is passed to 'eval'. E.g.:
        --ccut-fields="z:MCNeutrino/zenith,l6:IC86_Dunkman_L6/value" \
        --ccut-pass-if="(l6 == 1) & (z > pi/2)" '''
    )
    parser.add_argument(
        '--ccut-fields',
        metavar='FIELDS',
        type=str,
        default='',
        help= \
        '''Custom cut: String of comma-separated fields, each containing
        colon-separated (variable name : HDF5 address) tuples. For example,
        specifying:
        --ccut-fields="l5:IC86_Dunkman_L5/value,l6:IC86_Dunkman_L6/value"
        allows for a custom cut to be defined via --ccut-pass-if="(l5 == 1) &
        (l6 == 1)"
    '''
    )
  
    # NOTE:
    # For now I'm removing the aeff option so that all generated events files
    # will be of consistent format. The danger is that people will then use
    # weird joined-events combinations to produce Aeff plots. But since the
    # output HDF5 file is clearly and consistently labeled, I'll leave the
    # responsibility of how to work with the contents of the file up to the
    # user.
    
    #parser.add_argument(
    #    '--aeff',
    #    action='store_true',
    #    help='Add weighted_aeff field, which allows for performing effective area'
    #         ' computations using the resulting file'
    #)
    
    parser.add_argument(
        '-v', '--verbose',
        action='count',
        default=0,
        help='set verbosity level'
    )

    args = parser.parse_args()

    set_verbosity(args.verbose)
    
    det = args.det.lower()
    proc = args.proc.lower()
    run_settings = MCSRS.DetMCSimRunsSettings(
        RES.find_resource(args.run_settings),
        detector=det
    )
    proc_settings = DPP.DataProcParams(
        RES.find_resource(args.proc_settings),
        detector=det,
        proc_ver=proc
    )
   
    logging.info('Using detector %s, processing version %s.' % (det, proc))

    groupings = [
        # No events joined together
        None,
        # CC events unjoined; join nuall NC and nuallbar NC separately (used
        # for generating aeff param service's aeff parameterizations)
        'nuallnc;nuallbarnc',
        # CC events paried by flav--anti-flav; nuallNC+nuallbarNC all joined
        # together; used for reco services (MC and vbwkde)
        'nuecc+nuebarcc;numucc+numubarcc;nutaucc+nutaubarcc;nuallnc+nuallbarnc',
    ]
    ccut = None
    if args.ccut_pass_if:
        ccut = {'pass_if':args.ccut_pass_if, 'fields':args.ccut_fields.split(',')}
    for grouping in groupings:
        makeEventsFile(
            data_files={'nue':args.nue, 'numu':args.numu, 'nutau':args.nutau},
            outdir=args.outdir,
            run_settings=run_settings,
            proc_settings=proc_settings,
            join=grouping,
            cut=args.cut,
            cust_cuts=ccut,
            compute_aeff=True,
        )

if __name__ == "__main__":
    main()
