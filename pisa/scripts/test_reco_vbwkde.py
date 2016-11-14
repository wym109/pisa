#! /usr/bin/env python
# author: J.L. Lanfranchi
# date:   October 8, 2016
"""
Compares reco vbwkde vs. hist.
"""

from argparse import ArgumentParser

from pisa.core.pipeline import Pipeline
from pisa.utils.fileio import mkdir
from pisa.utils.log import logging, set_verbosity
from pisa.utils.tests import plot_cmp


if __name__ == '__main__':
    parser = ArgumentParser(
        description='''Compare reco.vbwkde against reco.hist.'''
    )
    parser.add_argument('--outdir', metavar='DIR', type=str, default=None,
                        required=False,
                        help='''Store all output plots to this directory. If
                        they don't exist, the script will make them, including
                        all subdirectories. If none is supplied no plots will
                        be saved.''')
    parser.add_argument('-v', action='count', default=None,
                        help='set verbosity level')
    args = parser.parse_args()
    set_verbosity(args.v)

    if args.outdir is not None:
        mkdir(args.outdir)

    hist_cfg = 'tests/settings/vbwkde_test_reco.hist.cfg'
    vbwkde_cfg = 'tests/settings/vbwkde_test_reco.vbwkde.cfg'

    hist_pipeline = Pipeline(
        config=hist_cfg
    )
    vbwkde_pipeline = Pipeline(
        config=vbwkde_cfg
    )
    hist_maps = hist_pipeline.get_outputs()
    vbwkde_maps = vbwkde_pipeline.get_outputs()
    assert vbwkde_maps.names == hist_maps.names
    for map_name in vbwkde_maps.names:
        vmap = vbwkde_maps[map_name]
        hmap = hist_maps[map_name]
        comparisons = vmap.compare(hmap)
        for k in ['max_diff_ratio', 'max_diff', 'nanmatch', 'infmatch']:
            print '%s: %s = %s' %(map_name, k, comparisons[k])

        if args.outdir is not None:
            plot_cmp(new=vmap, ref=hmap, new_label='reco.vbwkde',
                     ref_label='reco.hist', plot_label=vmap.tex,
                     file_label=vmap.name, outdir=args.outdir,
                     ftype='png')
