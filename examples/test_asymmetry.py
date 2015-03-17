#! /usr/bin/env python
#
# Quick unit test to make sure if you change something, you get a
# reasonable hierarchy asymmetry in the final result. Only input this
# script takes is the template settings file, and produces a plot of
# the hierarchy asymmetry (IMH - NMH)/sqrt(NMH) in each bin.

import numpy as np
from matplotlib import pyplot as plt

from pisa.analysis.TemplateMaker import TemplateMaker
from pisa.utils.params import get_values, select_hierarchy
from pisa.utils.log import set_verbosity,logging
from pisa.utils.jsons import from_json
from pisa.utils.plot import show_map

set_verbosity(0)
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
parser = ArgumentParser(description='''Quick check if all components are working reasonably well, by
making the final level hierarchy asymmetry plots from the input
settings file. ''')
parser.add_argument('template_settings',metavar='JSON',
                    help='Settings file to use for template generation')
parser.add_argument('--title',metavar="str",default='',
                    help="Title of the geometry or test in plots")
parser.add_argument('--save',action='store_true',default=False,
                    help="Save plots in cwd")
parser.add_argument('-v', '--verbose', action='count', default=0,
                    help='set verbosity level')
args = parser.parse_args()
set_verbosity(args.verbose)

template_settings = from_json(args.template_settings)

template_maker = TemplateMaker(get_values(template_settings['params']),
                               **template_settings['binning'])

# Make nmh template:
nmh_params = select_hierarchy(template_settings['params'],
                              normal_hierarchy=True)
imh_params = select_hierarchy(template_settings['params'],
                              normal_hierarchy=False)
nmh = template_maker.get_template(get_values(nmh_params))
imh = template_maker.get_template(get_values(imh_params))

print "keys: ",nmh.keys()
h_asym = {chan: {'map': (imh[chan]['map'] - nmh[chan]['map'])/np.sqrt(nmh[chan]['map']),
                 'ebins':nmh[chan]['ebins'],
                 'czbins': nmh[chan]['czbins'] }
          for chan in ['trck','cscd']}

print "  Total trck events (NMH): ",np.sum(nmh['trck']['map'])
print "  Total trck events (IMH): ",np.sum(imh['trck']['map'])
print "  Total cscd events (NMH): ",np.sum(nmh['cscd']['map'])
print "  Total cscd events (IMH): ",np.sum(imh['cscd']['map'])


for chan in ['trck','cscd']:
#chan = 'trck'
    plt.figure(figsize=(14,5))

    plt.subplot(1,3,1)
    show_map(nmh[chan])
    plt.title(args.title+' NMH, '+chan,fontsize='large')

    plt.subplot(1,3,2)
    show_map(imh[chan])
    plt.title(args.title+' IMH, '+chan,fontsize='large')

    plt.subplot(1,3,3)
    sigma = np.sqrt(np.sum(h_asym[chan]['map']**2))
    show_map(h_asym['trck'],cmap='RdBu_r')
    plt.title(args.title+' '+chan+r' asymmetry, $\sigma$ = %f'%sigma,
              fontsize='large')

    if args.save:
        print "Saving %s chan..."%chan
        plt.savefig(args.title+'_'+chan+'_asym.png',dpi=150)
plt.show()
