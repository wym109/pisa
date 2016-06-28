from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np

from pisa import ureg, Q_
from pisa.core.distribution_maker import DistributionMaker
from pisa.core.pipeline import Pipeline
from pisa.utils.fileio import from_file, to_file
from pisa.utils.log import set_verbosity
from pisa.utils.parse_config import parse_config
from pisa.utils.plotter import plotter
from pisa.core.map import MapSet

parser = ArgumentParser()
parser.add_argument('-t', '--template-settings', type=str,
                    metavar='configfile', required=True,
                    help='settings for the generation of templates')
parser.add_argument('-v', action='count', default=None,
                    help='set verbosity level')
args = parser.parse_args()

set_verbosity(args.v)

template_maker_settings = from_file(args.template_settings)
template_maker_configurator = parse_config(template_maker_settings)
template_maker = Pipeline(template_maker_configurator)

template = template_maker.get_outputs()

nutau_cc_cscd = template.pop('nutau_cc_cscd') + template.pop('nutaubar_cc_cscd')
nutau_cc_trck = template.pop('nutau_cc_trck') + template.pop('nutaubar_cc_trck')
nutau_cc_all = nutau_cc_trck + nutau_cc_cscd

cscd = sum([map for map in template if map.name.endswith('cscd')])
trck = sum([map for map in template if map.name.endswith('trck')])
all = cscd + trck

m = MapSet((nutau_cc_cscd/cscd.sqrt(), nutau_cc_trck/trck.sqrt(),
            nutau_cc_all/all.sqrt()))
m[0].tex = 'cascades'
m[1].tex = 'tracks'
m[2].tex = 'all'

my_plotter = plotter(stamp='nutau test', outdir='.', fmt='pdf', log=False,
        label=r'$s/\sqrt{b}$')
my_plotter.plot_2d_array(m, fname='nutau_test',cmap='OrRd')
