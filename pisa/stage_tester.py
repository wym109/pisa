#! /usr/bin/env python
# author: J.L. Lanfranchi
# date:   March 20, 2016

"""
test single stages
"""

import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from pisa.utils.log import logging, set_verbosity
from pisa.utils.fileio import from_file, to_file
from pisa.utils.parse_cfg import parse_cfg
import pisa.stage
import importlib
from copy import deepcopy

parser = ArgumentParser(
    description='''Test a single stage''',
    formatter_class=ArgumentDefaultsHelpFormatter
)
parser.add_argument('-t', '--template_settings', type=str,
                    metavar='configfile', required=True,
                    help='''settings for the template generation''')
parser.add_argument('-s', '--stage', type=str,
                    choices=['Flux','Osc','Aeff','Reco','PID'],
                    required=True, help='''stage to be tested''')
hselect = parser.add_mutually_exclusive_group(required=False)
parser.add_argument('-o', '--outfile', dest='outfile', metavar='FILE',
                    type=str, action='store', default="out.json",
                    help='file to store the output')
args = parser.parse_args()

# Load all the settings
config = from_file(args.template_settings)
config = parse_cfg(config) 

service = config[args.stage.lower()]['service']

# factory
# import stage service
module = importlib.import_module('pisa.%s.%s'%(args.stage.lower(), service))
# get class
cls = getattr(module,args.stage)
# instanciate object
stage = cls(**config[args.stage.lower()])
if isinstance(stage, pisa.stage.NoInputStage):
    output_map_set = stage.get_output_map_set()
elif isinstance(stage, pisa.stage.InputStage):
    output_map_set = stage.get_output_map_set(input_map_set)
