#!/usr/bin/env python
#
# Allow PISA to be distributed via distutils, i.e the user can just
#
#   pip install git+http://git@github.com/sboeser/pisa/master
#
# author: Sebastian Boeser
#         sboeser@physik.uni-bonn.de

from distutils.core import setup, Extension
from distutils.command import install_headers

setup(
  name='pisa',
  version='0.1',
  description='PINGU Simulation and Analysis',
  author='Sebastian Boeser',
  author_email='sboeser@physik.uni-bonn.de',
  url='http://github.com/sboeser/pisa',
  packages=['pisa','pisa.flux','pisa.oscillations','pisa.oscillations.prob3'],
  scripts=['examples/default_chain.sh',
           'pisa/flux/Flux.py',
           'pisa/oscillations/Oscillations.py',
           'pisa/trigger/EventRate.py',
           'pisa/reco/Reco.py',
           'pisa/pid/PID.py',
           ],
  ext_package='pisa.oscillations.prob3',
  ext_modules=[Extension('_BargerPropagator',
                   ['pisa/oscillations/prob3/BargerPropagator.i',
                    'pisa/oscillations/prob3/BargerPropagator.cc',
                    'pisa/oscillations/prob3/EarthDensity.cc',
                    'pisa/oscillations/prob3/mosc.c',
                    'pisa/oscillations/prob3/mosc3.c'],
                    swig_opts=['-c++'])]
)
