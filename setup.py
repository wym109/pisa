#!/usr/bin/env python
#
# Allow PISA to be distributed via distutils, i.e the user can just
#
#   pip install git+https://github.com/sboeser/pisa#egg=pisa
#
# author: Sebastian Boeser
#         sboeser@physik.uni-bonn.de

from distutils.core import setup, Extension

setup(
  name='pisa',
  version='2.0',
  description='PINGU Simulation and Analysis',
  author='Sebastian Boeser',
  author_email='sboeser@physik.uni-bonn.de',
  url='http://github.com/sboeser/pisa',
  packages=['pisa',
            'pisa.flux',
            'pisa.oscillations',
            'pisa.oscillations.prob3',
            'pisa.oscillations.nuCraft',
            'pisa.aeff',
            'pisa.reco',
            'pisa.pid',
            'pisa.analysis',
            'pisa.analysis.LLR',
            'pisa.utils',
            'pisa.resources'],
  scripts=['examples/default_chain.sh',
           'pisa/flux/Flux.py',
           'pisa/oscillations/Oscillation.py',
           'pisa/oscillations/OscillationProbs.py',
           'pisa/aeff/Aeff.py',
           'pisa/reco/Reco.py',
           'pisa/pid/PID.py',
           'pisa/analysis/TemplateMaker.py',
           'pisa/analysis/LLR/LLROptimizerAnalysis.py'
           ],
  ext_package='pisa.oscillations.prob3',
  ext_modules=[Extension('BargerPropagator',
                   ['pisa/oscillations/prob3/BargerPropagator.i',
                    'pisa/oscillations/prob3/BargerPropagator.cc',
                    'pisa/oscillations/prob3/EarthDensity.cc',
                    'pisa/oscillations/prob3/mosc.c',
                    'pisa/oscillations/prob3/mosc3.c'],
                    swig_opts=['-c++'])],
  package_data={'pisa.resources': ['logging.json',
                                   'aeff/*.json',
                                   'aeff/V15/cuts_V3/*.dat',
                                   'pid/*.json',
                                   'flux/*.d',
                                   'settings/*.json',
                                   'oscillations/*.hdf5',
                                   'oscillations/*.dat',
                                   'events/*.hdf5']}
)
