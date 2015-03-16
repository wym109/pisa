#!/usr/bin/env python
#
# Allow PISA to be distributed via distutils, i.e the user can just
#
#   pip install git+https://github.com/sboeser/pisa#egg=pisa
#
# author: Sebastian Boeser
#         sboeser@physik.uni-bonn.de

from distutils.core import setup, Extension
from distutils.command.build import build as _build

#Define custom build order, so that the python interface module
#created by SWIG is staged in build_py.
class build(_build):
    # different order: build_ext *before* build_py
    sub_commands = [('build_ext',     _build.has_ext_modules),
                    ('build_py',      _build.has_pure_modules),
                    ('build_clib',    _build.has_c_libraries),
                    ('build_scripts', _build.has_scripts),
                   ]


setup(
  name='pisa',
  version='2.0',
  description='PINGU Simulation and Analysis',
  author='Sebastian Boeser',
  author_email='sboeser@uni-mainz.de',
  url='http://github.com/sboeser/pisa',
  cmdclass = {'build': build },
  packages=['pisa',
            'pisa.flux',
            'pisa.oscillations',
            'pisa.oscillations.prob3',
            'pisa.oscillations.nuCraft',
            'pisa.aeff',
            'pisa.reco',
            'pisa.pid',
            'pisa.analysis',
            'pisa.analysis.llr',
            'pisa.analysis.scan',
            'pisa.analysis.fisher',
            'pisa.utils',
            'pisa.resources'],
  scripts=['examples/default_chain.sh',
           'examples/prob3_standalone_example.py',
           'examples/test_asymmetry.py',
           'pisa/flux/Flux.py',
           'pisa/oscillations/Oscillation.py',
           'pisa/oscillations/OscillationProbs.py',
           'pisa/aeff/Aeff.py',
           'pisa/reco/Reco.py',
           'pisa/pid/PID.py',
           'pisa/analysis/TemplateMaker.py',
           'pisa/analysis/scan/ScanAnalysis.py',
           'pisa/analysis/fisher/FisherAnalysis.py',
           'pisa/analysis/llr/LLROptimizerAnalysis.py'
           ],
  ext_package='pisa.oscillations.prob3',
  ext_modules=[Extension('_BargerPropagator',
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
