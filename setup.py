#!/usr/bin/env python
#
# Allow PISA to be distributed via distutils, i.e the user can just
#
# TODO: does the following work with requirements.txt file?
# --> some version of cython and numpy must already be installed, since they
#     are imported here
#
#   pip install git+https://github.com/sboeser/pisa#egg=pisa
#
# author: Sebastian Boeser
#         sboeser@physik.uni-bonn.de


from distutils.command.build import build as _build
from distutils.command.build_ext import build_ext as _build_ext
from distutils.core import setup, Extension
import os
import shutil
import subprocess
import sys
import tempfile


def cythonize(module):
    from Cython.Build import cythonize as cythonize_
    return cythonize_(module)


#def get_numpy_include():
#    """Obtain the numpy include directory. This logic works across numpy
#    versions."""
#    import numpy
#    try:
#        numpy_include = numpy.get_include()
#    except AttributeError:
#        numpy_include = numpy.get_numpy_include()
#    return numpy_include


def setup_cc():
    if 'CC' not in os.environ or os.environ['CC'].strip() == '':
        os.environ['CC'] = 'cc'


def has_cuda():
    # pycuda is present if it can be imported
    try:
        import pycuda.driver as cuda
    except:
        CUDA = False
    else:
        CUDA = True
    return CUDA


def has_openmp():
    # OpenMP is present if a test program can compile with -fopenmp flag
    # (e.g. Apple's compiler apparently doesn't support OpenMP, but gcc does)
    # nathan12343's solution: http://stackoverflow.com/questions/16549893
    OPENMP = False

    setup_cc()

    # see http://openmp.org/wp/openmp-compilers/
    omp_test = \
    r"""
    #include <omp.h>
    #include <stdio.h>
    int main() {
    #pragma omp parallel
    printf("Hello from thread %d, nthreads %d\n", omp_get_thread_num(), omp_get_num_threads());
    }
    """
    tmpfname = r'test.c'
    tmpdir = tempfile.mkdtemp()
    curdir = os.getcwd()
    os.chdir(tmpdir)
    cc = os.environ['CC']
    try:
        with open(tmpfname, 'w', 0) as file:
            file.write(omp_test)
        with open(os.devnull, 'w') as fnull:
            returncode = subprocess.call([cc, '-fopenmp', tmpfname],
                                         stdout=fnull, stderr=fnull)
        # Successful build (possibly with warnings) means we can use OpenMP
        OPENMP = (returncode == 0)
    finally:
        # Restore directory location and clean up
        os.chdir(curdir)
        shutil.rmtree(tmpdir)
    return OPENMP


class build(_build):
    """Define custom build order, so that the python interface module created
    by SWIG is staged in build_py.

    """
    # different order: build_ext *before* build_py
    sub_commands = [
        ('build_ext',     _build.has_ext_modules),
        ('build_py',      _build.has_pure_modules),
        ('build_clib',    _build.has_c_libraries),
        ('build_scripts', _build.has_scripts)
    ]


class build_ext(_build_ext):
    """Replace default build_ext to allow for numpy install before setup.py
    needs it to include its dir.

    Code copied from coldfix / http://stackoverflow.com/a/21621689

    """
    def finalize_options(self):
        _build_ext.finalize_options(self)
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


# python_requires
# setup_requires
# install_requires
# package_data
# include_package_data
# zip_safe
# eager_resources

if __name__ == '__main__':
    setup_cc()
    sys.stdout.write('Using compiler %s\n' %os.environ['CC'])

    CUDA = has_cuda()
    if not CUDA:
        sys.stderr.write('WARNING: Could not import pycuda; installing PISA'
                         ' without CUDA (GPU) support.\n')

    OPENMP = has_openmp()
    if not OPENMP:
        sys.stderr.write(
            'WARNING: Could not compile test program with -fopenmp;'
            ' installing PISA without OpenMP support.\n'
        )

    # Collect (build-able) external modules and package_data
    ext_modules = []
    package_data = {}

    # Prob3 oscillation code (pure C++, no CUDA)
    prob3cpu_module = Extension(
        name='pisa.stages.osc.prob3._BargerPropagator',
        sources=[
            'pisa/stages/osc/prob3/BargerPropagator.i',
            'pisa/stages/osc/prob3/BargerPropagator.cc',
            'pisa/stages/osc/prob3/EarthDensity.cc',
            'pisa/stages/osc/prob3/mosc.c',
            'pisa/stages/osc/prob3/mosc3.c'
        ],
        include_dirs=[
            #get_numpy_include(),
            'pisa/stages/osc/prob3/'
        ],
        extra_compile_args=['-Wall', '-O3', '-fPIC'],
        swig_opts=['-c++'],
    )
    ext_modules.append(prob3cpu_module)

    package_data['pisa.resources'] = [
        'aeff/*.json',
        'cross_sections/cross_sections.json',
        'events/*.hdf5',
        'events/*.json',
        'events/deepcore_ic86/MSU/1XXX/Joined/*.hdf5',
        'events/deepcore_ic86/MSU/1XXX/UnJoined/*.hdf5',
        'events/deepcore_ic86/MSU/1XXXX/Joined/*.hdf5',
        'events/deepcore_ic86/MSU/1XXXX/UnJoined/*.hdf5',
        'events/deepcore_ic86/MSU/icc/*.hdf5',
        'events/pingu_v36/*.hdf5',
        'events/pingu_v39/*.hdf5',
        'flux/*.d',
        'logging.json',
        'osc/*.hdf5',
        'osc/*.dat',
        'pid/*.json',
        'priors/*.json',
        'reco/*.json',
        'settings/discrete_sys_settings/*.ini',
        'settings/minimizer_settings/*.json',
        'settings/pipeline_settings/*.ini',
        'sys/*.json',
        'tests/data/aeff/*.json',
        'tests/data/flux/*.json',
        'tests/data/full/*.json',
        'tests/data/osc/*.json',
        'tests/data/pid/*.json',
        'tests/data/reco/*.json',
        'tests/data/xsec/*.root',
        'tests/data/oscfit/*.json',
        'tests/settings/*.ini'
    ]

    if CUDA:
        prob3gpu_module = Extension(
            name='pisa.stages.osc.grid_propagator._GridPropagator',
            sources=[
                'pisa/stages/osc/grid_propagator/GridPropagator.cpp',
                'pisa/stages/osc/prob3/EarthDensity.cc',
                'pisa/stages/osc/grid_propagator/GridPropagator.i'
            ],
            include_dirs=[
                get_numpy_include(),
                'pisa/stages/osc/prob3/'
            ],
            extra_compile_args=[
                '-xc++', '-lstdc++', '-shared-libgcc', '-c', '-Wall', '-O3',
                '-fPIC'
            ],
            swig_opts=[
                '-v', '-c++'
            ]
        )
        ext_modules.append(prob3gpu_module)
        package_data['pisa.stages.osc.grid_propagator'] = [
            'mosc3.cu',
            'mosc.cu',
            'mosc3.h',
            'mosc.h',
            'constants.h',
            'numpy.i',
            'GridPropagator.h',
            'OscUtils.h',
            'utils.h'
        ]

    if OPENMP:
        gaussians_module = Extension(
            'pisa.utils.gaussians',
            ['pisa/utils/gaussians.pyx'],
            libraries=['m'],
            extra_compile_args=[
                '-fopenmp', '-O2'
            ],
            extra_link_args=['-fopenmp']
        )
    else:
        gaussians_module = Extension(
            'pisa.utils.gaussians',
            ['pisa/utils/gaussians.pyx'],
            extra_compile_args=[
                '-O2'
            ],
            libraries=['m']
        )
    ext_modules.append(gaussians_module)

    # Now do the actual work
    setup(
        name='pisa',
        version='3.0.0',
        description='PINGU Simulation and Analysis',
        author='The IceCube/PINGU Collaboration',
        author_email='sboeser@uni-mainz.de',
        url='http://github.com/WIPACrepo/pisa',
        cmdclass = {
            'build': build,
            'build_ext': build_ext
        },
        setup_requires=[
            'pip>=1.8',
            'numpy>=1.11.0'
            'cython',
        ],
        install_requires=[
            'scipy>=0.17.0',
            'h5py',
            'matplotlib',
            'pint',
            'simplejson>=3.2.0',
            'tables',
            'uncertainties'
        ],
        extras_require = {
            'cuda':  ['pycuda'],
            'numba':  ['enum34', 'numba'],
            'dev': ['line_profiler', 'sphinx>1.3', 'recommonmark'],
        },
        packages=[
            'pisa',
            'pisa.analysis',
            'pisa.core',
            'pisa.resources',
            'pisa.stages',
            'pisa.stages.aeff',
            'pisa.stages.flux',
            'pisa.stages.osc',
            'pisa.stages.osc.grid_propagator',
            'pisa.stages.osc.nuCraft',
            'pisa.stages.osc.prob3',
            'pisa.stages.pid',
            'pisa.stages.reco',
            'pisa.stages.sys',
            'pisa.stages.xsec',
            'pisa.stages.mc',
            'pisa.stages.data',
            'pisa.utils',
            'pisa.resources'
        ],
        scripts=[
            'pisa/analysis/hypo_testing.py',
            'pisa/analysis/profile_llh_analysis.py',
            'pisa/core/distribution_maker.py',
            'pisa/core/pipeline.py',
            'pisa/scripts/make_events_file.py',
            'pisa/scripts/pisa_v2_v3_consistency_tests.py',
            'pisa/scripts/postprocess.py'
        ],
        ext_modules=cythonize(ext_modules),
        package_data=package_data
    )
