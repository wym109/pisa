#!/usr/bin/env python
#
# Allow PISA to be distributed via distutils, i.e the user can just
#
# TODO: does the following work with requirements.txt file?
#
#   pip install git+https://github.com/sboeser/pisa#egg=pisa
#
# author: Sebastian Boeser
#         sboeser@physik.uni-bonn.de


from distutils.command.build import build as _build
from distutils.core import setup, Extension
import os
import shutil
import subprocess
import sys
import tempfile

from Cython.Build import cythonize
import numpy


def setup_cc():
    if 'CC' not in os.environ or os.environ['CC'].strip() == '':
        os.environ['CC'] = 'cc'
    print 'using compiler %s' %os.environ['CC']


def has_cuda():
    # pycuda is present if it can be imported
    try:
        import pycuda.driver as cuda
    except:
        CUDA = False
        sys.stderr.write(
            'Could not import pycuda; installing without GPU support.\n'
        )
    else:
        CUDA = True

    return CUDA


def has_openmp():
    # OpenMP is present if a test program can compile with -fopenmp flag
    # (e.g. Apple's compiler apparently doesn't support OpenMP, but gcc does)
    # nathan12343's solution: http://stackoverflow.com/questions/16549893
    OPENMP = False

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
        OPENMP = returncode >= 0
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


if __name__ == '__main__':
    setup_cc()
    CUDA = has_cuda()
    OPENMP = has_openmp()

    if not OPENMP:
        sys.stderr.write(
            'Could not compile test program with OpenMP; installing without OpenMP'
            ' multithreading support.\n'
        )

    # Collect (build-able) external modules
    ext_modules = []

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
        extra_compile_args=['-Wall', '-O3', '-fPIC'],
        swig_opts=['-c++'],
    )
    ext_modules.append(prob3cpu_module)

    # Obtain the numpy include directory. This logic works across numpy versions.
    try:
        numpy_include = numpy.get_include()
    except AttributeError:
        numpy_include = numpy.get_numpy_include()

    if CUDA:
        prob3gpu_module = Extension(
            name='pisa.stages.osc.grid_propagator._GridPropagator',
            sources=[
                'pisa/stages/osc/grid_propagator/GridPropagator.cpp',
                'pisa/stages/osc/prob3/EarthDensity.cc',
                'pisa/stages/osc/grid_propagator/GridPropagator.i'
            ],
            include_dirs=[
                numpy_include,
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
        version='3.0',
        description='PINGU Simulation and Analysis',
        author='The IceCube/PINGU Collaboration',
        author_email='sboeser@uni-mainz.de',
        url='http://github.com/WIPACrepo/pisa',
        cmdclass = {'build': build},
        packages=[
            'pisa',
            'pisa.core',
            'pisa.utils',
            'pisa.stages',
            'pisa.stages.osc',
            'pisa.stages.osc.prob3',
            'pisa.stages.osc.nuCraft',
            'pisa.stages.osc.grid_propagator',
            'pisa.stages.aeff',
            'pisa.stages.reco',
            'pisa.stages.pid',
            'pisa.utils',
            'pisa.resources'
        ],
        scripts=[
            'pisa/core/analysis.py',
            'pisa/core/distribution_maker.py',
            'pisa/core/stage.py',
        ],
        ext_modules=cythonize(ext_modules),
        package_data={
            'pisa.resources': [
                'logging.json',
                'aeff/*.json',
                'reco/*.json',
                'pid/*.json',
                'flux/*.d',
                'settings/grid_settings/*.json',
                'settings/minimizer_settings/*.json',
                'settings/pipeline_settings/*.json',
                'osc/*.hdf5',
                'osc/*.dat',
                'events/*.hdf5'
            ]
        }
    )
