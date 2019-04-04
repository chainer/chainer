# This script is based on pybind11's example script. See the original via the
# following URL: https://github.com/pybind/cmake_example/blob/master/setup.py

import distutils
import os
import platform
import re
import subprocess
import sys

import setuptools
from setuptools.command import build_ext


class CMakeExtension(setuptools.Extension):

    def __init__(self, name, build_targets, sourcedir=''):
        setuptools.Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        self.build_targets = build_targets


class CMakeBuild(build_ext.build_ext):

    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError('CMake must be installed to build ChainerX')

        cmake_version = distutils.version.LooseVersion(
            re.search(r'version\s*([\d.]+)', out.decode()).group(1))
        if cmake_version < '3.1.0':
            raise RuntimeError('CMake >= 3.1.0 is required to build ChainerX')

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        # Decide the build type: release/debug
        build_type = os.getenv('CHAINERX_BUILD_TYPE', None)
        if build_type is not None:
            # Use environment variable
            pass
        elif self.debug:
            # Being built with `python setup.py build --debug`
            build_type = 'Debug'
        elif os.getenv('READTHEDOCS', None) == 'True':
            # on ReadTheDocs
            build_type = 'Debug'
        else:
            # default
            build_type = 'Release'

        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            '-DCHAINERX_BUILD_PYTHON=1',
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DPYTHON_EXECUTABLE=' + sys.executable,
            '-DCHAINERX_BUILD_TEST=OFF',
            '-DCMAKE_BUILD_TYPE=' + build_type,
        ]

        build_args = ['--config', build_type]

        if platform.system() == 'Windows':
            cmake_args += [
                '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                    build_type.upper(), extdir)]

            cmake_args += ['-G', 'Visual Studio 15 2017', '-T', 'llvm']

            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            build_args += ['--']
            build_args += ext.build_targets

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''), self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(
            ['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp,
            env=env)
        subprocess.check_call(
            ['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


def config_setup_kwargs(setup_kwargs, build_chainerx):
    if not build_chainerx:
        # `chainerx` package needs to be able to be imported even if ChainerX
        # is unavailable.
        setup_kwargs['packages'] += ['chainerx']
        return

    if sys.version_info < (3, 5):
        raise RuntimeError(
            'ChainerX is only available for Python 3.5 or later.')
    setup_kwargs['packages'] += [
        'chainerx',
        'chainerx._docs',
        'chainerx.creation',
        'chainerx.manipulation',
        'chainerx.math',
        'chainerx.random',
        'chainerx.testing',
    ]
    setup_kwargs['package_data'] = {
        'chainerx': ['py.typed', '*.pyi'],
    }

    setup_kwargs.update(dict(
        cmdclass={'build_ext': CMakeBuild},
        ext_modules=[CMakeExtension(
            name='chainerx._core',
            build_targets=['_core.so'],
            sourcedir='chainerx_cc')],
    ))
