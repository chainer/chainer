# This script is based on pybind11's example script. See the original via the following URL:
# https://github.com/pybind/cmake_example/blob/master/setup.py

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
        super(CMakeExtension, self).__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        self.build_targets = build_targets


class CMakeBuild(build_ext.build_ext):

    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError('CMake must be installed to build xChainer')

        cmake_version = distutils.version.LooseVersion(
            re.search(r'version\s*([\d.]+)', out.decode()).group(1))
        if cmake_version < '3.1.0':
            raise RuntimeError('CMake >= 3.1.0 is required to build xChainer')

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            '-DBUILD_PYTHON=1',
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DPYTHON_EXECUTABLE=' + sys.executable,
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        build_args += ext.build_targets

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


setuptools.setup(
    name='xchainer',
    version='0.0.1',
    author='xChainer authors',
    author_email='beam.web@gmail.com',
    description='',
    long_description='',
    ext_modules=[CMakeExtension('xchainer._core', ['_core.so'])],
    packages=['xchainer'],
    package_dir={'': 'python'},
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False,
)
