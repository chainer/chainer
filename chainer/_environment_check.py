from __future__ import absolute_import
import os
import sys
import warnings

import numpy.distutils.system_info
import pkg_resources

import chainer


def _check_python_350():
    if sys.version_info[:3] == (3, 5, 0):
        if not int(os.getenv('CHAINER_PYTHON_350_FORCE', '0')):
            msg = """
    Chainer does not work with Python 3.5.0.

    We strongly recommend to use another version of Python.
    If you want to use Chainer with Python 3.5.0 at your own risk,
    set 1 to CHAINER_PYTHON_350_FORCE environment variable."""

            raise Exception(msg)


def _check_python_2():
    if sys.version_info[:1] == (2,):
        warnings.warn('''
--------------------------------------------------------------------------------
Chainer is going to stop supporting Python 2 in v7.x releases.

Future releases of Chainer v7.x will not run on Python 2.
If you need to continue using Python 2, consider using Chainer v6.x, which
will be the last version that runs on Python 2.
--------------------------------------------------------------------------------
''')  # NOQA


def _check_osx_numpy_backend():
    if sys.platform != 'darwin':
        return

    blas_opt_info = numpy.distutils.system_info.get_info('blas_opt')
    if blas_opt_info:
        extra_link_args = blas_opt_info.get('extra_link_args')
        if extra_link_args and '-Wl,Accelerate' in extra_link_args:
            warnings.warn('''\
Accelerate has been detected as a NumPy backend library.
vecLib, which is a part of Accelerate, is known not to work correctly with Chainer.
We recommend using other BLAS libraries such as OpenBLAS.
For details of the issue, please see
https://docs.chainer.org/en/stable/tips.html#mnist-example-does-not-converge-in-cpu-mode-on-mac-os-x.

Please be aware that Mac OS X is not an officially supported OS.
''')  # NOQA


def _check_optional_dependencies():
    for dep in chainer._version._optional_dependencies:
        name = dep['name']
        pkgs = dep['packages']
        spec = dep['specifier']
        help = dep['help']
        installed = False
        for pkg in pkgs:
            found = False
            requirement = '{}{}'.format(pkg, spec)
            try:
                pkg_resources.require(requirement)
                found = True
            except pkg_resources.DistributionNotFound:
                continue
            except pkg_resources.VersionConflict:
                msg = '''
--------------------------------------------------------------------------------
{name} ({pkg}) version {version} may not be compatible with this version of Chainer.
Please consider installing the supported version by running:
  $ pip install '{requirement}'

See the following page for more details:
  {help}
--------------------------------------------------------------------------------
'''  # NOQA
                warnings.warn(msg.format(
                    name=name, pkg=pkg,
                    version=pkg_resources.get_distribution(pkg).version,
                    requirement=requirement, help=help))
                found = True
            except Exception:
                warnings.warn(
                    'Failed to check requirement: {}'.format(requirement))
                break

            if found:
                if installed:
                    warnings.warn('''
--------------------------------------------------------------------------------
Multiple installations of {name} package has been detected.
You should select only one package from from {pkgs}.
Follow these steps to resolve this issue:
  1. `pip list` to list {name} packages installed
  2. `pip uninstall <package name>` to uninstall all {name} packages
  3. `pip install <package name>` to install the proper one
--------------------------------------------------------------------------------
'''.format(name=name, pkgs=pkgs))
                installed = True


def check():
    _check_python_2()
    _check_python_350()
    _check_osx_numpy_backend()
    _check_optional_dependencies()
