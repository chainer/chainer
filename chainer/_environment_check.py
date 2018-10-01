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

Also note that Chainer does not officially support Mac OS X.
Please use it at your own risk.
''')  # NOQA


def _check_optional_dependencies():
    for dep in chainer._version._optional_dependencies:
        name = dep['name']
        pkgs = dep['packages']
        version = dep['version']
        help = dep['help']
        installed = False
        for pkg in pkgs:
            found = False
            requirement = '{}{}'.format(pkg, version)
            try:
                pkg_resources.require(requirement)
                found = True
            except pkg_resources.DistributionNotFound:
                continue
            except pkg_resources.VersionConflict:
                dist = pkg_resources.get_distribution(pkg).version
                warnings.warn('''
--------------------------------------------------------------------------------
{name} ({pkg}) version {version} may not be compatible with this version of Chainer.
Please consider installing the supported version by running:
  $ pip install '{requirement}'

See the the following page for more details:
  {help}
--------------------------------------------------------------------------------
'''.format(
    name=name, pkg=pkg, version=pkg_resources.get_distribution(pkg).version,
    requirement=requirement, help=dep['help']))  # NOQA
                found = True
            except Exception as e:
                warnings.warn(
                    'Failed to check requirement: {}'.format(requirement))
                break

            if found:
                if installed:
                    warnings.warn('''
--------------------------------------------------------------------------------
Multiple installation of {name} package has been detected.
You should install only one package from {pkgs}.
Run `pip list` to see the list of packages currentely installed, then
`pip uninstall <package name>` to uninstall unnecessary package(s).
--------------------------------------------------------------------------------
'''.format(name=name, pkgs=pkgs))
                installed = True


def check():
    _check_python_350()
    _check_osx_numpy_backend()
    _check_optional_dependencies()
