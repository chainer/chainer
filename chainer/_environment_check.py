import os
import sys
import warnings

import numpy.distutils.system_info


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


def check():
    _check_python_350()
    _check_osx_numpy_backend()
