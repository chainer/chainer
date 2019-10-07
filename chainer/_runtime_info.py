import platform
import sys

import numpy
import six

import chainer
from chainer.backends import cuda
from chainer.backends import intel64
import chainerx


class _RuntimeInfo(object):

    chainer_version = None
    numpy_version = None
    cuda_info = None
    ideep_version = None

    def __init__(self):
        self.chainer_version = chainer.__version__
        self.chainerx_available = chainerx.is_available()
        self.numpy_version = numpy.__version__
        self.platform_version = platform.platform()
        if cuda.available:
            self.cuda_info = cuda.cupyx.get_runtime_info()
        else:
            self.cuda_info = None
        if intel64.is_ideep_available():
            self.ideep_version = intel64.ideep.__version__
        else:
            self.ideep_version = None

    def __str__(self):
        s = six.StringIO()
        s.write('''Platform: {}\n'''.format(self.platform_version))
        s.write('''Chainer: {}\n'''.format(self.chainer_version))
        s.write('''ChainerX: {}\n'''.format(
            'Available' if self.chainerx_available else 'Not Available'))
        s.write('''NumPy: {}\n'''.format(self.numpy_version))
        if self.cuda_info is None:
            s.write('''CuPy: Not Available\n''')
        else:
            s.write('''CuPy:\n''')
            for line in str(self.cuda_info).splitlines():
                s.write('''  {}\n'''.format(line))
        if self.ideep_version is None:
            s.write('''iDeep: Not Available\n''')
        else:
            s.write('''iDeep: {}\n'''.format(self.ideep_version))
        return s.getvalue()


def _get_runtime_info():
    return _RuntimeInfo()


def print_runtime_info(out=None):
    """Shows Chainer runtime information.

    Runtime information includes:

    - OS platform

    - Chainer version

    - ChainerX version

    - NumPy version

    - CuPy version

      - CUDA information
      - cuDNN information
      - NCCL information

    - iDeep version

    Args:
        out: Output destination.
            If it is ``None``, runtime information
            will be shown in ``sys.stdout``.

    """
    if out is None:
        out = sys.stdout
    out.write(str(_get_runtime_info()))
    if hasattr(out, 'flush'):
        out.flush()
