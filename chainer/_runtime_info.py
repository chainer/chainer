import sys

import numpy
import six

import chainer
from chainer.backends import cuda


class _RuntimeInfo(object):

    chainer_version = None
    numpy_version = None
    cuda_info = None

    def __init__(self):
        self.chainer_version = chainer.__version__
        self.numpy_version = numpy.__version__
        if cuda.available:
            self.cuda_info = cuda.cupyx.get_runtime_info()
        else:
            self.cuda_info = None

    def __str__(self):
        s = six.StringIO()
        s.write('''Chainer: {}\n'''.format(self.chainer_version))
        s.write('''NumPy: {}\n'''.format(self.numpy_version))
        if self.cuda_info is None:
            s.write('''CuPy: Not Available\n''')
        else:
            s.write('''CuPy:\n''')
            for line in str(self.cuda_info).splitlines():
                s.write('''  {}\n'''.format(line))
        return s.getvalue()


def get_runtime_info():
    return _RuntimeInfo()


def print_runtime_info(out=None):
    if out is None:
        out = sys.stdout
    out.write(str(get_runtime_info()))
    if hasattr(out, 'flush'):
        out.flush()
