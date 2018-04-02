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
        self.cuda_info = cuda.get_runtime_info()

    def __str__(self):
        s = six.StringIO()
        s.write('''Chainer: {}\n'''.format(self.chainer_version))
        s.write('''NumPy: {}\n'''.format(self.numpy_version))
        s.write(str(self.cuda_info))
        return s.getvalue()


def get_runtime_info():
    return _RuntimeInfo()


def print_runtime_info(out=None):
    if out is None:
        out = sys.stdout
    out.write(str(get_runtime_info()))
