import sys

import numpy
import six

import chainer
from chainer.backends import cuda


class _RuntimeInfo(object):

    chainer_version = None
    numpy_version = None
    cupy_info = None

    def __init__(self):
        self.chainer_version = chainer.__version__
        self.numpy_version = numpy.__version__
        self.cupy_info = cuda.get_runtime_info()

    def __str__(self):
        s = six.StringIO()
        s.write('''Chainer: {}\n'''.format(self.chainer_version))
        s.write('''NumPy: {}\n'''.format(self.numpy_version))
        s.write(str(self.cupy_info))
        return s.getvalue()


def get_runtime_info(as_text=False):
    ri = _RuntimeInfo()
    if as_text:
        return str(ri)
    else:
        return ri


def print_runtime_info(out=None):
    if out is None:
        out = sys.stdout
    text = get_runtime_info(as_text=True)
    out.write(text)
