import sys

from chainer.backends import cuda


sys.modules[__name__] = cuda
