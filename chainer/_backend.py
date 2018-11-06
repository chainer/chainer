import numpy

import chainer
import chainerx
from chainer.backends import cuda


def _get_array_module(*args):
    if chainerx.is_available() or cuda.available:
        args = [arg.data if isinstance(arg, chainer.variable.Variable) else arg
                for arg in args]

    if (chainerx.is_available()
            and any([isinstance(a, chainerx.ndarray) for a in args])):
        return chainerx
    elif cuda.available:
        return cuda.cupy.get_array_module(*args)
    else:
        return numpy
