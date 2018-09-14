import numpy

from chainer.backends import cuda
from chainer.backends import intel64
import chainerx


def _array_to_numpy(array):
    if array is None:
        return None
    if isinstance(array, numpy.ndarray):
        return array
    if isinstance(array, (numpy.number, numpy.bool_, intel64.mdarray)):
        return numpy.asarray(array)
    if chainerx.is_available() and isinstance(array, chainerx.ndarray):
        return chainerx.tonumpy(array)
    if isinstance(array, cuda.ndarray):
        cuda.check_cuda_available()
        with cuda.get_device_from_array(array):
            return array.get()
    else:
        raise TypeError(
            'Array cannot be converted into an numpy.ndarray'
            '\nActual type: {0}.'.format(type(array)))


def to_numpy(array):
    if isinstance(array, (list, tuple)):
        d = {}
        ret = []
        for arr in array:
            if arr is None:
                ret.append(None)
            else:
                arr2 = d.get(id(arr))
                if arr2 is None:
                    arr2 = _array_to_numpy(arr)
                    d[id(arr)] = arr2
                ret.append(arr2)
        return type(array)(ret)
    else:
        return _array_to_numpy(array)
