import numpy

import chainer
from chainer.backends import cuda
from chainer.backends import intel64
import chainerx


def _contains_nan(x):
    """Returns whether the input array has NaN values.

    Args:
        x (numpy.ndarray or cupy.ndarray): Array to be checked.

    Returns:
        bool: True if the input has NaN values.

    """
    if x.dtype.kind in ('f', 'c'):
        with cuda.get_device_from_array(x):
            return cuda.get_array_module(x).isnan(x).any()
    else:
        return False


def copyto(dst, src):
    """Copies the elements of an ndarray to those of another one.

    This function can copy the CPU/GPU arrays to the destination arrays on
    another device.

    Args:
        dst (`numpy.ndarray`, `cupy.ndarray` or `ideep4py.mdarray`):
            Destination array.
        src (`numpy.ndarray`, `cupy.ndarray` or `ideep4py.mdarray`):
            Source array.

    """
    if isinstance(dst, numpy.ndarray):
        numpy.copyto(dst, numpy.asarray(cuda.to_cpu(src)))
    elif isinstance(dst, intel64.mdarray):
        intel64.ideep.basic_copyto(dst, cuda.to_cpu(src))
    elif isinstance(dst, cuda.ndarray):
        if isinstance(src, chainer.get_cpu_array_types()):
            src = numpy.asarray(src)
            if dst.flags.c_contiguous or dst.flags.f_contiguous:
                dst.set(src)
            else:
                cuda.cupy.copyto(dst, cuda.to_gpu(src, device=dst.device))
        elif isinstance(src, cuda.ndarray):
            cuda.cupy.copyto(dst, src)
        else:
            raise TypeError('cannot copy from non-array object of type {}'
                            .format(type(src)))
    else:
        raise TypeError('cannot copy to non-array object of type {}'.format(
            type(dst)))


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


def get_array_module(*args):
    """Gets an appropriate one from :mod:`numpy`, :mod:`cupy`, or
    :mod:`chainerx`.

    This function will return their data arrays' array module for
    :class:`~chainer.Variable` arguments.

    Args:
        args: Values to determine whether NumPy, CuPy, or ChainerX should be
        used.

    Returns:
        module: :mod:`cupy`, :mod:`numpy`, or :mod:`chainerx` is returned based
        on the types of the arguments.

    """
    if chainerx.is_available() or cuda.available:
        args = [arg.data if isinstance(arg, chainer.variable.Variable) else arg
                for arg in args]
    if chainerx.is_available() and \
            all([isinstance(a, chainerx.ndarray) for a in args]):
        return chainerx
    if cuda.available:
        return cuda.cupy.get_array_module(*args)
    else:
        return numpy
