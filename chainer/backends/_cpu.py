import numpy

from chainer import _backend
import chainer.backends
import chainerx


class CpuDevice(_backend.Device):

    @property
    def xp(self):
        return numpy

    @staticmethod
    def from_array(array):
        if isinstance(array, numpy.ndarray):
            return CpuDevice()
        return None

    def __eq__(self, other):
        return isinstance(other, CpuDevice)

    def __repr__(self):
        return '<{} (numpy)>'.format(self.__class__.__name__)

    def send_array(self, array):
        return _array_to_numpy(array)


def _get_device(device_spec):
    if device_spec is numpy:
        return CpuDevice()
    return None


def _to_numpy(array):
    """Converts an array or arrays to NumPy."""
    return _backend._convert_arrays(array, _array_to_numpy)


def _array_to_numpy(array):
    if array is None:
        return None
    if isinstance(array, numpy.ndarray):
        return array
    if isinstance(array, chainer.backends.intel64.mdarray):
        return numpy.asarray(array)
    if isinstance(array, chainerx.ndarray):
        return chainerx.to_numpy(array, copy=False)
    if isinstance(array, chainer.backends.cuda.ndarray):
        with chainer.backends.cuda.get_device_from_array(array):
            return array.get()
    if numpy.isscalar(array):
        return numpy.asarray(array)
    raise TypeError(
        'Array cannot be converted into an numpy.ndarray'
        '\nActual type: {0}.'.format(type(array)))
