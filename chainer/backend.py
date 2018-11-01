import contextlib

import numpy

import chainer
from chainer import backends
import chainerx as chainerx_module


def _contains_nan(x):
    """Returns whether the input array has NaN values.

    Args:
        x (numpy.ndarray or cupy.ndarray): Array to be checked.

    Returns:
        bool: True if the input has NaN values.

    """
    if x.dtype.kind in ('f', 'c'):
        with backends.cuda.get_device_from_array(x):
            return get_array_module(x).isnan(x).any()
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
        numpy.copyto(dst, backends.cpu.to_numpy(src))
    elif isinstance(dst, backends.intel64.mdarray):
        backends.intel64.ideep.basic_copyto(dst, backends.cpu.to_numpy(src))
    elif isinstance(dst, backends.cuda.ndarray):
        if isinstance(src, chainer.get_cpu_array_types()):
            src = numpy.asarray(src)
            if dst.flags.c_contiguous or dst.flags.f_contiguous:
                dst.set(src)
            else:
                backends.cuda.cupy.copyto(
                    dst, backends.cuda.to_gpu(src, device=dst.device))
        elif isinstance(src, backends.cuda.ndarray):
            backends.cuda.cupy.copyto(dst, src)
        else:
            raise TypeError('cannot copy from non-array object of type {}'
                            .format(type(src)))
    else:
        raise TypeError('cannot copy to non-array object of type {}'.format(
            type(dst)))


def _convert_arrays(array, func):
    # Converts array or arrays
    if isinstance(array, (list, tuple)):
        d = {}
        ret = []
        for arr in array:
            if arr is None:
                ret.append(None)
            else:
                arr2 = d.get(id(arr))
                if arr2 is None:
                    arr2 = func(arr)
                    d[id(arr)] = arr2
                ret.append(arr2)
        return type(array)(ret)
    else:
        return func(array)


# TODO(niboshi): Revisit API
def to_numpy(array):
    return chainer.backends.cpu.to_numpy(array)


# TODO(niboshi): Revisit API
def to_chainerx(array, device_spec=None):
    return chainer.backends.chainerx.to_chainerx(array)


class Device(object):
    """Device object.
    """

    @property
    def xp(self):
        """Array module corresponding to the device."""
        raise NotImplementedError('Derived class must override this property.')

    def __enter__(self):
        raise RuntimeError(
            'Device class does not support runtime context using `with` '
            'statement. Use chainer.using_device instead.')

    def __exit__(self):
        # Definition of __exit__ is needed to raise a custom error on
        # __enter__.
        pass

    def __eq__(self, other):
        raise NotImplementedError('Derived class must override this method.')

    def __repr__(self):
        class_name = '{}.{}'.format(
            self.__class__.__module__, self.__class__.__name__)
        if self.xp is numpy:
            return '<{}(numpy)>'.format(class_name)

        if self.xp is backends.cuda.cupy:
            assert isinstance(self.device, backends.cuda.Device)
            return '<{}(cupy, {})>'.format(class_name, self.device.id)

        if self.xp is chainerx_module:
            assert isinstance(self.device, chainerx_module.Device)
            return '<{}(chainerx, {})>'.format(class_name, self.device.name)

        assert False

    def create_context(self):
        # Returns an object that implements __enter__ and __exit__.
        return None

    def send(self, arrays):
        """Transfers given arrays to the device.

        Args:
            arrays: Array or arrays of NumPy, CuPy, or ChainerX.

        Returns:
            Transferred arrays.

        """
        return _convert_arrays(arrays, self.send_array)


def get_device(device_spec):
    """Returns a device object.

    Args:
        device_spec (object): Device specifier. If a :class:`chainer.Device`
            instance is given, it is returned intact. Otherwise the following
            values are supported:

            * ChainerX devices

              * A string representing a device.
                (ex. ``'native:0'``, ``'native'``)
              * A tuple of ChainerX backend name and device index.
                (ex. ``('native', 0)``)
              * A :class:`chainerx.Device` object.

            * CuPy

              * A tuple of :mod:`cupy` module object and device ID.
                (ex. ``(cupy, 0)``)
              * A :class:`chainer.backends.cuda.Device` object.

            * NumPy

              * :mod:`numpy` module object. (``numpy``)

            * NumPy with Intel Architecture

              * :mod:`chainer.backends.intel64` module object.
                (``chainer.backends.intel64``)
    """
    assert device_spec is not None

    if isinstance(device_spec, Device):
        return device_spec

    get_device_funcs = (
        backends.cpu._get_device,
        backends.cuda._get_device,
        backends.intel64._get_device,
        backends.chainerx._get_device,
    )

    for get_device_func in get_device_funcs:
        device = get_device_func(device_spec)
        if device is not None:
            return device

    raise ValueError('Invalid device specifier: {}'.format(device_spec))


def _get_device_compat(device_spec):
    # Backward-compatibility version of get_device.
    # It supports CUDA device index as an integer (numpy if negative)
    # Returns chainer.Device.
    if isinstance(device_spec, backends.cuda._integer_types):
        if device_spec < 0:
            return chainer.backends.cpu.CpuDevice()
        else:
            return backends.cuda.GpuDevice.from_device_id(device_spec)
    return get_device(device_spec)


@contextlib.contextmanager
def _dummy_context():
    yield


def using_device(device_spec):
    """Context manager to apply the thread-local device state.

    Args:
        device_spec (object): Device specifier. See :func:`chainer.get_device`
            for details.
    """

    # TODO(niboshi): Set default device (once this concept is introduced in
    # Chainer).
    device = get_device(device_spec)
    context = device.create_context()
    if context is None:
        return _dummy_context()
    return context


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
    if chainerx_module.is_available() or backends.cuda.available:
        args = [arg.data if isinstance(arg, chainer.variable.Variable) else arg
                for arg in args]

    if (chainerx_module.is_available()
            and any([isinstance(a, chainerx_module.ndarray) for a in args])):
        return chainerx_module
    elif backends.cuda.available:
        return backends.cuda.cupy.get_array_module(*args)
    else:
        return numpy


def get_device_from_array(*arrays):
    """Gets the device from arrays.

    The device on which the given array reside is returned.

    .. note::

        Unlike :func:`get_array_module`, this method does not recognize
        :class:`~chainer.Variable` objects.
        If you need to get device from the :class:`~chainer.Variable` instance
        ``v``, you need to use ``get_device_from_array(v.array)``.

    Args:
        arrays (array or list of arrays):
            Arrays to determine the device. If multiple arrays are given, the
            device correspoinding to the first array which is not NumPy array
            is returned.

    Returns:
        chainer.Device: Device instance.
    """
    for array in arrays:
        device = backends.cuda.GpuDevice.from_array(array)
        if device is not None:
            return device

        if isinstance(array, chainerx_module.ndarray):
            return backends.chainerx.ChainerxDevice(array.device)

        device = backends.intel64.Intel64Device.from_array(array)
        if device is not None:
            return device

    return backends.cpu.CpuDevice()
