import numpy

import chainer
from chainer.backends import _chainerx
from chainer.backends import _cpu
from chainer.backends import cuda
from chainer.backends import intel64
import chainerx

# Aliases
from chainer._backend import Device
from chainer.backends._chainerx import ChainerxDevice
from chainer.backends._chainerx import from_chainerx  # NOQA
from chainer.backends._chainerx import to_chainerx  # NOQA
from chainer.backends._cpu import CpuDevice
from chainer.backends.cuda import GpuDevice
from chainer.backends.intel64 import Intel64Device
from chainer import types  # NOQA


def _contains_nan(x):
    """Returns whether the input array has NaN values.

    Args:
        x (numpy.ndarray or cupy.ndarray): Array to be checked.

    Returns:
        bool: True if the input has NaN values.

    """
    if x.dtype.kind in ('f', 'c'):
        with cuda.get_device_from_array(x):
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
        numpy.copyto(dst, _cpu._to_cpu(src))
    elif isinstance(dst, intel64.mdarray):
        intel64.ideep.basic_copyto(
            dst, _cpu._to_cpu(src))
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


def get_device(device_spec):
    # type: (types.DeviceSpec) -> Device
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
    if device_spec is None:
        raise ValueError('Invalid dtype specifier: {}'.format(device_spec))

    if isinstance(device_spec, Device):
        return device_spec

    get_device_funcs = (
        _chainerx._get_device,
        _cpu._get_device,
        cuda._get_device,
        intel64._get_device,
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
    if isinstance(device_spec, cuda._integer_types):
        if device_spec < 0:
            return CpuDevice()
        else:
            return GpuDevice.from_device_id(device_spec)
    return get_device(device_spec)


def using_device(device_spec):
    """Context manager to apply the thread-local device state.

    Args:
        device_spec (object): Device specifier. See :func:`chainer.get_device`
            for details.
    """

    # TODO(niboshi): Set default device (once this concept is introduced in
    # Chainer).
    device = get_device(device_spec)
    return device.create_context()


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
    is_chainerx_available = chainerx.is_available()
    if is_chainerx_available or cuda.available:
        arrays = []
        for arg in args:
            # Unwrap arrays
            if isinstance(arg, chainer.variable.Variable):
                array = arg.data
            else:
                array = arg
            if is_chainerx_available and isinstance(array, chainerx.ndarray):
                return chainerx
            arrays.append(array)
        if cuda.available:
            return cuda.cupy.get_array_module(*arrays)
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
        device = GpuDevice.from_array(array)
        if device is not None:
            return device

        if isinstance(array, chainerx.ndarray):
            return ChainerxDevice(array.device)

        device = Intel64Device.from_array(array)
        if device is not None:
            return device

    return CpuDevice()
