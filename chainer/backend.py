import numpy
import six

import chainer
from chainer.backends import _chainerx
from chainer.backends import _cpu
from chainer.backends import cuda
from chainer.backends import intel64
import chainerx

# Aliases
from chainer._backend import Device
from chainer.backends._chainerx import ChainerxDevice
from chainer.backends._chainerx import from_chx  # NOQA
from chainer.backends._chainerx import to_chx  # NOQA
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


def _guess_device_from_array_module(xp):
    """Returns a plausible device from array module

    .. warning::

        There can be multiple devices for a module

    """
    if xp is cuda.cupy:
        return cuda.GpuDevice(cuda.Device())
    elif xp is chainerx:
        return _chainerx.ChainerxDevice(chainerx.get_default_device())
    else:
        # Cannot detect intel64, because xp of intel64 is numpy.
        return _cpu.CpuDevice()


def get_device(device_spec):
    # type: (types.DeviceSpec) -> Device
    """Returns a device object.

    Args:
        device_spec (object): Device specifier.
            If a :class:`chainer.backend.Device` instance is given, it is
            returned intact. Otherwise the following values are supported:

            * ChainerX devices

              * A string representing a device.
                (ex. ``'native:0'``, ``'native'``)
              * A :class:`chainerx.Device` object.

            * CuPy

              * A string starts with ``'@cupy:'``.
                (ex. ``'@cupy:0'``)
              * A :class:`chainer.backends.cuda.Device` object.

            * NumPy

              * The string ``'@numpy'``.

            * NumPy with Intel Architecture

              * The string ``'@intel64'``.
    """
    if isinstance(device_spec, Device):
        return device_spec

    if isinstance(device_spec, cuda._integer_types):
        return _get_device_cupy_or_numpy(device_spec)

    if chainerx.is_available() and isinstance(device_spec, chainerx.Device):
        return _chainerx.ChainerxDevice(device_spec)

    if cuda.available and isinstance(device_spec, cuda.Device):
        return cuda.GpuDevice(device_spec)

    if isinstance(device_spec, six.string_types):
        # '-1', '0', '1', ...
        try:
            int_device_spec = int(device_spec)
        except ValueError:
            pass
        else:
            return _get_device_cupy_or_numpy(int_device_spec)

        if device_spec.startswith('@'):
            # '@module:...'
            mod_name, colon, precise_spec = device_spec[1:].partition(':')
            if mod_name == 'numpy':
                if not colon:
                    return _cpu.CpuDevice()
            elif mod_name == 'cupy':
                if colon:
                    return cuda.GpuDevice.from_device_id(int(precise_spec))
            elif mod_name == 'intel64':
                if not colon:
                    return intel64.Intel64Device()

        elif chainerx.is_available():
            return _chainerx.ChainerxDevice(chainerx.get_device(device_spec))

    raise ValueError('Invalid device specifier: {}'.format(device_spec))


def _get_device_cupy_or_numpy(device_spec):
    # legacy spec of (gpu) device
    if device_spec >= 0:
        return cuda.GpuDevice.from_device_id(device_spec)
    else:
        return _cpu.CpuDevice()


def using_device(device_spec):
    """Context manager to apply the thread-local device state.

    Args:
        device_spec (object): Device specifier. See :func:`chainer.get_device`
            for details.

    .. admonition:: Example

        .. testcode::
           :skipif: doctest_helper.skipif_not_enough_cuda_devices(2)

           with chainer.using_device('@cupy:1'):
               a = cupy.empty((3, 2))

           assert a.device.id == 1

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
