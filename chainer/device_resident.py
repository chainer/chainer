import abc
import sys
import typing as tp  # NOQA

import numpy

import chainer
from chainer import backend
from chainer.backends import _cpu
from chainer.backends import cuda
from chainer.backends import intel64
from chainer import types  # NOQA
from chainer import utils
import chainerx


class DeviceResident(utils.enable_final(meta_base=abc.ABCMeta)):

    """A base class of objects with multi-device hierarchy."""

    _device = _cpu.CpuDevice()

    def __init__(self):
        # Store overridden to_device family method names.
        self._overridden_to_methods = tuple([
            m for m in ('to_cpu', 'to_gpu', 'to_intel64')
            if _is_to_device_method_overridden(self, m)])

    def device_resident_accept(self, visitor):
        """Applies the visitor to all the device objects in this instance.

        Args:
            visitor(~chainer.device_resident.DeviceResidentsVisitor): Visitor.

        This method should be overridden if the concrete class has custom
        sub-hierarchy of device resident objects.
        """
        visitor.visit_device_resident(self)

    @property
    def device(self):
        """:class:`~chainer.backend.Device` instance."""
        return self._device

    @property
    def xp(self):
        # type: () -> types.Xp
        """Array module corresponding to the device.

        Depending on the device in which this object resides, this property
        returns :mod:`numpy`, :mod:`cupy` or :mod:`chainerx`.

        """
        device = self.device
        if device is None:
            return None
        return device.xp

    def to_cpu(self):
        # type: () -> 'DeviceResident'
        """Copies parameter variables and persistent values to CPU.

        This method does not handle non-registered attributes. If some of such
        attributes must be copied to CPU, the link implementation should
        override :meth:`~DeviceResident.device_resident_accept` to do so.

        Returns: self

        """
        visitor = _ToDeviceVisitor(
            backend.CpuDevice(),
            entry_method_info=('to_cpu', {}),
            skip_between_cupy_devices=True,
            starting_device_resident=self)
        self.__to_device(visitor)
        return self

    def to_gpu(
            self,
            device=None,  # type: tp.Optional[types.CudaDeviceSpec]
    ):
        # type: (...) -> 'DeviceResident'
        """Copies parameter variables and persistent values to GPU.

        This method does not handle non-registered attributes. If some of such
        attributes must be copied to GPU, the link implementation must
        override :meth:`~DeviceResident.device_resident_accept` to do so.

        Args:
            device: Target device specifier. If omitted, the current device is
                used.

        Returns: self

        """
        cuda.check_cuda_available()
        cuda_device = cuda._get_device_or_current(device)
        device = chainer.backends.cuda.GpuDevice(cuda_device)
        visitor = _ToDeviceVisitor(
            device,
            entry_method_info=('to_gpu', {'device': device.device}),
            skip_between_cupy_devices=True,
            starting_device_resident=self)
        self.__to_device(visitor)
        return self

    def to_intel64(self):
        # type: () -> 'DeviceResident'
        """Copies parameter variables and persistent values to CPU."""
        intel64.check_ideep_available()
        visitor = _ToDeviceVisitor(
            chainer.get_device(intel64.Intel64Device()),
            entry_method_info=('to_intel64', {}),
            starting_device_resident=self)
        self.__to_device(visitor)
        return self

    @utils.final
    def to_chx(self):
        """Converts parameter variables and persistent values to ChainerX \
without any copy.

        This method does not handle non-registered attributes. If some of such
        attributes must be copied to ChainerX, the link implementation must
        override this method to do so.

        Returns: self
        """
        if not chainerx.is_available():
            raise RuntimeError('ChainerX is not available.')

        if self.xp is chainerx:
            return self

        self.device_resident_accept(_ToChxVisitor())
        return self

    @utils.final
    def from_chx(self):
        """Converts parameter variables and persistent values from ChainerX \
to NumPy/CuPy devices without any copy."""
        if isinstance(self._device, backend.ChainerxDevice):
            self._device = self._device.fallback_device

        self.device_resident_accept(_FromChxVisitor())
        return self

    def __to_device(self, to_device_visitor):
        self.device_resident_accept(to_device_visitor)

    @utils.final
    def to_device(
            self,
            device  # type: types.DeviceSpec
    ):
        # type: (...) -> 'DeviceResident'
        """Copies parameter variables and persistent values to the specified \
device.

        This method does not handle non-registered attributes. If some of such
        attributes must be copied to the device, the link implementation must
        override this method to do so.

        Args:
            device: Target device specifier. See
                :func:`~chainer.get_device` for available values.

        Returns: self

        """
        device = chainer.get_device(device)
        self.__to_device(_ToDeviceVisitor(device))
        return self


def _is_to_device_method_overridden(device_resident, method_name):
    # Returns whether the specified to_device family method is overridden.
    to_method = getattr(device_resident, method_name, None).__func__
    to_method_orig = getattr(DeviceResident, method_name)
    if sys.version_info < (3,):
        to_method_orig = to_method_orig.__func__
    if to_method is not to_method_orig:
        return True  # overridden
    return False


class DeviceResidentsVisitor(object):

    """Base class of visitors that visits device resident objects recursively.

    ..  seealso::
        :class:`chainer.DeviceResident`
    """

    def visit_device_resident(self, device_resident):
        """Processes a :class:`~chainer.DeviceResident` instance."""
        raise NotImplementedError()

    def visit_array(self, arr):
        """Processes an array and returns a new one.

        If the visitor does not create a new array, it can simply return the
        original array.
        """
        raise NotImplementedError()

    def visit_variable(self, param):
        """Processes a :class:`~chainer.Variable` or a \
:class:`~chainer.Parameter`."""
        raise NotImplementedError()


class _ToDeviceVisitor(DeviceResidentsVisitor):
    # A visitor that implements recursive to_device().
    # For backward compatibility, if any of to_cpu/to_gpu/to_intel64 are
    # overridden on a device resident, this visitor calls it instead of
    # `visit_device_resident`. That's true even if `to_device` was originally
    # called.

    def __init__(
            self, device, entry_method_info=None,
            skip_between_cupy_devices=False,
            starting_device_resident=None):

        assert isinstance(device, chainer.backend.Device)

        # `entry_method_info` is for backward compatibility workaround for
        # overridden methods.
        # It indicates which method originally causes this visitor.
        # If it is any of the to_??? method names, descendant resident's
        # respective method will be called if it's overridden
        # (instead of `device_resident_accept`).
        if entry_method_info is not None:
            assert len(entry_method_info) == 2
            assert entry_method_info[0] in ('to_cpu', 'to_gpu', 'to_intel64')

        # starting_device_resident is also for backward compatibility
        # workaround for overridden methods.
        # It is a DeviceResident if to_xxx methods were initially called
        # on this visitor. This is used to avoid infinite accept-visit loop
        # that would occur by calling to_xxx methods.
        assert (starting_device_resident is None
                or isinstance(starting_device_resident, DeviceResident))

        self._device = device
        self._entry_method_info = entry_method_info
        self._skip_between_cupy_devices = skip_between_cupy_devices
        self._starting_device_resident = starting_device_resident

    def visit_device_resident(self, device_resident):
        device_resident._device = self._device

        # Backward compatibility workaround for overridden methods
        if device_resident._overridden_to_methods:
            # Skip this device resident, if the visitor was initially triggered
            # from it.
            if device_resident is self._starting_device_resident:
                return

            if self._entry_method_info is not None:
                # Deprecated method is being called: e.g. to_cpu and to_gpu.
                method_name, kwargs = self._entry_method_info
            else:
                # to_device is being called
                method_name, kwargs = (
                    self._device_to_method_name_and_kwargs(self._device))
            if method_name in device_resident._overridden_to_methods:
                to_method = getattr(device_resident, method_name)
                to_method(**kwargs)
                return

    def _device_to_method_name_and_kwargs(self, device):
        # Converts a device instance to the corresponding combination of
        # to_??? method name and kwargs.

        # chainerx
        if device.xp is chainerx:
            return None, {}
        # cupy
        if device.xp is cuda.cupy:
            return 'to_gpu', {'device': device.device.id}
        # numpy
        assert device.xp is numpy
        if isinstance(device, _cpu.CpuDevice):
            return 'to_cpu', {}
        # intel64
        assert isinstance(device, intel64.Intel64Device)
        return 'to_intel64', {}

    def visit_array(self, arr):
        assert isinstance(arr, chainer.get_array_types())
        if not (self._skip_between_cupy_devices
                and self._device.xp is cuda.cupy
                and isinstance(arr, cuda.ndarray)):
            return self._device.send(arr)
        return arr

    def visit_variable(self, param):
        assert isinstance(param, chainer.Variable)
        if not (self._skip_between_cupy_devices
                and self._device.xp is cuda.cupy
                and param.device.xp is cuda.cupy):
            param.to_device(self._device)


class _ToChxVisitor(DeviceResidentsVisitor):
    # A visitor that recursively calls to_chx().

    def visit_device_resident(self, device_resident):
        device_resident._device = backend.ChainerxDevice.from_fallback_device(
            device_resident._device)

    def visit_array(self, arr):
        assert isinstance(arr, chainer.get_array_types())
        return backend.to_chx(arr)

    def visit_variable(self, param):
        assert isinstance(param, chainer.Variable)
        param.to_chx()


class _FromChxVisitor(DeviceResidentsVisitor):
    # A visitor that recursively calls from_chx().

    def visit_device_resident(self, device_resident):
        if isinstance(device_resident._device, backend.ChainerxDevice):
            device_resident._device = device_resident._device.fallback_device

    def visit_array(self, arr):
        assert isinstance(arr, chainer.get_array_types())
        return backend.from_chx(arr)

    def visit_variable(self, param):
        assert isinstance(param, chainer.Variable)
        param.from_chx()
