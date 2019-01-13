import sys
import typing as tp  # NOQA
import warnings

import numpy

import chainer
from chainer import backend
from chainer.backends import _cpu
from chainer.backends import cuda
from chainer.backends import intel64
from chainer import types  # NOQA
import chainerx


class DeviceResident(object):

    """A base class of objects with multi-device hierarchy."""

    _device = _cpu.CpuDevice()

    # Indicates the object is mixed-device.
    _MIXED_DEVICE = object()

    def __init__(self):
        # Check and store overridden to_device family method names.
        overridden_to_methods = tuple([
            m for m in ('to_device', 'to_cpu', 'to_gpu', 'to_intel64')
            if _is_to_device_method_overridden(self, m)])
        if overridden_to_methods:
            # to_device() cannot be overridden.
            if 'to_device' in overridden_to_methods:
                raise TypeError(
                    'to_device cannot be overridden (class {}).'.format(
                        self.__class__))
            # Overriding to_cpu/to_gpu/to_intel64 causes a warning.
            warnings.warn(
                'Overriding method(s) [{}] of class {} is deprecated. '
                'Override `visit_device_residents` instead.'.format(
                    ', '.join(overridden_to_methods),
                    self.__class__),
                DeprecationWarning)
        self._overridden_to_methods = overridden_to_methods

    def visit_device_residents(self, visitor):
        """Applies the visitor to all the device objects in this instance."""
        raise NotImplementedError(
            'Concrete implementation must override this method to describe '
            'the logic to traverse device objects.')

    @property
    def device(self):
        """Returns the sole device"""
        if self._device is None:
            visitor = _GetSoleDeviceVisitor()
            self.visit_device_residents(visitor)
            sole_device = visitor.sole_device
            if sole_device is None:
                self._device = self._MIXED_DEVICE
            else:
                self._device = sole_device

        if self._device is self._MIXED_DEVICE:
            return None
        return self._device

    @property
    def xp(self):
        # type: () -> types.Xp
        """Array module for this link.

        Depending on which of CPU/GPU this link is on, this property returns
        :mod:`numpy` or :mod:`cupy`.

        """
        device = self.device
        if device is None:
            return None
        return device.xp

    def to_cpu(self):
        # type: () -> 'DeviceResident'
        """Copies parameter variables and persistent values to CPU.

        This method does not handle non-registered attributes. If some of such
        attributes must be copied to CPU, the link implementation must
        override :meth:`Link.to_device` to do so.

        Returns: self

        """
        visitor = _ToDeviceVisitor(
            backend.CpuDevice(),
            entry_method_info=('to_cpu', {}),
            skip_between_cupy_devices=True)
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
        override :meth:`Link.to_device` to do so.

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
            entry_method_info=('to_gpu', {'device': device}),
            skip_between_cupy_devices=True)
        self.__to_device(visitor)
        return self

    def to_intel64(self):
        # type: () -> 'DeviceResident'
        """Copies parameter variables and persistent values to CPU."""
        intel64.check_ideep_available()
        visitor = _ToDeviceVisitor(
            chainer.get_device(intel64),
            entry_method_info=('to_intel64', {}))
        self.__to_device(visitor)
        return self

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

        self._device = None
        self.visit_device_residents(_ToChxVisitor())
        return self

    def from_chx(self):
        """Converts parameter variables and persistent values from ChainerX \
to NumPy/CuPy devices without any copy."""
        if isinstance(self._device, backend.ChainerxDevice):
            self._device = self._device.fallback_device

        self._device = None
        self.visit_device_residents(_FromChxVisitor())
        return self

    def __to_device(self, to_device_visitor):
        self._device = None
        self.visit_device_residents(to_device_visitor)

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
    """

    def visit_visitable(self, visitable, visitor):
        assert isinstance(visitable, DeviceResident)
        visitable.visit_device_residents(visitor)

    def visit_array(self, arr):
        """Processes an array and returns a new one.

        If the visitor does not create a new array, it can simply return the
        original array.
        """
        raise NotImplementedError()

    def visit_param(self, param):
        """Processes a parameter."""
        raise NotImplementedError()


class _GetSoleDeviceVisitor(DeviceResidentsVisitor):
    # A visitor for retrieving the hierarchie's single device.
    # If the hierarchy has multiple devices, this visitor only indicates that
    # by _MIXED_DEVICE special object. In this case actual devices cannot be
    # retrieved.

    _MIXED_DEVICE = object()
    _device = None

    @property
    def sole_device(self):
        # Returns the hierarchie's single device.
        # Returns None if multiple devices were found.

        if self._device is None:
            # Not a single object is visited
            return None
        if self._device is self._MIXED_DEVICE:
            # Multiple devices are found
            return None
        # Sole device is found
        return self._device

    def _visit_device(self, device):
        assert isinstance(device, backend.Device)
        if self._device is None:
            self._device = device
        elif self._device == device:
            pass
        else:
            # Different devices are found
            self._device = self._MIXED_DEVICE

    def visit_array(self, arr):
        assert isinstance(arr, chainer.get_array_types())
        self._visit_device(backend.get_device_from_array(arr))
        return arr

    def visit_param(self, param):
        assert isinstance(param, chainer.Variable)
        self._visit_device(param.device)


class _ToDeviceVisitor(DeviceResidentsVisitor):
    # A visitor that implements recursive to_device().
    # For backward compatibility, if any of to_cpu/to_gpu/to_intel64 are
    # overridden on a visitable, this visitor calls it instead of
    # `visit_visitable`. That's true even if `to_device` was originally called.

    def __init__(
            self, device, entry_method_info=None,
            skip_between_cupy_devices=False):

        assert isinstance(device, chainer.backend.Device)

        # `entry_method_info` is for backward compatibility workaround for
        # overridden methods.
        # It indicates which method originally causes this visitor.
        # If it is any of the to_??? method names, descendant resident's
        # respective method will be called if it's overridden
        # (instead of `visit_device_residents`).
        if entry_method_info is not None:
            assert len(entry_method_info) == 2
            assert entry_method_info[0] in ('to_cpu', 'to_gpu', 'to_intel64')

        self._device = device
        self._entry_method_info = entry_method_info
        self._skip_between_cupy_devices = skip_between_cupy_devices

    def visit_visitable(self, visitable, visitor):
        visitable._device = None

        # Backward compatibility workaround for overridden methods
        if visitable._overridden_to_methods:
            if self._entry_method_info is not None:
                # Deprecated method is being called: e.g. to_cpu and to_gpu.
                method_name, kwargs = self._entry_method_info
            else:
                # to_device is being called
                method_name, kwargs = (
                    self._device_to_method_name_and_kwargs(self._device))
            if method_name in visitable._overridden_to_methods:
                to_method = getattr(visitable, method_name)
                to_method(**kwargs)
                return

        super(_ToDeviceVisitor, self).visit_visitable(visitable, visitor)

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

    def visit_param(self, param):
        assert isinstance(param, chainer.Variable)
        if not (self._skip_between_cupy_devices
                and self._device.xp is cuda.cupy
                and param.device.xp is cuda.cupy):
            param.to_device(self._device)


class _ToChxVisitor(DeviceResidentsVisitor):
    # A visitor that recursively calls to_chx().

    def visit_visitable(self, visitable, visitor):
        visitable._device = None
        super(_ToChxVisitor, self).visit_visitable(visitable, visitor)

    def visit_array(self, arr):
        assert isinstance(arr, chainer.get_array_types())
        return backend.to_chx(arr)

    def visit_param(self, param):
        assert isinstance(param, chainer.Variable)
        param.to_chx()


class _FromChxVisitor(DeviceResidentsVisitor):
    # A visitor that recursively calls from_chx().

    def visit_visitable(self, visitable, visitor):
        visitable._device = None
        super(_FromChxVisitor, self).visit_visitable(visitable, visitor)

    def visit_array(self, arr):
        assert isinstance(arr, chainer.get_array_types())
        return backend.from_chx(arr)

    def visit_param(self, param):
        assert isinstance(param, chainer.Variable)
        param.from_chx()
