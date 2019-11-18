import numpy

import chainer
from chainer import _backend
from chainer.backends import _cpu
from chainer.backends import cuda
from chainer.backends import intel64
import chainerx


class ChainerxDevice(_backend.Device):

    """Device for ChainerX backend"""

    xp = chainerx
    supported_array_types = (chainerx.ndarray,)

    __hash__ = _backend.Device.__hash__

    def __init__(self, device: 'chainerx.Device') -> None:
        assert isinstance(device, chainerx.Device)
        super(ChainerxDevice, self).__init__()
        self.device = device  # type: chainerx.Device

    @staticmethod
    def from_array(array):
        if isinstance(array, chainerx.ndarray) and array.device is not None:
            return ChainerxDevice(array.device)
        return None

    @staticmethod
    def from_fallback_device(device):
        """Returns a :class:`~chainer.backend.ChainerxDevice` corresponding \
to the fallback device.

        .. seealso::
            :data:`~chainer.backend.ChainerxDevice.fallback_device`
        """
        assert isinstance(device, _backend.Device)
        if isinstance(device, _cpu.CpuDevice):
            return ChainerxDevice(chainerx.get_device('native', 0))
        if isinstance(device, cuda.GpuDevice):
            return ChainerxDevice(
                chainerx.get_device('cuda', device.device.id))
        raise RuntimeError(
            'Only CPU or GPU devices are allowed. '
            'Actual: {}'.format(device))

    @property
    def name(self):
        return self.device.name

    @property
    def fallback_device(self):
        """Fallback device.

        A fallback device is either a :class:`~chainer.backend.CpuDevice` or
        a :class:`~chainer.backend.GpuDevice` which shares the same physical
        device with the original ChainerX device.

        For example, the fallback device of ``native:0`` ChainerX device is
        :class:`~chainer.backend.CpuDevice`. The fallback device of ``cuda:1``
        ChainerX device is :class:`~chainer.backend.GpuDevice` with device ID
        1.
        """
        backend_name = self.device.backend.name
        if backend_name == 'native':
            return _cpu.CpuDevice()
        if backend_name == 'cuda':
            return cuda.GpuDevice.from_device_id(self.device.index)
        raise RuntimeError(
            'Only \'native\' or \'cuda\' devices have corresponding fallback '
            'devices. Actual: {}'.format(backend_name))

    def __eq__(self, other):
        return (
            isinstance(other, ChainerxDevice)
            and other.device == self.device)

    def __repr__(self):
        return '<{} {}>'.format(
            self.__class__.__name__, self.device.name)

    def create_context(self):
        # Returns a context that sets the default device.
        return chainerx.using_device(self.device)

    def send_array(self, array):
        device = self.device
        if isinstance(array, chainerx.ndarray):
            if array.device is device:
                return array
            return array.to_device(device)
        return _array_to_chainerx(array, device)

    def use(self):
        chainerx.set_default_device(self.device)

    def is_array_supported(self, array):
        return (
            isinstance(array, chainerx.ndarray)
            and self.device == array.device)


def to_chx(array):
    """Converts an array or arrays to ChainerX.

    Destination ChainerX devices are chosen according to the types of input
    arrays.
    """
    return _backend._convert_arrays(array, _array_to_chainerx)


def from_chx(array):
    """Converts an array or arrays from ChainerX to NumPy or CuPy ones.

    Destination array types are chosen such that no copies occur.
    """
    return _backend._convert_arrays(array, _array_from_chainerx)


def _get_chainerx_device(device_spec):
    # Returns chainerx.Device
    if isinstance(device_spec, chainerx.Device):
        return device_spec
    return chainerx.get_device(device_spec)


def _array_to_chainerx(array, device=None):
    # If device is None, appropriate device is chosen according to the input
    # arrays.
    assert device is None or isinstance(device, chainerx.Device)

    if array is None:
        return None

    if array.dtype not in chainerx.all_dtypes:
        raise TypeError(
            'Dtype {} is not supported in ChainerX.'.format(array.dtype.name))

    if isinstance(array, chainerx.ndarray):
        if device is None:
            return array
        if device is array.device:
            return array
        return array.to_device(device)
    if isinstance(array, numpy.ndarray):
        if device is None:
            device = chainerx.get_device('native', 0)
        return chainerx.array(array, device=device, copy=False)
    if isinstance(array, cuda.ndarray):
        if device is None:
            device = chainerx.get_device('cuda', array.device.id)
        elif device.backend.name != 'cuda':
            # cupy to non-cuda backend
            # TODO(niboshi): Remove conversion to numpy when both CuPy and
            # ChainerX support the array interface.
            array = _cpu._to_cpu(array)
            return chainerx.array(array, device=device, copy=False)
        elif device.index != array.device.id:
            # cupy to cuda backend but different device
            array = cuda.to_gpu(array, device=device.index)
        # cupy to cuda backend with the same device
        return chainerx._core._fromrawpointer(
            array.data.mem.ptr,
            array.shape,
            array.dtype,
            array.strides,
            device,
            array.data.ptr - array.data.mem.ptr,
            array)
    if isinstance(array, intel64.mdarray):
        return _array_to_chainerx(numpy.array(array), device)
    if numpy.isscalar(array):
        return chainerx.asarray(array)

    raise TypeError(
        'Array cannot be converted into chainerx.ndarray'
        '\nActual type: {0}.'.format(type(array)))


def _array_from_chainerx(array):
    if array is None:
        return None
    if not isinstance(array, chainerx.ndarray):
        if isinstance(array, chainer.get_array_types()):
            return array
        raise TypeError(
            'Tried to convert to a non-ChainerX array from an invalid type: '
            '{}'.format(type(array)))

    backend_name = array.device.backend.name
    if backend_name == 'native':
        return _cpu._to_cpu(array)
    if backend_name == 'cuda':
        return cuda.to_gpu(array, array.device.index)

    raise ValueError(
        'Only ChainerX arrays with native or cuda backends can be converted '
        'to non-ChainerX arrays.\nActual: {0}.'.format(backend_name))
