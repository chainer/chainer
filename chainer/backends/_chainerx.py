import numpy

from chainer import _backend
from chainer.backends import _cpu
from chainer.backends import cuda
from chainer.backends import intel64
import chainerx


class ChainerxDevice(_backend.Device):

    def __init__(self, device):
        assert isinstance(device, chainerx.Device)
        super(ChainerxDevice, self).__init__()
        self.device = device

    @property
    def xp(self):
        return chainerx

    def __eq__(self, other):
        return (
            isinstance(other, ChainerxDevice)
            and other.device == self.device)

    def __repr__(self):
        return '<{} {}>'.format(
            self.__class__.__name__, self.device.name)

    def create_context(self):
        # Returns a context that sets the default device.
        return chainerx.device_scope(self.device)

    def send_array(self, array):
        device = self.device
        if isinstance(array, chainerx.ndarray):
            if array.device is device:
                return array
            return array.to_device(device)
        return _array_to_chainerx(array, device)


def _get_device(device_spec):
    # Called from chainer.backend.get_device
    if not chainerx.is_available():
        return None
    if isinstance(device_spec, chainerx.Device):
        return ChainerxDevice(device_spec)
    if isinstance(device_spec, str):
        return ChainerxDevice(chainerx.get_device(device_spec))
    if (isinstance(device_spec, tuple) and len(device_spec) >= 1
            and isinstance(device_spec[0], str)):
        return ChainerxDevice(chainerx.get_device(*device_spec))
    return None


def _get_chainerx_device(device_spec):
    # Returns chainerx.Device
    if isinstance(device_spec, chainerx.Device):
        return device_spec
    return chainerx.get_device(device_spec)


def _to_chainerx(array):
    """Converts an array or arrays to ChainerX.

    Destination ChainerX devices are chosen according to the types of input
    arrays.
    """
    return _backend._convert_arrays(
        array, lambda arr: _array_to_chainerx(arr, None))


def _array_to_chainerx(array, device):
    # If device is None, appropriate device is chosen according to the input
    # arrays.
    assert device is None or isinstance(device, chainerx.Device)

    if array is None:
        return None
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
            array = _cpu._to_numpy(array)
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
