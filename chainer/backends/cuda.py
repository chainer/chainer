"""Device, context and memory management on CuPy.

.. note::
   The package ``chainer.cuda`` has been renamed to
   :mod:`chainer.backends.cuda` as of v4.0.0, but the previous module path
   ``chainer.cuda`` is also available.

Chainer uses `CuPy <https://cupy.chainer.org/>`_ (with very thin wrapper)
to exploit the speed of GPU computation. Following modules and classes defined
in CuPy are imported to :mod:`chainer.backends.cuda` module for convenience
(refer to this table when reading chainer's source codes).

===================================== =================================
 imported name                         original name
===================================== =================================
 ``chainer.backends.cuda.cupy``        :mod:`cupy`
 ``chainer.backends.cuda.cupyx``       :mod:`cupyx`
 ``chainer.backends.cuda.ndarray``     :class:`cupy.ndarray`
 ``chainer.backends.cuda.cupy.cuda``   :mod:`cupy.cuda`
 ``chainer.backends.cuda.Device``      :class:`cupy.cuda.Device`
 ``chainer.backends.cuda.Event``       :class:`cupy.cuda.Event`
 ``chainer.backends.cuda.Stream``      :class:`cupy.cuda.Stream`
===================================== =================================

Chainer replaces the default allocator of CuPy by its memory pool
implementation. It enables us to reuse the device memory over multiple
forward/backward computations, and temporary arrays for consecutive elementwise
operations.
"""

import binascii
import functools
import itertools
import os
import threading
import time
import typing as tp  # NOQA
import warnings

import numpy
import six

import chainer
from chainer import _backend
from chainer.backends import _cpu
from chainer.backends import intel64
from chainer.configuration import config
from chainer import types  # NOQA
import chainerx

available = False  # type: bool
cudnn_enabled = False  # type: bool

try:
    import cupy
    from cupy import cuda  # NOQA
    from cupy.cuda import cublas  # NOQA
    import cupyx  # NOQA
    import cupyx.scipy.linalg  # NOQA
    import cupyx.scipy.special  # NOQA

    from cupy import ndarray  # type: ignore # NOQA

    from cupy.cuda import Device  # type: ignore # NOQA
    from cupy.cuda import Event  # type: ignore # NOQA
    from cupy.cuda import Stream  # type: ignore # NOQA

    available = True
except Exception as e:
    _resolution_error = e

    class ndarray(object):  # type: ignore # for type testing
        @property
        def shape(self):
            # type: () -> types.Shape
            pass

        @property
        def device(self):
            # type: () -> 'Device'
            pass

        def get(self, stream=None):
            # type: (tp.Optional['Stream']) -> numpy.ndarray
            pass

        def set(self, arr, stream=None):
            # type: (numpy.ndarray, tp.Optional['Stream']) -> None
            pass

    class Device(object):  # type: ignore # for type testing
        def __init__(self, device=None):
            # type: (tp.Optional[int]) -> None
            pass

        def __enter__(self):
            # type: () -> 'Device'
            pass

        def __exit__(self, *args):
            # type: (*tp.Any) -> None
            pass

    class Event(object):  # type: ignore # for type testing
        pass

    class Stream(object):  # type: ignore # for type testing
        pass

    # for `xp is chainer.backends.cuda.cupy` to always work
    cupy = object()


if available:
    _cudnn_disabled_by_user = int(os.environ.get('CHAINER_CUDNN', '1')) == 0
    try:
        import cupy.cudnn
        cudnn = cupy.cudnn  # type: tp.Optional[types.ModuleType]
        libcudnn = cupy.cuda.cudnn  # type: tp.Any # NOQA
        cudnn_enabled = not _cudnn_disabled_by_user
    except Exception as e:
        _resolution_error = e

        # for `chainer.backends.cuda.libcudnn` to always work
        libcudnn = object()


def check_cuda_available():
    """Checks if CUDA is available.

    When CUDA is correctly set up, nothing happens.
    Otherwise it raises ``RuntimeError``.
    """
    if not available:
        msg = ('CUDA environment is not correctly set up\n'
               '(see https://github.com/chainer/chainer#installation).')
        msg += str(_resolution_error)
        raise RuntimeError(msg)
    if (not cudnn_enabled and
            not _cudnn_disabled_by_user and
            not getattr(check_cuda_available, '_already_warned', False)):
        warnings.warn(
            'cuDNN is not enabled.\n'
            'Please reinstall CuPy after you install cudnn\n'
            '(see https://docs-cupy.chainer.org/en/stable/install.html'
            '#install-cudnn).')
        check_cuda_available._already_warned = True


class DummyDeviceType(Device):

    """Dummy device class that does nothing with cupy.cuda.Device interface.

    This class is used to represent CPU device.

    """

    id = -1

    def __init__(self):
        pass

    def __int__(self):
        return -1

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def use(self):
        pass

    def synchronize(self):
        pass

    def __eq__(self, other):
        return isinstance(other, DummyDeviceType)

    def __ne__(self, other):
        return not (self == other)


DummyDevice = DummyDeviceType()


# ------------------------------------------------------------------------------
# Global states
# ------------------------------------------------------------------------------
if available:
    # This is for backward compatibility
    memory_pool = cupy.get_default_memory_pool()
    pinned_memory_pool = cupy.get_default_pinned_memory_pool()


_integer_types = six.integer_types + (numpy.integer,)


# ------------------------------------------------------------------------------
# Device
# ------------------------------------------------------------------------------
class GpuDevice(_backend.Device):

    """Device for GPU (CuPy) backend"""

    def __init__(self, device):
        check_cuda_available()
        assert isinstance(device, Device)

        super(GpuDevice, self).__init__()
        self.device = device

    @staticmethod
    def from_device_id(device_id):
        check_cuda_available()

        if not (isinstance(device_id, _integer_types) and device_id >= 0):
            raise ValueError('Invalid CUDA device ID: {}'.format(device_id))

        return GpuDevice(Device(device_id))

    @staticmethod
    def from_array(array):
        if isinstance(array, ndarray) and array.device is not None:
            return GpuDevice(array.device)
        return None

    def __eq__(self, other):
        return isinstance(other, GpuDevice) and other.device == self.device

    def __repr__(self):
        return '<{} (cupy):{}>'.format(
            self.__class__.__name__, self.device.id)

    def __str__(self):
        return '@cupy:{}'.format(self.device.id)

    @property
    def xp(self):
        return cupy

    @property
    def supported_array_types(self):
        return (ndarray,)

    def create_context(self):
        # Creates a new cuda.Device instance because a single cuda.Device
        # instance cannot be used across threads.
        return Device(self.device.id)

    def send_array(self, array):
        return _array_to_gpu(array, self.device, None)

    def use(self):
        self.device.use()


# ------------------------------------------------------------------------------
# Global states
# ------------------------------------------------------------------------------
def get_device_from_id(device_id):
    # type: (tp.Optional[int]) -> Device
    """Gets the device from an ID integer.

    Args:
        device_id (int or None): The ID of the device which this function
            returns.
    """
    if device_id is not None:
        if device_id >= 0:
            check_cuda_available()
            return Device(int(device_id))
    return DummyDevice


def get_device_from_array(*arrays):
    # type: (*ndarray) -> Device
    """Gets the device from a list of CuPy array or a single CuPy array.

    .. deprecated:: v6.0.0

        This API is deprecated. Please use
        :func:`~chainer.backend.get_device_from_array` instead.

    The device on which the given CuPy array reside is returned.

    .. note::

        This method only recognizes :class:`cupy.ndarray`\\ s in arguments.
        Especially note that, unlike :func:`get_array_module`, this method
        does not recognize :class:`~chainer.Variable` objects.
        If you need to get device from the :class:`~chainer.Variable` instance
        ``v``, you need to use ``get_device_from_array(v.array)``.

    Args:
        arrays (:class:`cupy.ndarray` or list of :class:`cupy.ndarray`):
            A CuPy array which this function returns the device corresponding
            to. If a list of :class:`cupy.ndarray`\\ s are given, it returns
            the first device object of an array in the list.
    """
    for array in arrays:
        if isinstance(array, ndarray) and array.device is not None:
            return array.device
    return DummyDevice


def get_device(*args):
    """Gets the device from a device object, an ID integer or an array object.

    .. note::

        This API is deprecated since v3.0.0. Please use
        :func:`~chainer.backends.cuda.get_device_from_id`
        or :func:`~chainer.backends.cuda.get_device_from_array` instead.

    This is a convenient utility to select a correct device if the type of
    ``arg`` is unknown (i.e., one can use this function on arrays that may be
    on CPU or GPU). The returned device object supports the context management
    protocol of Python for the *with* statement.

    Args:
        args: Values to specify a GPU device. The first device object, integer
            or :class:`cupy.ndarray` object is used to select a device.
            If it is a device object, it is returned. If it is an integer,
            the corresponding device is returned. If it is a CuPy array,
            the device on which this array reside is returned. If any
            arguments are neither integers nor CuPy arrays, a dummy device
            object representing CPU is returned.

    Returns:
        Device object specified by given ``args``.

    .. seealso::
       See :class:`cupy.cuda.Device` for the device selection not by arrays.

    """
    warnings.warn('get_device is deprecated. Please use get_device_from_id or'
                  ' get_device_from_array instead.', DeprecationWarning)
    return _get_cuda_device(*args)


def _get_cuda_device(*args):
    # Returns cuda.Device or DummyDevice.
    for arg in args:
        if type(arg) is not bool and isinstance(arg, _integer_types):
            check_cuda_available()
            return Device(arg)
        if isinstance(arg, ndarray):
            if arg.device is None:
                continue
            return arg.device
        if available and isinstance(arg, Device):
            return arg

    # NOTE: This function returns DummyDevice for both NumPy and ChainerX
    return DummyDevice


def _get_device_or_current(device):
    # type: (tp.Optional[types.CudaDeviceSpec]) -> Device

    # Returns cuda.Device.
    # - If cuda.Device instance, it's returned intact.
    # - If None, the current device is returned.
    # - If non-negative integer, cuda.Device is returned.
    # - Otherwise: error.
    if device is None:
        return cuda.Device()
    if isinstance(device, Device):
        return device
    if not (isinstance(device, _integer_types) and device >= 0):
        raise ValueError('Invalid CUDA device specifier: {}'.format(device))
    return cuda.Device(int(device))


# ------------------------------------------------------------------------------
# cupy.ndarray allocation and copy
# ------------------------------------------------------------------------------

def to_gpu(array, device=None, stream=None):
    """Copies the given CPU array to the specified device.

    Args:
        array (*array*, None, list or tuple):
            Array or arrays to be sent to GPU.
        device: CUDA device specifier. If ``None`` or :data:`cuda.DummyDevice`,
            the arrays will be copied to the current CUDA device.
        stream (~cupy.cuda.Stream): *(deprecated since v3.0.0)*
            CUDA stream. If not ``None``, the copy runs asynchronously.

    Returns:
        cupy.ndarray, list or tuple: Array or arrays on GPU.

        If some of the arrays are already on GPU, then this function just
        returns those arrays without performing any copy.

        If input arrays include `None`, it is returned as `None` as is.

    """
    if stream is not None:
        warnings.warn(
            'The stream option is deprecated in chainer.backends.cuda.to_gpu. '
            'Please remove it.', DeprecationWarning)

    check_cuda_available()
    if device is DummyDevice:
        device = cuda.Device()
    else:
        device = _get_device_or_current(device)

    return _backend._convert_arrays(
        array, lambda arr: _array_to_gpu(arr, device, stream))


def _array_to_gpu(array, device, stream):
    if array is None:
        return None

    if isinstance(array, chainerx.ndarray):
        # TODO(niboshi): Update this logic once both CuPy and ChainerX support
        # the array interface.
        if array.device.backend.name == 'cuda':
            # Convert to cupy.ndarray on the same device as source array
            array = cupy.ndarray(
                array.shape,
                array.dtype,
                cupy.cuda.MemoryPointer(
                    cupy.cuda.UnownedMemory(
                        array.data_ptr + array.offset,
                        array.data_size,
                        array,
                        array.device.index),
                    0),
                strides=array.strides)
        else:
            array = chainerx.to_numpy(array)
    elif isinstance(array, (numpy.number, numpy.bool_)):
        array = numpy.asarray(array)
    elif isinstance(array, intel64.mdarray):
        array = numpy.asarray(array)

    if isinstance(array, ndarray):
        if array.device == device:
            return array
        is_numpy = False
    elif isinstance(array, numpy.ndarray):
        is_numpy = True
    else:
        raise TypeError(
            'The array sent to gpu must be an array or a NumPy scalar.'
            '\nActual type: {0}.'.format(type(array)))

    if stream is not None:
        with device:
            with stream:
                if is_numpy:
                    return cupy.asarray(array)
                # Need to make a copy when an array is copied to another device
                return cupy.array(array, copy=True)

    with device:
        if is_numpy:
            return cupy.asarray(array)
        # Need to make a copy when an array is copied to another device
        return cupy.array(array, copy=True)


def to_cpu(array, stream=None):
    """Copies the given GPU array to host CPU.

    Args:
        array (*array*, None, list or tuple):
            Array or arrays to be sent to CPU.
        stream (cupy.cuda.Stream): CUDA stream.

    Returns:
        numpy.ndarray, list or tuple: Array on CPU.

        If some of the arrays are already on CPU, then this function just
        returns those arrays without performing any copy.

        If input arrays include `None`, it is returned as `None` as is.

    """
    return _backend._convert_arrays(
        array, lambda arr: _array_to_cpu(arr, stream))


def _array_to_cpu(array, stream):
    if array is None:
        return None
    if isinstance(array, ndarray):
        check_cuda_available()
        with get_device_from_array(array):
            return array.get(stream)
    return _cpu._array_to_cpu(array)


def copy(array, out=None, out_device=None, stream=None):
    """Copies a :class:`cupy.ndarray` object using the default stream.

    This function can copy the device array to the destination array on another
    device.

    Args:
        array (cupy.ndarray): Array to be copied.
        out (cupy.ndarray): Destination array.
            If it is not ``None``, then ``out_device`` argument is ignored.
        out_device: Destination device specifier. Actual device object is
            obtained by passing this value to :func:`get_device`.
        stream (cupy.cuda.Stream): CUDA stream.

    Returns:
        cupy.ndarray: Copied array.

        If ``out`` is not specified, then the array is allocated on the device
        specified by ``out_device`` argument.

    """
    # TODO(niboshi): Update docstring not to mention deprecated `get_device`
    check_cuda_available()
    assert stream is None  # TODO(beam2d): FIX IT

    if out is None:
        if out_device is None:
            out_device = array
        with chainer.get_device(out_device):
            out = cupy.empty_like(array)

    with get_device_from_array(array):
        cupy.copyto(out, array)

    return out


# ------------------------------------------------------------------------------
# Function result memoization
# ------------------------------------------------------------------------------
def memoize(for_each_device=False):
    """Makes a function memoizing the result for each argument and device.

    This is a similar version of :func:`cupy.memoize`. The difference is that
    this function can be used in the global scope even if CUDA is not
    available. In such case, this function does nothing.

    .. note::
       This decorator acts as a dummy if CUDA is not available. It cannot be
       used for general purpose memoization even if ``for_each_device`` is set
       to False.

    """
    if available:
        return cupy.memoize(for_each_device)

    def dummy_decorator(f):
        @functools.wraps(f)
        def ret(*args, **kwargs):
            return f(*args, **kwargs)
        return ret
    return dummy_decorator


def clear_memo():
    """Clears the memoized results for all functions decorated by memoize.

    This function works like :func:`cupy.clear_memo` as a counterpart for
    :func:`chainer.backends.cuda.memoize`. It can be used even if CUDA is
    not available. In such a case, this function does nothing.

    """
    if available:
        cupy.clear_memo()


# ------------------------------------------------------------------------------
# Kernel definition utility
# ------------------------------------------------------------------------------
@memoize()
def elementwise(in_params, out_params, operation, name, **kwargs):
    """Creates an elementwise kernel function.

    This function uses :func:`~chainer.backends.cuda.memoize` to cache the
    kernel object, i.e. the resulting kernel object is cached for each argument
    combination and CUDA device.

    The arguments are the same as those for
    :class:`cupy.ElementwiseKernel`, except that the ``name`` argument is
    mandatory.

    """
    check_cuda_available()
    return cupy.ElementwiseKernel(
        in_params, out_params, operation, name, **kwargs)


@memoize()
def reduce(in_params, out_params, map_expr, reduce_expr, post_map_expr,
           identity, name, **kwargs):
    """Creates a global reduction kernel function.

    This function uses :func:`~chainer.backends.cuda.memoize` to cache the
    resulting kernel object, i.e. the resulting kernel object is cached for
    each argument combination and CUDA device.

    The arguments are the same as those for
    :class:`cupy.ReductionKernel`, except that the ``name`` argument is
    mandatory.

    """
    check_cuda_available()
    return cupy.ReductionKernel(
        in_params, out_params, map_expr, reduce_expr, post_map_expr,
        identity, name, **kwargs)


@memoize()
def raw(code, name, *args, **kwargs):
    """Creates a raw kernel function.

    This function uses :func:`~chainer.backends.cuda.memoize` to cache the
    resulting kernel object, i.e. the resulting kernel object is cached for
    each argument combination and CUDA device.

    The arguments are the same as those for :class:`cupy.RawKernel`.

    """
    check_cuda_available()
    return cupy.RawKernel(code, name, *args, **kwargs)


# ------------------------------------------------------------------------------
# numpy/cupy compatible coding
# ------------------------------------------------------------------------------
def get_array_module(*args):
    """Gets an appropriate one from :mod:`numpy` or :mod:`cupy`.

    This is almost equivalent to :func:`cupy.get_array_module`. The differences
    are that this function can be used even if CUDA is not available and that
    it will return their data arrays' array module for
    :class:`~chainer.Variable` arguments.

    .. deprecated:: v5.0.0

        This API is deprecated. Please use
        :func:`~chainer.backend.get_array_module` instead.

    Args:
        args: Values to determine whether NumPy or CuPy should be used.

    Returns:
        module: :mod:`cupy` or :mod:`numpy` is returned based on the types of
        the arguments.

    """
    return chainer.backend.get_array_module(*args)


def get_max_workspace_size():
    """Gets the workspace size for cuDNN.

    Check "cuDNN Library User Guide" for detail.

    Returns:
        int: The workspace size for cuDNN.

    """
    # To avoid error on no cuDNN environment
    if cudnn_enabled:
        return cudnn.get_max_workspace_size()
    return 0


def set_max_workspace_size(size):
    """Sets the workspace size for cuDNN.

    Check "cuDNN Library User Guide" for detail.

    Args:
        size: The workspace size for cuDNN.

    """
    # To avoid error on no cuDNN environment
    if cudnn_enabled:
        cudnn.set_max_workspace_size(size)


def fuse(*args, **kwargs):
    """Function fusing decorator.

    It calls :func:`cupy.fuse` when CuPy is available to make fused function
    and does nothing otherwise.

    .. seealso::
       :func:`cupy.fuse`

    """
    if available:
        return cupy.fuse(*args, **kwargs)
    elif len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return args[0]
    else:
        return lambda f: f


# ------------------------------------------------------------------------------
# cuDNN
# ------------------------------------------------------------------------------
_SHOULD_USE_CUDNN = {
    '==always': {'always': True, 'auto': False, 'never': False},
    '>=auto':   {'always': True, 'auto': True,  'never': False},
}


_cudnn_version = cuda.cudnn.getVersion() if cudnn_enabled else -1


def should_use_cudnn(level, lowest_version=0):
    """Determines if we should use cuDNN.

    This function checks ``chainer.config.use_cudnn``,
    ``chainer.backends.cuda.cudnn_enabled``, and the cuDNN version. Note that
    ``cudnn_enabled`` flag is fixed at loading of :mod:`chainer` module.

    Args:
        level (str): cuDNN use level. It must be either ``'==always'`` or
            ``'>=auto'``. ``'==always'`` indicates that the ``use_cudnn``
            config must be ``'always'`` to use cuDNN.
        lowest_version (int): Required lowest cuDNN version. It must be
            non-negative.

    Returns:
        bool: ``True`` if the caller should use cuDNN.

    """
    if _cudnn_version < lowest_version:
        return False

    if level not in _SHOULD_USE_CUDNN:
        raise ValueError('invalid cuDNN use level: %s '
                         '(must be either of "==always" or ">=auto")' %
                         repr(level))
    flags = _SHOULD_USE_CUDNN[level]

    use_cudnn = config.use_cudnn
    if use_cudnn not in flags:
        raise ValueError('invalid use_cudnn configuration: %s '
                         '(must be either of "always", "auto", or "never")' %
                         repr(use_cudnn))
    return flags[use_cudnn]


_tensor_core_flag = {'always': True, 'auto': None, 'never': False}


def should_use_cudnn_tensor_core(dtype):
    """Determines if Tensor Core should be used.

    Args:
        dtype (numpy.dtype): data type of input tensor.

    Returns:
        bool: ``True`` if Tensor Core should be used.
    """

    use_cudnn_tensor_core = config.use_cudnn_tensor_core
    if use_cudnn_tensor_core not in _tensor_core_flag:
        raise ValueError('invalid use_cudnn_tensor_core configuration: %s '
                         '(must be either of "always", "auto", or "never")' %
                         repr(use_cudnn_tensor_core))
    use_tensor_core = _tensor_core_flag[use_cudnn_tensor_core]
    if use_tensor_core is None:
        use_tensor_core = cudnn.is_tensor_core_available(dtype)
    return use_tensor_core


# ------------------------------------------------------------------------------
# cupy.cudnn utility
# ------------------------------------------------------------------------------

def get_cudnn_dropout_states():
    if not cudnn_enabled:
        raise RuntimeError('cuDNN is not enabled.')

    thread_id = threading.current_thread().ident
    return get_cudnn_dropout_states_core(thread_id)


_dropout_states_count = itertools.count()


@memoize(for_each_device=True)
def get_cudnn_dropout_states_core(thread_id):
    states_id = next(_dropout_states_count)
    seed = os.getenv('CHAINER_SEED')
    if seed is None:
        try:
            seed_str = binascii.hexlify(os.urandom(8))
            seed = numpy.uint64(int(seed_str, 16))
        except NotImplementedError:
            seed = numpy.uint64(time.clock() * 1000000)
    else:
        seed = numpy.uint64(seed)

    seed += numpy.uint64(states_id)
    return cudnn.DropoutStates(None, seed)
