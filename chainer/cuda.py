"""Device, context and memory management on CuPy.

Chainer uses `CuPy <https://cupy.chainer.org/>`_ (with very thin wrapper)
to exploit the speed of GPU computation. Following modules and classes defined
in CuPy are imported to :mod:`chainer.cuda` module for convenience (refer to
this table when reading chainer's source codes).

============================ =================================
 imported name                original name
============================ =================================
 ``chainer.cuda.cupy``        :mod:`cupy`
 ``chainer.cuda.ndarray``     :class:`cupy.ndarray`
 ``chainer.cuda.cupy.cuda``   :mod:`cupy.cuda`
 ``chainer.cuda.Device``      :class:`cupy.cuda.Device`
 ``chainer.cuda.Event``       :class:`cupy.cuda.Event`
 ``chainer.cuda.Stream``      :class:`cupy.cuda.Stream`
============================ =================================

Chainer replaces the default allocator of CuPy by its memory pool
implementation. It enables us to reuse the device memory over multiple
forward/backward computations, and temporary arrays for consecutive elementwise
operations.
"""

import functools
import os
import warnings

import numpy
import six

import chainer


available = False
cudnn_enabled = False

try:
    import cupy
    from cupy import cuda  # NOQA
    from cupy.cuda import cublas  # NOQA

    from cupy import ndarray  # NOQA

    from cupy.cuda import Device  # NOQA
    from cupy.cuda import Event  # NOQA
    from cupy.cuda import Stream  # NOQA

    from . import cuda_fusion as fusion  # NOQA

    available = True
except Exception as e:
    _resolution_error = e
    fusion = numpy

    class ndarray(object):
        pass  # for type testing

if available:
    _cudnn_disabled_by_user = int(os.environ.get('CHAINER_CUDNN', '1')) == 0
    try:
        import cupy.cudnn
        cudnn = cupy.cudnn
        cudnn_enabled = not _cudnn_disabled_by_user
    except Exception as e:
        _resolution_error = e


def check_cuda_available():
    """Checks if CUDA is available.

    When CUDA is correctly set up, nothing happens.
    Otherwise it raises ``RuntimeError``.
    """
    if not available:
        msg = ('CUDA environment is not correctly set up\n'
               '(see https://github.com/pfnet/chainer#installation).')
        msg += str(_resolution_error)
        raise RuntimeError(msg)
    if (not cudnn_enabled and
            not _cudnn_disabled_by_user and
            not getattr(check_cuda_available, '_already_warned', False)):
        warnings.warn(
            'cuDNN is not enabled.\n'
            'Please reinstall chainer after you install cudnn\n'
            '(see https://github.com/pfnet/chainer#installation).')
        check_cuda_available._already_warned = True


class DummyDeviceType(object):

    """Dummy device class that does nothing with cupy.cuda.Device interface.

    This class is used to represent CPU device.

    """

    id = -1

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
    memory_pool = cuda.MemoryPool()
    cuda.set_allocator(memory_pool.malloc)
    pinned_memory_pool = cuda.PinnedMemoryPool()
    cuda.set_pinned_memory_allocator(pinned_memory_pool.malloc)


if six.PY2:
    try:
        from future.types.newint import newint as _newint
        _integer_types = six.integer_types + (_newint,)
    except ImportError:
        _integer_types = six.integer_types
else:
    _integer_types = six.integer_types


# ------------------------------------------------------------------------------
# Global states
# ------------------------------------------------------------------------------
def get_device_from_id(device_id):
    """Gets the device from an ID integer.

    Args:
        device_id (int or None): The ID of the device which this function
            returns.
    """
    if device_id is not None:
        check_cuda_available()
        return Device(device_id)
    else:
        return DummyDevice


def get_device_from_array(*arrays):
    """Gets the device from a list of CuPy array or a single CuPy array.

    The device on which the given CuPy array reside is returned.

    Args:
        array (cupy.ndarray or list of cupy.ndarray):
            A CuPy array which this function returns the device corresponding
            to. If a list of :class:`cupy.ndarray` s are given, it returns
            the first device object of an array in the list.
    """
    for array in arrays:
        if isinstance(array, ndarray) and array.device is not None:
            return array.device
    return DummyDevice


def get_device(*args):
    """Gets the device from a device object, an ID integer or an array object.

    .. note::

        This API is deprecated. Please use
        :func:`~chainer.cuda.get_device_from_id`
        or :func:`~chainer.cuda.get_device_from_array` instead.

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
    return _get_device(*args)


def _get_device(*args):
    for arg in args:
        if type(arg) in _integer_types:
            check_cuda_available()
            return Device(arg)
        if isinstance(arg, ndarray):
            if arg.device is None:
                continue
            return arg.device
        if available and isinstance(arg, Device):
            return arg

    return DummyDevice


# ------------------------------------------------------------------------------
# cupy.ndarray allocation and copy
# ------------------------------------------------------------------------------

def to_gpu(array, device=None, stream=None):
    """Copies the given CPU array to specified device.

    Args:
        array: Array to be sent to GPU.
        device: Device specifier.
        stream (cupy.cuda.Stream): CUDA stream. If not ``None``, the copy runs
            asynchronously.

    Returns:
        cupy.ndarray: Array on GPU.

        If ``array`` is already on GPU, then this function just returns
        ``array`` without performing any copy. Note that this function does not
        copy :class:`cupy.ndarray` into specified device.

    """
    check_cuda_available()
    with _get_device(device):
        array_dev = get_device_from_array(array)
        if array_dev.id == cupy.cuda.device.get_device_id():
            return array

        if stream is not None:
            ret = cupy.empty_like(array)
            mem = None
            if array_dev.id == -1:
                # cpu to gpu
                mem = cupy.cuda.alloc_pinned_memory(array.nbytes)
                src = numpy.frombuffer(
                    mem, array.dtype, array.size).reshape(array.shape)
                src[...] = array
                ret.set(src, stream)
            else:
                # gpu to gpu
                with array_dev:
                    src = array.copy()
                    event = cupy.cuda.Event()
                    event.record()
                stream.wait_event(event)
                ret.data.copy_from_device_async(src.data, src.nbytes, stream)

            # to hold a reference until the end of the asynchronous memcpy
            stream.add_callback(lambda *x: None, (src, mem, ret))

            return ret

        if array_dev.id == -1:
            return cupy.asarray(array)

        # Need to make a copy when an array is copied to another device
        return cupy.array(array, copy=True)


def to_cpu(array, stream=None):
    """Copies the given GPU array to host CPU.

    Args:
        array: Array to be sent to CPU.
        stream (cupy.cuda.Stream): CUDA stream.

    Returns:
        numpy.ndarray: Array on CPU.

        If given ``array`` is already on CPU, then this function just returns
        ``array`` without performing any copy.

    """
    if isinstance(array, ndarray):
        check_cuda_available()
        with get_device_from_array(array):
            return array.get(stream)
    elif isinstance(array, numpy.ndarray):
        return array
    else:
        raise TypeError(
            'The array sent to cpu must be numpy.ndarray or cupy.ndarray.'
            '\nActual type: {0}.'.format(type(array)))


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
    check_cuda_available()
    assert stream is None  # TODO(beam2d): FIX IT

    if out is None:
        if out_device is None:
            out_device = array
        with _get_device(out_device):
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
    :func:`chainer.cuda.memoize`. It can be used even if CUDA is not available.
    In such a case, this function does nothing.

    """
    if available:
        cupy.clear_memo()


# ------------------------------------------------------------------------------
# Kernel definition utility
# ------------------------------------------------------------------------------
@memoize(for_each_device=True)
def elementwise(in_params, out_params, operation, name, **kwargs):
    """Creates an elementwise kernel function.

    This function uses :func:`~chainer.cuda.memoize` to cache the
    kernel object, i.e. the resulting kernel object is cached for each argument
    combination and CUDA device.

    The arguments are the same as those for
    :class:`cupy.ElementwiseKernel`, except that the ``name`` argument is
    mandatory.

    """
    check_cuda_available()
    return cupy.ElementwiseKernel(
        in_params, out_params, operation, name, **kwargs)


@memoize(for_each_device=True)
def reduce(in_params, out_params, map_expr, reduce_expr, post_map_expr,
           identity, name,  **kwargs):
    """Creates a global reduction kernel function.

    This function uses :func:`~chainer.cuda.memoize` to cache the resulting
    kernel object, i.e. the resulting kernel object is cached for each argument
    combination and CUDA device.

    The arguments are the same as those for
    :class:`cupy.ReductionKernel`, except that the ``name`` argument is
    mandatory.

    """
    check_cuda_available()
    return cupy.ReductionKernel(
        in_params, out_params, map_expr, reduce_expr, post_map_expr,
        identity, name, **kwargs)


# ------------------------------------------------------------------------------
# numpy/cupy compatible coding
# ------------------------------------------------------------------------------
def get_array_module(*args):
    """Gets an appropriate one from :mod:`numpy` or :mod:`cupy`.

    This is almost equivalent to :func:`cupy.get_array_module`. The differences
    are that this function can be used even if CUDA is not available and that
    it will return their data arrays' array module for
    :class:`~chainer.Variable` arguments.

    Args:
        args: Values to determine whether NumPy or CuPy should be used.

    Returns:
        module: :mod:`cupy` or :mod:`numpy` is returned based on the types of
        the arguments.

    """
    if available:
        args = [arg.data if isinstance(arg, chainer.variable.Variable) else arg
                for arg in args]
        return cupy.get_array_module(*args)
    else:
        return numpy


_max_workspace_size = 8 * 1024 * 1024


def get_max_workspace_size():
    """Gets the workspace size for cuDNN.

    Check "cuDNN Library User Guide" for detail.

    Returns:
        int: The workspace size for cuDNN.

    """
    return _max_workspace_size


def set_max_workspace_size(size):
    """Sets the workspace size for cuDNN.

    Check "cuDNN Library User Guide" for detail.

    Args:
        size: The workspace size for cuDNN.

    """
    global _max_workspace_size
    _max_workspace_size = size
