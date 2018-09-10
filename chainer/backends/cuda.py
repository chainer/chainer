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

import functools
import os
import warnings

import numpy
import six

import chainer
from chainer.backends import intel64
from chainer.configuration import config

available = False
cudnn_enabled = False

try:
    import cupy
    from cupy import cuda  # NOQA
    from cupy.cuda import cublas  # NOQA
    import cupyx  # NOQA

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

    # for `xp is cuda.cupy` to always work
    cupy = object()

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
            '#install-cupy-with-cudnn-and-nccl).')
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
    # This is for backward compatibility
    memory_pool = cupy.get_default_memory_pool()
    pinned_memory_pool = cupy.get_default_pinned_memory_pool()


_integer_types = six.integer_types + (numpy.integer,)


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

        This API is deprecated. Please use
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
    return _get_device(*args)


def _get_device(*args):
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

    return DummyDevice


# ------------------------------------------------------------------------------
# cupy.ndarray allocation and copy
# ------------------------------------------------------------------------------

def to_gpu(array, device=None, stream=None):
    """Copies the given CPU array to the specified device.

    Args:
        array (*array*, None, list or tuple):
            Array or arrays to be sent to GPU.
        device: Device specifier.
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
    with _get_device(device) as device_:
        if isinstance(array, (list, tuple)):
            d = {}
            ret = []
            for arr in array:
                if arr is None:
                    ret.append(None)
                else:
                    arr2 = d.get(id(arr))
                    if arr2 is None:
                        arr2 = _array_to_gpu(arr, device_, stream)
                        d[id(arr)] = arr2
                    ret.append(arr2)
            return type(array)(ret)
        else:
            return _array_to_gpu(array, device_, stream)


def _array_to_gpu(array, device, stream):
    assert device is DummyDevice or isinstance(device, Device)
    if array is None:
        return None

    if isinstance(array, (numpy.number, numpy.bool_)):
        array = numpy.asarray(array)
    elif isinstance(array, intel64.mdarray):
        array = numpy.asarray(array)

    if not isinstance(array, (cupy.ndarray, numpy.ndarray)):
        raise TypeError(
            'The array sent to gpu must be an array or a NumPy scalar.'
            '\nActual type: {0}.'.format(type(array)))

    array_dev = get_device_from_array(array)
    if array_dev.id == cupy.cuda.device.get_device_id():
        return array

    if stream is not None and stream.ptr != 0:
        ret = cupy.empty_like(array)
        if array_dev.id == -1:
            # cpu to gpu
            mem = cupy.cuda.alloc_pinned_memory(array.nbytes)
            src = numpy.frombuffer(
                mem, array.dtype, array.size).reshape(array.shape)
            src[...] = array
            ret.set(src, stream)
            cupy.cuda.pinned_memory._add_to_watch_list(
                stream.record(), mem)
        else:
            # gpu to gpu
            with array_dev:
                src = array.copy()
                event = Stream.null.record()
            stream.wait_event(event)
            ret.data.copy_from_device_async(
                src.data, src.nbytes, stream)

            # to hold a reference until the end of the asynchronous
            # memcpy
            stream.add_callback(lambda *x: None, (src, ret))
        return ret

    if array_dev.id == -1:
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
    if isinstance(array, (list, tuple)):
        d = {}
        ret = []
        for arr in array:
            if arr is None:
                ret.append(None)
            else:
                arr2 = d.get(id(arr))
                if arr2 is None:
                    arr2 = _array_to_cpu(arr, stream)
                    d[id(arr)] = arr2
                ret.append(arr2)
        return type(array)(ret)
    else:
        return _array_to_cpu(array, stream)


def _array_to_cpu(array, stream):
    if array is None:
        return None
    if isinstance(array, ndarray):
        check_cuda_available()
        with get_device_from_array(array):
            return array.get(stream)
    elif isinstance(array, (numpy.number, numpy.bool_)):
        return numpy.asarray(array)
    elif isinstance(array, chainer.get_cpu_array_types()):
        return array
    else:
        raise TypeError(
            'The array sent to cpu must be numpy.ndarray or cupy.ndarray, '
            'or a NumPy scalar.'
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
    :func:`chainer.backends.cuda.memoize`. It can be used even if CUDA is
    not available. In such a case, this function does nothing.

    """
    if available:
        cupy.clear_memo()


# ------------------------------------------------------------------------------
# Kernel definition utility
# ------------------------------------------------------------------------------
@memoize(for_each_device=True)
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


@memoize(for_each_device=True)
def reduce(in_params, out_params, map_expr, reduce_expr, post_map_expr,
           identity, name,  **kwargs):
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
