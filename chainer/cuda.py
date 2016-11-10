"""Device, context and memory management on CuPy.

Chainer uses CuPy (with very thin wrapper) to exploit the speed of GPU
computation. Following modules and classes are imported to :mod:`cuda`
module for convenience (refer to this table when reading chainer's source
codes).

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


available = False
cudnn_enabled = False

try:
    import cupy
    from cupy.core import fusion
    import cupy.cuda
    import cupy.cuda.cublas

    cuda = cupy.cuda
    cublas = cuda.cublas

    ndarray = cupy.ndarray
    Device = cuda.Device
    Event = cuda.Event
    Stream = cuda.Stream

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


def init(arg=None):
    warnings.warn(
        'chainer.cuda.init is deprecated. You need to call nothing to '
        'initialize your environment. Call chainer.cuda.check_cuda_available '
        'to check availability of CUDA.',
        DeprecationWarning)
    check_cuda_available()


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
def get_device(*args):
    """Gets the device from a device object, an ID integer or an array object.

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
    with get_device(device):
        array_dev = get_device(array)
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
        with get_device(array):
            return array.get(stream)
    elif isinstance(array, numpy.ndarray):
        return array
    else:
        raise TypeError(
            'The array sent to cpu must be numpy.ndarray or cupy.ndarray.'
            '\nActual type: {0}.'.format(type(array)))


def empty(shape, dtype=numpy.float32):
    """Creates an uninitialized :class:`cupy.ndarray` object.

    Args:
        shape (tuple of ints): The shape of array.
        dtype (numpy.dtype): Element type.

    Returns:
        cupy.ndarray: Uninitialized GPU array allocated by the memory pool.

    """
    warnings.warn(
        'chainer.cuda.empty is deprecated. Use cupy.empty instead.',
        DeprecationWarning)
    check_cuda_available()
    return cupy.empty(shape, dtype)


def full(shape, fill_value, dtype=numpy.float32, stream=None):
    """Creates a constant-filled :class:`cupy.ndarray` object.

    Args:
        shape (tuple of ints): The shape of array.
        fill_value: Constant to fill the array by.
        dtype (numpy.dtype): Element type.
        stream (cupy.cuda.Stream): CUDA stream.

    Returns:
        cupy.ndarray: Constant-filled GPU array allocated by the memory pool.

    """
    warnings.warn(
        'chainer.cuda.full is deprecated. Use cupy.full instead.',
        DeprecationWarning)
    check_cuda_available()
    assert stream is None
    return cupy.full(shape, fill_value, dtype=dtype)


def zeros(shape, dtype=numpy.float32, stream=None):
    """Creates a zero-filled :class:`cupy.ndarray` object.

    This function is equivalent to ``full(shape, 0, dtype, stream)``.

    """
    warnings.warn(
        'chainer.cuda.zeros is deprecated. Use cupy.zeros instead.',
        DeprecationWarning)
    check_cuda_available()
    assert stream is None
    return cupy.zeros(shape, dtype=dtype)


def ones(shape, dtype=numpy.float32, stream=None):
    """Creates a zero-filled :class:`cupy.ndarray` object.

    This function is equivalent to ``full(shape, 1, dtype, stream)``.

    """
    warnings.warn(
        'chainer.cuda.ones is deprecated. Use cupy.ones instead.',
        DeprecationWarning)
    check_cuda_available()
    assert stream is None
    return cupy.ones(shape, dtype=dtype)


def empty_like(array):
    """Creates an uninitialized GPU array like the given one.

    Args:
        array (cupy.ndarray or numpy.ndarray): Base array.

    Returns:
        cupy.ndarray: GPU array of the same shape and dtype as `array`.

    """
    warnings.warn(
        'chainer.cuda.empty_like is deprecated. Use cupy.empty_like instead.',
        DeprecationWarning)
    check_cuda_available()
    if isinstance(array, cupy.ndarray):
        return cupy.empty_like(array)
    return cupy.empty(array.shape, dtype=array.dtype)


def full_like(array, fill_value, stream=None):
    """Creates a constant-filled :class:`cupy.ndarray` object like the given array.

    Args:
        array (cupy.ndarray or numpy.ndarray): Base array.
        fill_value: Constant value to fill the array by.
        stream (cupy.cuda.Stream): CUDA stream.

    Returns:
        cupy.ndarray: Constant-filled array.

    """
    warnings.warn(
        'chainer.cuda.full_like is deprecated. Use cupy.full_like instead.',
        DeprecationWarning)
    check_cuda_available()
    assert stream is None
    if isinstance(array, cupy.ndarray):
        return cupy.full_like(array, fill_value)
    return cupy.full(array.shape, fill_value, dtype=array.dtype)


def zeros_like(array, stream=None):
    """Creates a zero-filled :class:`cupy.ndarray` object like the given array.

    Args:
        array (cupy.ndarray or numpy.ndarray): Base array.
        stream (cupy.cuda.Stream): CUDA stream.

    Returns:
        cupy.ndarray: Zero-filled array.

    """
    warnings.warn(
        'chainer.cuda.zeros_like is deprecated. Use cupy.zeros_like instead.',
        DeprecationWarning)
    check_cuda_available()
    assert stream is None
    if isinstance(array, cupy.ndarray):
        return cupy.zeros_like(array)
    return cupy.zeros(array.shape, dtype=array.dtype)


def ones_like(array, stream=None):
    """Creates a one-filled :class:`cupy.ndarray` object like the given array.

    Args:
        array (cupy.ndarray or numpy.ndarray): Base array.
        stream (cupy.cuda.Stream): CUDA stream.

    Returns:
        cupy.ndarray: One-filled array.

    """
    warnings.warn(
        'chainer.cuda.ones_like is deprecated. Use cupy.ones_like instead.',
        DeprecationWarning)
    check_cuda_available()
    assert stream is None
    if isinstance(array, cupy.ndarray):
        return cupy.ones_like(array)
    return cupy.ones(array.shape, dtype=array.dtype)


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
        with get_device(out_device):
            out = cupy.empty_like(array)

    with get_device(array):
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

    This is almost equivalent to :func:`cupy.get_array_module`. The only
    difference is that this function can be used even if CUDA is not available.

    Args:
        args: Values to determine whether NumPy or CuPy should be used.

    Returns:
        module: :mod:`cupy` or :mod:`numpy` is returned based on the types of
        the arguments.

    """
    if available:
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

# ------------------------------------------------------------------------------
# numpy/cupy compatible ufunc
# ------------------------------------------------------------------------------


class Fusion(object):

    def __init__(self, func, input_num, reduce_f, post_map):
        self.func = func
        self.input_num = input_num
        self.reduce = reduce_f
        self.post_map = post_map

    def __call__(self, *args, **kwargs):
        if self.reduce is None:
            return self.func(*args)
        else:
            axis = kwargs['axis'] if 'axis' in kwargs else None
            reduce_f = self.reduce
            if axis is None:
                return self.post_map(reduce_f(self.func(*args)))
            else:
                return self.post_map(reduce_f(self.func(*args), axis=axis))


if available:
    fuse = fusion.fuse
    sqrt_fixed = fusion.sqrt_fixed

else:
    def fuse(input_num=None, reduce=None, post_map=lambda x: x):
        """Function fusing decorator.

        This decorator can be used to define an elementwise or reduction kernel
        more easily than `ElementwiseKernel` class or `ReductionKernel` class.

        This decorator makes `Fusion` class from the given function.

        Args:
            input_num (int): Number of input arguments of the given function.
            immutable_num (int): Number of immutable input arguments of
                the given function.
            reduce (function): The reduce function which is applied after
                pre-mapping step. If not assigned, reduction step is skipped.
            post_map (function): Mapping function for reduced values.
                If not assigned, post_map step is skipped.
    """
        return lambda f: Fusion(f, input_num, reduce, post_map)

    sqrt_fixed = numpy.sqrt


where = fusion.where
clip = fusion.clip

bitwise_and = fusion.bitwise_and
bitwise_or = fusion.bitwise_or
bitwise_xor = fusion.bitwise_xor
invert = fusion.invert
left_shift = fusion.left_shift
right_shift = fusion.right_shift

greater = fusion.greater
greater_equal = fusion.greater_equal
less = fusion.less
less_equal = fusion.less_equal
equal = fusion.equal
not_equal = fusion.not_equal

isfinite = fusion.isfinite
isinf = fusion.isinf
isnan = fusion.isnan

logical_and = fusion.logical_and
logical_or = fusion.logical_or
logical_not = fusion.logical_not
logical_xor = fusion.logical_xor

sin = fusion.sin
cos = fusion.cos
tan = fusion.tan
arcsin = fusion.arcsin
arccos = fusion.arccos
arctan = fusion.arctan
hypot = fusion.hypot
deg2rad = fusion.deg2rad
rad2deg = fusion.rad2deg
degrees = fusion.degrees
radians = fusion.radians

sinh = fusion.sinh
cosh = fusion.cosh
tanh = fusion.tanh
arcsinh = fusion.arcsinh
arccosh = fusion.arccosh
arctanh = fusion.arctanh

rint = fusion.rint
floor = fusion.floor
ceil = fusion.ceil
trunc = fusion.trunc

exp = fusion.exp
expm1 = fusion.expm1
exp2 = fusion.exp2
log = fusion.log
log10 = fusion.log10
log2 = fusion.log2
log1p = fusion.log1p
logaddexp = fusion.logaddexp
logaddexp2 = fusion.logaddexp2

signbit = fusion.signbit
copysign = fusion.copysign
ldexp = fusion.ldexp
frexp = fusion.frexp
nextafter = fusion.nextafter

add = fusion.add
reciprocal = fusion.reciprocal
negative = fusion.negative
multiply = fusion.multiply
divide = fusion.divide
power = fusion.power
subtract = fusion.subtract
true_divide = fusion.true_divide
floor_divide = fusion.floor_divide
fmod = fusion.fmod
mod = fusion.mod
modf = fusion.modf
remainder = fusion.remainder

clip = fusion.clip
sqrt = fusion.sqrt
square = fusion.square
absolute = fusion.absolute
abs = fusion.abs
sign = fusion.sign
maximum = fusion.maximum
minimum = fusion.minimum
fmax = fusion.fmax
fmin = fusion.fmin

all = fusion.all
any = fusion.any
sum = fusion.sum
prod = fusion.prod
amax = fusion.amax
amin = fusion.amin
