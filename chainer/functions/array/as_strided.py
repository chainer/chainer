from chainer.backends import cuda
from chainer import function_node
from chainer.utils import type_check

import numpy as np

from six import moves

index_dtype = {t().itemsize: t for t in np.sctypes['int']}


def _byte2step(iterable, itemsize):
    for i in iterable:
        assert i % itemsize == 0
    return tuple([i // itemsize for i in iterable])


def _step2byte(iterable, itemsize):
    return tuple([i * itemsize for i in iterable])


def _maybe_overlapping_memory(shape, strides):
    """Returns bool value indicating the array with such shape and strides
    might have overlapping memory.

    Args:
    shape (tuple of int): The shape of output.
    strides (tuple of int): The strides of output, given in the unit of steps.
    storage_offset (int):
        The offset between the head of allocated memory and the pointer of
        first element, given in the unit of steps.

    Returns:
        bool: Existence of the overlapping memory
    """
    max_ptr_in_slice = 0
    for stride, size in sorted(zip([abs(s) for s in strides], shape)):
        if stride <= max_ptr_in_slice:
            return True
        max_ptr_in_slice += stride * (size - 1)
    return False


def _min_index(shape, strides, storage_offset):
    """Returns the leftest index in the array (in the unit-steps)

    Args:
        shape (tuple of int): The shape of output.
        strides (tuple of int):
            The strides of output, given in the unit of steps.
        storage_offset (int):
            The offset between the head of allocated memory and the pointer of
            first element, given in the unit of steps.

    Returns:
        int: The leftest pointer in the array
    """
    sh_st_neg = [sh_st for sh_st in zip(shape, strides) if sh_st[1] < 0]
    if len(sh_st_neg) == 0:
        return storage_offset
    else:
        return storage_offset + moves.reduce(
            lambda base, sh_st: base + (sh_st[0] - 1) * sh_st[1], sh_st_neg, 0)


def _max_index(shape, strides, storage_offset):
    """Returns the rightest index in the array

    Args:
        shape (tuple of int): The shape of output.
        strides (tuple of int): The strides of output, given in unit-steps.
        storage_offset (int):
            The offset between the head of allocated memory and the pointer of
            first element, given in the unit of steps.

    Returns:
        int: The rightest pointer in the array
    """
    sh_st_pos = [sh_st for sh_st in zip(shape, strides) if sh_st[1] > 0]
    if len(sh_st_pos) == 0:
        return storage_offset
    else:
        return storage_offset + moves.reduce(
            lambda base, sh_st: base + (sh_st[0] - 1) * sh_st[1], sh_st_pos, 0)


def _index_add(augend, indices, addend):
    """Wrapper of :func:`cupyx.scatter_add` and :func:`numpy.add.at`

    Args:
        augend (:class:`numpy.ndarray` or :class:`cupy.ndarray`):
            The array modified in-place.
        indices (:class:`numpy.ndarray` or :class:`cupy.ndarray`):
            The indices of ``augend``. The shape is the same to the ``addend``.
        addend (:class:`numpy.ndarray` or :class:`cupy.ndarray`):
            The array to be added.

    Returns:
        None
    """
    if isinstance(augend, cuda.ndarray):
        cuda.cupyx.scatter_add(augend, indices, addend)
    elif isinstance(augend, np.ndarray):
        np.add.at(augend, indices, addend)


def _get_base_array(array):
    """Get the founder of :class:`numpy.ndarray`.

    Args:
        array (:class:`numpy.ndarray`):
            The view of the base array.

    Returns:
        :class:`numpy.ndarray`:
            The base array.
    """
    base_array_candidate = array
    while base_array_candidate.base is not None:
        base_array_candidate = base_array_candidate.base
    return base_array_candidate


def _stride_array(array, shape, strides, storage_offset):
    """Wrapper of :func:`numpy.lib.stride_tricks.as_strided`.

    .. note:
        ``strides`` and ``storage_offset`` is given in the unit of steps
        instead the unit of bytes. This specification differs from that of
        :func:`numpy.lib.stride_tricks.as_strided`.

    Args:
        array (:class:`numpy.ndarray` of :class:`cupy.ndarray`):
            The base array for the returned view.
        shape (tuple of int):
            The shape of the returned view.
        strides (tuple of int):
            The strides of the returned view, given in the unit of steps.
        storage_offset (int):
            The offset from the leftest pointer of allocated memory to
            the first element of returned view, given in the unit of steps.

    Returns:
        :class:`numpy.ndarray` or :class:`cupy.ndarray`:
            The new view for the base array.
    """

    min_index = _min_index(shape, strides, storage_offset)
    max_index = _max_index(shape, strides, storage_offset)

    strides = _step2byte(strides, array.itemsize)
    storage_offset, = _step2byte((storage_offset,), array.itemsize)

    if min_index < 0:
        raise ValueError('Out of buffer: too small index was specified')

    if isinstance(array, cuda.ndarray):
        pooled_memory = array.data.mem
        if (max_index + 1) * array.itemsize > pooled_memory.size:
            raise ValueError('Out of buffer: too large index was specified')

        memptr = cuda.cupy.cuda.memory.MemoryPointer(pooled_memory,
                                                     storage_offset)
        return cuda.cupy.ndarray(shape, array.dtype, memptr, strides)
    elif isinstance(array, np.ndarray):
        base_array = _get_base_array(array)
        if (max_index + 1) * base_array.itemsize > base_array.nbytes:
            raise ValueError('Out of buffer: too large index was specified')

        return np.ndarray(shape, base_array.dtype, base_array.data,
                          storage_offset, strides)
    else:
        raise TypeError('Only (np|cp).ndarray is accepted')


class TensorGeometry(object):
    def __init__(self, array):
        self.shape = array.shape
        self.strides = _byte2step(array.strides, array.itemsize)
        if isinstance(array, np.ndarray):
            base_array = _get_base_array(array)
            array_ptr = array.__array_interface__['data'][0]
            base_array_ptr = base_array.__array_interface__['data'][0]
            offset_bytes = array_ptr - base_array_ptr
        elif isinstance(array, cuda.ndarray):
            offset_bytes = array.data.ptr - array.data.mem.ptr
        else:
            raise ValueError('only (np|cp).ndarray is supported')
        self.storage_offset, = _byte2step((offset_bytes,), array.itemsize)
        self.itemsize = array.itemsize

    @property
    def ndim(self):
        return len(self.shape)


class AsStrided(function_node.FunctionNode):
    """Transportation of :func:`torch.Tensor.as_strided`.
    While :func:`torch.Tensor.as_strided` does not support nagative strides,
    this implementation does support it.
    """

    def __init__(self, shape, strides, storage_offset=None):
        self.shape = shape
        self.strides = strides
        self.storage_offset = storage_offset
        self.input_geometry = None

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)

    def forward(self, inputs):
        assert len(inputs) > 0

        x = inputs[0]

        self.input_geometry = TensorGeometry(x)

        if self.storage_offset is None:
            self.storage_offset = self.input_geometry.storage_offset

        return _stride_array(x, self.shape, self.strides, self.storage_offset),

    def backward(self, _, grad_outputs):
        """Backward computation which calls :class:`AsStridedGrad`.

        .. note:
            While this implementation is based on *New-Style Function
            Implementation*, the backward computation does not support
            double-backpropagation due to *layout agnostic* algorithm (
            originally named in the note of pytorch).
        """
        return AsStridedGrad(self.input_geometry, self.shape, self.strides,
                             self.storage_offset).apply(grad_outputs)


class AsStridedGrad(function_node.FunctionNode):
    """Backward of :func:`~chainer.functions.as_strided`.
    """

    def __init__(self, input_geometry, shape, strides, storage_offset):
        self.input_geometry = input_geometry
        self.shape = shape
        self.strides = strides
        self.storage_offset = storage_offset

    def forward(self, grads):
        assert len(grads) > 0
        gy = grads[0]

        if gy.dtype not in np.sctypes['float']:
            raise TypeError('Only float is supported for back propagation')

        xp = cuda.get_array_module(gy)
        input_geometry = self.input_geometry
        itemsize = input_geometry.itemsize

        if 0 in input_geometry.shape:
            return xp.zeros(input_geometry.shape)

        #  1. remove redundant axis from input/output
        #  [redundant axis]
        #  axis with shape==0, shape==1 or strides==0
        if 0 in gy.shape:
            return cuda.get_array_module(gy).zeros(input_geometry.shape)
        else:
            out_shape = tuple([self.shape[i] for i in moves.range(gy.ndim) if
                               self.shape[i] != 1 and self.strides[i] != 0])
            out_strides = tuple([self.strides[i] for i in moves.range(gy.ndim)
                                 if self.shape[i] != 1
                                 and self.strides[i] != 0])
            gy = gy.sum(
                tuple([i for i in moves.range(gy.ndim)
                       if self.strides[i] == 0]))
            gy = gy.squeeze()

        out_storage_offset = self.storage_offset

        inp_shape = tuple([input_geometry.shape[i]
                           for i in moves.range(input_geometry.ndim)
                           if input_geometry.shape[i] != 1])
        inp_strides = tuple([input_geometry.strides[i]
                             for i in moves.range(input_geometry.ndim)
                             if input_geometry.shape[i] != 1])
        inp_storage_offset = input_geometry.storage_offset

        #  2. calculate minimum required storage for gradient computation
        inp_min_ptr = _min_index(inp_shape, inp_strides,
                                 input_geometry.storage_offset)
        out_min_ptr = _min_index(out_shape, out_strides, self.storage_offset)
        common_min_ptr = min(inp_min_ptr, out_min_ptr)

        inp_max_ptr = _max_index(inp_shape, inp_strides,
                                 input_geometry.storage_offset)
        out_max_ptr = _max_index(out_shape, out_strides, self.storage_offset)
        common_max_ptr = max(inp_max_ptr, out_max_ptr)

        base_size = (common_max_ptr - common_min_ptr) + 1

        storage = xp.zeros(base_size, dtype=gy.dtype)
        flatten_full_indices = xp.arange(base_size,
                                         dtype=index_dtype[itemsize])

        out_maybe_overlap = _maybe_overlapping_memory(out_shape, out_strides)

        if out_maybe_overlap:
            out_indices = _stride_array(flatten_full_indices, out_shape,
                                        out_strides,
                                        out_storage_offset - common_min_ptr)
            _index_add(storage, out_indices, gy)
        else:
            storage_view = _stride_array(storage, out_shape, out_strides,
                                         out_storage_offset - common_min_ptr)
            storage_view[:] = gy[:]

        inp_maybe_overlap = _maybe_overlapping_memory(inp_shape, inp_strides)
        if inp_maybe_overlap:
            count = xp.zeros_like(storage)
            inp_indices = _stride_array(flatten_full_indices, inp_shape,
                                        inp_strides,
                                        inp_storage_offset - common_min_ptr)
            _index_add(count, inp_indices, xp.ones(1))
            with np.errstate(divide='ignore', invalid='ignore'):
                storage /= count

        return _stride_array(storage, inp_shape, inp_strides,
                             inp_storage_offset - common_min_ptr),

    def backward(self, target_input_indexes, grad_outputs):
        raise NotImplementedError


def as_strided(x, shape, strides, storage_offset=None):
    """Create a new view of array with the given shape, strides, and offset.

    Args:
        x (tuple of :class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            The array pointing a memory buffer. Its view is totally ignored.
        shape (tuple of int):
            The shape of output.
        strides (tuple of int):
            The strides of output, given in the unit of steps.
        storage_offset (int):
            The offset between the head of allocated memory and the pointer of
            first element, given in the unit of steps.

    Returns:
        ~chainer.Variable: The strided variable.

    .. warning::
        Users should be aware that this function potentially causes unintended
        side effects. See `numpy.lib.stride_tricks.as_strided`_ for the detail.

    .. note::
        The backward algorithm is borrowed from `torch.Tensor.as_strided`.
        Therefore, the returned gradient of ``backward`` is *layout-agnostic*
        when ``x`` contains memory overlap. See notes in pytorch's source
        code (as_strided Backward and layout-aware/agnostic autograd) too.

    .. note::
        In this function ``strides`` and ``storage_offset`` are given in the
        unit of steps instead of bytes. This specification differs from
        :func:`numpy.lib.stride_tricks.as_strided`.

    .. admonition:: Example

        >>> from chainer import functions as F, Variable
        >>> x = Variable(np.arange(4, dtype=np.float32))
        >>> x
        variable([0., 1., 2., 3.])
        >>> y = F.as_strided(x, (3, 2), (1, 1), 0)
        >>> y
        variable([[0., 1.],
                  [1., 2.],
                  [2., 3.]])
        >>> y.grad = np.ones((3, 2), dtype=np.float32)
        >>> y.backward()
        >>> x.grad
        array([1., 2., 2., 1.], dtype=float32)

    .. _numpy.lib.stride_tricks.as_strided:
        https://docs.scipy.org/doc/numpy/reference/generated/\
        numpy.lib.stride_tricks.as_strided.html

    """
    return AsStrided(shape, strides, storage_offset).apply((x,))[0]
