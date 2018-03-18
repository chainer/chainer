import numpy

import chainer
from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check
from chainer import variable


class GetItem(function_node.FunctionNode):

    """Function that slices array and extract elements."""

    def __init__(self, slices):
        if isinstance(slices, list):
            if all([isinstance(s, int) for s in slices]):
                slices = slices,
            slices = tuple(slices)
        elif not isinstance(slices, tuple):
            slices = slices,

        if chainer.is_debug():
            n_ellipses = 0
            for s in slices:
                if s is Ellipsis:
                    n_ellipses += 1
            if n_ellipses > 1:
                raise ValueError('Only one Ellipsis is allowed')

        self.slices = slices

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        n_nones = len([item for item in self.slices if item is None])
        valid_slice = len(self.slices) - n_nones
        type_check.expect(in_types[0].ndim >= valid_slice)

    def forward(self, xs):
        return utils.force_array(xs[0][self.slices]),

    def backward(self, indexes, gy):
        return GetItemGrad(
            self.slices, self.inputs[0].shape, self.inputs[0].dtype).apply(gy)


class GetItemGrad(function_node.FunctionNode):

    def __init__(self, slices, in_shape, in_dtype):
        self.slices = slices
        self._in_shape = in_shape
        self._in_dtype = in_dtype

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        gx = xp.zeros(self._in_shape, self._in_dtype)
        if xp is numpy:
            numpy.add.at(gx, self.slices, inputs[0])
        else:
            gx.scatter_add(self.slices, inputs[0])
        return gx,

    def backward(self, indexes, ggx):
        return GetItem(self.slices).apply(ggx)


def get_item(x, slices):
    """Extract elements from array with specified shape, axes and offsets.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): A variable to be sliced.
        slices (int, slice, Ellipsis, None, integer array-like, boolean\
        array-like or tuple of them):
            An object to specify the selection of elements.

    Returns:
        A :class:`~chainer.Variable` object which contains sliced array of
        ``x``.

    .. note::

        It only supports types that are supported by CUDA's atomicAdd when
        an integer array is included in ``slices``.
        The supported types are ``numpy.float32``, ``numpy.int32``,
        ``numpy.uint32``, ``numpy.uint64`` and ``numpy.ulonglong``.

    .. note::

        It does not support ``slices`` that contains multiple boolean arrays.

    .. note::

       See NumPy document for details of `indexing
       <https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html>`_.

    .. admonition:: Example

        >>> x = np.arange(12).reshape((2, 2, 3))
        >>> x
        array([[[ 0,  1,  2],
                [ 3,  4,  5]],
        <BLANKLINE>
               [[ 6,  7,  8],
                [ 9, 10, 11]]])
        >>> F.get_item(x, 0)
        variable([[0, 1, 2],
                  [3, 4, 5]])
        >>> F.get_item(x, (0, 0, slice(0, 2, 1)))  # equals x[0, 0, 0:2:1]
        variable([0, 1])
        >>> F.get_item(x, (Ellipsis, 2))  # equals x[..., 2]
        variable([[ 2,  5],
                  [ 8, 11]])
        >>> F.get_item(x, (1, np.newaxis, 1, 0))  # equals x[1, None, 1, 0]
        variable([9])

    """
    return GetItem(slices).apply((x,))[0]


def install_variable_get_item():
    variable.Variable.__getitem__ = get_item
