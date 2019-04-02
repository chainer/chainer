import numpy

import chainer
from chainer import backend
from chainer import function_node
from chainer import utils
from chainer.utils import type_check
from chainer import variable
import chainerx


_numpy_supports_0d_bool_index = \
    numpy.lib.NumpyVersion(numpy.__version__) >= '1.13.0'


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
        type_check._argname(in_types, ('x',))

    def forward(self, xs):
        slices = tuple([
            backend.from_chx(s) if isinstance(s, chainerx.ndarray) else s
            for s in self.slices])
        return utils.force_array(xs[0][slices]),

    def backward(self, indexes, gy):
        return GetItemGrad(
            self.slices, self.inputs[0].shape).apply(gy)


class GetItemGrad(function_node.FunctionNode):

    def __init__(self, slices, in_shape):
        self.slices = slices
        self._in_shape = in_shape

    def forward(self, inputs):
        gy, = inputs
        xp = backend.get_array_module(*inputs)
        gx = xp.zeros(self._in_shape, gy.dtype)
        if xp is numpy:
            try:
                numpy.add.at(gx, self.slices, gy)
            except IndexError:
                done = False
                # In numpy<1.13, 0-dim boolean index is not supported in
                # numpy.add.at and it's supported for 0-dim arr in
                # arr.__getitem__.
                if not _numpy_supports_0d_bool_index and len(self.slices) == 1:
                    idx = numpy.asanyarray(self.slices[0])
                    if idx.dtype == numpy.dtype(bool):
                        # Convert the array and the mask to 1-dim.
                        # numpy.add.at with them is supported in older numpy.
                        numpy.add.at(gx[None], idx[None], gy)
                        done = True

                if not done:
                    msg = '''
GetItem does not support backward for this slices. The slices argument is not
supported by numpy.add.at, while it is supported by numpy.ndarray.__getitem__.

Please report this error to the issue tracker with the stack trace,
the information of your environment, and your script:
https://github.com/chainer/chainer/issues/new.
'''
                    raise IndexError(msg)
        else:
            gx.scatter_add(self.slices, inputs[0])
        return gx,

    def backward(self, indexes, ggx):
        return GetItem(self.slices).apply(ggx)


def get_item(x, slices):
    """Extract elements from array with specified shape, axes and offsets.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            A variable to be sliced.
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

       See NumPy documentation for details of `indexing
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
