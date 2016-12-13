import collections

import numpy

import chainer
from chainer import cuda
from chainer import function
from chainer.utils import type_check
from chainer import utils
from chainer import variable


class SetItem(function.Function):

    """Function that slices array and set given value there."""

    def __init__(self, slices):
        if not isinstance(slices, collections.Iterable):
            slices = tuple([slices])

        if chainer.is_debug():
            n_ellipses = 0
            for s in slices:
                if numpy.isscalar(s) or s is None or isinstance(s, slice):
                    pass
                elif s is Ellipsis:
                    n_ellipses += 1
                else:
                    raise ValueError('Only basic indexing is supported')
            if n_ellipses > 1:
                raise ValueError('Only one Ellipsis is allowed')

        self.slices = slices

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        valid_slice = len(self.slices) - self.slices.count(None)
        type_check.expect(in_types[0].ndim >= valid_slice)

    def forward(self, xs):
        ary = xs[0]
        value = xs[1]
        ary[tuple(self.slices)] = value
        return ary, value

    def backward(self, xs, gys):
        xp = cuda.get_array_module(*xs)
        gy = gys[0]
        gx = xp.zeros_like(xs[0])
        gvalue = utils.force_array(gy[tuple(self.slices)])
        gx[tuple(self.slices)] = gvalue
        return gx, gvalue


def set_item(x, slices, value):
    """Set value to array with specified shape, axes and offsets.

    Args:
        x (tuple of Variables): Variable to be sliced.
        slices (int, slice, None or Ellipsis or tuple of them): Basic slicing
            to slice a variable. It supports ``int``, ``slice``, ``newaxis``
            (equivalent to ``None``) and ``Ellipsis``.
        value (Variable): Variable to set as elements.


    Returns:
        Variable: :class:`~chainer.Variable` object
            which has been updated the elements by ``value``.

    .. note::

       See NumPy document for details of `indexing
       <http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html>`_.

    """
    return SetItem(slices)(x, value)


def install_variable_set_item():
    variable.Variable.__setitem__ = set_item
