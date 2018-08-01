import chainer
from chainer import function_node
from chainer.utils import type_check


class CopiedSetItem(function_node.FunctionNode):

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
        type_check.expect(in_types.size() == 2)
        type_check.expect(in_types[0].dtype == in_types[1].dtype)

    def forward(self, inputs):
        lhs, rhs = inputs
        y = lhs.copy()
        y[self.slices] = rhs
        self.rhs_shape = rhs.shape
        return y,

    def backward(self, indexes, grad_outputs):
        gy, = grad_outputs
        ret = []
        if 0 in indexes:
            ret.append(copied_set_item(
                gy, self.slices, gy.xp.array(0, dtype=gy.dtype)))
        if 1 in indexes:
            # TODO(kataoka): Allow duplicate value in indices (self.slices)
            ret.append(chainer.functions.sum_to(
                gy[self.slices], self.rhs_shape))
        return tuple(ret)


def copied_set_item(x, slices, rhs):
    """Copy array and do setitem

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): A variable to be sliced.
        slices (int, slice, Ellipsis, None, integer array-like, boolean\
        array-like or tuple of them):
            An object to specify the selection of elements.
        rhs (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): A variable to be sliced.

    Returns:
        A :class:`~chainer.Variable` object which contains the new array.

    """
    return CopiedSetItem(slices).apply((x, rhs))[0]
