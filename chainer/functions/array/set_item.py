import chainer
from chainer.backends import cuda
from chainer import function_node
from chainer.utils import argument
from chainer.utils import type_check


class CopiedSetItem(function_node.FunctionNode):

    """Sets given values to specified elements of an array"""

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
        xp = cuda.get_array_module(gy)
        ret = []
        if 0 in indexes:
            ret.append(_copied_set_item(
                gy, self.slices, xp.array(0, dtype=gy.dtype)))
        if 1 in indexes:
            if chainer.is_debug():
                # Check duplicate indexes
                tmp = xp.arange(gy.size).reshape(gy.shape)
                tmp = xp.sort(tmp[self.slices], axis=None)
                if (tmp[:-1] == tmp[1:]).any():
                    raise ValueError('setitem to an index more than once')
            ret.append(chainer.functions.sum_to(
                gy[self.slices], self.rhs_shape))
        return tuple(ret)


def _copied_set_item(x, slices, rhs):
    return CopiedSetItem(slices).apply((x, rhs))[0]


def set_item(x, slices, rhs, **kwargs):
    """set_item(x, slices, rhs, *, inplace=True)

    Copies array and does setitem

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): A variable to be sliced.
        slices (int, slice, Ellipsis, None, integer array-like, boolean\
        array-like or tuple of them):
            An object to specify the selection of elements.
        rhs (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): A variable to be set.

    Returns:
        A :class:`~chainer.Variable` object which contains the new array.

    """
    inplace, = argument.parse_kwargs(
        kwargs, ('inplace', True)
    )
    if inplace:
        raise NotImplementedError(
            'set_item currently supports only inplace=False')
    return _copied_set_item(x, slices, rhs)
