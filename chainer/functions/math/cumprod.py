from chainer.backends import cuda
from chainer import function_node
from chainer.functions.array import concat
from chainer.functions.array import flatten
from chainer.functions.array import flip
from chainer.utils import type_check


class Cumprod(function_node.FunctionNode):
    """Cumulative prod of array elements over a given axis."""

    def __init__(self, axis):
        if isinstance(axis, int) or axis is None:
            self.axis = axis
        else:
            raise TypeError('axis must be int or None')

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
        )
        if self.axis >= 0:
            type_check.expect(self.axis < in_types[0].ndim)
        else:
            type_check.expect(-self.axis - 1 < in_types[0].ndim)

    def forward(self, inputs):
        self.retain_inputs((0,))
        self.retain_outputs((0,))
        x, = inputs
        self._in_shape = x.shape
        xp = cuda.get_array_module(x)
        return xp.cumprod(x, axis=self.axis),

    def backward(self, indexes, grad_outputs):
        x, = self.get_retained_inputs()
        axis = self.axis
        if x.shape[axis] == 0:
            return None,
        if axis < 0:
            axis += x.ndim

        xp = cuda.get_array_module(x)
        y, = self.get_retained_outputs()
        gy, = grad_outputs

        expander = (slice(None),) * axis
        z, = Cumprodsum(axis).apply((
            x[expander + (slice(None, 0, -1),)],
            flip.flip(gy, axis),
        ))

        y_flip = concat.concat([
            xp.ones_like(y.array[expander + (slice(0, 1),)]),
            y[expander + (slice(None, -1),)]
        ], axis=axis)

        return flip.flip(z, axis) * y_flip,


class Cumprodsum(function_node.FunctionNode):

    def __init__(self, axis):
        self.axis = axis

    def forward(self, inputs):
        self.retain_inputs((0,))
        self.retain_outputs((0,))
        xmul, xadd = inputs
        xp = cuda.get_array_module(xmul)
        y = xp.empty_like(xadd)

        axis = self.axis
        expander = (slice(None),) * axis
        cum = xp.zeros_like(xadd[expander + (0,)])

        i = 0
        while True:
            ix = expander + (i,)
            cum += xadd[ix]
            y[ix] = cum
            if i >= xmul.shape[axis]:
                break
            cum *= xmul[ix]
            i += 1

        return y,

    def backward(self, indexes, grad_outputs):
        xmul, = self.get_retained_inputs()
        xp = cuda.get_array_module(xmul)
        y, = self.get_retained_outputs()
        gy, = grad_outputs
        axis = self.axis

        z, = Cumprodsum(axis).apply((
            flip.flip(xmul, axis),
            flip.flip(gy, axis),
        ))
        gxadd = flip.flip(z, axis)

        expander = (slice(None),) * axis
        ix = expander + (slice(1, None),)
        gxmul = gxadd[ix] * y[expander + (slice(None, -1),)]
        return gxmul, gxadd


def cumprod(x, axis=None):
    """Cumulative prod of array elements over a given axis.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
           :class:`cupy.ndarray`):
            Elements to calculate the cumulative prod.
        axis (int or None):
            Axis along which the cumulative prod is taken.
            If it is not specified, the input is flattened.

    Returns:
        ~chainer.Variable: Output variable.

    """
    if axis is None:
        x = flatten.flatten(x)
        axis = 0
    return Cumprod(axis).apply((x,))[0]
