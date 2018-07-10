import chainer
from chainer.backends import cuda
from chainer import function_node
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
        if self.axis is not None:
            if self.axis >= 0:
                type_check.expect(self.axis < in_types[0].ndim)
            else:
                type_check.expect(-self.axis - 1 < in_types[0].ndim)

    def forward(self, inputs):
        self.retain_inputs((0,))
        self.retain_outputs((0,))
        x, = inputs
        xp = cuda.get_array_module(x)
        return xp.cumprod(x, axis=self.axis),

    def backward(self, indexes, grad_outputs):
        x, = self.get_retained_inputs()
        xp = cuda.get_array_module(x)
        y, = self.get_retained_outputs()
        gy, = grad_outputs
        F = chainer.functions

        axis = self.axis
        if axis is None:
            axis = 0
            shape = x.shape
            x = F.flatten(x)
        else:
            shape = None
        if y.shape[axis] == 0:
            return None,
        if axis < 0:
            axis += y.ndim

        _, x = F.split_axis(x, (1,), axis)
        z, = Cumprodsum(axis).apply((
            F.flip(x, axis),
            F.flip(gy, axis),
        ))

        y, ylast = F.split_axis(y, (-1,), axis)
        y = F.concat([xp.ones_like(ylast.array), y], axis=axis)

        gx = F.flip(z, axis) * y
        if shape is not None:
            gx = F.reshape(gx, shape)
        return gx,


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
        F = chainer.functions
        xmul, = self.get_retained_inputs()
        y, = self.get_retained_outputs()
        gy, = grad_outputs
        axis = self.axis

        z, = Cumprodsum(axis).apply((
            F.flip(xmul, axis),
            F.flip(gy, axis),
        ))
        gxadd = F.flip(z, axis)
        _, gxmul = F.split_axis(gxadd, (1,), axis)
        y, _ = F.split_axis(y, (-1,), axis)
        gxmul *= y
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
    return Cumprod(axis).apply((x,))[0]
