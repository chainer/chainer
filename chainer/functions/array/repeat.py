import six

from chainer import cuda
from chainer import function_node
from chainer.utils import type_check


class Repeat(function_node.FunctionNode):

    """Repeat elements of an array."""

    def __init__(self, repeats, axis=None):
        if isinstance(repeats, six.integer_types):
            self.repeats = (repeats,)
        elif isinstance(repeats, tuple) and all(
                isinstance(x, six.integer_types) for x in repeats):
            self.repeats = repeats
        else:
            raise TypeError('repeats must be int or tuple of ints')

        if not all(x >= 0 for x in self.repeats):
            raise ValueError('all elements in repeats must be zero or larger')

        self.axis = axis

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)

    def forward(self, inputs):
        self.retain_inputs((0,))
        x = inputs[0]
        xp = cuda.get_array_module(x)
        repeats = self.repeats
        if self.axis is None or len(self.repeats) == 1:
            repeats = self.repeats[0]
        return xp.repeat(x, repeats, self.axis),

    def backward(self, indexes, grad_outputs):
        x = self.get_retained_inputs()[0]
        return RepeatGrad(self.repeats, self.axis, x.shape, x.dtype).apply(
            grad_outputs)


class RepeatGrad(function_node.FunctionNode):

    def __init__(self, repeats, axis, in_shape, in_dtype):
        self.repeats = repeats
        self.axis = axis
        self.in_shape = in_shape
        self.in_dtype = in_dtype

    def forward(self, inputs):
        gy = inputs[0]
        xp = cuda.get_array_module(gy)
        repeats = self.repeats
        axis = self.axis

        if len(gy) == 0:
            gx = xp.zeros(self.in_shape, self.in_dtype)
            return gx,
        elif axis is None:
            gx = gy.reshape(-1, repeats[0]).sum(axis=1).reshape(self.in_shape)
            return gx,
        elif len(repeats) == 1:
            shape = list(self.in_shape)
            shape[axis:axis + 1] = [-1, repeats[0]]
            gx = gy.reshape(shape).sum(axis=axis + 1)
            return gx,

        gx = xp.zeros(self.in_shape, self.in_dtype)
        slices = [slice(None) for _ in six.moves.range(self.axis)]
        pos = 0
        for (i, r) in enumerate(repeats):
            src = slices + [slice(pos, pos + r)]
            dst = slices + [slice(i, i + 1)]
            gx[dst] = gy[src].sum(axis=self.axis, keepdims=True)
            pos += r
        return gx,

    def backward(self, indexes, grad_outputs):
        return Repeat(self.repeats, self.axis).apply(grad_outputs)


def repeat(x, repeats, axis=None):
    """Construct an array by repeating a given array.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Input variable.
        repeats (:class:`int` or :class:`tuple` of :class:`int` s):
            The number of times which each element of ``x`` is repeated.
        axis (:class:`int`):
            The axis along which to repeat values.

    Returns:
        ~chainer.Variable: The repeated output Variable.

    .. admonition:: Example

        >>> x = np.array([0, 1, 2])
        >>> x.shape
        (3,)
        >>> y = np.repeat(x, 2)
        >>> y.shape
        (6,)
        >>> y
        array([0, 0, 1, 1, 2, 2])
        >>> x = np.array([[1,2], [3,4]])
        >>> x.shape
        (2, 2)
        >>> y = np.repeat(x, 3, axis=1)
        >>> y.shape
        (2, 6)
        >>> y
        array([[1, 1, 1, 2, 2, 2],
               [3, 3, 3, 4, 4, 4]])
        >>> y = np.repeat(x, [1, 2], axis=0)
        >>> y.shape
        (3, 2)
        >>> y
        array([[1, 2],
               [3, 4],
               [3, 4]])

    """
    return Repeat(repeats, axis).apply((x,))[0]
