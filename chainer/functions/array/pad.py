import numpy

from chainer import backend
from chainer import function_node
from chainer.utils import type_check


class Pad(function_node.FunctionNode):

    """Padding of an array."""

    def __init__(self, pad_width, mode, **keywords):
        self.mode = mode
        self.keywords = keywords
        self.pad_width = pad_width
        self.pad_bw = numpy.asarray(pad_width)
        if self.pad_bw.size == 1:
            self.pad_bw = numpy.repeat(self.pad_bw, 2)

    def check_type_forward(self, in_types):
        # Depending on the arguments, pad_width and keywords, the input value
        # may be inappropriate. In that case, numpy.pad or cupy.pad will raise
        # errors, so that only check the size and the dtype in this function.
        type_check._argname(in_types, ('x',))
        x_type = in_types[0]
        type_check.expect(x_type.dtype.kind == 'f')

    def forward(self, inputs):
        xp = backend.get_array_module(*inputs)
        return xp.pad(inputs[0], self.pad_width, mode=self.mode,
                      **self.keywords),

    def backward(self, inputs, grad_outputs):
        gy, = grad_outputs
        in_shape = self.inputs[0].shape
        if self.pad_bw.ndim == 1:
            self.pad_bw = numpy.tile(self.pad_bw, (len(in_shape), 1))
        input_idxs = tuple(
            slice(p[0], p[0] + dim) for dim, p in zip(in_shape, self.pad_bw))
        return gy[input_idxs],


def pad(x, pad_width, mode, **keywords):
    """Pad an input variable.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            Input data.
        pad_width (int or array-like):
            Number of values padded to the edges of each axis.
        mode (str):
            Specifies how the function fills the periphery of the array.
            The mode is passed to :func:`numpy.pad` or :func:`cupy.pad`.
            If it is ``'constant'``, the input is padded by a constant value
            specified by ``constant_values``.
        constant_values (int or array-like):
            Constant values to fill the periphery in the ``'constant'`` mode.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return Pad(pad_width, mode, **keywords).apply((x,))[0]
