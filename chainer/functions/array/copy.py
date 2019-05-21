from chainer.backends import cuda
from chainer import function_node
from chainer.utils import type_check


class Copy(function_node.FunctionNode):

    """Copies the input variable onto the specified device."""

    def __init__(self, out_device):
        self.out_device = out_device

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))

    def forward(self, inputs):
        x, = inputs
        self._in_device = cuda.get_device_from_array(x).id
        if int(self.out_device) == -1:
            return cuda.to_cpu(x),
        else:
            return cuda.to_gpu(x, self.out_device),

    def backward(self, indexes, grad_outputs):
        return Copy(self._in_device).apply(grad_outputs)


def copy(x, dst):
    """Copies the input variable onto the specified device.

    If the input ``x`` already resides on the device specified by ``dst``, no
    copy will actually take place and the returned variable will hold a view
    of the input. In other cases, the input will be copied to ``dst``.
    When ``dst == -1``, the array is copied to the host memory.
    This function supports copies from host to host, from host to device,
    from device to device and from device to host.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            Variable to be copied.
        dst (int): Target device specifier.

    Returns:
        ~chainer.Variable: Output variable.

    .. admonition:: Example

        >>> import chainer.backends.cuda as cuda
        >>> x = np.random.uniform(-1, 1, (5, 10))
        >>> cuda.get_device_from_array(x).id
        -1
        >>> y = F.copy(x, 0) # from host to device0
        >>> cuda.get_device_from_array(y.array).id
        0
        >>> z = F.copy(y, -1) # from device0 to host
        >>> cuda.get_device_from_array(z.array).id
        -1

    """
    y, = Copy(dst).apply((x,))
    return y
