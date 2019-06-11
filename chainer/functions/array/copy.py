import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import function_node
from chainer.utils import type_check
import chainerx


class Copy(function_node.FunctionNode):

    """Copies the input variable onto the specified device."""

    def __init__(self, in_device, out_device):
        self._in_device = in_device
        self.out_device = out_device

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))

    def forward(self, inputs):
        x, = inputs
        return self.out_device.send(x),

    def forward_chainerx(self, inputs):
        x, = inputs
        return x.to_device(self.out_device.device),

    def backward(self, indexes, grad_outputs):
        f = Copy(self.out_device, self._in_device)
        return f.apply(grad_outputs)


# TODO(niboshi): Link from `dst` to an appropriate device specifier docs.
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
        dst: Target device specifier.

    Returns:
        ~chainer.Variable: Output variable.

    .. admonition:: Example

        >>> import chainer.backends.cuda as cuda
        >>> x_arr = np.random.uniform(-1, 1, (5, 10))
        >>> x = chainer.Variable(x_arr)
        >>> x.device
        <CpuDevice (numpy)>
        >>> y = F.copy(x, '@cupy:0') # from CPU (NumPy) to GPU 0 (CuPy)
        >>> y.device
        <GpuDevice (cupy):0>

    .. note::
        Copies between non-ChainerX devices and ChainerX devices are not
        supported.

    """
    # For backward compatibility
    if dst is cuda.DummyDevice:
        dst = chainer.get_device('@numpy')

    x_is_var = isinstance(x, chainer.Variable)
    in_device = backend.get_device_from_array(x.array if x_is_var else x)
    out_device = chainer.get_device(dst)

    if in_device.xp is chainerx:
        x_arr = x.chx_array if x_is_var else x
        if out_device.xp is not chainerx:
            # ChainerX to non-ChainerX
            if x_arr.is_backprop_required():
                raise RuntimeError(
                    'F.copy does not support copy from a ChainerX array with '
                    'backprop required to a non-ChainerX device.\n'
                    'From: {}\n'
                    'To: {}'.format(in_device, out_device))
            return chainer.Variable(
                out_device.send(x_arr), requires_grad=False)

        # ChainerX to ChainerX
        return chainer.Variable(
            out_device.send(x_arr), requires_grad=x_arr.is_backprop_required())

    if out_device.xp is chainerx:
        # Non-ChainerX to ChainerX
        if x_is_var and x.requires_grad:
            raise RuntimeError(
                'F.copy does not support copy from a non-ChainerX array with '
                'backprop required to a ChainerX device.\n'
                'From: {}\n'
                'To: {}'.format(in_device, out_device))
        x_arr = x.array if x_is_var else x
        return chainer.Variable(out_device.send(x_arr), requires_grad=False)

    # Non-ChainerX to non-ChainerX
    y, = Copy(in_device, out_device).apply((x,))
    return y
