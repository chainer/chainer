from chainer import cuda
from chainer import function_node
from chainer.utils import type_check


class FFT(function_node.FunctionNode):

    """Fast Fourie transform."""

    def __init__(self, method):
        self._method = method

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        r_type, i_type = in_types
        type_check.expect(
            r_type.dtype.kind == 'f',
            r_type.ndim > 0,
            r_type.shape == i_type.shape,
            r_type.dtype == i_type.dtype,
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        real, imag = inputs
        x = real + imag * 1j
        y = getattr(xp.fft, self._method)(x)
        real_y = y.real.astype(real.dtype)
        imag_y = y.imag.astype(imag.dtype)
        return real_y, imag_y

    def backward(self, inputs, grads):
        gr, gi = grads
        gxi, gxr = FFT(self._method).apply((gi, gr))
        return gxr, gxi


def fft(real, imag):
    """Fast Fourie transform.

    Args:
        real (chainer.Variable): Real part of the input.
        imag (chainer.Variable): Imaginary part of the input.

    Returns:
        tuple: Returns ``(ry, ri)`` where ``ry`` is the real part of
        the result and ``ri`` is the imaginary part of the result.

    """
    return FFT('fft').apply((real, imag))


def ifft(real, imag):
    """Inverse fast Fourie transform.

    Args:
        real (chainer.Variable): Real part of the input.
        imag (chainer.Variable): Imaginary part of the input.

    Returns:
        tuple: Returns ``(ry, ri)`` where ``ry`` is the real part of
        the result and ``ri`` is the imaginary part of the result.

    """
    return FFT('ifft').apply((real, imag))
