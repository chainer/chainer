from chainer import cuda
from chainer import function_node
from chainer.utils import type_check


class FFT(function_node.FunctionNode):

    """Fast Fourier transform."""

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
        xp = cuda.get_array_module(*grads)
        if gr is None:
            gr = xp.zeros_like(gi.data)
        if gi is None:
            gi = xp.zeros_like(gr.data)
        gxi, gxr = FFT(self._method).apply((gi, gr))
        return gxr, gxi


def fft(x):
    """Fast Fourier transform.

    Args:
        x (tuple): ``(real, imag)`` where ``real`` is a
            :class:`~chainer.Variable` storing the real part and
            ``imag`` is a :class:`~chainer.Variable` storing the imaginary
            part.

    Returns:
        tuple: Returns ``(ry, ri)`` where ``ry`` is the real part of
        the result and ``ri`` is the imaginary part of the result.

    .. note::
       Currently this function supports a tuple as input. It will supports a
       complex numbers directly in the future.

    """
    real, imag = x
    return FFT('fft').apply((real, imag))


def ifft(x):
    """Inverse fast Fourier transform.

    Args:
        x (tuple): ``(real, imag)`` where ``real`` is a
            :class:`~chainer.Variable` storing the real part and
            ``imag`` is a :class:`~chainer.Variable` storing the imaginary
            part.

    Returns:
        tuple: Returns ``(ry, ri)`` where ``ry`` is the real part of
        the result and ``ri`` is the imaginary part of the result.

    .. note::
       Currently this function supports a tuple as input. It will supports a
       complex numbers directly in the future.

    """
    real, imag = x
    return FFT('ifft').apply((real, imag))
