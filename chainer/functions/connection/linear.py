import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check


def _as_mat(x, n_batch_axes):
    if n_batch_axes == 1:
        return x
    elif 1 < n_batch_axes < x.ndim:
        return x.reshape(x.shape[:n_batch_axes] + (-1,))
    else:
        raise ValueError('n_batch_axes should be less than x.ndim and greater '
                         'than 0 but {} was given.'.format(n_batch_axes))


class LinearFunction(function.Function):

    def __init__(self, n_batch_axes=1):
        self._n_batch_axes = n_batch_axes

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)
        x_type, w_type = in_types[:2]

        type_check.expect(
            x_type.dtype.kind == 'f',
            w_type.dtype.kind == 'f',
            x_type.ndim > self._n_batch_axes,
            w_type.ndim == 2,
            type_check.prod(
                x_type.shape[self._n_batch_axes:]) == w_type.shape[1],
        )
        if type_check.eval(n_in) == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == x_type.dtype,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[0],
            )

    def forward(self, inputs):
        x = _as_mat(inputs[0], self._n_batch_axes)
        W = inputs[1]

        if not type_check.same_types(*inputs):
            raise ValueError('numpy and cupy must not be used together\n'
                             'type(W): {0}, type(x): {1}'
                             .format(type(W), type(x)))

        y = x.dot(W.T).astype(x.dtype, copy=False)
        if len(inputs) == 3:
            b = inputs[2]
            y += b
        return y,

    def backward(self, inputs, grad_outputs):
        x = _as_mat(inputs[0], self._n_batch_axes)
        W = inputs[1]
        gy = grad_outputs[0]

        if not type_check.same_types(*inputs):
            raise ValueError('numpy and cupy must not be used together\n'
                             'type(W): {0}, type(x): {1}'
                             .format(type(W), type(x)))

        if self._n_batch_axes == 1:
            gx = gy.dot(W).astype(x.dtype, copy=False).reshape(inputs[0].shape)
            gW = gy.T.dot(x).astype(W.dtype, copy=False)
        else:
            xp = cuda.get_array_module(*inputs)
            gy_ax = six.moves.range(self._n_batch_axes, gy.ndim)
            gx = xp.tensordot(
                gy, W, axes=(gy_ax, 0)).astype(x.dtype, copy=False)
            ax = six.moves.range(self._n_batch_axes)
            gW = xp.tensordot(gy, x, axes=(ax, ax)).astype(W.dtype, copy=False)
        if len(inputs) == 3:
            gb = gy.sum(tuple(six.moves.range(self._n_batch_axes)))
            return gx, gW, gb
        else:
            return gx, gW


def linear(x, W, b=None, n_batch_axes=1):
    """Linear function, or affine transformation.

    It accepts two or three arguments: an input minibatch ``x``, a weight
    matrix ``W``, and optionally a bias vector ``b``. It computes
    .. math:: Y = xW^\\top + b.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable, which is a :math:`(s_1, \
            s_2, ..., s_n)`-shaped float array. In the default setting, its
            first ``n_batch_axes`` dimensions are treated as the *minibatch
            dimensions*. The other dimensions are treated as concatenated one
            dimension whose size must be :math:`(s_{\rm n_batch_axes + 1} * \
            ... * s_n = N)`.
        W (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Weight variable of shape :math:`(M, N)`,
            where :math:`(N = s_1 * ... * s_n)`.
        b (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Bias variable (optional) of shape
            :math:`(M,)`.
        n_batch_axes (int): The number of batch axes. The default is 1. The
            input variable is reshaped into
            :math:`{\\rm n_batch_sxes} + 1`-dimensional tensor.

    Returns:
        ~chainer.Variable: Output variable. A float array with shape
        of :math:`(s_B, M)`.

    .. seealso:: :class:`~chainer.links.Linear`

    .. admonition:: Example

        >>> x = np.random.uniform(0, 1, (3, 4)).astype('f')
        >>> W = np.random.uniform(0, 1, (5, 4)).astype('f')
        >>> b = np.random.uniform(0, 1, (5,)).astype('f')
        >>> y = F.linear(x, W, b)
        >>> y.shape
        (3, 5)

    """
    if b is None:
        return LinearFunction(n_batch_axes)(x, W)
    else:
        return LinearFunction(n_batch_axes)(x, W, b)
