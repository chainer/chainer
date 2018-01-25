import six

import numpy

from chainer import cuda
from chainer import function_node
import chainer.functions
from chainer.utils import type_check


class LinearFunction(function_node.FunctionNode):

    def __init__(self, n_batch_axes=1):
        super(LinearFunction, self).__init__()
        if n_batch_axes < 1:
            raise ValueError(
                'n_batch_axes should be less than x.ndim and greater '
                'than 0 but {} was given.'.format(n_batch_axes))
        self._n_batch_axes = n_batch_axes

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)
        x_type, w_type = in_types[:2]

        type_check.expect(
            x_type.dtype.kind == 'f',
            w_type.dtype.kind == 'f',
            x_type.ndim == self._n_batch_axes + 1,
            w_type.ndim == 2,
            x_type.shape[self._n_batch_axes] == w_type.shape[1],
        )
        if type_check.eval(n_in) == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == x_type.dtype,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[0],
            )

    def forward(self, inputs):
        x = inputs[0]
        W = inputs[1]

        if not type_check.same_types(*inputs):
            raise ValueError('numpy and cupy must not be used together\n'
                             'type(W): {0}, type(x): {1}'
                             .format(type(W), type(x)))

        # NumPy raises an error when the array is not contiguous.
        # See: https://github.com/chainer/chainer/issues/2744
        # TODO(niboshi): Remove this code when NumPy is fixed.
        if (isinstance(x, numpy.ndarray) and
                not (x.flags.c_contiguous or x.flags.f_contiguous) and
                1 in x.shape):
            x = numpy.ascontiguousarray(x)

        if x.ndim != 2:
            x = x.reshape(x.shape[:self._n_batch_axes] + (-1,))

        y = x.dot(W.T).astype(x.dtype, copy=False)
        if len(inputs) == 3:
            b = inputs[2]
            y += b
        self.retain_inputs((0, 1))  # b is not retained
        return y,

    def backward(self, indexes, grad_outputs):
        x, W = self.get_retained_inputs()
        gy, = grad_outputs

        ret = []
        if 0 in indexes:
            gx, = LinearGradData().apply((W, gy))
            ret.append(chainer.functions.cast(gx, x.dtype))
        if 1 in indexes:
            gW, = LinearGradWeight(self._n_batch_axes).apply((x, gy))
            ret.append(chainer.functions.cast(gW, W.dtype))
        if 2 in indexes:
            gb = chainer.functions.sum(
                gy, axis=tuple(six.moves.range(self._n_batch_axes)))
            ret.append(gb)

        return ret


class LinearGradData(function_node.FunctionNode):

    def forward(self, inputs):
        self.retain_inputs((0, 1))
        W, gy = inputs

        if (isinstance(gy, numpy.ndarray) and
                not (gy.flags.c_contiguous or gy.flags.f_contiguous) and
                1 in gy.shape):
            gy = numpy.ascontiguousarray(gy)

        gx = gy.dot(W).astype(gy.dtype, copy=False)
        return gx,

    def backward(self, indexes, grad_outputs):
        W, gy = self.get_retained_inputs()
        ggx, = grad_outputs

        ret = []

        if 0 in indexes:
            gw, = LinearGradWeight().apply((ggx, gy))
            ret.append(chainer.functions.cast(gw, W.dtype))
        if 1 in indexes:
            ggy = linear(ggx, W)
            ret.append(chainer.functions.cast(ggy, gy.dtype))
        return ret


class LinearGradWeight(function_node.FunctionNode):

    def __init__(self, n_batch_axes=1):
        super(LinearGradWeight, self).__init__()
        if n_batch_axes < 1:
            raise ValueError(
                'n_batch_axes should be less than x.ndim and greater '
                'than 0 but {} was given.'.format(n_batch_axes))
        self._n_batch_axes = n_batch_axes

    def forward(self, inputs):
        self.retain_inputs((0, 1))
        x, gy = inputs

        if (isinstance(gy, numpy.ndarray) and
                not (gy.flags.c_contiguous or gy.flags.f_contiguous) and
                1 in gy.shape):
            gy = numpy.ascontiguousarray(gy)

        if self._n_batch_axes > 1:
            xp = cuda.get_array_module(*inputs)
            ax = six.moves.range(self._n_batch_axes)
            gW = xp.tensordot(
                gy, x, axes=(ax, ax)).astype(W.dtype, copy=False)
        else:
            gW = gy.T.dot(x).astype(W.dtype, copy=False)
        return gW,

    def backward(self, indexes, grad_outputs):
        x, gy = self.get_retained_inputs()
        ggW, = grad_outputs

        ret = []
        if 0 in indexes:
            gx, = LinearGradData().apply((ggW, gy))
            ret.append(chainer.functions.cast(gx, x.dtype))
        if 1 in indexes:
            ggy = linear(x, ggW)
            ret.append(chainer.functions.cast(ggy, gy.dtype))
        return ret


def linear(x, W, b=None, n_batch_axes=1):
    """Linear function, or affine transformation.

    It accepts two or three arguments: an input minibatch ``x``, a weight
    matrix ``W``, and optionally a bias vector ``b``. It computes

    .. math:: Y = xW^\\top + b.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable, which is a :math:`(s_1, s_2, \
            ..., s_n)`-shaped float array. Its first ``n_batch_axes``
            dimensions are handled as *minibatch dimensions*. The
            other dimensions are handled as concatenated one dimension whose
            size must be :math:`(s_{\\rm n_batch_axes} * ... * s_n = N)`.
        W (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Weight variable of shape :math:`(M, N)`,
            where :math:`(N = s_{\\rm n_batch_axes} * ... * s_n)`.
        b (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Bias variable (optional) of shape
            :math:`(M,)`.
        n_batch_axes (int): The number of batch axes. The default is 1. The
            input variable is reshaped into
            :math:`{\\rm n_batch_axes} + 1`-dimensional tensor.

    Returns:
        ~chainer.Variable: Output variable. A float array with shape
        of :math:`(s_1, ..., s_{\\rm n_batch_axes}, M)`.

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
        args = x, W
    else:
        args = x, W, b

    y, = LinearFunction(n_batch_axes).apply(args)
    return y
