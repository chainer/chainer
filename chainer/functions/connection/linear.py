import numpy

from chainer import function_node
import chainer.functions
from chainer.utils import type_check

from chainer.graph_optimimzations import static_backward
from chainer.graph_optimimzations import static_return_none

class LinearFunction(function_node.FunctionNode):

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)
        x_type, w_type = in_types[:2]

        type_check.expect(
            x_type.dtype.kind == 'f',
            w_type.dtype.kind == 'f',
            x_type.ndim == 2,
            w_type.ndim == 2,
            x_type.shape[1] == w_type.shape[1],
        )
        if type_check.eval(n_in) == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == x_type.dtype,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[0],
            )

    @static_return_none
    def static_linear(self, x, W, bias, y):
        """y = x*W^T + bias

        """
        numpy.dot(x, W.T, out=y)
        y += bias

    @static_return_none
    def static_linear_no_bias(self, x, W, y):
        """y = x*W^T

        """
        numpy.dot(x, W.T, out=y)

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

        # Notes:
        # In order to be compatible with the "static graph" feature, it is
        # required that all output arrays of this forward
        # function be allocated explicitly:
        y = numpy.empty((x.shape[0], W.shape[0])).astype(x.dtype)
        # This is required because all of the "static_*()" functions
        # use the convention that any output arrays are supplied
        # as input arguments to the function. That is because it is
        # not allowed for a "static_*()" function to return anything
        # other than `None`. The reason is to prevent dynamic allocation
        # of output arrays during execution of the static schedule
        # because it would break the model.
        if len(inputs) == 3:
            bias = inputs[2]
            # Note: `y` is the output array.
            self.static_linear(x, W, bias, y)
        else:
            # Note: `y` is the output array.
            self.static_linear_no_bias(x, W, y)
        self.retain_inputs((0, 1))  # b is not retained

        # old code:
        #y = x.dot(W.T).astype(x.dtype, copy=False)
        #if len(inputs) == 3:
        #    b = inputs[2]
        #    y += b
        #self.retain_inputs((0, 1))  # b is not retained
        return y,

    # fixme: move decorator to FunctionNode
    @static_backward
    def backward(self, indexes, grad_outputs):
        x, W = self.get_retained_inputs()
        gy, = grad_outputs

        ret = []
        if 0 in indexes:
            gx = linear(gy, W.T)
            ret.append(chainer.functions.cast(gx, x.dtype))
        if 1 in indexes:
            gW = linear(gy.T, x.T)
            ret.append(chainer.functions.cast(gW, W.dtype))
        if 2 in indexes:
            gb = chainer.functions.sum(gy, axis=0)
            ret.append(gb)

        return ret


def linear(x, W, b=None):
    """Linear function, or affine transformation.

    It accepts two or three arguments: an input minibatch ``x``, a weight
    matrix ``W``, and optionally a bias vector ``b``. It computes

    .. math:: Y = xW^\\top + b.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable, which is a :math:`(s_B, s_1, \
            s_2, ..., s_n)`-shaped float array. Its first dimension
            :math:`(s_B)` is assumed to be the *minibatch dimension*. The
            other dimensions are treated as concatenated one dimension whose
            size must be :math:`(s_1 * ... * s_n = N)`.
        W (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Weight variable of shape :math:`(M, N)`,
            where :math:`(N = s_1 * ... * s_n)`.
        b (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Bias variable (optional) of shape
            :math:`(M,)`.

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
    if x.ndim > 2:
        print('fixme: implement static optimizations.')
        x = x.reshape(len(x), -1)

    if b is None:
        args = x, W
    else:
        args = x, W, b

    y, = LinearFunction().apply(args)
    return y
