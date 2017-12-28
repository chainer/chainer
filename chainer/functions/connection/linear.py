import numpy

from chainer.backends import cuda
from chainer import function_node
import chainer.functions
from chainer.utils import type_check

from chainer.graph_optimimzations.static_graph import static_schedule_func


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

    # fixme: remove
    @static_schedule_func
    def static_linear_old(self, x, W, bias, y):
        """y = x*W^T + bias

        """
        # NumPy raises an error when the array is not contiguous.
        # See: https://github.com/chainer/chainer/issues/2744
        # TODO(niboshi): Remove this code when NumPy is fixed.
        if (isinstance(x, numpy.ndarray) and
                not (x.flags.c_contiguous or x.flags.f_contiguous) and
                1 in x.shape):
            x = numpy.ascontiguousarray(x)
        numpy.dot(x, W.T, out=y)
        y += bias

    # fixme: remove
    @static_schedule_func
    def static_linear_no_bias_old(self, x, W, y):
        """y = x*W^T

        """
        numpy.dot(x, W.T, out=y)

    @static_schedule_func
    def static_linear_no_bias_naive(self, x, W, y):
        # todo: this performs unnecessary memory allocations.
        # consider optimizing further.

        # NumPy raises an error when the array is not contiguous.
        # See: https://github.com/chainer/chainer/issues/2744
        # TODO(niboshi): Remove this code when NumPy is fixed.
        if (isinstance(x, numpy.ndarray) and
                not (x.flags.c_contiguous or x.flags.f_contiguous) and
                1 in x.shape):
            x = numpy.ascontiguousarray(x)
        # todo (vogel): optimize to prevent unnecessary allocation.
        y[:] = x.dot(W.T).astype(x.dtype, copy=False)

    @static_schedule_func
    def static_add_bias(self, y, bias):
        y += bias

    def forward(self, inputs):
        x = inputs[0]
        W = inputs[1]

        if not type_check.same_types(*inputs):
            raise ValueError('numpy and cupy must not be used together\n'
                             'type(W): {0}, type(x): {1}'
                             .format(type(W), type(x)))



        # Notes:
        # In order to be compatible with the "static graph" feature, it is
        # required that all output arrays of this forward
        # function be allocated explicitly:
        xp = cuda.get_array_module(x)
        y = xp.empty((x.shape[0], W.shape[0])).astype(x.dtype)

        # This is required because all of the "static_*()" functions
        # use the convention that any output arrays are supplied
        # as input arguments to the function. That is because it is
        # not allowed for a "static_*()" function to return anything
        # other than `None`. The reason is to prevent dynamic allocation
        # of output arrays during execution of the static schedule
        # because it would break the model.

        self.static_linear_no_bias_naive(x, W, y)
        if len(inputs) == 3:
            bias = inputs[2]
            self.static_add_bias(y, bias)


        ##########################33
        # old code:

        #if (isinstance(x, numpy.ndarray) and
        #        not (x.flags.c_contiguous or x.flags.f_contiguous) and
        #        1 in x.shape):
        #    x = numpy.ascontiguousarray(x)

        #y = x.dot(W.T).astype(x.dtype, copy=False)
        #if len(inputs) == 3:
        #    b = inputs[2]
        #    y += b
        #########################

        self.retain_inputs((0, 1))  # b is not retained
        return y,

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
        x = x.reshape(len(x), -1)

    if b is None:
        args = x, W
    else:
        args = x, W, b

    y, = LinearFunction().apply(args)
    return y
