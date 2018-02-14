import numpy

from chainer.backends import intel64
from chainer import function_node
import chainer.functions
from chainer.utils import type_check


class LinearFunction(function_node.FunctionNode):

    _use_ideep = False

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

    def forward(self, inputs):
        if (intel64.should_use_ideep('>=auto')
                and intel64.inputs_all_ready(inputs)):
            # iDeep implementation
            self._use_ideep = True
            return self._forward_ideep(inputs)

        # Generic implementation
        if len(inputs) == 3:
            x, W, b = inputs
        else:
            (x, W), b = inputs, None

        # NumPy raises an error when the array is not contiguous.
        # See: https://github.com/chainer/chainer/issues/2744
        # TODO(niboshi): Remove this code when NumPy is fixed.
        if (isinstance(x, numpy.ndarray) and
                not (x.flags.c_contiguous or x.flags.f_contiguous) and
                1 in x.shape):
            x = numpy.ascontiguousarray(x)

        y = x.dot(W.T).astype(x.dtype, copy=False)
        if b is not None:
            y += b
        self.retain_inputs((0, 1))  # b is not retained
        return y,

    def _forward_ideep(self, inputs):
        if len(inputs) == 3:
            x, W, b = inputs
        else:
            (x, W), b = inputs, None

        y = intel64.ideep.linear.Forward(
            intel64.ideep.array(x),
            intel64.ideep.array(W),
            intel64.ideep.array(b) if b is not None else None)

        self.retain_inputs((0, 1))
        return y,

    def backward(self, indexes, grad_outputs):
        ret = []
        gy, = grad_outputs

        x, W = self.get_retained_inputs()

        if self._use_ideep:
            # iDeep implementation
            if 0 in indexes:  # grad_x
                gx = LinearGradDIdeep().apply((W, gy))
                ret.append(gx[0])
            if 1 in indexes:  # grad_W
                gW = LinearGradWIdeep().apply((x, gy))
                ret.append(gW[0])
            if 2 in indexes:  # grad_b
                gb = chainer.functions.sum(gy, axis=0)
                ret.append(gb)
        else:
            # Generic implementation
            if 0 in indexes:
                gx, = LinearGradData().apply((W, gy))
                ret.append(chainer.functions.cast(gx, x.dtype))
            if 1 in indexes:
                gW, = LinearGradWeight().apply((x, gy))
                ret.append(chainer.functions.cast(gW, W.dtype))
            if 2 in indexes:
                gb = chainer.functions.sum(gy, axis=0)
                ret.append(gb)

        return ret


class LinearGradDIdeep(function_node.FunctionNode):

    def forward_cpu(self, inputs):
        W, gy = inputs

        gx = intel64.ideep.linear.BackwardData(
            intel64.ideep.array(W),
            intel64.ideep.array(gy))

        self.retain_inputs((0, 1))
        return gx,

    def backward(self, indexes, grad_outputs):
        inputs = self.get_retained_inputs()
        W, gy = inputs
        ggx, = grad_outputs

        ret = []
        if 0 in indexes:  # grad_W
            gg = linear(gy.T, ggx.T)
            ret.append(gg)
        if 1 in indexes:  # grad_gy
            gg = linear(ggx, W)
            ret.append(gg)

        return ret


class LinearGradWIdeep(function_node.FunctionNode):

    def forward_cpu(self, inputs):
        x, gy = inputs
        gW = intel64.ideep.linear.BackwardWeights(
            intel64.ideep.array(x),
            intel64.ideep.array(gy))
        self.retain_inputs((0, 1))
        return gW,

    def backward(self, indexes, grad_outputs):
        inputs = self.get_retained_inputs()
        x, gy = inputs
        if len(grad_outputs) == 2:
            ggW, ggb = grad_outputs
        else:
            ggW, = grad_outputs
            ggb = None

        ret = []
        if 0 in indexes:  # grad_x
            gg = linear(gy, ggW.T)
            ret.append(gg)
        if 1 in indexes:  # grad_gy
            gg = linear(x, ggW)
            if ggb is not None:
                gg += chainer.functions.broadcast_to(ggb, gg.shape)
            ret.append(gg)

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

    def forward(self, inputs):
        self.retain_inputs((0, 1))
        x, gy = inputs

        if (isinstance(gy, numpy.ndarray) and
                not (gy.flags.c_contiguous or gy.flags.f_contiguous) and
                1 in gy.shape):
            gy = numpy.ascontiguousarray(gy)

        gW = gy.T.dot(x).astype(gy.dtype, copy=False)
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
