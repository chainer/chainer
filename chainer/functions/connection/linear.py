import numpy

from chainer import backend
from chainer.backends import intel64
from chainer import function_node
import chainer.functions
from chainer.graph_optimizations import static_code
from chainer import utils
from chainer.utils import type_check
import chainerx


class LinearFunction(function_node.FunctionNode):

    _config_use_ideep = None
    _supports_static_optimizations = True

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)
        x_type, w_type = in_types[:2]
        type_check._argname((x_type, w_type), ('x', 'W'))

        type_check.expect(
            x_type.dtype.kind == 'f',
            w_type.dtype.kind == 'f',
            x_type.ndim == 2,
            w_type.ndim == 2,
            x_type.shape[1] == w_type.shape[1],
        )
        if type_check.eval(n_in) == 3:
            b_type = in_types[2]
            type_check._argname((b_type,), ('b',))
            type_check.expect(
                b_type.dtype == x_type.dtype,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[0],
            )

    @static_code
    def static_linear_no_bias(self, xp, optimized, inputs, outputs):
        x, W = inputs
        y = outputs[0]
        # NumPy raises an error when the array is not contiguous.
        # See: https://github.com/chainer/chainer/issues/2744
        # TODO(niboshi): Remove this code when NumPy is fixed.
        if (isinstance(x, numpy.ndarray) and
                not (x.flags.c_contiguous or x.flags.f_contiguous) and
                1 in x.shape):
            x = numpy.ascontiguousarray(x)

        if optimized:
            # Note: We can only call this function when both x and W
            # have the same dtype. Otherwise, the output type (for y)
            # may not be as expected (i.e., not the same dtype as x).
            xp.dot(x, W.T, out=y)
        else:
            y[:] = x.dot(W.T).astype(x.dtype, copy=False)

    @static_code
    def static_add_bias(self, inputs, outputs):
        bias = inputs[0]
        y = outputs[0]
        y += bias

    def forward_chainerx(self, inputs):
        # TODO(niboshi): Support dtype casting in ChainerX
        if inputs[0].dtype != inputs[1].dtype:
            return chainer.Fallback

        # Generic implementation
        if len(inputs) == 3:
            x, W, b = inputs
            if x.dtype != b.dtype:
                return chainer.Fallback
            return chainerx.linear(x, W, b),
        else:
            x, W = inputs
            return chainerx.linear(x, W),

    def forward(self, inputs):
        self._config_use_ideep = chainer.config.use_ideep
        if (intel64.should_use_ideep('>=auto')
                and intel64.inputs_all_ready(inputs)):
            # iDeep implementation
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

        # In order to be compatible with the "static graph" feature, it is
        # required that all output arrays of this forward
        # function be allocated explicitly:
        xp = backend.get_array_module(x)
        y = xp.empty((x.shape[0], W.shape[0]), dtype=x.dtype)

        # This is required because all of the "static_*()" functions
        # use the convention that any output arrays are supplied
        # as input arguments to the function. That is because it is
        # not allowed for a "static_*()" function to return anything
        # other than `None`. The reason is to prevent dynamic allocation
        # of output arrays during execution of the static schedule
        # because it would break the model.
        self.static_linear_no_bias(xp, x.dtype == W.dtype, inputs=[x, W],
                                   outputs=[y])
        if len(inputs) == 3:
            self.static_add_bias(inputs=[b], outputs=[y])

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
        x, W = self.get_retained_inputs()
        gy, = grad_outputs
        ret = []
        with chainer.using_config('use_ideep', self._config_use_ideep):
            if 0 in indexes:
                gx, = LinearGradData().apply((W, gy))
                ret.append(chainer.functions.cast(gx, x.dtype))
            if 1 in indexes:
                gW, = LinearGradWeight(W.dtype).apply((x, gy))
                ret.append(chainer.functions.cast(gW, W.dtype))
            if 2 in indexes:
                gb = chainer.functions.sum(gy, axis=0)
                ret.append(gb)

        return ret


class LinearGradData(function_node.FunctionNode):

    _config_use_ideep = None

    def forward(self, inputs):
        self._config_use_ideep = chainer.config.use_ideep
        if (intel64.should_use_ideep('>=auto')
                and intel64.inputs_all_ready(inputs)):
            # iDeep implementation
            return self._forward_ideep(inputs)

        # Generic implementation
        self.retain_inputs((0, 1))
        W, gy = inputs

        if (isinstance(gy, numpy.ndarray) and
                not (gy.flags.c_contiguous or gy.flags.f_contiguous) and
                1 in gy.shape):
            gy = numpy.ascontiguousarray(gy)

        gx = gy.dot(W).astype(gy.dtype, copy=False)
        return gx,

    def _forward_ideep(self, inputs):
        self.retain_inputs((0, 1))
        W, gy = inputs
        gx = intel64.ideep.linear.BackwardData(
            intel64.ideep.array(W),
            intel64.ideep.array(gy))
        return gx,

    def backward(self, indexes, grad_outputs):
        W, gy = self.get_retained_inputs()
        ggx, = grad_outputs

        ret = []
        with chainer.using_config('use_ideep', self._config_use_ideep):
            if 0 in indexes:
                gw, = LinearGradWeight(W.dtype).apply((ggx, gy))
                ret.append(chainer.functions.cast(gw, W.dtype))
            if 1 in indexes:
                ggy = linear(ggx, W)
                ret.append(chainer.functions.cast(ggy, gy.dtype))
        return ret


class LinearGradWeight(function_node.FunctionNode):

    _config_use_ideep = None

    def __init__(self, w_dtype):
        self._w_dtype = w_dtype

    def forward(self, inputs):
        self._config_use_ideep = chainer.config.use_ideep
        if (intel64.should_use_ideep('>=auto')
                and self._w_dtype == numpy.float32
                and intel64.inputs_all_ready(inputs)):
            # iDeep implementation
            return self._forward_ideep(inputs)

        # Generic implementation
        self.retain_inputs((0, 1))
        x, gy = inputs

        if (isinstance(gy, numpy.ndarray) and
                not (gy.flags.c_contiguous or gy.flags.f_contiguous) and
                1 in gy.shape):
            gy = numpy.ascontiguousarray(gy)

        gW = gy.T.dot(x).astype(self._w_dtype, copy=False)
        return gW,

    def _forward_ideep(self, inputs):
        self.retain_inputs((0, 1))
        x, gy = inputs
        gW = intel64.ideep.linear.BackwardWeights(
            intel64.ideep.array(x),
            intel64.ideep.array(gy))
        return gW,

    def backward(self, indexes, grad_outputs):
        x, gy = self.get_retained_inputs()
        ggW, = grad_outputs

        ret = []
        with chainer.using_config('use_ideep', self._config_use_ideep):
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

    .. math:: y_i = W x_i + b.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable,
            which is a :math:`(s_1, s_2, ..., s_n)`-shaped float array.
            Its first ``n_batch_axes`` dimensions are handled as
            *minibatch dimensions*. The other dimensions are handled as
            concatenated one dimension whose size must be
            :math:`(s_{\\rm n\\_batch\\_axes} * ... * s_n = N)`.
        W (:class:`~chainer.Variable` or :ref:`ndarray`):
            Weight variable of shape :math:`(M, N)`,
            where :math:`(N = s_{\\rm n\\_batch\\_axes} * ... * s_n)`.
        b (:class:`~chainer.Variable` or :ref:`ndarray`):
            Bias variable (optional) of shape :math:`(M,)`.
        n_batch_axes (int): The number of batch axes. The default is 1. The
            input variable is reshaped into
            (:math:`{\\rm n\\_batch\\_axes} + 1`)-dimensional tensor.
            This should be greater than 0.

    Returns:
        ~chainer.Variable: Output variable. A float array with shape
        of :math:`(s_1, ..., s_{\\rm n\\_batch\\_axes}, M)`.

    .. seealso::

        :class:`~chainer.links.Linear` to manage the model parameters
        ``W`` and ``b``.

    .. admonition:: Example

        >>> x = np.random.uniform(0, 1, (3, 4)).astype(np.float32)
        >>> W = np.random.uniform(0, 1, (5, 4)).astype(np.float32)
        >>> b = np.random.uniform(0, 1, (5,)).astype(np.float32)
        >>> y = F.linear(x, W, b)
        >>> y.shape
        (3, 5)

    """
    if n_batch_axes <= 0:
        raise ValueError('n_batch_axes should be greater than 0.')
    if n_batch_axes > 1:
        batch_shape = x.shape[:n_batch_axes]
        batch_size = utils.size_of_shape(batch_shape)
        x = x.reshape(batch_size, -1)
    elif x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    if b is None:
        args = x, W
    else:
        args = x, W, b

    y, = LinearFunction().apply(args)
    if n_batch_axes > 1:
        y = y.reshape(batch_shape + (-1,))
    return y
