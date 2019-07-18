import numpy

import chainer
from chainer.backends import cuda
from chainer.backends import intel64
from chainer import configuration
from chainer import function_node
from chainer.utils import argument
from chainer.utils import type_check

if cuda.cudnn_enabled:
    cudnn = cuda.cudnn


class Dropout(function_node.FunctionNode):

    """Dropout regularization."""

    def __init__(self, dropout_ratio, mask=None, return_mask=False):
        if not 0.0 <= dropout_ratio < 1.0:
            raise ValueError('dropout_ratio must be in the range [0, 1)')
        self.dropout_ratio = dropout_ratio
        self.mask = mask
        self.return_mask = return_mask
        self._use_cudnn = False

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, x):
        if (intel64.should_use_ideep('>=auto')
                and intel64.inputs_all_ready(x)
                and self.mask is None):
            return self._forward_ideep(x)

        if self.mask is not None:
            y = x[0] * self.mask
        else:
            scale = x[0].dtype.type(1. / (1 - self.dropout_ratio))
            flag = numpy.random.rand(*x[0].shape) >= self.dropout_ratio
            self.mask = scale * flag
            y = x[0] * self.mask
        return y,

    def forward_gpu(self, x):
        if (chainer.should_use_cudnn('>=auto', 5000)
                and x[0].flags.c_contiguous
                and self.mask is None
                and not self.return_mask):
            self._use_cudnn = True

            if hasattr(self, 'states'):
                # if we already have a dropout mask,
                # the forward operation is equal to backward.
                return cuda.get_cudnn_dropout_states().backward(
                    None, x[0], self.dropout_ratio, self.states),

            self.states, y = cuda.get_cudnn_dropout_states().forward(
                None, x[0], self.dropout_ratio)
            return y,
        else:
            if self.mask is not None:
                y = x[0] * self.mask
            else:
                rand = cuda.cupy.random.rand(*x[0].shape, dtype=numpy.float32)
                scale = x[0].dtype.type(1. / (1 - self.dropout_ratio))
                self.mask, y = cuda.elementwise(
                    'T x, R r, T scale, T ratio', 'T mask, T y',
                    '''
                    mask = (r >= ratio) * scale;
                    y = x * mask;
                    ''',
                    'dropout_fwd',
                )(x[0], rand, scale, self.dropout_ratio)
            return y,

    def _forward_ideep(self, x):
        mask, y = intel64.ideep.dropout.Forward(
            intel64.ideep.array(x[0]),
            self.dropout_ratio)
        self.mask = mask
        return y,

    def backward(self, x, gy):
        if chainer.should_use_cudnn('>=auto', 5000) and self._use_cudnn:
            return DropoutGradCuDNN(self.states, self.dropout_ratio).apply(gy)
        else:
            return DropoutGrad(self.mask).apply(gy)


class DropoutGrad(function_node.FunctionNode):
    """Computes the gradient of the Dropout function."""

    def __init__(self, mask):
        self.mask = mask

    def forward(self, inputs):
        if (intel64.should_use_ideep('>=auto')
                and intel64.inputs_all_ready(inputs)):
            return self._forward_ideep(inputs)

        y = inputs[0] * self.mask
        return y,

    def _forward_ideep(self, inputs):
        return intel64.ideep.dropout.Backward(
            intel64.ideep.array(self.mask),
            intel64.ideep.array(inputs[0])),

    def backward(self, indexes, gy):
        return DropoutGrad(self.mask).apply(gy)


class DropoutGradCuDNN(function_node.FunctionNode):
    """Computes the gradient of the Dropout function with cuDNN support."""

    def __init__(self, states, dropout_ratio):
        self.states = states
        self.dropout_ratio = dropout_ratio

    def forward(self, inputs):
        return cuda.get_cudnn_dropout_states().backward(
            None, inputs[0], self.dropout_ratio, self.states),

    def backward(self, indexes, gy):
        return DropoutGradCuDNN(self.states, self.dropout_ratio).apply(gy)


def dropout(x, ratio=.5, **kwargs):
    """dropout(x, ratio=.5, *, mask=None, return_mask=False)

    Drops elements of input variable randomly.

    This function drops input elements randomly with probability ``ratio`` and
    scales the remaining elements by factor ``1 / (1 - ratio)``. In testing
    mode (i.e., ``chainer.config.train`` is set to ``False``), it does nothing
    and just returns ``x``.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            Input variable. A :math:`(s_1, s_2, ..., s_N)` -shaped float array.
        ratio (float):
            Dropout ratio. The ``ratio`` must be ``0.0 <= ratio < 1.0``.
        mask (:ref:`ndarray` or None):
            The mask to be used for dropout.
            You do not have to specify this value, unless you need to make
            results deterministic.
            If ``mask`` is not specified or set to ``None``, a mask will be
            generated randomly according to the given ``ratio``.
            If ``mask`` is specified, ``ratio`` will be ignored.
            The shape and dtype must be the same as ``x`` and should be on the
            same device.
            Note that iDeep and cuDNN will not be used for this function if
            mask is specified, as iDeep and cuDNN do not support it.
        return_mask (bool):
            If ``True``, the mask used for dropout is returned together with
            the output variable.
            The returned mask can later be reused by passing it to ``mask``
            argument.

    Returns:
        ~chainer.Variable or tuple:
            When ``return_mask`` is ``False`` (default), returns the output
            variable.
            When ``True``, returns the tuple of the output variable and
            mask (:ref:`ndarray`). The mask will be on the same device as the
            input. The mask will become ``None`` when ``chainer.config.train``
            is set to ``False``.

    See the paper by G. Hinton: `Improving neural networks by preventing
    co-adaptation of feature detectors <https://arxiv.org/abs/1207.0580>`_.

    .. admonition:: Example

        >>> x = np.array([[-1, 0], [2, -3], [-2, 1]], np.float32)
        >>> with chainer.using_config('train', True):
        ...     y = F.dropout(x)
        >>> y.array
        array([[-2.,  0.],
               [ 4., -6.],
               [-0.,  2.]], dtype=float32)
        >>> with chainer.using_config('train', True):
        ...     y = F.dropout(x, ratio=0.0) \
# dropout returns original input if ratio=0.0
        >>> (x == y.array).all()
        True
        >>> with chainer.using_config('train', False):
        ...     y = F.dropout(x) \
# dropout in test mode returns original input
        >>> (x == y.array).all()
        True

    """
    mask = None
    return_mask = False
    if kwargs:
        mask, return_mask = argument.parse_kwargs(
            kwargs, ('mask', mask), ('return_mask', return_mask),
            train='train argument is not supported anymore. '
                  'Use chainer.using_config')

    if configuration.config.train:
        func = Dropout(ratio, mask, return_mask)
        out, = func.apply((x,))
        mask = func.mask
    else:
        out = chainer.as_variable(x)
        mask = None

    if return_mask:
        return out, mask
    return out
