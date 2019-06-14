import numpy
import six

from chainer.backends import cuda
from chainer.backends import intel64
from chainer import function_node
from chainer.utils import type_check


def _cu_conv_sum(y, x, n):
    # Convolutional sum
    # TODO(beam2d): Use scan computation
    rdim = x.size // (x.shape[0] * x.shape[1])
    cuda.elementwise(
        'raw T x, int32 rdim, int32 N, int32 n_', 'raw T y',
        '''
          int half_n = n_ / 2;
          int offset = i / rdim * N * rdim + i % rdim;

          float sum_part = 0;
          for (int j = 0; j < N + half_n; ++j) {
            if (j < N) {
              sum_part += x[offset + j * rdim];
            }
            if (j >= n_) {
              sum_part -= x[offset + (j - n_) * rdim];
            }
            if (j >= half_n) {
              y[offset + (j - half_n) * rdim] = sum_part;
            }
          }
        ''', 'lrn_conv_sum')(x, rdim, x.shape[1], n, y,
                             size=x.shape[0] * rdim)


class LocalResponseNormalization(function_node.FunctionNode):

    """Cross-channel normalization function used in AlexNet."""

    _use_ideep = False

    def __init__(self, n=5, k=2, alpha=1e-4, beta=.75):
        self.n = n
        self.k = k
        self.alpha = alpha
        self.beta = beta

        self.scale = None
        self.indexes = None
        self.unit_scale = None

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim >= 2,
        )

    def forward_cpu(self, inputs):
        if (intel64.should_use_ideep('>=auto')
                and intel64.inputs_all_ready(inputs, (4,))):
            self._use_ideep = True
            return self.forward_ideep(inputs)

        x, = inputs
        self.retain_inputs((0,))
        self.retain_outputs((0,))

        half_n = self.n // 2
        x2 = numpy.square(x)
        sum_part = x2.copy()
        for i in six.moves.range(1, half_n + 1):
            sum_part[:, i:] += x2[:, :-i]
            sum_part[:, :-i] += x2[:, i:]
        self.unit_scale = self.k + self.alpha * sum_part
        self.scale = self.unit_scale ** -self.beta
        y = x * self.scale
        return y,

    def forward_ideep(self, inputs):
        x, = inputs
        self.retain_inputs((0,))
        self.retain_outputs((0,))

        param = intel64.ideep.localResponseNormalizationParam(
            self.n, self.k, self.n * self.alpha, self.beta,
            intel64.ideep.localResponseNormalizationParam.lrn_across_channels)
        y, indexes = intel64.ideep.localResponseNormalization.Forward(
            intel64.ideep.array(x), param)
        self.indexes = indexes
        return y,

    def forward_gpu(self, inputs):
        x, = inputs
        self.retain_inputs((0,))
        self.retain_outputs((0,))

        self.y = cuda.cupy.square(x)  # temporary
        self.scale = cuda.cupy.empty_like(self.y)
        _cu_conv_sum(self.scale, self.y, self.n)
        cuda.elementwise(
            'T x, T k, T alpha, T beta',
            'T y, T scale',
            '''scale = k + alpha * scale;
               y = x * pow(scale, -beta);''',
            'lrn_fwd')(x, self.k, self.alpha, self.beta,
                       self.y, self.scale)
        return self.y,

    def backward(self, indexes, grad_outputs):
        x, = self.get_retained_inputs()
        y, = self.get_retained_outputs()
        gy, = grad_outputs

        f = LocalResponseNormalizationGrad(
            self.n, self.k, self.alpha, self.beta, self._use_ideep,
            self.scale, self.indexes, self.unit_scale,)
        return f.apply((x, y, gy))


class LocalResponseNormalizationGrad(function_node.FunctionNode):

    def __init__(self, n, k, alpha, beta, use_ideep,
                 scale=None, indexes=None, unit_scale=None):
        self.n = n
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self._use_ideep = use_ideep

        self.scale = scale
        self.indexes = indexes
        self.unit_scale = unit_scale

    def forward_cpu(self, inputs):
        if self._use_ideep:
            return self._backward_ideep(inputs)

        x, y, gy = inputs
        half_n = self.n // 2
        summand = y * gy / self.unit_scale
        sum_part = summand.copy()
        for i in six.moves.range(1, half_n + 1):
            sum_part[:, i:] += summand[:, :-i]
            sum_part[:, :-i] += summand[:, i:]

        gx = gy * self.scale - 2 * self.alpha * self.beta * x * sum_part
        return gx,

    def _backward_ideep(self, inputs):
        x, y, gy = inputs

        param = intel64.ideep.localResponseNormalizationParam(
            self.n, self.k, self.n * self.alpha, self.beta,
            intel64.ideep.localResponseNormalizationParam.lrn_across_channels
        )
        gx = intel64.ideep.localResponseNormalization.Backward(
            intel64.ideep.array(x),
            intel64.ideep.array(gy),
            self.indexes,
            param)
        return gx,

    def forward_gpu(self, inputs):
        x, y, gy = inputs
        summand = cuda.elementwise(
            'T scale, T y, T gy', 'T summand',
            'summand = y * gy / scale',
            'lrn_bwd_summand')(self.scale, y, gy)
        gx = cuda.cupy.empty_like(x)
        _cu_conv_sum(gx, summand, self.n)
        cuda.elementwise(
            ' T x, T gy, T scale, T beta, T coeff', 'T gx',
            'gx = pow(scale, -beta) * gy - coeff * x * gx',
            'lrn_bwd')(x, gy, self.scale,
                       self.beta, 2 * self.alpha * self.beta, gx)
        return gx,

    def backward(self, indexes, grad_outputs):
        # No trivial way to implement double-backward for this function.
        raise NotImplementedError


def local_response_normalization(x, n=5, k=2, alpha=1e-4, beta=.75):
    """Local response normalization across neighboring channels.

    This function implements normalization across channels. Let :math:`x` an
    input image with :math:`N` channels. Then, this function computes an output
    image :math:`y` by following formula:

    .. math::
       y_i = {x_i \\over \\left( k + \\
              \\alpha \\sum_{j=\\max{1, i - n/2}}^{\\min{N, i + n/2}} \\
              x_j^2 \\right)^\\beta}.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.
        n (int): Normalization window width.
        k (float): Smoothing parameter.
        alpha (float): Normalizer scaling parameter.
        beta (float): Normalizer power parameter.

    Returns:
        ~chainer.Variable: Output variable.

    See: Section 3.3 of `ImageNet Classification with Deep Convolutional
    Neural Networks <https://www.cs.toronto.edu/~fritz/absps/imagenet.pdf>`_

    """
    return LocalResponseNormalization(n, k, alpha, beta).apply((x,))[0]
