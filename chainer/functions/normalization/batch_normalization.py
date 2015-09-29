import numpy

from chainer import cuda
from chainer import function
from chainer import link
from chainer.utils import type_check
from chainer import variable


class BatchNormalizationFunction(function.Function):

    """Batch normalization function.

    Args:
        eps (float): Epsilon value for numerical stability.

    .. seealso::
        See :class:`BatchNormalization` for details.

    """
    def __init__(self, eps=1e-5):
        self.eps = eps

    def check_type_forward(self, in_types):
        n_in = in_types.size().eval()
        if n_in != 3 and n_in != 5:
            raise type_check.InvalidType(
                '%s or %s' % (in_types.size() == 3, in_types.size() == 5),
                '%s == %s' % (in_types.size(), n_in))

        x_type, gamma_type, beta_type = in_types[:3]
        type_check.expect(
            x_type.dtype == numpy.float32,
            x_type.ndim >= gamma_type.ndim + 1,
            # TODO(beam2d): Check shape
            gamma_type.dtype == numpy.float32,
            beta_type.dtype == numpy.float32,
            gamma_type.shape == beta_type.shape,
        )

        if len(in_types) == 5:
            mean_type, var_type = in_types[3:]
            type_check.expect(
                mean_type.dtype == numpy.float32,
                mean_type.shape == gamma_type.shape,
                var_type.dtype == numpy.float32,
                var_type.shape == gamma_type.shape,
            )
            

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x, gamma, beta = inputs[:3]

        head_ndim = gamma.ndim + 1
        expander = (None, Ellipsis) + (None,) * (x.ndim - head_ndim)
        gamma = gamma[expander]
        beta = beta[expander]

        if len(inputs) == 5:
            mean = inputs[3]
            var = inputs[4]
        else:
            axis = (0,) + tuple(range(head_ndim, x.ndim))
            mean = x.mean(axis=axis)
            var = x.var(axis=axis)
            var += self.eps
            self.mean = mean
            self.var = var

        self.std = xp.sqrt(var, dtype=var.dtype)
        x_mu = x - mean[expander]
        self.x_hat = x_mu / self.std[expander]
        y = gamma * self.x_hat
        y += beta
        return y,

    def backward(self, inputs, grad_outputs):
        if len(inputs) == 5:
            # TODO(beam2d): Support it
            raise RuntimeError('BatchNormalization does not support backprop '
                               'with fixed mean/var.')

        x, gamma = inputs[:2]
        gy = grad_outputs[0]

        head_ndim = gamma.ndim + 1
        expander = (None, Ellipsis) + (None,) * (x.ndim - head_ndim)
        m = gamma.dtype.type(x.size // gamma.size)

        axis = (0,) + tuple(range(head_ndim, x.ndim))
        gbeta = gy.sum(axis=axis)
        ggamma = (gy * self.x_hat).sum(axis=axis)

        gx = (gamma / self.std)[expander] * (
            gy - (self.x_hat * ggamma[expander] + gbeta[expander]) / m)
        return gx, ggamma, gbeta


class BatchNormalization(link.Link):

    """Batch normalization on outputs of linear or convolution functions.

    Args:
        size (int or tuple of ints): Size (or shape) of channel
            dimensions.
        decay (float): Decay rate of moving average.
        eps (float): Epsilon value for numerical stability.
        dtype (numpy.dtype): Type to use in computing.

    See: `Batch Normalization: Accelerating Deep Network Training by Reducing\
          Internal Covariate Shift <http://arxiv.org/abs/1502.03167>`_

    """
    def __init__(self, size, decay=0.9, eps=1e-5, dtype=numpy.float32):
        super(BatchNormalization, self).__init__()
        self._func = BatchNormalizationFunction(eps)

        if isinstance(size, tuple):
            self.size = size
        elif isinstance(size, int):
            self.size = size,
        else:
            raise TypeError('size must be tuple or int')

        self.dtype = numpy.dtype(dtype)

        avg_mean = numpy.zeros(size, dtype=self.dtype)
        self.states['avg_mean'] = avg_mean
        self.states['avg_var'] = numpy.zeros_like(avg_mean)
        self.states['N'] = 0

        self.params['gamma'] = variable.Variable(numpy.ones_like(avg_mean))
        self.params['beta'] = variable.Variable(numpy.zeros_like(avg_mean))

        self.decay = decay
        self.eps = eps

    def __call__(self, x, test=False, finetune=False):
        """Invokes the forward propagation of BatchNormalization.

        BatchNormalization accepts additional arguments, which controlls three
        different running mode.

        Args:
            x (Variable): An input variable.
            test (bool): If ``True``, BatchNormalization runs in testing mode;
                it normalizes the input using precomputed statistics.
            finetune (bool): If ``True``, BatchNormalization runs in finetuning
                mode; it accumulates the input array to compute population
                statistics for normalization, and normalizes the input using
                batch statistics.

        If ``test`` and ``finetune`` are both ``False``, then
        BatchNormalization runs in training mode; it computes moving averages
        of mean and variance for evaluation during training, and normalizes the
        input using batch statistics.

        """
        use_batch_mean = not test or finetune

        gamma = self.params['gamma']
        beta = self.params['beta']
        avg_mean = self.states['avg_mean']
        avg_var = self.states['avg_var']

        if use_batch_mean:
            ret = self._func(x, gamma, beta)
            func = ret.creator

            if finetune:
                self.states['N'] += 1
                decay = 1. / self.states['N']
            else:
                decay = self.decay

            m = x.data.size // gamma.data.size
            adjust = m / max(m - 1., 1.)  # unbiased estimation
            avg_mean *= decay
            func.mean *= 1 - decay  # reuse buffer as a temporary
            avg_mean += func.mean
            del func.mean
            avg_var *= decay
            func.var *= (1 - decay) * adjust  # reuse buffer as a temporary
            avg_var += func.var
            del func.var
        else:
            ret = self._func(x, gamma, beta, avg_mean, avg_var)
        return ret

    def start_finetuning(self):
        self.states['N'] = 0
