import numpy

from chainer.functions.normalization import batch_normalization
from chainer import link
from chainer import variable


class BatchNormalization(link.Link):

    """Batch normalization layer on outputs of linear or convolution functions.

    This is a callable link that wraps the
    :func:`functions.batch_normalization` and
    :func:`functions.fixed_batch_normalization` functions.

    This link runs in three modes: training mode, finetuning mode, and testing
    mode.

    In training mode, it normalizes the input by *batch statistics*, and
    computes approximated population statistics by moving averages.

    In finetuning mode, it accumulates the input to compute *population
    statistics*. In order to correctly compute the population statistics, a
    user must use this mode to feed mini batches running through whole training
    dataset.

    In testing mode, it uses precmoputed population statistics to normalize the
    input variable. The population statistics is approximated if it is computed
    by training mode, or accurate if it is correctly computed by finetuning
    mode.

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
        self.eps = eps

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
            ret = batch_normalization.batch_normalization(
                x, gamma, beta, self.eps)
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
            ret = batch_normalization.fixed_batch_normalization(
                x, gamma, beta, avg_mean, avg_var, self.eps)
        return ret

    def start_finetuning(self):
        """Resets the population count for collecting population statistics.

        This method can be skipped if it is the first time to use the
        finetuning mode. Otherwise, this method should be called before
        starting the finetuning mode again.

        """
        self.states['N'] = 0
