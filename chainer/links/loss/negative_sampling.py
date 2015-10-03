import numpy

from chainer import cuda
from chainer.functions.loss import negative_sampling
from chainer import link
from chainer.utils import walker_alias
from chainer import variable


class NegativeSampling(link.Link):

    """Negative sampling loss with parameters.

    This is a primitive link that wraps the
    :func:`~chainer.functions.negative_sampling` function. It holds the weight
    matrix as a parameter.
    It also builds a sampler internally given a list of word counts.

    Args:
        in_size (int): Dimension of input vectors.
        counts (int list): Number of each identifiers.
        sample_size (int): Number of negative samples.
        power (float): Power factor :math:`\\alpha`.

    .. seealso:: :func:`~chainer.functions.negative_sampling` for more detail.

    """
    def __init__(self, in_size, counts, sample_size, power=0.75):
        super(NegativeSampling, self).__init__()
        vocab_size = len(counts)
        self.params['W'] = variable.Variable(
            numpy.zeros((vocab_size, in_size)).astype(numpy.float32))

        self.sample_size = sample_size
        power = numpy.float32(power)
        p = numpy.array(counts, power.dtype)
        numpy.power(p, power, p)
        self.sampler = walker_alias.WalkerAlias(p)

    def to_cpu(self):
        super(NegativeSampling, self).to_cpu()
        self.sampler.to_cpu()

    def to_gpu(self, device=None):
        super(NegativeSampling, self).to_gpu(device)
        with cuda.get_device(device):
            self.sampler.to_gpu()

    def __call__(self, x, t):
        """Computes the loss value for given input and groundtruth labels.

        Args:
            x (~chainer.Variable): Input of the weight matrix multiplication.
            t (~chainer.Variable): Batch of groundtruth labels.

        Returns:
            ~chainer.Variable: Loss value.

        """
        return negative_sampling.negative_sampling(
            x, t, self.params['W'], self.sampler.sample, self.sample_size)
