import numpy

from chainer import cuda
from chainer.functions.loss import black_out
from chainer import link
from chainer.utils import walker_alias
from chainer import variable


class BlackOut(link.Link):

    """BlackOut loss layer.

    .. seealso:: :func:`~chainer.functions.black_out` for more detail.

    Args:
        in_size (int): Dimension of input vectors.
        counts (int list): Number of each identifiers.
        sample_size (int): Number of negative samples.

    Attributes:
        W (~chainer.Parameter): Weight parameter matrix.

    """

    def __init__(self, in_size, counts, sample_size):
        super(BlackOut, self).__init__()
        vocab_size = len(counts)
        p = numpy.array(counts, dtype=numpy.float32)
        self.sampler = walker_alias.WalkerAlias(p)
        self.sample_size = sample_size

        with self.init_scope():
            self.W = variable.Parameter(shape=(vocab_size, in_size))

    def to_cpu(self):
        super(BlackOut, self).to_cpu()
        self.sampler.to_cpu()

    def to_gpu(self, device=None):
        with cuda._get_device(device):
            super(BlackOut, self).to_gpu()
            self.sampler.to_gpu()

    def __call__(self, x, t):
        """Computes the loss value for given input and ground truth labels.

        Args:
            x (~chainer.Variable): Input of the weight matrix multiplication.
            t (~chainer.Variable): Batch of ground truth labels.

        Returns:
            ~chainer.Variable: Loss value.

        """

        batch_size = x.shape[0]
        if hasattr(self, 'sample_data'):
            # for test
            sample_data = self.sample_data
        else:
            shape = (batch_size, self.sample_size)
            sample_data = self.sampler.sample(shape)
        samples = variable.Variable(sample_data)
        return black_out.black_out(x, t, self.W, samples)
