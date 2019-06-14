import numpy

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

    sample_data = None

    def __init__(self, in_size, counts, sample_size):
        super(BlackOut, self).__init__()
        vocab_size = len(counts)
        p = numpy.array(counts, dtype=numpy.float32)
        self.sampler = walker_alias.WalkerAlias(p)
        self.sample_size = sample_size

        with self.init_scope():
            self.W = variable.Parameter(shape=(vocab_size, in_size))

    def device_resident_accept(self, visitor):
        super(BlackOut, self).device_resident_accept(visitor)
        self.sampler.device_resident_accept(visitor)

    def forward(self, x, t):
        """Computes the loss value for given input and ground truth labels.

        Args:
            x (~chainer.Variable): Input of the weight matrix multiplication.
            t (~chainer.Variable): Batch of ground truth labels.

        Returns:
            ~chainer.Variable: Loss value.

        """

        batch_size = x.shape[0]
        if self.sample_data is not None:
            # for test
            sample_data = self.sample_data
        else:
            shape = (batch_size, self.sample_size)
            sample_data = self.sampler.sample(shape)
        samples = variable.Variable(sample_data, requires_grad=False)
        return black_out.black_out(x, t, self.W, samples)
