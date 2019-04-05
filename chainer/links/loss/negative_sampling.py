import numpy

import chainer
from chainer.functions.loss import negative_sampling
from chainer import link
from chainer.utils import argument
from chainer.utils import walker_alias
from chainer import variable


class NegativeSampling(link.Link):

    """Negative sampling loss layer.

    This link wraps the :func:`~chainer.functions.negative_sampling` function.
    It holds the weight matrix as a parameter. It also builds a sampler
    internally given a list of word counts.

    Args:
        in_size (int): Dimension of input vectors.
        counts (int list): Number of each identifiers.
        sample_size (int): Number of negative samples.
        power (float): Power factor :math:`\\alpha`.
        dtype (numpy.dtype): Type to use in computing.

    .. seealso:: :func:`~chainer.functions.negative_sampling` for more detail.

    Attributes:
        W (~chainer.Variable): Weight parameter matrix.

    """

    def __init__(self, in_size, counts, sample_size, power=0.75, dtype=None):
        super(NegativeSampling, self).__init__()
        dtype = chainer.get_dtype(dtype)
        vocab_size = len(counts)
        self.sample_size = sample_size
        power = dtype.type(power)
        p = numpy.array(counts, dtype)
        numpy.power(p, power, p)
        self.sampler = walker_alias.WalkerAlias(p)

        with self.init_scope():
            self.W = variable.Parameter(0, (vocab_size, in_size))

    def device_resident_accept(self, visitor):
        super(NegativeSampling, self).device_resident_accept(visitor)
        self.sampler.device_resident_accept(visitor)

    def forward(self, x, t, reduce='sum', **kwargs):
        """forward(x, t, reduce='sum', *, return_samples=False)

        Computes the loss value for given input and ground truth labels.

        Args:
            x (~chainer.Variable): Input of the weight matrix multiplication.
            t (~chainer.Variable): Batch of ground truth labels.
            reduce (str): Reduction option. Its value must be either
                ``'sum'`` or ``'no'``. Otherwise, :class:`ValueError` is
                raised.
            return_samples (bool):
                If ``True``, the sample array is also returned.
                The sample array is a
                :math:`(\\text{batch_size}, \\text{sample_size} + 1)`-array of
                integers whose first column is fixed to the ground truth labels
                and the other columns are drawn from the
                :class:`chainer.utils.WalkerAlias` sampler.

        Returns:
            ~chainer.Variable or tuple:
                If ``return_samples`` is ``False`` (default), loss value is
                returned.

                Otherwise, a tuple of the loss value and the sample array
                is returned.

        """
        return_samples = False
        if kwargs:
            return_samples, = argument.parse_kwargs(
                kwargs, ('return_samples', return_samples))

        ret = negative_sampling.negative_sampling(
            x, t, self.W, self.sampler.sample, self.sample_size,
            reduce=reduce, return_samples=return_samples)
        return ret
