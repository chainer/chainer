import numpy

import chainer
from chainer import function
from chainer.utils import aggregator
from chainer.utils import type_check


class MeanSquaredError(function.Function):

    """Mean squared error (a.k.a. Euclidean loss) function."""

    def __init__(self, *args, **kwargs):
        aggregate_option = kwargs.pop('aggregate_axes', 'mean')
        self.aggregator = aggregator.Aggregator(aggregate_option)
        super(MeanSquaredError, self).__init__(*args, **kwargs)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype == numpy.float32,
            in_types[1].dtype == numpy.float32,
            in_types[0].shape == in_types[1].shape
        )

    def forward_cpu(self, inputs):
        x0, x1 = inputs
        self.diff = x0 - x1
        diff = self.diff.ravel()
        diff = diff * diff
        return self.aggregator.forward(diff),

    def forward_gpu(self, inputs):
        x0, x1 = inputs
        self.diff = x0 - x1
        diff = self.diff.ravel()
        return self.aggregator.forward(diff),

    def backward(self, inputs, gy):
        gy = self.aggregator.backward(gy[0])
        gx0 = 2 * gy * self.diff
        return gx0, -gx0


def mean_squared_error(x0, x1):
    """Mean squared error function.

    This function computes mean squared error between two variables. The mean
    is taken over the minibatch. Note that the error is not scaled by 1/2.

    """
    return MeanSquaredError()(x0, x1)
