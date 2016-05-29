import six
import numpy

from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check


class AxesAggregator(object):

    def __init__(self, non_aggregate_axes):
        self.non_aggregate_axes = non_aggregate_axes

    def forward(self, x):
        xp = cuda.get_array_module(x)
        self.in_shape = x.shape
        self.aggregate_axes = tuple(numpy.setdiff1d(six.moves.range(x.ndim), self.non_aggregate_axes))
        return utils.force_array(xp.sum(x, self.aggregate_axes))

    def backward(self, gy):
        xp = cuda.get_array_module(gy)
        new_shape = list(s for s in self.in_shape)
        for a in self.aggregate_axes:
            new_shape[a] = 1
        gy = gy.reshape(new_shape)
        ret = xp.zeros(self.in_shape, dtype=gy.dtype)
        ret[...] = gy
        return ret


class Aggregator(object):

    def __init__(self, aggregate_option):
        if aggregate_option == 'elementwise':
            self.aggregator = None
        elif aggregate_option == 'samplewise':
            self.aggregator = AxesAggregator((0,))
        elif aggregate_option == 'sum':
            self.aggregator = AxesAggregator(())
        elif aggregate_option == 'mean':
            self.aggregator = AxesAggregator(())
        elif aggregate_option == 'divide':
            self.aggregator = AxesAggregator(())
        else:
            raise ValueError('No such aggregate_option:{}'.format(aggregate_option))
        self.aggregate_option = aggregate_option

    def forward(self, x, *args, **kwargs):
        self.sample_size = len(x)
        if self.aggregator:
            x = self.aggregator.forward(x)

        if self.aggregate_option == 'mean':
            x /= self.sample_size
        elif self.aggregate_option == 'divide':
            count = kwargs.pop('count', 1)
            x /= count
        return x

    def backward(self, gy, *args, **kwargs):
        if self.aggregate_option == 'mean':
            gy /= self.sample_size
        elif self.aggregate_option == 'divide':
            count = kwargs.pop('count', 1)
            gy /= count

        if self.aggregator:
            gy = self.aggregator.backward(gy)
        return gy


class AggregateFunction(function.Function):

    def __init__(self, aggregate_option):
        self.aggregator = Aggregator(aggregate_option)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)

    def forward(self, x):
        return self.aggregator.forward(x[0]),

    def backward(self, x, gy):
        return self.aggregator.backward(gy[0]),


def aggregate(x, aggregate_option):
    return AggregateFunction(aggregate_option)(x)
