import collections

import numpy
import six

from chainer import cuda


class Summary(object):

    """Online summarization of a sequence of scalars.

    TODO(beam2d): document it.

    """
    def __init__(self):
        self._x = 0
        self._x2 = 0
        self._n = 0

    def add(self, value):
        self._x += value
        self._x2 += value * value
        self._n += 1

    @property
    def mean(self):
        return self._x / self._n

    @property
    def variance(self):
        mean = self.mean
        return (self._x2 / self._n) - (mean * mean)

    @property
    def std(self):
        return self.variance ** 0.5


class DictSummary(object):

    """Online summarization of a sequence of dictionaries.

    TODO(beam2d): document it.

    """
    def __init__(self):
        self._summaries = collections.defaultdict(Summary)

    def add(self, d):
        for key, value in six.iteritems(d):
            if isinstance(value, (float, numpy.ndarray, cuda.ndarray)):
                self._summaries[key].add(value)

    @property
    def mean(self):
        return {key: summary.mean
                for key, summary in six.iteritems(self._summaries)}

    @property
    def variance(self):
        return {key: summary.variance
                for key, summary in six.iteritems(self._summaries)}

    @property
    def std(self):
        return {key: summay.std
                for key, summary in six.iteritems(self._summaries)}
