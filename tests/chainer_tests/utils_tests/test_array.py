import unittest

import numpy

from chainer import testing
from chainer.utils import array


@testing.parameterize(
    {'in_shape': (), 'out_shape': (), 'axis': (), 'lead': 0},
    {'in_shape': (3,), 'out_shape': (), 'axis': (0,), 'lead': 1},
    {'in_shape': (3,), 'out_shape': (1,), 'axis': (0,), 'lead': 0},
    {'in_shape': (3,), 'out_shape': (3,), 'axis': (), 'lead': 0},
    {'in_shape': (2, 3), 'out_shape': (), 'axis': (0, 1), 'lead': 2},
    {'in_shape': (2, 3), 'out_shape': (3,), 'axis': (0,), 'lead': 1},
    {'in_shape': (2, 3), 'out_shape': (1, 3), 'axis': (0,), 'lead': 0},
    {'in_shape': (2, 3), 'out_shape': (2, 1), 'axis': (1,), 'lead': 0},
    {'in_shape': (2, 3), 'out_shape': (2, 3), 'axis': (), 'lead': 0},
    {'in_shape': (2, 3, 4), 'out_shape': (3, 1), 'axis': (0, 2), 'lead': 1},
)
class TestSumTo(unittest.TestCase):

    def test_sum_to(self):
        n_elems = numpy.prod(self.in_shape)
        x = numpy.arange(1, n_elems + 1, dtype=numpy.float32).reshape(self.in_shape)
        y_actual = array.sum_to(x, self.out_shape)

        y_expect = numpy.squeeze(x.sum(self.axis, keepdims=True), tuple(range(self.lead)))
        numpy.testing.assert_array_equal(y_expect, y_actual)


testing.run_module(__name__, __file__)
