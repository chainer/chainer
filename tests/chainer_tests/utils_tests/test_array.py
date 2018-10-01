import unittest

import numpy

from chainer import testing
from chainer.utils import array


@testing.parameterize(
    {'in_shape': (), 'out_shape': ()},
    {'in_shape': (3,), 'out_shape': ()},
    {'in_shape': (3,), 'out_shape': (1,)},
    {'in_shape': (3,), 'out_shape': (3,)},
    {'in_shape': (2, 3), 'out_shape': ()},
    {'in_shape': (2, 3), 'out_shape': (3,)},
    {'in_shape': (2, 3), 'out_shape': (1, 3)},
    {'in_shape': (2, 3), 'out_shape': (2, 1)},
    {'in_shape': (2, 3), 'out_shape': (2, 3)},
    {'in_shape': (2, 3, 4), 'out_shape': (3, 1)},
)
class TestSumTo(unittest.TestCase):

    def test_sum_to(self):
        n_elems = numpy.prod(self.in_shape)
        x = numpy.arange(1, n_elems + 1, dtype=numpy.float32).reshape(
            self.in_shape)
        y_actual = array.sum_to(x, self.out_shape)

        y_expect = numpy.zeros(self.out_shape, x.dtype)
        for dst, src in numpy.nditer(
                [y_expect, x], ['reduce_ok'], [['readwrite'], ['readonly']]):
            dst += src
        numpy.testing.assert_array_equal(y_expect, y_actual)


testing.run_module(__name__, __file__)
