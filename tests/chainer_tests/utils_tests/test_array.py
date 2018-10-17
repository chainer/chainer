import unittest

import numpy

from chainer import testing
from chainer.utils import array


@testing.parameterize(
    {'shape': ()},
    {'shape': (2, 3)},
    {'shape': (1,)},
    {'shape': (0,)},
    {'shape': (5, 6, 7)},
    {'shape': (0, 3)},
    {'shape': (2, 0)},
    {'shape': (5, 0, 7)},
)
class TestSizeOfShape(unittest.TestCase):

    def test_size_of_shape(self):
        arr = numpy.empty(self.shape)
        size = array.size_of_shape(arr.shape)
        size_expect = arr.size
        assert type(size) == type(size_expect)
        assert size == size_expect


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
