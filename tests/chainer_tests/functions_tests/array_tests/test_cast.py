import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product_dict(
    [
        {'shape': (3, 4)},
        {'shape': ()},
    ],
    [
        {'in_type': numpy.bool_},
        {'in_type': numpy.uint8},
        {'in_type': numpy.uint64},
        {'in_type': numpy.int8},
        {'in_type': numpy.int64},
        {'in_type': numpy.float16},
        {'in_type': numpy.float32},
        {'in_type': numpy.float64},
    ],
    [
        {'out_type': numpy.bool_},
        {'out_type': numpy.uint8},
        {'out_type': numpy.uint64},
        {'out_type': numpy.int8},
        {'out_type': numpy.int64},
        {'out_type': numpy.float16},
        {'out_type': numpy.float32},
        {'out_type': numpy.float64},
    ]
))
class TestCast(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.in_type)
        self.g = numpy.random.uniform(-1, 1, self.shape).astype(self.out_type)

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = functions.cast(x, self.out_type)
        assert y.data.shape == x.data.shape
        assert y.data.dtype == self.out_type

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, g_data):
        if (numpy.dtype(self.in_type).kind != 'f'
                or numpy.dtype(self.out_type).kind != 'f'):
            raise unittest.SkipTest('Non-float dtypes')

        def func(x):
            return functions.cast(x, self.out_type)

        gradient_check.check_backward(
            func, x_data, g_data, dtype='d',
            eps=2.0 ** -2, atol=1e-2, rtol=1e-3)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.g)


class TestNoCast(unittest.TestCase):

    def setUp(self):
        self.dtype = numpy.float32
        self.x = numpy.empty(1, self.dtype)

    def check_forward_no_cast(self, x_data):
        y = functions.cast(x_data, self.dtype)
        assert isinstance(y, chainer.Variable)
        assert y.data is x_data

    def test_forward_no_cast_array(self):
        y = functions.cast(self.x, self.dtype)
        assert isinstance(y, chainer.Variable)
        assert y.data is self.x

    def test_forward_no_cast_variable(self):
        # If backprop is disabled, it's safe to simply return the input
        # variable for no-op casts.
        x = chainer.Variable(self.x)
        with chainer.using_config('enable_backprop', False):
            y = functions.cast(x, self.dtype)
        assert y is x

    def test_forward_no_cast_grad(self):
        # This test would fail if F.cast does not create new function nodes for
        # no-op casts
        x = chainer.Variable(self.x)
        y1 = functions.cast(x, self.dtype)
        y2 = functions.cast(x, self.dtype)
        z = y1 + y2
        gy1, gy2 = chainer.grad([z], [y1, y2], [numpy.ones_like(z.data)])
        assert gy1.dtype == self.dtype
        assert gy2.dtype == self.dtype
        numpy.testing.assert_array_equal(gy1.data, numpy.ones_like(y1.data))
        numpy.testing.assert_array_equal(gy2.data, numpy.ones_like(y2.data))


testing.run_module(__name__, __file__)
