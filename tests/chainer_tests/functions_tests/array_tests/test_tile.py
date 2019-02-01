import unittest

import numpy

from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product({
    'in_shape': [(), 2, (2, 3)],
    'reps': [(), 0, 2, (0, 0), (1, 2), (2, 2), (2, 0)],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestTile(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.in_shape).astype(self.dtype)
        out_shape = numpy.tile(self.x, self.reps).shape
        self.gy = numpy.random.uniform(-1, 1, out_shape).astype(self.dtype)
        self.ggx = numpy.random.uniform(-1, 1, self.in_shape) \
            .astype(self.dtype)

        self.check_forward_options = {}
        self.check_backward_options = {'dtype': numpy.float64}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 5e-4, 'rtol': 5e-3}
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 2 ** -4, 'rtol': 2 ** -4}

    def check_forward(self, x_data):
        y = functions.tile(x_data, self.reps)
        y_expected = numpy.tile(self.x, self.reps)
        self.assertEqual(y.dtype, y_expected.dtype)
        testing.assert_allclose(
            y.data, y_expected, **self.check_forward_options)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        def f(x):
            return functions.tile(x, self.reps)

        gradient_check.check_backward(
            f, x_data, y_grad, **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    def check_double_backward(self, x_data, y_grad, x_grad_grad):
        def f(x):
            return functions.tile(x, self.reps)

        gradient_check.check_double_backward(
            f, x_data, y_grad, x_grad_grad, **self.check_backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(self.x, self.gy, self.ggx)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy),
                                   cuda.to_gpu(self.ggx))


@testing.parameterize(*testing.product({
    'reps': [-1, (-1, -1)],
}))
class TestTileValueError(unittest.TestCase):

    def test_value_error(self):
        x = numpy.random.uniform(-1, 1, (2,)).astype(numpy.float32)
        with self.assertRaises(ValueError):
            functions.tile(x, self.reps)


class TestTileTypeError(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2,)).astype(numpy.float32)

    def test_reps_not_int(self):
        with self.assertRaises(TypeError):
            functions.tile(self.x, 'a')

    def test_x_not_ndarray_or_variable(self):
        with self.assertRaises(TypeError):
            functions.tile((self.x, self.x), 2)


testing.run_module(__name__, __file__)
