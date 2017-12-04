import unittest

import numpy

from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product({
    # repeats is any of (int, bool or tuple) and
    # axis is any of (int or None).
    'shape_repeats_axis': [
        # 1-D
        (2, 0, None),
        (2, 1, None),
        (2, 2, 0),
        (2, True, None),
        (2, (2,), None),
        (2, (True,), 0),
        (2, (1, 2), None),
        (2, (True, 2), 0),
        # 2-D
        ((3, 2), 2, 0),
        ((3, 2), (2,), None),
        ((3, 2), 2, 1),
        ((3, 2), (3, 4, 3), 0),
        ((3, 2), (3, True), 1),
        ((3, 2), (True,) * 6, None),
        # 3-D
        ((3, 2, 3), (3, 2, True), 0),
        ((3, 2, 3), (3, 4), 1),
        ((3, 2, 3), (3, 2, 1), 2),
    ],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestRepeat(unittest.TestCase):

    def setUp(self):
        (self.in_shape, self.repeats, self.axis) = self.shape_repeats_axis
        self.x = numpy.random.uniform(-1, 1, self.in_shape).astype(self.dtype)
        out_shape = numpy.repeat(self.x, self.repeats, self.axis).shape
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
        y = functions.repeat(x_data, self.repeats, self.axis)
        y_expected = numpy.repeat(self.x, self.repeats, self.axis)
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
            return functions.repeat(x, self.repeats, self.axis)

        gradient_check.check_backward(
            f, x_data, y_grad, **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    def check_double_backward(self, x_data, y_grad, x_grad_grad):
        def f(x):
            y = functions.repeat(x, self.repeats, self.axis)
            return y * y

        gradient_check.check_double_backward(
            f, x_data, y_grad, x_grad_grad, **self.check_backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(self.x, self.gy, self.ggx)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy),
                                   cuda.to_gpu(self.ggx))


@testing.parameterize(*testing.product({
    'repeats': [-1, (-1, -1)],
    'axis': [-1],
}))
class TestRepeatValueError(unittest.TestCase):

    def test_value_error(self):
        x = numpy.random.uniform(-1, 1, (2,)).astype(numpy.float32)
        with self.assertRaises(ValueError):
            functions.repeat(x, self.repeats, self.axis)


class TestRepeatTypeError(unittest.TestCase):

    def test_type_error_repeats_str(self):
        x = numpy.random.uniform(-1, 1, (2,)).astype(numpy.float32)
        with self.assertRaises(TypeError):
            functions.repeat(x, 'a')

    def test_type_error_axis_str(self):
        x = numpy.random.uniform(-1, 1, (2,)).astype(numpy.float32)
        with self.assertRaises(TypeError):
            functions.repeat(x, 1, 'a')

    def test_type_error_axis_bool(self):
        x = numpy.random.uniform(-1, 1, (2,)).astype(numpy.float32)
        with self.assertRaises(TypeError):
            functions.repeat(x, 1, True)


testing.run_module(__name__, __file__)
