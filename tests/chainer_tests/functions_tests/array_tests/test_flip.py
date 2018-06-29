import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.utils import type_check


@testing.parameterize(*testing.product_dict(
    [
        {'shape': (1,), 'axis': 0},
        {'shape': (2, 3, 4), 'axis': 0},
        {'shape': (2, 3, 4), 'axis': 1},
        {'shape': (2, 3, 4), 'axis': 2},
        {'shape': (2, 3, 4), 'axis': -3},
        {'shape': (2, 3, 4), 'axis': -2},
        {'shape': (2, 3, 4), 'axis': -1},
    ],
    [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64},
    ],
))
class TestFlip(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.ggx = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)

        self.check_double_backward_options = {'atol': 1e-3, 'rtol': 1e-2}
        if self.dtype == numpy.float16:
            self.check_double_backward_options.update(dtype=numpy.float64)

    def check_forward(self, x_data, axis):
        x = chainer.Variable(x_data)
        y = functions.flip(x, axis)

        flip_func = getattr(numpy, 'flip', functions.array.flip._flip)
        expected_y = flip_func(x_data, axis)
        testing.assert_allclose(y.data, expected_y)

    def test_forward_cpu(self):
        self.check_forward(self.x, self.axis)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), self.axis)

    def check_backward(self, x_data, axis, y_grad):
        gradient_check.check_backward(lambda x: functions.flip(x, axis),
                                      x_data, y_grad, dtype=numpy.float64)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.axis, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.x), self.axis, cuda.to_gpu(self.gy))

    def check_double_backward(self, x_data, axis, y_grad, x_grad_grad):
        def f(x):
            return functions.flip(x, axis)

        gradient_check.check_double_backward(
            f, x_data, y_grad, x_grad_grad,
            **self.check_double_backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(self.x, self.axis, self.gy, self.ggx)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(cuda.to_gpu(self.x), self.axis,
                                   cuda.to_gpu(self.gy),
                                   cuda.to_gpu(self.ggx))


@testing.parameterize(
    {'axis': 3},
    {'axis': -4},
)
class TestFlipInvalidTypeAxis(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f')

    def check_type_error(self, x):
        with self.assertRaises(type_check.InvalidType):
            functions.flip(x, self.axis)

    def test_type_error_cpu(self):
        self.check_type_error(self.x)

    @attr.gpu
    def test_type_error_gpu(self):
        self.check_type_error(cuda.to_gpu(self.x))


class TestFlipInvalidTypeError(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f')

    def test_invalid_axis(self):
        with self.assertRaises(TypeError):
            functions.flip(self.x, 'a')


@testing.parameterize(*testing.product_dict(
    [
        {'shape': (1,), 'axis': 0},
        {'shape': (2, 3, 4), 'axis': 0},
        {'shape': (2, 3, 4), 'axis': 1},
        {'shape': (2, 3, 4), 'axis': 2},
        {'shape': (2, 3, 4), 'axis': -3},
        {'shape': (2, 3, 4), 'axis': -2},
        {'shape': (2, 3, 4), 'axis': -1},
    ],
    [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64},
    ],
))
@testing.with_requires('numpy>=1.12.0')
class TestFlipFunction(unittest.TestCase):

    def test_equal_to_numpy_flip(self):
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        numpy.testing.assert_array_equal(
            functions.array.flip._flip(x, self.axis),
            numpy.flip(x, self.axis))


testing.run_module(__name__, __file__)
