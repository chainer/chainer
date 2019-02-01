import unittest

import numpy

import chainer
from chainer.backends import cuda
import chainer.functions as F
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product({
    'func_name': ['cos', 'sin', 'tan'],
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TrigonometricFunctionsTest(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(.5, 1, self.shape).astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.ggx = numpy.random.uniform(.5, 1, self.shape).astype(self.dtype)
        self.func = getattr(F, self.func_name)
        camel_name = self.func_name[0].upper() + self.func_name[1:]
        self.func_class = getattr(
            chainer.functions.math.trigonometric, camel_name)
        self.np_func = getattr(numpy, self.func_name)

        if self.dtype == numpy.float16:
            self.backward_options = {'eps': 1e-3, 'atol': 1e-2, 'rtol': 1e-2}
            self.double_backward_options = {
                'eps': 1e-3, 'atol': 1e-2, 'rtol': 1e-2}
        else:
            self.backward_options = {'atol': 1e-4, 'rtol': 1e-3}
            self.double_backward_options = {'atol': 1e-4, 'rtol': 1e-3}

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = self.func(x)
        testing.assert_allclose(
            self.np_func(self.x), y.data, atol=1e-4, rtol=1e-4)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            self.func, x_data, y_grad, dtype='d', **self.backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    def check_double_backward(self, x_data, y_grad, x_grad_grad):
        gradient_check.check_double_backward(
            self.func, x_data, y_grad, x_grad_grad, dtype='d',
            **self.double_backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(self.x, self.gy, self.ggx)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(cuda.to_gpu(
            self.x), cuda.to_gpu(self.gy), cuda.to_gpu(self.ggx))

    def test_label(self):
        self.assertEqual(self.func_class().label, self.func_name)


def make_data(shape, dtype):
    x = numpy.random.uniform(-.9, .9, shape).astype(dtype)
    gy = numpy.random.uniform(-1, 1, shape).astype(dtype)
    ggx = numpy.random.uniform(-.9, .9, shape).astype(dtype)
    return x, gy, ggx


@testing.unary_math_function_unittest(
    F.arcsin,
    make_data=make_data,
    forward_options={'atol': 1e-3, 'rtol': 1e-3},
    double_backward_options={'eps': 1e-3},
)
class TestArcsin(unittest.TestCase):
    pass


@testing.unary_math_function_unittest(
    F.arccos,
    make_data=make_data,
    forward_options={'atol': 1e-3, 'rtol': 1e-3},
    double_backward_options={'eps': 1e-3},
)
class TestArccos(unittest.TestCase):
    pass


@testing.unary_math_function_unittest(F.arctan, make_data=make_data)
class TestArctan(unittest.TestCase):
    pass


@testing.parameterize(*testing.product({
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestArctan2(unittest.TestCase):

    def setUp(self):
        self.x1 = numpy.random.uniform(
            -10.0, 10.0, self.shape).astype(self.dtype)
        self.x2 = numpy.random.uniform(
            -10.0, 10.0, self.shape).astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.ggx1 = numpy.random.uniform(
            -10.0, 10.0, self.shape).astype(self.dtype)
        self.ggx2 = numpy.random.uniform(
            -10.0, 10.0, self.shape).astype(self.dtype)
        if self.dtype == numpy.float16:
            self.backward_options = {
                'eps': 1e-3, 'atol': 2 ** -4, 'rtol': 2 ** -4}
            self.double_backward_options = {
                'eps': 1e-3, 'atol': 2 ** -4, 'rtol': 2 ** -4}
        else:
            self.backward_options = {
                'atol': 1e-3, 'rtol': 1e-3}
            self.double_backward_options = {
                'atol': 1e-3, 'rtol': 1e-3}

        # Avoid non-differentiable point
        self.x1[(abs(self.x1) < 1e-2) & (self.x2 < 0)] = 1
        self.ggx1[(abs(self.ggx1) < 1e-2) & (self.ggx2 < 0)] = 1

    def check_forward(self, x1_data, x2_data):
        y = F.arctan2(x1_data, x2_data)
        numpy.testing.assert_array_less(
            cuda.to_cpu(y.data),
            numpy.full(y.shape, numpy.pi))
        numpy.testing.assert_array_less(
            numpy.full(y.shape, -numpy.pi),
            cuda.to_cpu(y.data))
        testing.assert_allclose(
            numpy.arctan2(self.x1, self.x2), y.data, atol=1e-4, rtol=1e-4)

    def test_forward_cpu(self):
        self.check_forward(self.x1, self.x2)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x1), cuda.to_gpu(self.x2))

    def check_backward(self, x1_data, x2_data, y_grad):
        gradient_check.check_backward(
            F.arctan2, (x1_data, x2_data), y_grad, dtype='d',
            **self.backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x1, self.x2, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x1),
                            cuda.to_gpu(self.x2),
                            cuda.to_gpu(self.gy))

    def check_double_backward(
            self, x1_data, x2_data, y_grad, x1_grad_grad, x2_grad_grad):
        gradient_check.check_double_backward(
            F.arctan2, (x1_data, x2_data), y_grad,
            (x1_grad_grad, x2_grad_grad), dtype='d',
            **self.double_backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(
            self.x1, self.x2, self.gy, self.ggx1, self.ggx2)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.x1),
            cuda.to_gpu(self.x2),
            cuda.to_gpu(self.gy),
            cuda.to_gpu(self.ggx1),
            cuda.to_gpu(self.ggx2))


testing.run_module(__name__, __file__)
