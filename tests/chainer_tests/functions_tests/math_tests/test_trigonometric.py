import unittest

import numpy

import chainer
from chainer import cuda
import chainer.functions as F
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


@testing.parameterize(*testing.product({
    'func_name': ['cos', 'sin', 'tan'],
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TrigonometricFunctionsTest(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(.5, 1, self.shape).astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.func = getattr(F, self.func_name)
        camel_name = self.func_name[0].upper() + self.func_name[1:]
        self.func_class = getattr(F, camel_name)
        self.np_func = getattr(numpy, self.func_name)

        if self.dtype == numpy.float16:
            self.backward_options = {
                'eps': 2 ** -4, 'atol': 2 ** -4, 'rtol': 2 ** -4}
        else:
            self.backward_options = {}

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = self.func(x)
        testing.assert_allclose(
            self.np_func(self.x), y.data, atol=1e-4, rtol=1e-4)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            self.func, x_data, y_grad, **self.backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    def test_label(self):
        self.assertEqual(self.func_class().label, self.func_name)


def make_data(shape, dtype):
    x = numpy.random.uniform(-.9, .9, shape).astype(dtype)
    gy = numpy.random.uniform(-1, 1, shape).astype(dtype)
    return x, gy


@testing.unary_math_function_unittest(F.Arcsin(), make_data=make_data)
class TestArcsin(unittest.TestCase):
    pass


@testing.unary_math_function_unittest(F.Arccos(), make_data=make_data)
class TestArccos(unittest.TestCase):
    pass


@testing.unary_math_function_unittest(F.Arctan())
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
        if self.dtype == numpy.float16:
            self.backward_options = {
                'eps': 2 ** -4, 'atol': 2 ** -4, 'rtol': 2 ** -4}
        else:
            self.backward_options = {
                'atol': 1e-3, 'rtol': 1e-3}

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

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x1, self.x2)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x1), cuda.to_gpu(self.x2))

    def check_backward(self, x1_data, x2_data, y_grad):
        gradient_check.check_backward(
            F.arctan2, (x1_data, x2_data), y_grad, **self.backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x1, self.x2, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x1),
                            cuda.to_gpu(self.x2),
                            cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)
