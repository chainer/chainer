import unittest

import numpy

import chainer
from chainer.backends import cuda
import chainer.functions as F
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
import chainerx


class UnaryFunctionsTestBase(unittest.TestCase):

    def make_data(self):
        raise NotImplementedError

    def setUp(self):
        self.x, self.gy = self.make_data()
        self.check_forward_options = {'atol': 1e-7, 'rtol': 1e-7}
        if self.dtype == numpy.float16:
            self.check_backward_options = {'atol': 3e-3, 'rtol': 1e-2}
            self.check_double_backward_options = {'atol': 3e-3, 'rtol': 1e-2}
        else:
            self.check_backward_options = {'atol': 1e-4, 'rtol': 1e-3}
            self.check_double_backward_options = {'atol': 1e-4, 'rtol': 1e-3}

    def check_forward(self, op, op_np, x_data):
        x = chainer.Variable(x_data)
        y = op(x)
        self.assertEqual(x.data.dtype, y.data.dtype)
        testing.assert_allclose(
            op_np(self.x), y.data, **self.check_forward_options)

    def check_forward_cpu(self, op, op_np):
        self.check_forward(op, op_np, self.x)

    def check_forward_gpu(self, op, op_np):
        self.check_forward(op, op_np, cuda.to_gpu(self.x))

    def check_forward_chainerx(self, op, op_np):
        self.check_forward(op, op_np, chainerx.array(self.x))

    def check_backward(self, op, x_data, y_grad):
        gradient_check.check_backward(
            op, x_data, y_grad, dtype=numpy.float64,
            **self.check_backward_options)

    def check_backward_cpu(self, op):
        self.check_backward(op, self.x, self.gy)

    def check_backward_gpu(self, op):
        self.check_backward(op, cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    def check_backward_chainerx(self, op):
        self.check_backward(
            op, chainerx.array(self.x), chainerx.array(self.gy))

    def check_double_backward(self, op, x_data, y_grad, y_grad_grad):
        gradient_check.check_double_backward(
            op, x_data, y_grad, y_grad_grad, dtype=numpy.float64,
            **self.check_double_backward_options)

    def check_double_backward_cpu(self, op):
        self.check_double_backward(op, self.x, self.gy, self.ggy)

    def check_double_backward_gpu(self, op):
        self.check_double_backward(op, cuda.to_gpu(
            self.x), cuda.to_gpu(self.gy), cuda.to_gpu(self.ggy))

    def check_double_backward_chainerx(self, op):
        self.check_double_backward(op, chainerx.array(
            self.x), chainerx.array(self.gy), chainerx.array(self.ggy))

    def check_label(self, op, expected):
        self.assertEqual(op().label, expected)


@testing.parameterize(*testing.product({
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestExp(UnaryFunctionsTestBase):

    def make_data(self):
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.ggy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return x, gy

    def test_forward_cpu(self):
        self.check_forward_cpu(F.exp, numpy.exp)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward_gpu(F.exp, numpy.exp)

    @attr.chainerx
    def test_forward_chainerx(self):
        self.check_forward_chainerx(F.exp, numpy.exp)

    def test_backward_cpu(self):
        self.check_backward_cpu(F.exp)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward_gpu(F.exp)

    @attr.chainerx
    def test_backward_chainerx(self):
        self.check_backward_chainerx(F.exp)

    def test_label(self):
        self.check_label(chainer.functions.math.exponential.Exp, 'exp')

    def test_double_backward_cpu(self):
        self.check_double_backward_cpu(F.exp)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward_gpu(F.exp)

    @attr.chainerx
    def test_double_backward_chainerx(self):
        self.check_double_backward_chainerx(F.exp)


@testing.parameterize(*testing.product({
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestLog(UnaryFunctionsTestBase):

    def make_data(self):
        x = numpy.random.uniform(.5, 1, self.shape).astype(self.dtype)
        gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.ggy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return x, gy

    def test_forward_cpu(self):
        self.check_forward_cpu(F.log, numpy.log)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward_gpu(F.log, numpy.log)

    @attr.chainerx
    def test_forward_chainerx(self):
        self.check_forward_chainerx(F.log, numpy.log)

    def test_backward_cpu(self):
        self.check_backward_cpu(F.log)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward_gpu(F.log)

    @attr.chainerx
    def test_backward_chainerx(self):
        self.check_backward_chainerx(F.log)

    def test_label(self):
        self.check_label(chainer.functions.math.exponential.Log, 'log')

    def test_double_backward_cpu(self):
        self.check_double_backward_cpu(F.log)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward_gpu(F.log)

    @attr.chainerx
    def test_double_backward_chainerx(self):
        self.check_double_backward_chainerx(F.log)


@testing.parameterize(*testing.product({
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestLog2(UnaryFunctionsTestBase):

    def make_data(self):
        x = numpy.random.uniform(.5, 1, self.shape).astype(self.dtype)
        gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.ggy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return x, gy

    def test_forward_cpu(self):
        self.check_forward_cpu(F.log2, numpy.log2)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward_gpu(F.log2, numpy.log2)

    def test_backward_cpu(self):
        self.check_backward_cpu(F.log2)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward_gpu(F.log2)

    def test_label(self):
        self.check_label(chainer.functions.math.exponential.Log2, 'log2')

    def test_double_backward_cpu(self):
        self.check_double_backward_cpu(F.log2)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward_gpu(F.log2)


@testing.parameterize(*testing.product({
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestLog10(UnaryFunctionsTestBase):

    def make_data(self):
        x = numpy.random.uniform(.5, 1, self.shape).astype(self.dtype)
        gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.ggy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return x, gy

    def test_forward_cpu(self):
        self.check_forward_cpu(F.log10, numpy.log10)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward_gpu(F.log10, numpy.log10)

    def test_backward_cpu(self):
        self.check_backward_cpu(F.log10)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward_gpu(F.log10)

    def test_label(self):
        self.check_label(chainer.functions.math.exponential.Log10, 'log10')

    def test_double_backward_cpu(self):
        self.check_double_backward_cpu(F.log2)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward_gpu(F.log2)


testing.run_module(__name__, __file__)
