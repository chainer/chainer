import unittest

import numpy

import chainer
from chainer import basic_math
from chainer import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


class BinaryOpTestBase(object):

    def make_data(self):
        raise NotImplementedError()

    def setUp(self):
        self.x1, self.x2, self.gy = self.make_data()

    def check_forward(self, op, x1_data, x2_data):
        x1 = chainer.Variable(x1_data)
        x2 = chainer.Variable(x2_data)
        y = op(x1, x2)
        gradient_check.assert_allclose(op(self.x1, self.x2), y.data)

    def forward_cpu(self, op):
        self.check_forward(op, self.x1, self.x2)

    @condition.retry(3)
    def test_add_forward_cpu(self):
        self.forward_cpu(lambda x, y: x + y)

    @condition.retry(3)
    def test_sub_forward_cpu(self):
        self.forward_cpu(lambda x, y: x - y)

    @condition.retry(3)
    def test_mul_forward_cpu(self):
        self.forward_cpu(lambda x, y: x * y)

    @condition.retry(3)
    def test_div_forward_cpu(self):
        self.forward_cpu(lambda x, y: x / y)

    @condition.retry(3)
    def test_pow_forward_cpu(self):
        self.forward_cpu(lambda x, y: x ** y)

    @condition.retry(3)
    def test_radd_forward_cpu(self):
        self.forward_cpu(lambda x, y: y.__radd__(x))

    @condition.retry(3)
    def test_rsub_forward_cpu(self):
        self.forward_cpu(lambda x, y: y.__rsub__(x))

    @condition.retry(3)
    def test_rmul_forward_cpu(self):
        self.forward_cpu(lambda x, y: y.__rmul__(x))

    @condition.retry(3)
    def test_rdiv_forward_cpu(self):
        self.forward_cpu(lambda x, y: y.__rtruediv__(x))

    @condition.retry(3)
    def test_rpow_forward_cpu(self):
        self.forward_cpu(lambda x, y: y.__rpow__(x))

    def forward_gpu(self, op):
        self.check_forward(op, cuda.to_gpu(self.x1), cuda.to_gpu(self.x2))

    @attr.gpu
    @condition.retry(3)
    def test_add_forward_gpu(self):
        self.forward_gpu(lambda x, y: x + y)

    @attr.gpu
    @condition.retry(3)
    def test_sub_forward_gpu(self):
        self.forward_gpu(lambda x, y: x - y)

    @attr.gpu
    @condition.retry(3)
    def test_mul_forward_gpu(self):
        self.forward_gpu(lambda x, y: x * y)

    @attr.gpu
    @condition.retry(3)
    def test_div_forward_gpu(self):
        self.forward_gpu(lambda x, y: x / y)

    @attr.gpu
    @condition.retry(3)
    def test_pow_forward_gpu(self):
        self.forward_gpu(lambda x, y: x ** y)

    @attr.gpu
    @condition.retry(3)
    def test_radd_forward_gpu(self):
        self.forward_gpu(lambda x, y: y.__radd__(x))

    @attr.gpu
    @condition.retry(3)
    def test_rsub_forward_gpu(self):
        self.forward_gpu(lambda x, y: y.__rsub__(x))

    @attr.gpu
    @condition.retry(3)
    def test_rmul_forward_gpu(self):
        self.forward_gpu(lambda x, y: y.__rmul__(x))

    @attr.gpu
    @condition.retry(3)
    def test_rdiv_forward_gpu(self):
        self.forward_gpu(lambda x, y: y.__rtruediv__(x))

    @attr.gpu
    @condition.retry(3)
    def test_rpow_forward_gpu(self):
        self.forward_gpu(lambda x, y: y.__rpow__(x))

    @attr.gpu
    def test_add_constant_allocation(self):
        x = 0
        y = chainer.Variable(cuda.cupy.ones((1,)))
        z = y + x
        self.assertEqual(1, z.data.get()[0])

    def check_backward(self, op, x1_data, x2_data, y_grad, atol):
        gradient_check.check_backward(op, (x1_data, x2_data), y_grad,
                                      atol=atol)

    def backward_cpu(self, op, atol=1e-5):
        self.check_backward(op, self.x1, self.x2, self.gy, atol)

    @condition.retry(3)
    def test_add_backward_cpu(self):
        self.backward_cpu(lambda x, y: x + y)

    @condition.retry(3)
    def test_sub_backward_cpu(self):
        self.backward_cpu(lambda x, y: x - y)

    @condition.retry(3)
    def test_mul_backward_cpu(self):
        self.backward_cpu(lambda x, y: x * y)

    @condition.retry(3)
    def test_div_backward_cpu(self):
        self.backward_cpu(lambda x, y: x / y)

    @condition.retry(3)
    def test_pow_backward_cpu(self):
        self.backward_cpu(lambda x, y: x ** y, atol=1e-4)

    def backward_gpu(self, op, atol=1e-5):
        self.check_backward(
            op, cuda.to_gpu(self.x1), cuda.to_gpu(self.x2),
            cuda.to_gpu(self.gy), atol)

    @attr.gpu
    @condition.retry(3)
    def test_add_backward_gpu(self):
        self.backward_gpu(lambda x, y: x + y)

    @attr.gpu
    @condition.retry(3)
    def test_sub_backward_gpu(self):
        self.backward_gpu(lambda x, y: x - y)

    @attr.gpu
    @condition.retry(3)
    def test_mul_backward_gpu(self):
        self.backward_gpu(lambda x, y: x * y)

    @attr.gpu
    @condition.retry(3)
    def test_div_backward_gpu(self):
        self.backward_gpu(lambda x, y: x / y)

    @attr.gpu
    @condition.retry(3)
    def test_pow_backward_gpu(self):
        self.backward_gpu(lambda x, y: x ** y, atol=1e-4)


class TestBinaryOpSimple(BinaryOpTestBase, unittest.TestCase):

    def make_data(self):
        x1 = numpy.random.uniform(.5, 1, (3, 2)).astype(numpy.float32)
        x2 = numpy.random.uniform(.5, 1, (3, 2)).astype(numpy.float32)
        gy = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)
        return x1, x2, gy


class TestBinaryOpZeroDimension(BinaryOpTestBase, unittest.TestCase):

    def make_data(self):
        x1 = numpy.random.uniform(.5, 1, ()).astype(numpy.float32)
        x2 = numpy.random.uniform(.5, 1, ()).astype(numpy.float32)
        gy = numpy.random.uniform(-1, 1, ()).astype(numpy.float32)
        return x1, x2, gy


class TestBinaryOpConstant(unittest.TestCase):

    def _test_constant_one(self, func, lhs, rhs, gpu=False):
        if gpu:
            lhs = cuda.to_gpu(lhs)
        x = chainer.Variable(lhs)
        y = func(x, rhs)
        self.assertEqual(y.data.dtype, numpy.float32)
        y.backward()
        self.assertEqual(x.grad.dtype, numpy.float32)

    def _test_constant(self, func):
        x_data = numpy.array(1, numpy.float32)

        self._test_constant_one(func, x_data, 1)
        self._test_constant_one(func, x_data, 1.0)
        self._test_constant_one(func, x_data, numpy.int64(1))
        self._test_constant_one(func, x_data, numpy.float64(1.0))

    def _test_constant_gpu(self, func):
        x_data = numpy.array(1, numpy.float32)

        self._test_constant_one(func, x_data, 1, True)
        self._test_constant_one(func, x_data, 1.0, True)
        self._test_constant_one(func, x_data, numpy.int64(1), True)
        self._test_constant_one(func, x_data, numpy.float64(1), True)

    def _test_constant_array_one(self, func, lhs, rhs):
        x = chainer.Variable(lhs)
        y = func(x, rhs)
        self.assertEqual(y.data.dtype, numpy.float32)
        y.grad = numpy.ones_like(y.data, numpy.float32)
        y.backward()
        self.assertEqual(x.grad.dtype, numpy.float32)

    def _test_constant_array(self, func):
        x_data = numpy.array([1.0, 2.0], numpy.float32)

        self._test_constant_array_one(
            func, x_data, numpy.array([3.0, 4.0], numpy.int32))
        self._test_constant_array_one(
            func, x_data, numpy.array([3.0, 4.0], numpy.int64))
        self._test_constant_array_one(
            func, x_data, numpy.array([3.0, 4.0], numpy.float32))
        self._test_constant_array_one(
            func, x_data, numpy.array([3.0, 4.0], numpy.float64))

        with self.assertRaises(ValueError):
            self._test_constant_array_one(func, x_data, [3.0, 4.0])
        with self.assertRaises(ValueError):
            self._test_constant_array_one(func, x_data, (3.0, 4.0))

        with self.assertRaises(ValueError):
            self._test_constant_array_one(func, x_data, [3.0, 4.0, 5.0])
        with self.assertRaises(ValueError):
            self._test_constant_array_one(func, x_data, (3.0, 4.0, 5.0))
        with self.assertRaises(ValueError):
            self._test_constant_array_one(
                func, x_data, numpy.array([3.0, 4.0, 5.0], numpy.float32))

    def _test_constant_array_gpu_one(self, func, lhs, rhs):
        x = chainer.Variable(cuda.to_gpu(lhs))
        y = func(x, rhs)
        self.assertEqual(y.data.dtype, numpy.float32)
        y.grad = chainer.cuda.cupy.ones_like(y.data).astype(numpy.float32)
        y.backward()
        self.assertEqual(x.grad.dtype, numpy.float32)

    def _test_constant_array_gpu(self, func, exception=TypeError):
        x_data = numpy.array([1.0, 2.0], numpy.float32)

        self._test_constant_array_gpu_one(
            func, x_data, cuda.to_gpu(numpy.array([3.0, 4.0], numpy.int32)))
        self._test_constant_array_gpu_one(
            func, x_data, cuda.to_gpu(numpy.array([3.0, 4.0], numpy.int64)))
        self._test_constant_array_gpu_one(
            func, x_data, cuda.to_gpu(numpy.array([3.0, 4.0], numpy.float32)))
        self._test_constant_array_gpu_one(
            func, x_data, cuda.to_gpu(numpy.array([3.0, 4.0], numpy.float64)))

        with self.assertRaises(exception):
            self._test_constant_array_one(
                func, x_data, cuda.to_gpu(
                    numpy.array([3.0, 4.0, 5.0], numpy.float32)))

    def test_add_constant(self):
        self._test_constant(lambda x, y: x + y)

    @attr.gpu
    def test_add_constant_gpu(self):
        self._test_constant_gpu(lambda x, y: x + y)

    def test_add_constant_array(self):
        self._test_constant_array(lambda x, y: x + y)

    @attr.gpu
    def test_add_constant_array_gpu(self):
        self._test_constant_array_gpu(lambda x, y: x + y)

    def test_radd_constant(self):
        self._test_constant(lambda x, y: y + x)

    @attr.gpu
    def test_radd_constant_gpu(self):
        self._test_constant_gpu(lambda x, y: y + x)

    def test_radd_constant_array(self):
        self._test_constant_array(lambda x, y: y + x)

    @attr.gpu
    def test_radd_constant_array_gpu(self):
        self._test_constant_array_gpu(lambda x, y: y + x)

    def test_sub_constant(self):
        self._test_constant(lambda x, y: x - y)

    @attr.gpu
    def test_sub_constant_gpu(self):
        self._test_constant_gpu(lambda x, y: x - y)

    def test_sub_constant_array(self):
        self._test_constant_array(lambda x, y: x - y)

    @attr.gpu
    def test_sub_constant_array_gpu(self):
        self._test_constant_array_gpu(lambda x, y: x - y)

    def test_rsub_constant(self):
        self._test_constant(lambda x, y: y - x)

    @attr.gpu
    def test_rsub_constant_gpu(self):
        self._test_constant_gpu(lambda x, y: y - x)

    def test_rsub_constant_array(self):
        self._test_constant_array(lambda x, y: y - x)

    @attr.gpu
    def test_rsub_constant_array_gpu(self):
        self._test_constant_array_gpu(lambda x, y: y - x)

    def test_mul_constant(self):
        self._test_constant(lambda x, y: x * y)

    @attr.gpu
    def test_mul_constant_gpu(self):
        self._test_constant_gpu(lambda x, y: x * y)

    def test_mul_constant_array(self):
        self._test_constant_array(lambda x, y: x * y)

    @attr.gpu
    def test_mul_constant_array_gpu(self):
        self._test_constant_array(lambda x, y: x * y)

    def test_rmul_constant(self):
        self._test_constant(lambda x, y: y * x)

    @attr.gpu
    def test_rmul_constant_gpu(self):
        self._test_constant_gpu(lambda x, y: y * x)

    def test_rmul_constant_array(self):
        self._test_constant_array(lambda x, y: y * x)

    @attr.gpu
    def test_rmul_constant_array_gpu(self):
        # _test_constant_array_one throws pycuda._pvt_struct.error
        self._test_constant_array_gpu(lambda x, y: y * x, exception=Exception)

    def test_div_constant(self):
        self._test_constant(lambda x, y: x / y)

    @attr.gpu
    def test_div_constant_gpu(self):
        self._test_constant_gpu(lambda x, y: x / y)

    def test_div_constant_array(self):
        self._test_constant_array(lambda x, y: x / y)

    @attr.gpu
    def test_div_constant_array_gpu(self):
        # _test_constant_array_one throws pycuda._pvt_struct.error
        self._test_constant_array_gpu(lambda x, y: x / y, exception=Exception)

    def test_rdiv_constant(self):
        self._test_constant(lambda x, y: y / x)

    @attr.gpu
    def test_rdiv_constant_gpu(self):
        self._test_constant_gpu(lambda x, y: y / x)

    def test_rdiv_constant_array(self):
        self._test_constant_array(lambda x, y: y / x)

    @attr.gpu
    def test_rdiv_constant_array_gpu(self):
        self._test_constant_array_gpu(lambda x, y: y / x)

    def test_pow_constant(self):
        self._test_constant(lambda x, y: x ** y)

    @attr.gpu
    def test_pow_constant_gpu(self):
        self._test_constant_gpu(lambda x, y: x ** y)

    def test_pow_constant_array(self):
        self._test_constant_array(lambda x, y: x ** y)

    @attr.gpu
    def test_pow_constant_array_gpu(self):
        self._test_constant_array_gpu(lambda x, y: x ** y, exception=TypeError)

    def test_rpow_constant(self):
        self._test_constant(lambda x, y: y ** x)

    @attr.gpu
    def test_rpow_constant_gpu(self):
        self._test_constant_gpu(lambda x, y: y ** x)

    def test_rpow_constant_array(self):
        self._test_constant_array(lambda x, y: y ** x)

    @attr.gpu
    def test_rpow_constant_array_gpu(self):
        # _test_constant_array_one throws pycuda._pvt_struct.error
        self._test_constant_array_gpu(lambda x, y: y ** x, exception=Exception)


class VariableConstantOpTestBase(object):

    def make_date(self):
        raise NotImplementedError()

    def setUp(self):
        self.x, self.gy, self.value = self.make_data()

    def check_forward(self, op, x_data):
        x = chainer.Variable(x_data)
        y = op(x, self.value)
        gradient_check.assert_allclose(
            op(self.x, self.value), y.data, atol=1e-7, rtol=1e-7)

    def forward_cpu(self, op):
        self.check_forward(op, self.x)

    @condition.retry(3)
    def test_add_forward_cpu(self):
        self.forward_cpu(lambda x, y: x + y)

    @condition.retry(3)
    def test_radd_forward_cpu(self):
        self.forward_cpu(lambda x, y: y + x)

    @condition.retry(3)
    def test_sub_forward_cpu(self):
        self.forward_cpu(lambda x, y: x - y)

    @condition.retry(3)
    def test_rsub_forward_cpu(self):
        self.forward_cpu(lambda x, y: y - x)

    @condition.retry(3)
    def test_mul_forward_cpu(self):
        self.forward_cpu(lambda x, y: x * y)

    @condition.retry(3)
    def test_rmul_forward_cpu(self):
        self.forward_cpu(lambda x, y: y * x)

    @condition.retry(3)
    def test_div_forward_cpu(self):
        self.forward_cpu(lambda x, y: x / y)

    @condition.retry(3)
    def test_rdiv_forward_cpu(self):
        self.forward_cpu(lambda x, y: y / x)

    @condition.retry(3)
    def test_pow_forward_cpu(self):
        self.forward_cpu(lambda x, y: x ** y)

    @condition.retry(3)
    def test_rpow_forward_cpu(self):
        self.forward_cpu(lambda x, y: y ** x)

    def forward_gpu(self, op):
        self.check_forward(op, cuda.to_gpu(self.x))

    @attr.gpu
    @condition.retry(3)
    def test_add_forward_gpu(self):
        self.forward_gpu(lambda x, y: x + y)

    @attr.gpu
    @condition.retry(3)
    def test_radd_forward_gpu(self):
        self.forward_gpu(lambda x, y: y + x)

    @attr.gpu
    @condition.retry(3)
    def test_sub_forward_gpu(self):
        self.forward_gpu(lambda x, y: x - y)

    @attr.gpu
    @condition.retry(3)
    def test_rsub_forward_gpu(self):
        self.forward_gpu(lambda x, y: y - x)

    @attr.gpu
    @condition.retry(3)
    def test_mul_forward_gpu(self):
        self.forward_gpu(lambda x, y: x * y)

    @attr.gpu
    @condition.retry(3)
    def test_rmul_forward_gpu(self):
        self.forward_gpu(lambda x, y: y * x)

    @attr.gpu
    @condition.retry(3)
    def test_div_forward_gpu(self):
        self.forward_gpu(lambda x, y: x / y)

    @attr.gpu
    @condition.retry(3)
    def test_rdiv_forward_gpu(self):
        self.forward_gpu(lambda x, y: y / x)

    @attr.gpu
    @condition.retry(3)
    def test_pow_forward_gpu(self):
        self.forward_gpu(lambda x, y: x ** y)

    @attr.gpu
    @condition.retry(3)
    def test_rpow_forward_gpu(self):
        self.forward_gpu(lambda x, y: y ** x)

    def check_backward(self, op, x_data, y_grad):
        gradient_check.check_backward(lambda x: op(x, self.value),
                                      x_data, y_grad)

    def backward_cpu(self, op):
        self.check_backward(op, self.x, self.gy)

    @condition.retry(3)
    def test_add_backward_cpu(self):
        self.backward_cpu(lambda x, y: x + y)

    @condition.retry(3)
    def test_radd_backward_cpu(self):
        self.backward_cpu(lambda x, y: y + x)

    @condition.retry(3)
    def test_sub_backward_cpu(self):
        self.backward_cpu(lambda x, y: x - y)

    @condition.retry(3)
    def test_rsub_backward_cpu(self):
        self.backward_cpu(lambda x, y: y - x)

    @condition.retry(3)
    def test_mul_backward_cpu(self):
        self.backward_cpu(lambda x, y: x * y)

    @condition.retry(3)
    def test_rmul_backward_cpu(self):
        self.backward_cpu(lambda x, y: y * x)

    @condition.retry(3)
    def test_div_backward_cpu(self):
        self.backward_cpu(lambda x, y: x / y)

    @condition.retry(3)
    def test_rdiv_backward_cpu(self):
        self.backward_cpu(lambda x, y: y / x)

    @condition.retry(3)
    def test_pow_backward_cpu(self):
        self.backward_cpu(lambda x, y: x ** y)

    @condition.retry(3)
    def test_rpow_backward_cpu(self):
        self.backward_cpu(lambda x, y: y ** x)

    def backward_gpu(self, op):
        self.check_backward(op, cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    @attr.gpu
    @condition.retry(3)
    def test_add_backward_gpu(self):
        self.backward_gpu(lambda x, y: x + y)

    @attr.gpu
    @condition.retry(3)
    def test_radd_backward_gpu(self):
        self.backward_gpu(lambda x, y: y + x)

    @attr.gpu
    @condition.retry(3)
    def test_sub_backward_gpu(self):
        self.backward_gpu(lambda x, y: x - y)

    @attr.gpu
    @condition.retry(3)
    def test_rsub_backward_gpu(self):
        self.backward_gpu(lambda x, y: y - x)

    @attr.gpu
    @condition.retry(3)
    def test_mul_backward_gpu(self):
        self.backward_gpu(lambda x, y: x * y)

    @attr.gpu
    @condition.retry(3)
    def test_rmul_backward_gpu(self):
        self.backward_gpu(lambda x, y: y * x)

    @attr.gpu
    @condition.retry(3)
    def test_div_backward_gpu(self):
        self.backward_gpu(lambda x, y: x / y)

    @attr.gpu
    @condition.retry(3)
    def test_rdiv_backward_gpu(self):
        self.backward_gpu(lambda x, y: y / x)

    @attr.gpu
    @condition.retry(3)
    def test_pow_backward_gpu(self):
        self.backward_gpu(lambda x, y: x ** y)

    @attr.gpu
    @condition.retry(3)
    def test_rpow_backward_gpu(self):
        self.backward_gpu(lambda x, y: y ** x)


class TestVariableConstantOpSimple(VariableConstantOpTestBase,
                                   unittest.TestCase):

    def make_data(self):
        x = numpy.random.uniform(.5, 1, (3, 2)).astype(numpy.float32)
        gy = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)
        value = .5
        return x, gy, value


class TestVariableConstantOpZeroDimension(VariableConstantOpTestBase,
                                          unittest.TestCase):

    def make_data(self):
        x = numpy.random.uniform(.5, 1, ()).astype(numpy.float32)
        gy = numpy.random.uniform(-1, 1, ()).astype(numpy.float32)
        value = .5
        return x, gy, value


class TestVariableConstantArrayOp(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(.5, 1, (3, 2)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)
        self.value = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)

    def check_forward(self, op, x_data, gpu, positive):
        value = self.value
        if positive:
            value = numpy.abs(value)
        v = value
        if gpu:
            v = cuda.to_gpu(v)
        x = chainer.Variable(x_data)
        y = op(x, v)
        gradient_check.assert_allclose(
            op(self.x, value), y.data, atol=1e-6, rtol=1e-6)

    def forward_cpu(self, op, positive=False):
        self.check_forward(op, self.x, False, positive)

    @condition.retry(3)
    def test_add_forward_cpu(self):
        self.forward_cpu(lambda x, y: x + y)

    @condition.retry(3)
    def test_radd_forward_cpu(self):
        self.forward_cpu(lambda x, y: y + x)

    @condition.retry(3)
    def test_sub_forward_cpu(self):
        self.forward_cpu(lambda x, y: x - y)

    @condition.retry(3)
    def test_rsub_forward_cpu(self):
        self.forward_cpu(lambda x, y: y - x)

    @condition.retry(3)
    def test_mul_forward_cpu(self):
        self.forward_cpu(lambda x, y: x * y)

    @condition.retry(3)
    def test_rmul_forward_cpu(self):
        self.forward_cpu(lambda x, y: y * x)

    @condition.retry(3)
    def test_div_forward_cpu(self):
        self.forward_cpu(lambda x, y: x / y)

    @condition.retry(3)
    def test_rdiv_forward_cpu(self):
        self.forward_cpu(lambda x, y: y / x)

    @condition.retry(3)
    def test_pow_forward_cpu(self):
        self.forward_cpu(lambda x, y: x ** y)

    @condition.retry(3)
    def test_rpow_forward_cpu(self):
        self.forward_cpu(lambda x, y: y ** x, positive=True)

    def forward_gpu(self, op, positive=False):
        self.check_forward(op, cuda.to_gpu(self.x), True, positive)

    @attr.gpu
    @condition.retry(3)
    def test_add_forward_gpu(self):
        self.forward_gpu(lambda x, y: x + y)

    @attr.gpu
    @condition.retry(3)
    def test_radd_forward_gpu(self):
        self.forward_gpu(lambda x, y: y + x)

    @attr.gpu
    @condition.retry(3)
    def test_sub_forward_gpu(self):
        self.forward_gpu(lambda x, y: x - y)

    @attr.gpu
    @condition.retry(3)
    def test_rsub_forward_gpu(self):
        self.forward_gpu(lambda x, y: y - x)

    @attr.gpu
    @condition.retry(3)
    def test_mul_forward_gpu(self):
        self.forward_gpu(lambda x, y: x * y)

    @attr.gpu
    @condition.retry(3)
    def test_rmul_forward_gpu(self):
        self.forward_gpu(lambda x, y: y * x)

    @attr.gpu
    @condition.retry(3)
    def test_div_forward_gpu(self):
        self.forward_gpu(lambda x, y: x / y)

    @attr.gpu
    @condition.retry(3)
    def test_rdiv_forward_gpu(self):
        self.forward_gpu(lambda x, y: y / x)

    @attr.gpu
    @condition.retry(3)
    def test_pow_forward_gpu(self):
        self.forward_gpu(lambda x, y: x ** y)

    @attr.gpu
    @condition.retry(3)
    def test_rpow_forward_gpu(self):
        self.forward_gpu(lambda x, y: y ** x, positive=True)

    def check_backward(self, op, x_data, y_grad, gpu, positive):
        value = self.value
        if positive:
            value = numpy.abs(value)
        if gpu:
            value = cuda.to_gpu(value)

        gradient_check.check_backward(lambda x: op(x, value), x_data, y_grad,
                                      atol=1e-4, rtol=1e-4)

    def backward_cpu(self, op, positive=False):
        self.check_backward(op, self.x, self.gy, False, positive)

    @condition.retry(3)
    def test_add_backward_cpu(self):
        self.backward_cpu(lambda x, y: x + y)

    @condition.retry(3)
    def test_radd_backward_cpu(self):
        self.backward_cpu(lambda x, y: y + x)

    @condition.retry(3)
    def test_sub_backward_cpu(self):
        self.backward_cpu(lambda x, y: x - y)

    @condition.retry(3)
    def test_rsub_backward_cpu(self):
        self.backward_cpu(lambda x, y: y - x)

    @condition.retry(3)
    def test_mul_backward_cpu(self):
        self.backward_cpu(lambda x, y: x * y)

    @condition.retry(3)
    def test_rmul_backward_cpu(self):
        self.backward_cpu(lambda x, y: y * x)

    @condition.retry(3)
    def test_div_backward_cpu(self):
        self.backward_cpu(lambda x, y: x / y)

    @condition.retry(3)
    def test_rdiv_backward_cpu(self):
        self.backward_cpu(lambda x, y: y / x)

    @condition.retry(3)
    def test_pow_backward_cpu(self):
        self.backward_cpu(lambda x, y: x ** y)

    @condition.retry(3)
    def test_rpow_backward_cpu(self):
        self.backward_cpu(lambda x, y: y ** x, positive=True)

    def backward_gpu(self, op, positive=False):
        self.check_backward(
            op, cuda.to_gpu(self.x), cuda.to_gpu(self.gy), True, positive)

    @attr.gpu
    @condition.retry(3)
    def test_add_backward_gpu(self):
        self.backward_gpu(lambda x, y: x + y)

    @attr.gpu
    @condition.retry(3)
    def test_radd_backward_gpu(self):
        self.backward_gpu(lambda x, y: y + x)

    @attr.gpu
    @condition.retry(3)
    def test_sub_backward_gpu(self):
        self.backward_gpu(lambda x, y: x - y)

    @attr.gpu
    @condition.retry(3)
    def test_mul_backward_gpu(self):
        self.backward_gpu(lambda x, y: x * y)

    @attr.gpu
    @condition.retry(3)
    def test_rmul_backward_gpu(self):
        self.backward_gpu(lambda x, y: y * x)

    @attr.gpu
    @condition.retry(3)
    def test_div_backward_gpu(self):
        self.backward_gpu(lambda x, y: x / y)

    @attr.gpu
    @condition.retry(3)
    def test_rdiv_backward_gpu(self):
        self.backward_gpu(lambda x, y: y / x)

    @attr.gpu
    @condition.retry(3)
    def test_pow_backward_gpu(self):
        self.backward_gpu(lambda x, y: x ** y)

    @attr.gpu
    @condition.retry(3)
    def test_rpow_backward_gpu(self):
        self.backward_gpu(lambda x, y: y ** x, positive=True)


class UnaryFunctionsTestBase(object):

    def make_data(self):
        raise NotImplementedError()

    def setUp(self):
        self.x, self.gy = self.make_data()

    def check_forward(self, op, op_np, x_data):
        x = chainer.Variable(x_data)
        y = op(x)
        gradient_check.assert_allclose(
            op_np(self.x), y.data, atol=1e-7, rtol=1e-7)

    def forward_cpu(self, op, op_np):
        self.check_forward(op, op_np, self.x)

    @condition.retry(3)
    def test_neg_forward_cpu(self):
        self.forward_cpu(lambda x: -x, lambda x: -x)

    @condition.retry(3)
    def test_abs_forward_cpu(self):
        self.forward_cpu(lambda x: abs(x), lambda x: abs(x))

    def forward_gpu(self, op, op_np):
        self.check_forward(op, op_np, cuda.to_gpu(self.x))

    @attr.gpu
    @condition.retry(3)
    def test_neg_forward_gpu(self):
        self.forward_gpu(lambda x: -x, lambda x: -x)

    @attr.gpu
    @condition.retry(3)
    def test_abs_forward_gpu(self):
        self.forward_gpu(lambda x: abs(x), lambda x: abs(x))

    def check_backward(self, op, x_data, y_grad):
        gradient_check.check_backward(op, x_data, y_grad)

    def backward_cpu(self, op):
        self.check_backward(op, self.x, self.gy)

    @condition.retry(3)
    def test_neg_backward_cpu(self):
        self.backward_cpu(lambda x: -x)

    @condition.retry(3)
    def test_abs_backward_cpu(self):
        self.backward_cpu(lambda x: abs(x))

    def backward_gpu(self, op):
        self.check_backward(op, cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    @attr.gpu
    @condition.retry(3)
    def test_neg_backward_gpu(self):
        self.backward_gpu(lambda x: -x)

    @attr.gpu
    @condition.retry(3)
    def test_abs_backward_gpu(self):
        self.backward_gpu(lambda x: abs(x))


class TestUnaryFunctionsSimple(UnaryFunctionsTestBase, unittest.TestCase):

    def make_data(self):
        x = numpy.random.uniform(.5, 1, (3, 2)).astype(numpy.float32)
        gy = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)
        return x, gy


class TestUnaryFunctionsZeroDimension(UnaryFunctionsTestBase,
                                      unittest.TestCase):

    def make_data(self):
        x = numpy.random.uniform(.5, 1, ()).astype(numpy.float32)
        gy = numpy.random.uniform(-1, 1, ()).astype(numpy.float32)
        return x, gy


class TestNegativePow(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 0, (3, 2)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            lambda x: x ** 2, x_data, y_grad, atol=1e-4, rtol=1e-4)

    def test_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


class TestNotSupportOperation(unittest.TestCase):

    def setUp(self):
        self.x = chainer.Variable(numpy.zeros(10))
        self.y = chainer.Variable(numpy.zeros(10))

    def test_lt(self):
        with self.assertRaises(NotImplementedError):
            self.x < self.y

    def test_le(self):
        with self.assertRaises(NotImplementedError):
            self.x <= self.y

    def test_eq(self):
        with self.assertRaises(NotImplementedError):
            self.x == self.y

    def test_ne(self):
        with self.assertRaises(NotImplementedError):
            self.x != self.y

    def test_gt(self):
        with self.assertRaises(NotImplementedError):
            self.x > self.y

    def test_ge(self):
        with self.assertRaises(NotImplementedError):
            self.x >= self.y

    def test_nonzero(self):
        with self.assertRaises(NotImplementedError):
            if self.x:
                pass


class ConvertValueToStringTest(unittest.TestCase):

    def _check_scalar(self, value, string):
        self.assertEqual(basic_math._convert_value_to_string(value), string)

    def test_integer_positive(self):
        self._check_scalar(2, '2')

    def test_integer_zero(self):
        self._check_scalar(0, '0')

    def test_integer_negative(self):
        self._check_scalar(-2, '(-2)')

    def test_float_positive(self):
        self._check_scalar(2.0, '2.0')

    def test_float_zero(self):
        self._check_scalar(0.0, '0.0')

    def test_float_negative(self):
        self._check_scalar(-2.0, '(-2.0)')

    def test_numpy_scalar(self):
        self._check_scalar(numpy.float32(2), '2.0')

    def _check_array(self, value, string):
        self.assertEqual(basic_math._convert_value_to_string(value), string)
        value = chainer.Variable(value)
        self.assertEqual(basic_math._convert_value_to_string(value), string)

    def test_array_cpu(self):
        self._check_array(numpy.array([1, 2]), 'constant array')

    @attr.gpu
    def test_array_gpu(self):
        self._check_array(cuda.ndarray([1, 2]), 'constant array')


class TestLabel(unittest.TestCase):

    def test_neg(self):
        self.assertEqual(basic_math.Neg().label, '__neg__')

    def test_absolute(self):
        self.assertEqual(basic_math.Absolute().label, '|_|')

    def test_add(self):
        self.assertEqual(basic_math.Add().label, '_ + _')

    def test_add_constant(self):
        self.assertEqual(basic_math.AddConstant(2.0).label, '_ + 2.0')

    def test_sub(self):
        self.assertEqual(basic_math.Sub().label, '_ - _')

    def test_sub_from_constant(self):
        self.assertEqual(basic_math.SubFromConstant(2.0).label, '2.0 - _')

    def test_mul(self):
        self.assertEqual(basic_math.Mul().label, '_ * _')

    def test_mul_constant(self):
        self.assertEqual(basic_math.MulConstant(2.0).label, '_ * 2.0')

    def test_div(self):
        self.assertEqual(basic_math.Div().label, '_ / _')

    def test_div_from_constant(self):
        self.assertEqual(basic_math.DivFromConstant(2.0).label, '_ / 2.0')

    def test_pow_var_var(self):
        self.assertEqual(basic_math.PowVarVar().label, '_ ** _')

    def test_pow_var_const(self):
        self.assertEqual(basic_math.PowVarConst(2.0).label, '_ ** 2.0')

    def test_pow_const_var(self):
        self.assertEqual(basic_math.PowConstVar(2.0).label, '2.0 ** _')


testing.run_module(__name__, __file__)
