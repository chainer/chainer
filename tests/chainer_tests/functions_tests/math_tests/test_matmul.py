import unittest

import numpy

import chainer
from chainer.backends import cuda
import chainer.functions as F
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.utils import type_check
import chainerx


def _matmul_tol(x1_dtype, x2_dtype):
    if x1_dtype == numpy.float16 or x2_dtype == numpy.float16:
        return {'atol': 2e-3, 'rtol': 2e-3}
    else:
        return {'atol': 1e-4, 'rtol': 1e-5}


@testing.parameterize(*testing.product_dict(
    [
        # matmul
        {'x1_shape': (2, 5), 'x2_shape': (5, 10), 'gy_shape': (2, 10),
         'transa': False, 'transb': False},
        {'x1_shape': (5, 2), 'x2_shape': (5, 10), 'gy_shape': (2, 10),
         'transa': True, 'transb': False},
        {'x1_shape': (2, 5), 'x2_shape': (10, 5), 'gy_shape': (2, 10),
         'transa': False, 'transb': True},
        {'x1_shape': (5, 2), 'x2_shape': (10, 5), 'gy_shape': (2, 10),
         'transa': True, 'transb': True},

        # vector
        {'x1_shape': (5,), 'x2_shape': (5,), 'gy_shape': (),
         'transa': True, 'transb': False},
        {'x1_shape': (5,), 'x2_shape': (5,), 'gy_shape': (),
         'transa': False, 'transb': True},

        # matrix-vector
        {'x1_shape': (5,), 'x2_shape': (5, 2), 'gy_shape': (2,),
         'transa': False, 'transb': False},
        {'x1_shape': (5,), 'x2_shape': (5, 2), 'gy_shape': (2,),
         'transa': True, 'transb': False},
        {'x1_shape': (5,), 'x2_shape': (2, 5), 'gy_shape': (2,),
         'transa': False, 'transb': True},
        {'x1_shape': (2, 5), 'x2_shape': (5,), 'gy_shape': (2,),
         'transa': False, 'transb': False},
        {'x1_shape': (5, 2), 'x2_shape': (5,), 'gy_shape': (2,),
         'transa': True, 'transb': False},
        {'x1_shape': (2, 5), 'x2_shape': (5,), 'gy_shape': (2,),
         'transa': False, 'transb': True},

        # batched matmul
        {'x1_shape': (6, 2, 5), 'x2_shape': (6, 5, 10), 'gy_shape': (6, 2, 10),
         'transa': False, 'transb': False},
        {'x1_shape': (6, 5, 2), 'x2_shape': (6, 5, 10), 'gy_shape': (6, 2, 10),
         'transa': True, 'transb': False},
        {'x1_shape': (6, 2, 5), 'x2_shape': (6, 10, 5), 'gy_shape': (6, 2, 10),
         'transa': False, 'transb': True},
        {'x1_shape': (6, 5, 2), 'x2_shape': (6, 10, 5), 'gy_shape': (6, 2, 10),
         'transa': True, 'transb': True},
        {'x1_shape': (2, 3, 4), 'x2_shape': (4,), 'gy_shape': (2, 3),
         'transa': False, 'transb': False},
        {'x1_shape': (4,), 'x2_shape': (2, 4, 3), 'gy_shape': (2, 3),
         'transa': False, 'transb': False},

        # batchsize = 1
        {'x1_shape': (1, 2, 5), 'x2_shape': (1, 5, 10), 'gy_shape': (1, 2, 10),
         'transa': False, 'transb': False},

        # 4dim batched matmul
        {'x1_shape': (2, 3, 4, 5), 'x2_shape': (2, 3, 5, 6),
         'gy_shape': (2, 3, 4, 6), 'transa': False, 'transb': False},
    ],
    [
        {'x1_dtype': numpy.float16},
        {'x1_dtype': numpy.float32},
        {'x1_dtype': numpy.float64},
    ],
    [
        {'x2_dtype': numpy.float16},
        {'x2_dtype': numpy.float32},
        {'x2_dtype': numpy.float64},
    ]
))
class TestMatMul(unittest.TestCase):

    def setUp(self):
        self.x1 = numpy.random.uniform(.5, 1, self.x1_shape)
        self.x1 = self.x1.astype(self.x1_dtype)
        self.x2 = numpy.random.uniform(.5, 1, self.x2_shape)
        self.x2 = self.x2.astype(self.x2_dtype)
        ret_dtype = numpy.result_type(self.x1_dtype, self.x2_dtype)
        self.gy = numpy.random.uniform(-1, 1, self.gy_shape).astype(ret_dtype)
        self.ggx1 = numpy.random.uniform(
            .5, 1, self.x1_shape).astype(self.x1_dtype)
        self.ggx2 = numpy.random.uniform(
            .5, 1, self.x2_shape).astype(self.x2_dtype)

        self.op = lambda x, y: F.matmul(x, y, transa=self.transa,
                                        transb=self.transb)
        self.forward_answer = self._get_forward_answer(self.x1, self.x2,
                                                       self.transa,
                                                       self.transb)

    def _get_forward_answer(self, x1, x2, transa, transb):
        if transa and x1.ndim >= 2:
            x1 = x1.swapaxes(-1, -2)

        if transb and x2.ndim >= 2:
            x2 = x2.swapaxes(-1, -2)

        if x1.ndim <= 2 or x2.ndim <= 2:
            return numpy.dot(x1, x2)
        else:
            return numpy.einsum('...ij,...jk->...ik', x1, x2)

    def check_forward(self, x1_data, x2_data):
        tol = _matmul_tol(x1_data.dtype, x2_data.dtype)
        x1 = chainer.Variable(x1_data)
        x2 = chainer.Variable(x2_data)
        y = self.op(x1, x2)
        testing.assert_allclose(self.forward_answer, y.data, **tol)

    def test_matmul_forward_cpu(self):
        self.check_forward(self.x1, self.x2)

    @attr.gpu
    def test_matmul_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x1), cuda.to_gpu(self.x2))

    @attr.chainerx
    def test_matmul_forward_chainerx(self):
        self.check_forward(chainerx.array(self.x1), chainerx.array(self.x2))

    def check_backward(self, x1_data, x2_data, y_grad, atol, rtol):
        gradient_check.check_backward(
            self.op, (x1_data, x2_data), y_grad, atol=atol, rtol=rtol,
            dtype=numpy.float32)

    def test_matmul_backward_cpu(self):
        self.check_backward(self.x1, self.x2, self.gy, atol=1e-2, rtol=5e-2)

    @attr.gpu
    def test_matmul_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.x1), cuda.to_gpu(self.x2),
            cuda.to_gpu(self.gy), atol=1e-2, rtol=1e-2)

    @attr.chainerx
    def test_matmul_backward_chainerx(self):
        self.check_backward(
            chainerx.array(self.x1), chainerx.array(self.x2),
            chainerx.array(self.gy), atol=1e-2, rtol=1e-2)

    def check_double_backward(
            self, x1_data, x2_data, y_grad, x1_grad_grad, x2_grad_grad,
            atol, rtol):
        gradient_check.check_double_backward(
            self.op, (x1_data, x2_data), y_grad, (x1_grad_grad, x2_grad_grad),
            atol=atol, rtol=rtol, dtype=numpy.float32)

    def test_matmul_double_backward_cpu(self):
        self.check_double_backward(
            self.x1, self.x2, self.gy, self.ggx1, self.ggx2,
            atol=1e-2, rtol=5e-2)

    @attr.gpu
    def test_matmul_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.x1), cuda.to_gpu(self.x2),
            cuda.to_gpu(self.gy), cuda.to_gpu(self.ggx1),
            cuda.to_gpu(self.ggx2), atol=1e-2, rtol=1e-2)

    @attr.chainerx
    def test_matmul_double_backward_chainerx(self):
        self.check_double_backward(
            chainerx.array(self.x1), chainerx.array(self.x2),
            chainerx.array(self.gy), chainerx.array(self.ggx1),
            chainerx.array(self.ggx2), atol=1e-2, rtol=1e-2)


@testing.parameterize(*testing.product_dict(
    [
        # batched matmul 2d x 2d
        {'x1_shape': (2, 3), 'x2_shape': (2, 3), 'gy_shape': (2, 1, 1),
         'transa': True, 'transb': False},
        {'x1_shape': (2, 3), 'x2_shape': (2, 3), 'gy_shape': (2, 3, 3),
         'transa': False, 'transb': True},

        # batched matmul 3d x 3d
        {'x1_shape': (3, 2, 5), 'x2_shape': (3, 5, 4), 'gy_shape': (3, 2, 4),
         'transa': False, 'transb': False},
        {'x1_shape': (3, 5, 2), 'x2_shape': (3, 5, 4), 'gy_shape': (3, 2, 4),
         'transa': True, 'transb': False},
        {'x1_shape': (3, 2, 5), 'x2_shape': (3, 4, 5), 'gy_shape': (3, 2, 4),
         'transa': False, 'transb': True},
        {'x1_shape': (3, 5, 2), 'x2_shape': (3, 4, 5), 'gy_shape': (3, 2, 4),
         'transa': True, 'transb': True},

        # batched matmul 2d x 3d
        {'x1_shape': (3, 5), 'x2_shape': (3, 1, 4), 'gy_shape': (3, 5, 4),
         'transa': False, 'transb': False},
        {'x1_shape': (3, 5), 'x2_shape': (3, 5, 4), 'gy_shape': (3, 1, 4),
         'transa': True, 'transb': False},
        {'x1_shape': (3, 5), 'x2_shape': (3, 4, 1), 'gy_shape': (3, 5, 4),
         'transa': False, 'transb': True},
        {'x1_shape': (3, 5), 'x2_shape': (3, 4, 5), 'gy_shape': (3, 1, 4),
         'transa': True, 'transb': True},

        # batched matmul 3d x 2d
        {'x1_shape': (3, 2, 5), 'x2_shape': (3, 5), 'gy_shape': (3, 2, 1),
         'transa': False, 'transb': False},
        {'x1_shape': (3, 5, 2), 'x2_shape': (3, 5), 'gy_shape': (3, 2, 1),
         'transa': True, 'transb': False},
        {'x1_shape': (3, 2, 1), 'x2_shape': (3, 5), 'gy_shape': (3, 2, 5),
         'transa': False, 'transb': True},
        {'x1_shape': (3, 1, 2), 'x2_shape': (3, 5), 'gy_shape': (3, 2, 5),
         'transa': True, 'transb': True},

        # batchsize = 1
        {'x1_shape': (1, 2, 5), 'x2_shape': (1, 5, 4), 'gy_shape': (1, 2, 4),
         'transa': False, 'transb': False},
    ]
))
class TestBatchMatMul(unittest.TestCase):
    x1_dtype = numpy.float32
    x2_dtype = numpy.float32

    def setUp(self):
        self.x1 = numpy.random.uniform(.5, 1, self.x1_shape)
        self.x1 = self.x1.astype(self.x1_dtype)
        self.x2 = numpy.random.uniform(.5, 1, self.x2_shape)
        self.x2 = self.x2.astype(self.x2_dtype)
        ret_dtype = numpy.result_type(self.x1_dtype, self.x2_dtype)
        self.gy = numpy.random.uniform(-1, 1, self.gy_shape).astype(ret_dtype)
        self.ggx1 = numpy.random.uniform(.5, 1, self.x1_shape).astype(
            self.x1_dtype)
        self.ggx2 = numpy.random.uniform(.5, 1, self.x2_shape).astype(
            self.x2_dtype)

        self.op = lambda x, y: F.batch_matmul(
            x, y, transa=self.transa, transb=self.transb)
        self.forward_answer = self._get_forward_answer(
            self.x1, self.x2, self.transa, self.transb)

    def _get_forward_answer(self, x1, x2, transa, transb):
        x1 = x1.reshape(x1.shape[:2] + (-1,))
        if transa:
            x1 = x1.swapaxes(-1, -2)

        x2 = x2.reshape(x2.shape[:2] + (-1,))
        if transb:
            x2 = x2.swapaxes(-1, -2)

        return numpy.einsum('...ij,...jk->...ik', x1, x2)

    def check_forward(self, x1_data, x2_data):
        tol = _matmul_tol(x1_data.dtype, x2_data.dtype)
        x1 = chainer.Variable(x1_data)
        x2 = chainer.Variable(x2_data)
        with testing.assert_warns(DeprecationWarning):
            y = self.op(x1, x2)
        testing.assert_allclose(self.forward_answer, y.data, **tol)

    def test_matmul_forward_cpu(self):
        self.check_forward(self.x1, self.x2)

    @attr.gpu
    def test_matmul_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x1), cuda.to_gpu(self.x2))

    def check_backward(self, x1_data, x2_data, y_grad, atol, rtol):
        with testing.assert_warns(DeprecationWarning):
            gradient_check.check_backward(
                self.op, (x1_data, x2_data), y_grad, atol=atol, rtol=rtol,
                dtype=numpy.float32)

    def test_matmul_backward_cpu(self):
        self.check_backward(self.x1, self.x2, self.gy, atol=1e-2, rtol=5e-2)

    @attr.gpu
    def test_matmul_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.x1), cuda.to_gpu(self.x2),
            cuda.to_gpu(self.gy), atol=1e-2, rtol=1e-2)

    def check_double_backward(
            self, x1_data, x2_data, y_grad, x1_grad_grad, x2_grad_grad,
            atol, rtol):
        with testing.assert_warns(DeprecationWarning):
            gradient_check.check_double_backward(
                self.op, (x1_data, x2_data), y_grad,
                (x1_grad_grad, x2_grad_grad),
                atol=atol, rtol=rtol, dtype=numpy.float32)

    def test_matmul_double_backward_cpu(self):
        self.check_double_backward(
            self.x1, self.x2, self.gy, self.ggx1, self.ggx2,
            atol=1e-2, rtol=5e-2)

    @attr.gpu
    def test_matmul_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.x1), cuda.to_gpu(self.x2),
            cuda.to_gpu(self.gy), cuda.to_gpu(self.ggx1),
            cuda.to_gpu(self.ggx2), atol=1e-2, rtol=1e-2)


class TestMatMulInvalid(unittest.TestCase):

    def test_invalid_shape(self):
        x_data = numpy.zeros((2, 3, 4), dtype=numpy.float32)
        y_data = numpy.zeros((3, 4, 3), dtype=numpy.float32)
        x = chainer.Variable(x_data)
        y = chainer.Variable(y_data)

        with self.assertRaises(type_check.InvalidType):
            F.matmul(x, y)

    def test_invalid_ndim(self):
        x_data = numpy.zeros((3, 2, 5), dtype=numpy.float32)
        y_data = numpy.zeros((3, 5), dtype=numpy.float32)
        x = chainer.Variable(x_data)
        y = chainer.Variable(y_data)

        with self.assertRaises(type_check.InvalidType):
            F.matmul(x, y)


testing.run_module(__name__, __file__)
