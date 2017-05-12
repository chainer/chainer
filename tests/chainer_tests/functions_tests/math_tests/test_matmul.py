import unittest

import numpy
import six

import chainer
from chainer import cuda
import chainer.functions as F
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import type_check


class _TestMatMul(unittest.TestCase):

    def check_forward(self, x1_data, x2_data, atol=1e-4, rtol=1e-5):
        x1 = chainer.Variable(x1_data)
        x2 = chainer.Variable(x2_data)
        y = self.op(x1, x2)
        testing.assert_allclose(self.forward_answer, y.data, atol, rtol)

    @condition.retry(3)
    def test_matmul_forward_cpu(self):
        if self.x1.dtype == numpy.float16 or self.x2.dtype == numpy.float16:
            self.check_forward(self.x1, self.x2, atol=1e-3, rtol=1e-3)
        else:
            self.check_forward(self.x1, self.x2)

    @attr.gpu
    @condition.retry(3)
    def test_matmul_forward_gpu(self):
        if self.x1.dtype == numpy.float16 or self.x2.dtype == numpy.float16:
            self.check_forward(cuda.to_gpu(self.x1), cuda.to_gpu(self.x2),
                               atol=1e-3, rtol=1e-3)
        else:
            self.check_forward(cuda.to_gpu(self.x1), cuda.to_gpu(self.x2))

    def check_backward(self, x1_data, x2_data, y_grad, atol, rtol):
        gradient_check.check_backward(
            self.op, (x1_data, x2_data), y_grad, atol=atol, rtol=rtol,
            dtype=numpy.float32)

    @condition.retry(3)
    def test_matmul_backward_cpu(self):
        self.check_backward(self.x1, self.x2, self.gy, atol=1e-2, rtol=5e-2)

    @attr.gpu
    @condition.retry(3)
    def test_matmul_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.x1), cuda.to_gpu(self.x2),
            cuda.to_gpu(self.gy), atol=1e-2, rtol=1e-2)


m = 2
k = 5
n = 10


class TestMatMulMatrixMatrix(_TestMatMul):

    def setUp(self):
        self.x1 = numpy.random.uniform(.5, 1, (m, k)).astype(numpy.float32)
        self.x2 = numpy.random.uniform(.5, 1, (k, n)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (m, n)).astype(numpy.float32)
        self.op = lambda x, y: F.matmul(x, y)
        self.forward_answer = numpy.dot(self.x1, self.x2)


class TestMatMulMatrixMatrixFP16(_TestMatMul):

    def setUp(self):
        self.x1 = numpy.random.uniform(.5, 1, (m, k)).astype(numpy.float16)
        self.x2 = numpy.random.uniform(.5, 1, (k, n)).astype(numpy.float16)
        self.gy = numpy.random.uniform(-1, 1, (m, n)).astype(numpy.float16)
        self.op = lambda x, y: F.matmul(x, y)
        self.forward_answer = numpy.dot(self.x1, self.x2)


class TestMatMulMatrixMatrixFP64(_TestMatMul):

    def setUp(self):
        self.x1 = numpy.random.uniform(.5, 1, (m, k)).astype(numpy.float64)
        self.x2 = numpy.random.uniform(.5, 1, (k, n)).astype(numpy.float64)
        self.gy = numpy.random.uniform(-1, 1, (m, n)).astype(numpy.float64)
        self.op = lambda x, y: F.matmul(x, y)
        self.forward_answer = numpy.dot(self.x1, self.x2)


class TestMatMulMatrixTMatrix(_TestMatMul):

    def setUp(self):
        self.x1 = numpy.random.uniform(.5, 1, (k, m)).astype(numpy.float32)
        self.x2 = numpy.random.uniform(.5, 1, (k, n)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (m, n)).astype(numpy.float32)
        self.op = lambda x, y: F.matmul(x, y, transa=True)
        self.forward_answer = numpy.dot(self.x1.T, self.x2)


class TestMatMulMatrixMatrixT(_TestMatMul):

    def setUp(self):
        self.x1 = numpy.random.uniform(.5, 1, (m, k)).astype(numpy.float32)
        self.x2 = numpy.random.uniform(.5, 1, (n, k)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (m, n)).astype(numpy.float32)
        self.op = lambda x, y: F.matmul(x, y, transb=True)
        self.forward_answer = numpy.dot(self.x1, self.x2.T)


class TestMatMulMatrixTMatrixT(_TestMatMul):

    def setUp(self):
        self.x1 = numpy.random.uniform(.5, 1, (k, m)).astype(numpy.float32)
        self.x2 = numpy.random.uniform(.5, 1, (n, k)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (m, n)).astype(numpy.float32)
        self.op = lambda x, y: F.matmul(x, y, transa=True, transb=True)
        self.forward_answer = numpy.dot(self.x1.T, self.x2.T)


class TestMatMulVectorTVector(_TestMatMul):

    def setUp(self):
        self.x1 = numpy.random.uniform(.5, 1, (m,)).astype(numpy.float32)
        self.x2 = numpy.random.uniform(.5, 1, (m,)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (1, 1)).astype(numpy.float32)
        self.op = lambda x, y: F.matmul(x, y, transa=True)
        self.forward_answer = numpy.dot(self.x1, self.x2).reshape(1, 1)


class TestMatMulVectorVectorT(_TestMatMul):

    def setUp(self):
        self.x1 = numpy.random.uniform(.5, 1, (m,)).astype(numpy.float32)
        self.x2 = numpy.random.uniform(.5, 1, (m,)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (m, m)).astype(numpy.float32)
        self.op = lambda x, y: F.matmul(x, y, transb=True)
        self.forward_answer = numpy.dot(
            self.x1.reshape(m, 1), self.x2.reshape(1, m))


batch_size = 10


class TestBatchMatMulMatrixMatrix(_TestMatMul):

    def setUp(self):
        self.x1 = numpy.random.uniform(
            .5, 1, (batch_size, m, k)).astype(numpy.float32)
        self.x2 = numpy.random.uniform(
            .5, 1, (batch_size, k, n)).astype(numpy.float32)
        self.gy = numpy.random.uniform(
            -1, 1, (batch_size, m, n)).astype(numpy.float32)
        self.op = lambda x, y: F.batch_matmul(x, y)
        self.forward_answer = numpy.array([
            numpy.dot(self.x1[i], self.x2[i])
            for i in six.moves.range(batch_size)])


class TestBatchMatMulMatrixTMatrix(_TestMatMul):

    def setUp(self):
        self.x1 = numpy.random.uniform(
            .5, 1, (batch_size, k, m)).astype(numpy.float32)
        self.x2 = numpy.random.uniform(
            .5, 1, (batch_size, k, n)).astype(numpy.float32)
        self.gy = numpy.random.uniform(
            -1, 1, (batch_size, m, n)).astype(numpy.float32)
        self.op = lambda x, y: F.batch_matmul(x, y, transa=True)
        self.forward_answer = numpy.array([
            numpy.dot(self.x1[i].T, self.x2[i])
            for i in six.moves.range(batch_size)])


class TestBatchMatMulMatrixMatrixT(_TestMatMul):

    def setUp(self):
        self.x1 = numpy.random.uniform(
            .5, 1, (batch_size, m, k)).astype(numpy.float32)
        self.x2 = numpy.random.uniform(
            .5, 1, (batch_size, n, k)).astype(numpy.float32)
        self.gy = numpy.random.uniform(
            -1, 1, (batch_size, m, n)).astype(numpy.float32)
        self.op = lambda x, y: F.batch_matmul(x, y, transb=True)
        self.forward_answer = numpy.array([
            numpy.dot(self.x1[i], self.x2[i].T)
            for i in six.moves.range(batch_size)])


class TestBatchMatMulMatrixTMatrixT(_TestMatMul):

    def setUp(self):
        self.x1 = numpy.random.uniform(
            .5, 1, (batch_size, k, m)).astype(numpy.float32)
        self.x2 = numpy.random.uniform(
            .5, 1, (batch_size, n, k)).astype(numpy.float32)
        self.gy = numpy.random.uniform(
            -1, 1, (batch_size, m, n)).astype(numpy.float32)
        self.op = lambda x, y: F.batch_matmul(x, y, transa=True, transb=True)
        self.forward_answer = numpy.array([
            numpy.dot(self.x1[i].T, self.x2[i].T)
            for i in six.moves.range(batch_size)])


class TestBatchMatMulVectorTVector(_TestMatMul):

    def setUp(self):
        self.x1 = numpy.random.uniform(
            .5, 1, (batch_size, m,)).astype(numpy.float32)
        self.x2 = numpy.random.uniform(
            .5, 1, (batch_size, m,)).astype(numpy.float32)
        self.gy = numpy.random.uniform(
            -1, 1, (batch_size, 1, 1)).astype(numpy.float32)
        self.op = lambda x, y: F.batch_matmul(x, y, transa=True)
        self.forward_answer = numpy.array([
            numpy.dot(self.x1[i], self.x2[i])
            for i in six.moves.range(batch_size)]).reshape(batch_size, 1, 1)


class TestBatchMatMulVectorVectorT(_TestMatMul):

    def setUp(self):
        self.x1 = numpy.random.uniform(
            .5, 1, (batch_size, m,)).astype(numpy.float32)
        self.x2 = numpy.random.uniform(
            .5, 1, (batch_size, m,)).astype(numpy.float32)
        self.gy = numpy.random.uniform(
            -1, 1, (batch_size, m, m)).astype(numpy.float32)
        self.op = lambda x, y: F.batch_matmul(x, y, transb=True)
        self.forward_answer = numpy.array([
            numpy.dot(self.x1[i].reshape(m, 1), self.x2[i].reshape(1, m))
            for i in six.moves.range(batch_size)])


class TestBatchMatMulMatrixMatrixBatchSize1(_TestMatMul):

    def setUp(self):
        self.x1 = numpy.random.uniform(
            .5, 1, (1, m, k)).astype(numpy.float32)
        self.x2 = numpy.random.uniform(
            .5, 1, (1, k, n)).astype(numpy.float32)
        self.gy = numpy.random.uniform(
            -1, 1, (1, m, n)).astype(numpy.float32)
        self.op = lambda x, y: F.batch_matmul(x, y)
        self.forward_answer = numpy.array([
            numpy.dot(self.x1[i], self.x2[i])
            for i in six.moves.range(1)])


class TestBatchMatMulBroadcastedMatrix1(_TestMatMul):

    def setUp(self):
        self.x1 = numpy.random.uniform(
            .5, 1, (batch_size, m, k)).astype(numpy.float32)
        self.x2 = numpy.random.uniform(
            .5, 1, (1, k, n)).astype(numpy.float32)
        self.gy = numpy.random.uniform(
            -1, 1, (batch_size, m, n)).astype(numpy.float32)
        self.op = lambda x, y: F.batch_matmul(
            x, F.broadcast_to(y, (batch_size, k, n)))
        self.forward_answer = numpy.array([
            numpy.dot(self.x1[i], self.x2[0])
            for i in six.moves.range(batch_size)])


class TestBatchMatMulBroadcastedMatrix2(_TestMatMul):

    def setUp(self):
        self.x1 = numpy.random.uniform(
            .5, 1, (batch_size, m, k)).astype(numpy.float32)
        self.x2 = numpy.random.uniform(
            .5, 1, (k, n)).astype(numpy.float32)
        self.gy = numpy.random.uniform(
            -1, 1, (batch_size, m, n)).astype(numpy.float32)
        self.op = lambda x, y: F.batch_matmul(
            x, F.broadcast_to(F.expand_dims(y, 0), (batch_size, k, n)))
        self.forward_answer = numpy.array([
            numpy.dot(self.x1[i], self.x2)
            for i in six.moves.range(batch_size)])


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

        # batched matmul
        {'x1_shape': (6, 2, 5), 'x2_shape': (6, 5, 10), 'gy_shape': (6, 2, 10),
         'transa': False, 'transb': False},
        {'x1_shape': (6, 5, 2), 'x2_shape': (6, 5, 10), 'gy_shape': (6, 2, 10),
         'transa': True, 'transb': False},
        {'x1_shape': (6, 2, 5), 'x2_shape': (6, 10, 5), 'gy_shape': (6, 2, 10),
         'transa': False, 'transb': True},
        {'x1_shape': (6, 5, 2), 'x2_shape': (6, 10, 5), 'gy_shape': (6, 2, 10),
         'transa': True, 'transb': True},

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
class TestNumpyLikeMatMul(_TestMatMul):

    def setUp(self):
        self.x1 = numpy.random.uniform(.5, 1, self.x1_shape)
        self.x1 = self.x1.astype(self.x1_dtype)
        self.x2 = numpy.random.uniform(.5, 1, self.x2_shape)
        self.x2 = self.x2.astype(self.x2_dtype)
        ret_dtype = numpy.result_type(self.x1_dtype, self.x2_dtype)
        self.gy = numpy.random.uniform(-1, 1, self.gy_shape).astype(ret_dtype)

        self.op = lambda x, y: F.numpy_like_matmul(x, y, transa=self.transa,
                                                   transb=self.transb)
        if self.x1.ndim == 1:
            x1 = self.x1
        else:
            x1 = self.x1.swapaxes(-1, -2) if self.transa else self.x1

        if self.x2.ndim == 1:
            x2 = self.x2
        else:
            x2 = self.x2.swapaxes(-1, -2) if self.transb else self.x2
        if x1.ndim <= 2:
            self.forward_answer = numpy.dot(x1, x2)
        else:
            self.forward_answer = numpy.einsum('...ij,...jk->...ik', x1, x2)


class TestNumpyLikeMatMulInvalid(unittest.TestCase):

    def test_invalid_shape(self):
        x_data = numpy.zeros((2, 3, 4), dtype=numpy.float32)
        y_data = numpy.zeros((1, 4, 3), dtype=numpy.float32)
        x = chainer.Variable(x_data)
        y = chainer.Variable(y_data)

        with self.assertRaises(type_check.InvalidType):
            F.numpy_like_matmul(x, y)

    def test_invalid_ndim(self):
        x_data = numpy.zeros((3, 2, 5), dtype=numpy.float32)
        y_data = numpy.zeros((3, 5), dtype=numpy.float32)
        x = chainer.Variable(x_data)
        y = chainer.Variable(y_data)

        with self.assertRaises(type_check.InvalidType):
            F.numpy_like_matmul(x, y)


testing.run_module(__name__, __file__)
