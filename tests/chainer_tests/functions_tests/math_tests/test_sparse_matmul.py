import unittest

import numpy

import chainer
from chainer import cuda
import chainer.functions as F
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product_dict(
    [
        {'m': 3, 'n': 4, 'k': 2},
        {'m': 2, 'n': 3, 'k': 4},
    ],
    [
        {'transa': False, 'transb': False},
        {'transa': False, 'transb': True},
        {'transa': True, 'transb': False},
        {'transa': True, 'transb': True},
    ],
    [
        {'nbatch': 0},
        {'nbatch': 1},
        {'nbatch': 4},
    ],
    [
        {'a_dtype': numpy.float16},
        {'a_dtype': numpy.float32},
        {'a_dtype': numpy.float64},
    ],
    [
        {'b_dtype': numpy.float16},
        {'b_dtype': numpy.float32},
        {'b_dtype': numpy.float64},
    ]
))
class TestSparseMatMul(unittest.TestCase):

    def setUp(self):
        a_shape = self._set_shape([self.m, self.k], self.transa)
        b_shape = self._set_shape([self.k, self.n], self.transb)
        c_shape = self._set_shape([self.m, self.n], False)
        self.c_dtype = numpy.result_type(self.a_dtype, self.b_dtype)
        self.a = self._setup_tensor(.5, 1, a_shape, self.a_dtype)
        self.a[numpy.where(self.a < .75)] = 0
        self.b = self._setup_tensor(.5, 1, b_shape, self.b_dtype)
        self.b[numpy.where(self.b < .75)] = 0
        self.gc = self._setup_tensor(-1, 1, c_shape, self.c_dtype)
        # self.gga = self._setup_tensor(.5, 1, a_shape, self.a_dtype)
        # self.ggb = self._setup_tensor(.5, 1, b_shape, self.b_dtype)
        self.forward_answer = self._matmul(self.a, self.b)

    def _set_shape(self, shape, trans):
        if trans:
            shape = [shape[1], shape[0]]
        if self.nbatch > 0:
            return [self.nbatch, shape[0], shape[1]]
        else:
            return shape

    def _setup_tensor(self, _min, _max, shape, dtype):
        return numpy.random.uniform(_min, _max, shape).astype(dtype)

    def _matmul(self, a, b):
        if self.transa:
            a = a.swapaxes(-1, -2)
        if self.transb:
            b = b.swapaxes(-1, -2)
        return numpy.matmul(a, b)

    #
    # SPDN: sparse A * dense B
    #
    def check_SPDN_forward(self, a_data, b_data, atol=1e-4, rtol=1e-5):
        sp_a = F.sparse_dense2coo(a_data, use_variable=True)
        b = chainer.Variable(b_data)
        c = F.sparse_matmul(sp_a, b, transa=self.transa, transb=self.transb)
        testing.assert_allclose(self.forward_answer, c.data, atol, rtol)

    @attr.gpu
    def test_SPDN_sparse_matmul_forward_gpu(self):
        a = cuda.to_gpu(self.a)
        b = cuda.to_gpu(self.b)
        if self.a_dtype == numpy.float16 or self.b_dtype == numpy.float16:
            self.check_SPDN_forward(a, b, atol=1e-3, rtol=1e-3)
        else:
            self.check_SPDN_forward(a, b)

    def check_SPDN_backward(self, a_data, b_data, c_grad, atol, rtol):
        sp_a = F.sparse_dense2coo(a_data)
        func = F.math.sparse_matmul.SparseMatMul(
            sp_a.row, sp_a.col, sp_a.shape,
            transa=self.transa, transb=self.transb, transc=False)
        op = lambda a, b: func.apply((a, b))[0]
        gradient_check.check_backward(
            op, (sp_a.data, b_data), c_grad, atol=atol, rtol=rtol,
            dtype=numpy.float32)

    @attr.gpu
    def test_SPDN_sparse_matmul_backward_gpu(self):
        self.check_SPDN_backward(
            cuda.to_gpu(self.a), cuda.to_gpu(self.b),
            cuda.to_gpu(self.gc), atol=1e-2, rtol=1e-2)

    #
    # DNSP: dense A * sparse B
    #
    def check_DNSP_forward(self, a_data, b_data, atol=1e-4, rtol=1e-5):
        a = chainer.Variable(a_data)
        sp_b = F.sparse_dense2coo(b_data, use_variable=True)
        c = F.sparse_matmul(a, sp_b, transa=self.transa, transb=self.transb)
        testing.assert_allclose(self.forward_answer, c.data, atol, rtol)

    @attr.gpu
    def test_DNSP_sparse_matmul_forward_gpu(self):
        a = cuda.to_gpu(self.a)
        b = cuda.to_gpu(self.b)
        if self.a_dtype == numpy.float16 or self.b_dtype == numpy.float16:
            self.check_DNSP_forward(a, b, atol=1e-3, rtol=1e-3)
        else:
            self.check_DNSP_forward(a, b)

    def check_DNSP_backward(self, a_data, b_data, c_grad, atol, rtol):
        sp_b = F.sparse_dense2coo(b_data)
        func = F.math.sparse_matmul.SparseMatMul(
            sp_b.row, sp_b.col, sp_b.shape,
            transa=not self.transb, transb=not self.transa, transc=True)
        op = lambda b, a: func.apply((b, a))[0]
        gradient_check.check_backward(
            op, (sp_b.data, a_data), c_grad, atol=atol, rtol=rtol,
            dtype=numpy.float32)

    @attr.gpu
    def test_DNSP_tensordot_backward_gpu(self):
        self.check_DNSP_backward(
            cuda.to_gpu(self.a), cuda.to_gpu(self.b),
            cuda.to_gpu(self.gc), atol=1e-2, rtol=1e-2)


testing.run_module(__name__, __file__)
