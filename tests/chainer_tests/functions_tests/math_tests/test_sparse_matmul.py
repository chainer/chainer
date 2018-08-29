import unittest

import numpy

import chainer
from chainer import cuda
import chainer.functions as F
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer import utils
from chainer.utils import type_check

_scipy_available = True
try:
    from scipy import sparse  # NOQA
except ImportError:
    _scipy_available = False


def _setup_tensor(_min, _max, shape, dtype, threshold=None):
    y = numpy.random.uniform(_min, _max, shape).astype(dtype)
    if threshold is not None:
        y[y < threshold] = 0
    return y


@testing.parameterize(*testing.product_dict(
    [
        {'m': 2, 'n': 3, 'k': 4},
        {'m': 3, 'n': 4, 'k': 2},
    ],
    [
        {'transa': False}, {'transa': True},
    ],
    [
        {'transb': False}, {'transb': True},
    ],
    [
        {'nbatch': 0}, {'nbatch': 1}, {'nbatch': 4},
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
class TestCooMatMul(unittest.TestCase):

    def setUp(self):
        a_shape = self._set_shape([self.m, self.k], self.transa)
        b_shape = self._set_shape([self.k, self.n], self.transb)
        c_shape = self._set_shape([self.m, self.n], False)
        self.c_dtype = numpy.result_type(self.a_dtype, self.b_dtype)
        self.a = _setup_tensor(.5, 1, a_shape, self.a_dtype, .75)
        self.b = _setup_tensor(.5, 1, b_shape, self.b_dtype, .75)
        self.gc = _setup_tensor(-1, 1, c_shape, self.c_dtype)
        self.gga = _setup_tensor(.5, 1, a_shape, self.a_dtype)
        self.gga[numpy.where(self.a < .75)] = 0
        self.ggb = _setup_tensor(.5, 1, b_shape, self.b_dtype)
        self.ggb[numpy.where(self.b < .75)] = 0
        self.forward_answer = self._matmul(self.a, self.b)

    def _set_shape(self, shape, trans):
        if trans:
            shape = [shape[1], shape[0]]
        if self.nbatch > 0:
            shape = [self.nbatch, shape[0], shape[1]]
        return shape

    def _matmul(self, a, b):
        if self.transa:
            a = a.swapaxes(-1, -2)
        if self.transb:
            b = b.swapaxes(-1, -2)
        if hasattr(numpy, 'matmul'):
            return numpy.matmul(a, b)
        elif a.ndim == 2:
            return numpy.dot(a, b)
        else:
            return numpy.einsum('...ij,...jk->...ik', a, b)

    #
    # SPDN: sparse A * dense B
    #
    def check_SPDN_forward(self, a_data, b_data, atol=1e-4, rtol=1e-5):
        sp_a = utils.to_coo(a_data, requires_grad=True)
        b = chainer.Variable(b_data)
        c = F.sparse_matmul(sp_a, b, transa=self.transa, transb=self.transb)
        testing.assert_allclose(self.forward_answer, c.data, atol, rtol)

    def test_SPDN_sparse_matmul_forward_cpu(self):
        if not _scipy_available:
            return
        if self.a_dtype == numpy.float16 or self.b_dtype == numpy.float16:
            self.check_SPDN_forward(self.a, self.b, atol=1e-3, rtol=1e-3)
        else:
            self.check_SPDN_forward(self.a, self.b)

    @attr.gpu
    def test_SPDN_sparse_matmul_forward_gpu(self):
        a = cuda.to_gpu(self.a)
        b = cuda.to_gpu(self.b)
        if self.a_dtype == numpy.float16 or self.b_dtype == numpy.float16:
            self.check_SPDN_forward(a, b, atol=1e-3, rtol=1e-3)
        else:
            self.check_SPDN_forward(a, b)

    def check_SPDN_backward(self, a_data, b_data, c_grad, atol, rtol):
        sp_a = utils.to_coo(a_data)
        func = F.math.sparse_matmul.CooMatMul(
            sp_a.row, sp_a.col, sp_a.shape, sp_a.order,
            transa=self.transa, transb=self.transb, transc=False)

        def op(a, b):
            return func.apply((a, b))[0]
        gradient_check.check_backward(
            op, (sp_a.data.data, b_data), c_grad, atol=atol, rtol=rtol,
            dtype=numpy.float32)

    def test_SPDN_sparse_matmul_backward_cpu(self):
        if not _scipy_available:
            return
        self.check_SPDN_backward(
            self.a, self.b, self.gc, atol=1e-2, rtol=1e-2)

    @attr.gpu
    def test_SPDN_sparse_matmul_backward_gpu(self):
        self.check_SPDN_backward(
            cuda.to_gpu(self.a), cuda.to_gpu(self.b),
            cuda.to_gpu(self.gc), atol=1e-2, rtol=1e-2)

    def check_SPDN_double_backward(
            self, a_data, b_data, c_grad, a_grad_grad, b_grad_grad,
            atol, rtol):
        sp_a = utils.to_coo(a_data)
        sp_gga = utils.to_coo(a_grad_grad)
        func = F.math.sparse_matmul.CooMatMul(
            sp_a.row, sp_a.col, sp_a.shape, sp_a.order,
            transa=self.transa, transb=self.transb, transc=False)

        def op(a, b):
            return func.apply((a, b))[0]
        gradient_check.check_double_backward(
            op, (sp_a.data.data, b_data),
            c_grad, (sp_gga.data.data, b_grad_grad),
            atol=atol, rtol=rtol, dtype=numpy.float32)

    def test_SPDN_sparse_matmul_double_backward_cpu(self):
        if not _scipy_available:
            return
        self.check_SPDN_double_backward(
            self.a, self.b, self.gc, self.gga, self.ggb,
            atol=1e-2, rtol=1e-2)

    @attr.gpu
    def test_SPDN_sparse_matmul_double_backward_gpu(self):
        self.check_SPDN_double_backward(
            cuda.to_gpu(self.a), cuda.to_gpu(self.b),
            cuda.to_gpu(self.gc), cuda.to_gpu(self.gga),
            cuda.to_gpu(self.ggb), atol=1e-2, rtol=1e-2)

    #
    # DNSP: dense A * sparse B
    #
    def check_DNSP_forward(self, a_data, b_data, atol=1e-4, rtol=1e-5):
        a = chainer.Variable(a_data)
        sp_b = utils.to_coo(b_data, requires_grad=True)
        c = F.sparse_matmul(a, sp_b, transa=self.transa, transb=self.transb)
        testing.assert_allclose(self.forward_answer, c.data, atol, rtol)

    def test_DNSP_sparse_matmul_forward_cpu(self):
        if not _scipy_available:
            return
        if self.a_dtype == numpy.float16 or self.b_dtype == numpy.float16:
            self.check_DNSP_forward(self.a, self.b, atol=1e-3, rtol=1e-3)
        else:
            self.check_DNSP_forward(self.a, self.b)

    @attr.gpu
    def test_DNSP_sparse_matmul_forward_gpu(self):
        a = cuda.to_gpu(self.a)
        b = cuda.to_gpu(self.b)
        if self.a_dtype == numpy.float16 or self.b_dtype == numpy.float16:
            self.check_DNSP_forward(a, b, atol=1e-3, rtol=1e-3)
        else:
            self.check_DNSP_forward(a, b)

    def check_DNSP_backward(self, a_data, b_data, c_grad, atol, rtol):
        sp_b = utils.to_coo(b_data)
        func = F.math.sparse_matmul.CooMatMul(
            sp_b.row, sp_b.col, sp_b.shape, sp_b.order,
            transa=not self.transb, transb=not self.transa, transc=True)

        def op(b, a):
            return func.apply((b, a))[0]
        gradient_check.check_backward(
            op, (sp_b.data.data, a_data), c_grad, atol=atol, rtol=rtol,
            dtype=numpy.float32)

    def test_DNSP_tensordot_backward_cpu(self):
        if not _scipy_available:
            return
        self.check_DNSP_backward(
            self.a, self.b, self.gc, atol=1e-2, rtol=1e-2)

    @attr.gpu
    def test_DNSP_tensordot_backward_gpu(self):
        self.check_DNSP_backward(
            cuda.to_gpu(self.a), cuda.to_gpu(self.b),
            cuda.to_gpu(self.gc), atol=1e-2, rtol=1e-2)

    def check_DNSP_double_backward(
            self, a_data, b_data, c_grad, a_grad_grad, b_grad_grad,
            atol, rtol):
        sp_b = utils.to_coo(b_data)
        sp_ggb = utils.to_coo(b_grad_grad)
        func = F.math.sparse_matmul.CooMatMul(
            sp_b.row, sp_b.col, sp_b.shape, sp_b.order,
            transa=not self.transb, transb=not self.transa, transc=True)

        def op(b, a):
            return func.apply((b, a))[0]
        gradient_check.check_double_backward(
            op, (sp_b.data.data, a_data),
            c_grad, (sp_ggb.data.data, a_grad_grad),
            atol=atol, rtol=rtol, dtype=numpy.float32)

    def test_DNSP_sparse_matmul_double_backward_cpu(self):
        if not _scipy_available:
            return
        self.check_DNSP_double_backward(
            self.a, self.b, self.gc, self.gga, self.ggb,
            atol=1e-2, rtol=1e-2)

    @attr.gpu
    def test_DNSP_sparse_matmul_double_backward_gpu(self):
        self.check_DNSP_double_backward(
            cuda.to_gpu(self.a), cuda.to_gpu(self.b),
            cuda.to_gpu(self.gc), cuda.to_gpu(self.gga),
            cuda.to_gpu(self.ggb), atol=1e-2, rtol=1e-2)


@testing.parameterize(*testing.product_dict(
    [
        {'transa': False}, {'transa': True},
    ],
    [
        {'transb': False}, {'transb': True},
    ],
))
class TestCooMatMulInvalid(unittest.TestCase):

    def test_invalid_ndim(self):
        a = _setup_tensor(.5, 1, (2, 3, 3), numpy.float32, .75)
        b = _setup_tensor(.5, 1, (3, 3), numpy.float32, .75)
        sp_a = utils.to_coo(a)
        sp_b = utils.to_coo(b)
        with self.assertRaises(type_check.InvalidType):
            F.sparse_matmul(sp_a, b, self.transa, self.transb)
        with self.assertRaises(type_check.InvalidType):
            F.sparse_matmul(a, sp_b, self.transa, self.transb)

    def test_invalid_nbatch(self):
        a = _setup_tensor(.5, 1, (2, 3, 3), numpy.float32, .75)
        b = _setup_tensor(.5, 1, (3, 3, 3), numpy.float32, .75)
        sp_a = utils.to_coo(a)
        sp_b = utils.to_coo(b)
        with self.assertRaises(type_check.InvalidType):
            F.sparse_matmul(sp_a, b, self.transa, self.transb)
        with self.assertRaises(type_check.InvalidType):
            F.sparse_matmul(a, sp_b, self.transa, self.transb)

    def test_invalid_shape(self):
        a = _setup_tensor(.5, 1, (1, 2, 3), numpy.float32, .75)
        b = _setup_tensor(.5, 1, (1, 4, 5), numpy.float32, .75)
        sp_a = utils.to_coo(a)
        sp_b = utils.to_coo(b)
        with self.assertRaises(type_check.InvalidType):
            F.sparse_matmul(sp_a, b, self.transa, self.transb)
        with self.assertRaises(type_check.InvalidType):
            F.sparse_matmul(a, sp_b, self.transa, self.transb)

    def test_invalid_inputs(self):
        a = _setup_tensor(.5, 1, (1, 3, 3), numpy.float32, .75)
        b = _setup_tensor(.5, 1, (1, 3, 3), numpy.float32, .75)
        sp_a = utils.to_coo(a)
        sp_b = utils.to_coo(b)
        with self.assertRaises(ValueError):
            F.sparse_matmul(sp_a, sp_b, self.transa, self.transb)
        with self.assertRaises(ValueError):
            F.sparse_matmul(a, b, self.transa, self.transb)


testing.run_module(__name__, __file__)
