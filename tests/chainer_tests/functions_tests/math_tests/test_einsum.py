import unittest

import numpy

import chainer
from chainer.backends import cuda
import chainer.functions as F
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product_dict(
    [
        {'subscripts': 'ij,jk->ik', 'shapes': ((2, 3), (3, 4))},
        {'subscripts': 'kj,ji->ik', 'shapes': ((2, 3), (3, 4))},
        {'subscripts': 'ij,jk,kl->il', 'shapes': ((5, 2), (2, 3), (3, 4))},
        {'subscripts': 'ij,ij->i', 'shapes': ((2, 3), (2, 3))},
        # {'subscripts': 'ij,jk', 'shapes': ((2, 3), (3, 4))},
    ],
    [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64},
    ]
))
class TestEinSum(unittest.TestCase):

    def setUp(self):
        self.inputs = [
            self._setup_tensor(-1, 1, shape, self.dtype)
            for shape in self.shapes
        ]
        self.forward_answer = numpy.einsum(self.subscripts, *self.inputs)
        self.g = self._setup_tensor(-1, 1, self.forward_answer.shape, self.dtype)
        self.op = lambda *xs: F.einsum(self.subscripts, *xs)

    def _setup_tensor(self, _min, _max, shape, dtype):
        return numpy.random.uniform(_min, _max, shape).astype(dtype)

    """
    def check_forward(self, a_data, b_data, atol=1e-4, rtol=1e-5):
        a = chainer.Variable(a_data)
        b = chainer.Variable(b_data)
        c = self.op(a, b)
        testing.assert_allclose(self.forward_answer, c.data, atol, rtol)

    def test_tensordot_forward_cpu(self):
        if self.a.dtype == numpy.float16 or self.b.dtype == numpy.float16:
            self.check_forward(self.a, self.b, atol=1e-3, rtol=1e-3)
        else:
            self.check_forward(self.a, self.b)

    @attr.gpu
    def test_tensordot_forward_gpu(self):
        a = cuda.to_gpu(self.a)
        b = cuda.to_gpu(self.b)
        if self.a.dtype == numpy.float16 or self.b.dtype == numpy.float16:
            self.check_forward(a, b, atol=1e-3, rtol=1e-3)
        else:
            self.check_forward(a, b)
    """

    def check_backward(self, inputs_data, output_grad, atol, rtol):
        gradient_check.check_backward(
            self.op, inputs_data, output_grad, atol=atol, rtol=rtol,
            dtype=numpy.float32)

    def test_einsum_backward_cpu(self):
        self.check_backward(self.inputs, self.g, atol=1e-2, rtol=5e-2)

    @attr.gpu
    def test_einsum_backward_gpu(self):
        self.check_backward(
            tuple(cuda.to_gpu(x) for x in self.inputs),
            self.g, atol=1e-2, rtol=5e-2)

    """
    def check_double_backward(
            self, a_data, b_data, y_grad, a_grad_grad, b_grad_grad,
            atol, rtol):
        gradient_check.check_double_backward(
            self.op, (a_data, b_data), y_grad, (a_grad_grad, b_grad_grad),
            atol=atol, rtol=rtol, dtype=numpy.float32)

    def test_tensordot_double_backward_cpu(self):
        self.check_double_backward(
            self.a, self.b, self.gc, self.gga, self.ggb,
            atol=1e-2, rtol=5e-2)

    @attr.gpu
    def test_tensordot_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.a), cuda.to_gpu(self.b),
            cuda.to_gpu(self.gc), cuda.to_gpu(self.gga),
            cuda.to_gpu(self.ggb), atol=1e-2, rtol=1e-2)
    """


"""
class TestEinSumInvalid(unittest.TestCase):

    def test_invalid_shape(self):
        a_data = numpy.zeros((4, 3, 2), dtype=numpy.float32)
        b_data = numpy.zeros((2, 3, 5), dtype=numpy.float32)
        a = chainer.Variable(a_data)
        b = chainer.Variable(b_data)
        with self.assertRaises(ValueError):
            F.tensordot(a, b)
        with self.assertRaises(ValueError):
            F.tensordot(a, b, axes=((1, 2), (0, 1)))
        with self.assertRaises(ValueError):
            F.tensordot(a, b, axes=((0), (0)))
        with self.assertRaises(ValueError):
            F.tensordot(a, b, axes=((2), (2)))

    def test_invalid_axes(self):
        a_data = numpy.zeros((4, 3, 2), dtype=numpy.float32)
        b_data = numpy.zeros((3, 2, 5), dtype=numpy.float32)
        a = chainer.Variable(a_data)
        b = chainer.Variable(b_data)
        with self.assertRaises(ValueError):
            F.tensordot(a, b, axes=((1, 2), (0)))
        with self.assertRaises(ValueError):
            F.tensordot(a, b, axes=((2), (0, 1)))
        with self.assertRaises(ValueError):
            F.tensordot(a, b, axes=((0, 1, 2, 3), (0, 1, 2, 3)))
        with self.assertRaises(ValueError):
            F.tensordot(a, b, axes=(()))
        with self.assertRaises(ValueError):
            F.tensordot(a, b, axes=((), (), ()))
        with self.assertRaises(TypeError):
            F.tensordot(a, b, axes=1.0)
"""


testing.run_module(__name__, __file__)
