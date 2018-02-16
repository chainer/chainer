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
        {'a_shape': (4, 3, 2), 'b_shape': (3, 2, 5), 'axes': 2, 'gc_shape': (4, 5)},  # NOQA
        {'a_shape': (4, 3, 2), 'b_shape': (3, 2, 5), 'axes': ([1, 2], [0, 1]), 'gc_shape': (4, 5)},  # NOQA
        {'a_shape': (4, 2, 3), 'b_shape': (3, 5, 2), 'axes': ([2, 1], [0, 2]), 'gc_shape': (4, 5)},  # NOQA
        {'a_shape': (2, 4, 3), 'b_shape': (5, 3, 2), 'axes': ([2, 0], [1, 2]), 'gc_shape': (4, 5)},  # NOQA
        {'a_shape': (2, 3, 4), 'b_shape': (5, 2, 3), 'axes': ([1, 0], [2, 1]), 'gc_shape': (4, 5)},  # NOQA
        {'a_shape': (3, 2, 4), 'b_shape': (2, 5, 3), 'axes': ([0, 1], [2, 0]), 'gc_shape': (4, 5)},  # NOQA
        {'a_shape': (3, 4, 2), 'b_shape': (2, 3, 5), 'axes': ([0, 2], [1, 0]), 'gc_shape': (4, 5)},  # NOQA

        {'a_shape': (3, 4, 2), 'b_shape': (2, 5, 6), 'axes': 1, 'gc_shape': (3, 4, 5, 6)},  # NOQA
        {'a_shape': (3, 4, 2), 'b_shape': (2, 5, 6), 'axes': ([2, 0]), 'gc_shape': (3, 4, 5, 6)},  # NOQA
        {'a_shape': (3, 2, 4), 'b_shape': (5, 2, 6), 'axes': ([1, 1]), 'gc_shape': (3, 4, 5, 6)},  # NOQA
        {'a_shape': (2, 3, 4), 'b_shape': (5, 6, 2), 'axes': ([0, 2]), 'gc_shape': (3, 4, 5, 6)},  # NOQA

        {'a_shape': (4, 5, 3, 2), 'b_shape': (3, 2, 6), 'axes': 2, 'gc_shape': (4, 5, 6)},  # NOQA
        {'a_shape': (4, 5, 3, 2), 'b_shape': (3, 2, 6), 'axes': ([2, 3], [0, 1]), 'gc_shape': (4, 5, 6)},  # NOQA
        {'a_shape': (4, 5, 2, 3), 'b_shape': (3, 6, 2), 'axes': ([3, 2], [0, 2]), 'gc_shape': (4, 5, 6)},  # NOQA
        {'a_shape': (4, 2, 5, 3), 'b_shape': (6, 3, 2), 'axes': ([3, 1], [1, 2]), 'gc_shape': (4, 5, 6)},  # NOQA
        {'a_shape': (2, 4, 5, 3), 'b_shape': (6, 2, 3), 'axes': ([3, 0], [2, 1]), 'gc_shape': (4, 5, 6)},  # NOQA
        {'a_shape': (2, 4, 3, 5), 'b_shape': (2, 6, 3), 'axes': ([2, 0], [2, 0]), 'gc_shape': (4, 5, 6)},  # NOQA
        {'a_shape': (2, 3, 4, 5), 'b_shape': (2, 3, 6), 'axes': ([1, 0], [1, 0]), 'gc_shape': (4, 5, 6)},  # NOQA
        {'a_shape': (3, 2, 4, 5), 'b_shape': (3, 2, 6), 'axes': ([0, 1], [0, 1]), 'gc_shape': (4, 5, 6)},  # NOQA
        {'a_shape': (3, 2, 5, 4), 'b_shape': (3, 6, 2), 'axes': ([0, 1], [0, 2]), 'gc_shape': (5, 4, 6)},  # NOQA
        {'a_shape': (3, 5, 2, 4), 'b_shape': (6, 3, 2), 'axes': ([0, 2], [1, 2]), 'gc_shape': (5, 4, 6)},  # NOQA
        {'a_shape': (5, 3, 2, 4), 'b_shape': (6, 2, 3), 'axes': ([1, 2], [2, 1]), 'gc_shape': (5, 4, 6)},  # NOQA

        {'a_shape': (5, 4, 3, 2), 'b_shape': (4, 3, 2, 6), 'axes': 3, 'gc_shape': (5, 6)},  # NOQA
        {'a_shape': (5, 4, 3, 2), 'b_shape': (4, 3, 2, 6), 'axes': ([1, 2, 3], [0, 1, 2]), 'gc_shape': (5, 6)},  # NOQA
        {'a_shape': (5, 4, 2, 3), 'b_shape': (4, 3, 6, 2), 'axes': ([1, 3, 2], [0, 1, 3]), 'gc_shape': (5, 6)},  # NOQA
        {'a_shape': (5, 2, 4, 3), 'b_shape': (4, 6, 3, 2), 'axes': ([2, 3, 1], [0, 2, 3]), 'gc_shape': (5, 6)},  # NOQA
        {'a_shape': (2, 5, 4, 3), 'b_shape': (4, 6, 2, 3), 'axes': ([2, 3, 0], [0, 3, 2]), 'gc_shape': (5, 6)},  # NOQA
        {'a_shape': (2, 5, 3, 4), 'b_shape': (6, 4, 2, 3), 'axes': ([3, 2, 0], [1, 3, 2]), 'gc_shape': (5, 6)},  # NOQA
        {'a_shape': (2, 3, 5, 4), 'b_shape': (6, 2, 4, 3), 'axes': ([3, 1, 0], [2, 3, 1]), 'gc_shape': (5, 6)},  # NOQA
        {'a_shape': (3, 2, 5, 4), 'b_shape': (6, 2, 3, 4), 'axes': ([3, 0, 1], [3, 2, 1]), 'gc_shape': (5, 6)},  # NOQA
        {'a_shape': (3, 2, 4, 5), 'b_shape': (2, 6, 3, 4), 'axes': ([2, 0, 1], [3, 2, 0]), 'gc_shape': (5, 6)},  # NOQA
        {'a_shape': (3, 4, 2, 5), 'b_shape': (2, 3, 6, 4), 'axes': ([1, 0, 2], [3, 1, 0]), 'gc_shape': (5, 6)},  # NOQA
        {'a_shape': (4, 3, 2, 5), 'b_shape': (2, 3, 4, 6), 'axes': ([0, 1, 2], [2, 1, 0]), 'gc_shape': (5, 6)},  # NOQA
        {'a_shape': (4, 3, 5, 2), 'b_shape': (3, 2, 4, 6), 'axes': ([0, 1, 3], [2, 0, 1]), 'gc_shape': (5, 6)},  # NOQA
        {'a_shape': (4, 5, 3, 2), 'b_shape': (3, 4, 2, 6), 'axes': ([0, 2, 3], [1, 0, 2]), 'gc_shape': (5, 6)},  # NOQA

        {'a_shape': (3, 2), 'b_shape': (2, 4), 'axes': 1, 'gc_shape': (3, 4)},  # NOQA
        {'a_shape': (3, 2), 'b_shape': (2, 4), 'axes': (1, 0), 'gc_shape': (3, 4)},  # NOQA
        {'a_shape': (3, 2), 'b_shape': (4, 2), 'axes': (1, 1), 'gc_shape': (3, 4)},  # NOQA
        {'a_shape': (2, 3), 'b_shape': (4, 2), 'axes': (0, 1), 'gc_shape': (3, 4)},  # NOQA
        {'a_shape': (2, 3), 'b_shape': (2, 4), 'axes': (0, 0), 'gc_shape': (3, 4)},  # NOQA

        {'a_shape': (), 'b_shape': (), 'axes': 0, 'gc_shape': ()},  # NOQA
        {'a_shape': (2), 'b_shape': (3), 'axes': 0, 'gc_shape': (2, 3)},  # NOQA
        {'a_shape': (), 'b_shape': (2, 3), 'axes': 0, 'gc_shape': (2, 3)},  # NOQA
        {'a_shape': (2, 3), 'b_shape': (), 'axes': 0, 'gc_shape': (2, 3)},  # NOQA
        {'a_shape': (2, 3), 'b_shape': (4), 'axes': 0, 'gc_shape': (2, 3, 4)},  # NOQA
        {'a_shape': (2), 'b_shape': (3, 4), 'axes': 0, 'gc_shape': (2, 3, 4)},  # NOQA
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
class TestTensorDot(unittest.TestCase):

    def setUp(self):
        self.a = self._setup_tensor(.5, 1, self.a_shape, self.a_dtype)
        self.b = self._setup_tensor(.5, 1, self.b_shape, self.b_dtype)
        ret_dtype = numpy.result_type(self.a_dtype, self.b_dtype)
        self.gc = self._setup_tensor(-1, 1, self.gc_shape, ret_dtype)
        self.gga = self._setup_tensor(.5, 1, self.a_shape, self.a_dtype)
        self.ggb = self._setup_tensor(.5, 1, self.b_shape, self.b_dtype)

        self.op = lambda a, b: F.tensordot(a, b, axes=self.axes)
        self.forward_answer = numpy.tensordot(self.a, self.b, self.axes)

    def _setup_tensor(self, _min, _max, shape, dtype):
        return numpy.random.uniform(_min, _max, shape).astype(dtype)

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

    def check_backward(self, a_data, b_data, c_grad, atol, rtol):
        gradient_check.check_backward(
            self.op, (a_data, b_data), c_grad, atol=atol, rtol=rtol,
            dtype=numpy.float32)

    def test_tensordot_backward_cpu(self):
        self.check_backward(self.a, self.b, self.gc, atol=1e-2, rtol=5e-2)

    @attr.gpu
    def test_tensordot_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.a), cuda.to_gpu(self.b),
            cuda.to_gpu(self.gc), atol=1e-2, rtol=1e-2)

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


class TestTensorDotInvalid(unittest.TestCase):

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


testing.run_module(__name__, __file__)
