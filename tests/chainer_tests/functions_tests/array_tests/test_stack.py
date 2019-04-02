import unittest

import numpy

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
import chainerx


@testing.parameterize(*testing.product_dict(
    [
        {'shape': (3, 4), 'axis': 0, 'y_shape': (2, 3, 4)},
        {'shape': (3, 4), 'axis': 1, 'y_shape': (3, 2, 4)},
        {'shape': (3, 4), 'axis': 2, 'y_shape': (3, 4, 2)},
        {'shape': (3, 4), 'axis': -1, 'y_shape': (3, 4, 2)},
        {'shape': (3, 4), 'axis': -2, 'y_shape': (3, 2, 4)},
        {'shape': (3, 4), 'axis': -3, 'y_shape': (2, 3, 4)},
        {'shape': (), 'axis': 0, 'y_shape': (2,)},
        {'shape': (), 'axis': -1, 'y_shape': (2,)},
    ],
    [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64},
    ]
))
class TestStack(unittest.TestCase):

    def setUp(self):
        self.xs = [
            numpy.random.uniform(-1, 1, self.shape).astype(self.dtype),
            numpy.random.uniform(-1, 1, self.shape).astype(self.dtype),
        ]
        self.g = numpy.random.uniform(-1, 1, self.y_shape).astype(self.dtype)
        self.ggs = [
            numpy.random.uniform(-1, 1, self.shape).astype(self.dtype),
            numpy.random.uniform(-1, 1, self.shape).astype(self.dtype),
        ]

    def check_forward(self, xs_data):
        xs = [chainer.Variable(x) for x in xs_data]
        y = functions.stack(xs, axis=self.axis)

        if hasattr(numpy, 'stack'):
            # run test only with numpy>=1.10
            expect = numpy.stack(self.xs, axis=self.axis)
            testing.assert_allclose(y.data, expect)

        y_data = backend.CpuDevice().send(y.data)
        self.assertEqual(y_data.shape[self.axis], 2)
        numpy.testing.assert_array_equal(
            y_data.take(0, axis=self.axis), self.xs[0])
        numpy.testing.assert_array_equal(
            y_data.take(1, axis=self.axis), self.xs[1])

    def test_forward_cpu(self):
        self.check_forward(self.xs)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward([cuda.to_gpu(x) for x in self.xs])

    @attr.chainerx
    def test_forward_chainerx(self):
        self.check_forward([chainerx.array(x) for x in self.xs])

    def check_backward(self, xs_data, g_data):
        def func(*xs):
            return functions.stack(xs, self.axis)

        gradient_check.check_backward(
            func, xs_data, g_data, eps=2.0 ** -2, dtype='d')

    def test_backward_cpu(self):
        self.check_backward(self.xs, self.g)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(
            [cuda.to_gpu(x) for x in self.xs], cuda.to_gpu(self.g))

    @attr.chainerx
    def test_backward_chainerx(self):
        self.check_backward(
            [chainerx.array(x) for x in self.xs], chainerx.array(self.g))

    def check_double_backward(self, xs_data, g_data, ggs_data):
        def func(*xs):
            return functions.stack(xs, self.axis)

        gradient_check.check_double_backward(
            func, xs_data, g_data, ggs_data, dtype=numpy.float64)

    def test_double_backward_cpu(self):
        self.check_double_backward(self.xs, self.g, self.ggs)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.xs), cuda.to_gpu(self.g), cuda.to_gpu(self.ggs))

    @attr.chainerx
    def test_double_backward_chainerx(self):
        self.check_double_backward(
            backend.to_chx(self.xs),
            backend.to_chx(self.g),
            backend.to_chx(self.ggs))


testing.run_module(__name__, __file__)
