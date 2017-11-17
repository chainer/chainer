import unittest

import numpy
import six

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.utils import type_check


@testing.parameterize(*testing.product_dict(
    [
        {'shape': (2, 3, 4), 'y_shape': (4, 3, 4), 'xs_length': 2},
        {'shape': (3, 4), 'y_shape': (6, 4), 'xs_length': 2},
        {'shape': (3), 'y_shape': (2, 3), 'xs_length': 2},
        {'shape': (), 'y_shape': (2, 1), 'xs_length': 2},
        {'shape': (2, 3, 4), 'y_shape': (2, 3, 4), 'xs_length': 1},
        {'shape': (3, 4), 'y_shape': (3, 4), 'xs_length': 1},
        {'shape': (3), 'y_shape': (1, 3), 'xs_length': 1},
        {'shape': (), 'y_shape': (1, 1), 'xs_length': 1},
    ],
    [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64},
    ]
))
class TestVstack(unittest.TestCase):

    def setUp(self):
        self.xs = [
            numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
            for i in six.moves.range(self.xs_length)
        ]
        self.gy = numpy.random.uniform(-1, 1, self.y_shape).astype(self.dtype)
        self.ggxs = [
            numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
            for i in six.moves.range(self.xs_length)
        ]
        self.check_double_backward_options = {}
        if self.dtype == numpy.float16:
            self.check_double_backward_options = {'atol': 5e-4, 'rtol': 5e-3}

    def check_forward(self, xs_data):
        xs = [chainer.Variable(x) for x in xs_data]
        y = functions.vstack(xs)

        expect = numpy.vstack(self.xs)
        testing.assert_allclose(y.data, expect)

    def test_forward_cpu(self):
        self.check_forward(self.xs)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward([cuda.to_gpu(x) for x in self.xs])

    def check_backward(self, xs_data, y_grad):
        def func(*xs):
            return functions.vstack(xs)

        gradient_check.check_backward(
            func, xs_data, y_grad, eps=2.0 ** -2, dtype='d')

    def test_backward_cpu(self):
        self.check_backward(self.xs, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(
            [cuda.to_gpu(x) for x in self.xs], cuda.to_gpu(self.gy))

    def check_double_backward(self, xs_data, y_grad, xs_grad_grad):
        def func(*xs):
            y = functions.vstack(xs)
            return y * y

        gradient_check.check_double_backward(
            func, xs_data, y_grad, xs_grad_grad, eps=2.0 ** -2, dtype='d',
            **self.check_double_backward_options)

    def test_doble_backward_cpu(self):
        self.check_double_backward(self.xs, self.gy, self.ggxs)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(
            [cuda.to_gpu(x) for x in self.xs], cuda.to_gpu(self.gy),
            [cuda.to_gpu(ggx) for ggx in self.ggxs])


@testing.parameterize(
    {'a_shape': (2, 4, 5), 'b_shape': (3, 4, 5), 'valid': True},
    {'a_shape': (3, 4, 6), 'b_shape': (3, 4, 5), 'valid': False},
    {'a_shape': (3, 6, 5), 'b_shape': (3, 4, 5), 'valid': False},
    {'a_shape': (3, 4), 'b_shape': (4, 4), 'valid': True},
    {'a_shape': (3, 4), 'b_shape': (3, 3), 'valid': False},
    {'a_shape': (3,), 'b_shape': (4,), 'valid': False},
    {'a_shape': (3), 'b_shape': (3, 3), 'valid': False},
    {'a_shape': (), 'b_shape': (1), 'valid': False},
)
class TestVstackTypeCheck(unittest.TestCase):

    def setUp(self):
        self.xs = [
            numpy.random.uniform(-1, 1, self.a_shape).astype(numpy.float32),
            numpy.random.uniform(-1, 1, self.b_shape).astype(numpy.float32),
        ]

    def check_value_check(self):
        if self.valid:
            # Check if it throws nothing
            functions.vstack(self.xs)
        else:
            with self.assertRaises(type_check.InvalidType):
                functions.vstack(self.xs)

    def test_value_check_cpu(self):
        self.check_value_check()

    @attr.gpu
    def test_value_check_gpu(self):
        self.check_value_check()


testing.run_module(__name__, __file__)
