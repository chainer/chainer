import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


class TestMax(unittest.TestCase):

    def setUp(self):
        eps = 1e-5

        # Sample x with single maximum value
        while True:
            self.x = numpy.random.uniform(
                -1, 1, (3, 2, 4)).astype(numpy.float32)
            if (self.x > (numpy.max(self.x) - 2 * eps)).sum() == 1:
                break

        self.gy = numpy.array(2, dtype=numpy.float32)
        self.ggx = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(numpy.float32)

        self.check_backward_options = {
            'eps': eps, 'atol': 1e-3, 'rtol': 1e-2}
        self.check_double_backward_options = {
            'eps': eps, 'atol': 1e-3, 'rtol': 1e-2}

    def check_forward(self, x_data, axis=None, keepdims=False):
        x = chainer.Variable(x_data)
        y = functions.max(x, axis=axis, keepdims=keepdims)
        self.assertEqual(y.data.dtype, numpy.float32)
        y_expect = self.x.max(axis=axis, keepdims=keepdims)
        self.assertEqual(y.data.shape, y_expect.shape)
        testing.assert_allclose(y_expect, y.data)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    def test_forward_axis_cpu(self):
        for i in range(self.x.ndim):
            self.check_forward(self.x, axis=i)

    def test_forward_negative_axis_cpu(self):
        self.check_forward(self.x, axis=-1)

    def test_forward_multi_axis_cpu(self):
        self.check_forward(self.x, axis=(0, 1))

    def test_forward_multi_axis_invert_cpu(self):
        self.check_forward(self.x, axis=(1, 0))

    def test_forward_negative_multi_axis_cpu(self):
        self.check_forward(self.x, axis=(0, -1))

    def test_forward_negative_multi_axis_invert_cpu(self):
        self.check_forward(self.x, axis=(-2, 0))

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    @attr.gpu
    def test_forward_axis_gpu(self):
        for i in range(self.x.ndim):
            self.check_forward(cuda.to_gpu(self.x), axis=i)

    @attr.gpu
    def test_forward_negative_axis_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), axis=-1)

    @attr.gpu
    def test_forward_multi_axis_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), axis=(0, 1))

    @attr.gpu
    def test_forward_multi_axis_invert_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), axis=(1, 0))

    @attr.gpu
    def test_forward_negative_multi_axis_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), axis=(0, -1))

    @attr.gpu
    def test_forward_negative_multi_axis_invert_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), axis=(-2, 0))

    def check_backward(self, x_data, y_grad, axis=None, keepdims=False):
        gradient_check.check_backward(
            lambda x: functions.max(x, axis, keepdims),
            x_data, y_grad, dtype='d',
            **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    def test_backward_axis_cpu(self):
        for i in range(self.x.ndim):
            gy = numpy.ones_like(self.x.max(axis=i)) * self.gy
            self.check_backward(self.x, gy, axis=i)

    def test_backward_negative_axis_cpu(self):
        gy = numpy.ones_like(self.x.max(axis=-1)) * self.gy
        self.check_backward(self.x, gy, axis=-1)

    def test_backward_multi_axis_cpu(self):
        gy = numpy.ones_like(self.x.max(axis=(0, 1))) * self.gy
        self.check_backward(self.x, gy, axis=(0, 1))

    def test_backward_multi_axis_invert_cpu(self):
        gy = numpy.ones_like(self.x.max(axis=(1, 0))) * self.gy
        self.check_backward(self.x, gy, axis=(1, 0))

    def test_backward_negative_multi_axis_cpu(self):
        gy = numpy.ones_like(self.x.max(axis=(0, -1))) * self.gy
        self.check_backward(self.x, gy, axis=(0, -1))

    def test_backward_negative_multi_axis_invert_cpu(self):
        gy = numpy.ones_like(self.x.max(axis=(-2, 0))) * self.gy
        self.check_backward(self.x, gy, axis=(-2, 0))

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    @attr.gpu
    def test_backward_axis_gpu(self):
        for i in range(self.x.ndim):
            gy = numpy.ones_like(self.x.sum(axis=i)) * self.gy
            self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(gy), axis=i)

    @attr.gpu
    def test_backward_negative_axis_gpu(self):
        for i in range(self.x.ndim):
            gy = numpy.ones_like(self.x.sum(axis=-1)) * self.gy
            self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(gy), axis=-1)

    @attr.gpu
    def test_backward_multi_axis_gpu(self):
        gy = numpy.ones_like(self.x.sum(axis=(0, 1))) * self.gy
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(gy), axis=(0, 1))

    @attr.gpu
    def test_backward_multi_axis_invert_gpu(self):
        gy = numpy.ones_like(self.x.sum(axis=(1, 0))) * self.gy
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(gy), axis=(1, 0))

    @attr.gpu
    def test_backward_negative_multi_axis_gpu(self):
        gy = numpy.ones_like(self.x.sum(axis=(0, -1))) * self.gy
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(gy), axis=(0, -1))

    @attr.gpu
    def test_backward_negative_multi_axis_invert_gpu(self):
        gy = numpy.ones_like(self.x.sum(axis=(-2, 0))) * self.gy
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(gy), axis=(-2, 0))

    def check_double_backward(
            self, x_data, y_grad, x_grad_grad, axis=None, keepdims=False):
        def f(x):
            x = functions.max(x, axis, keepdims)
            return x * x

        gradient_check.check_double_backward(
            f, x_data, y_grad, x_grad_grad, dtype='d',
            **self.check_double_backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(self.x, self.gy, self.ggx)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.gy), cuda.to_gpu(self.ggx))

    def test_invalid_axis_type(self):
        with self.assertRaises(TypeError):
            functions.Max([0])

    def test_invalid_axis_type_in_tuple(self):
        with self.assertRaises(TypeError):
            functions.Max((1, 'x'))

    def test_duplicate_axis(self):
        with self.assertRaises(ValueError):
            functions.Max((0, 0))

    def test_pos_neg_duplicate_axis(self):
        with self.assertRaises(ValueError):
            self.x.max(axis=(1, -2))


class TestMin(unittest.TestCase):

    def setUp(self):
        eps = 1e-5

        # Sample x with single minimum value
        while True:
            self.x = numpy.random.uniform(
                -1, 1, (3, 2, 4)).astype(numpy.float32)
            if (self.x < (numpy.min(self.x) + 2 * eps)).sum() == 1:
                break

        self.gy = numpy.array(2, dtype=numpy.float32)
        self.ggx = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(numpy.float32)

        self.check_backward_options = {
            'eps': eps, 'atol': 1e-3, 'rtol': 1e-2}
        self.check_double_backward_options = {
            'eps': eps, 'atol': 1e-3, 'rtol': 1e-2}

    def check_forward(self, x_data, axis=None, keepdims=False):
        x = chainer.Variable(x_data)
        y = functions.min(x, axis=axis, keepdims=keepdims)
        self.assertEqual(y.data.dtype, numpy.float32)
        y_expect = self.x.min(axis=axis, keepdims=keepdims)
        self.assertEqual(y.data.shape, y_expect.shape)
        testing.assert_allclose(y_expect, y.data)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    def test_forward_axis_cpu(self):
        for i in range(self.x.ndim):
            self.check_forward(self.x, axis=i)

    def test_forward_negative_axis_cpu(self):
        self.check_forward(self.x, axis=-1)

    def test_forward_multi_axis_cpu(self):
        self.check_forward(self.x, axis=(0, 1))

    def test_forward_multi_axis_invert_cpu(self):
        self.check_forward(self.x, axis=(1, 0))

    def test_forward_negative_multi_axis_cpu(self):
        self.check_forward(self.x, axis=(0, -1))

    def test_forward_negative_multi_axis_invert_cpu(self):
        self.check_forward(self.x, axis=(-2, 0))

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    @attr.gpu
    def test_forward_axis_gpu(self):
        for i in range(self.x.ndim):
            self.check_forward(cuda.to_gpu(self.x), axis=i)

    @attr.gpu
    def test_forward_negative_axis_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), axis=-1)

    @attr.gpu
    def test_forward_multi_axis_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), axis=(0, 1))

    @attr.gpu
    def test_forward_multi_axis_invert_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), axis=(1, 0))

    @attr.gpu
    def test_forward_negative_multi_axis_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), axis=(0, -1))

    @attr.gpu
    def test_forward_negative_multi_axis_invert_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), axis=(-2, 0))

    def check_backward(self, x_data, y_grad, axis=None, keepdims=False):
        gradient_check.check_backward(
            lambda x: functions.min(x, axis=axis, keepdims=keepdims),
            x_data, y_grad, dtype='d',
            **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    def test_backward_axis_cpu(self):
        for i in range(self.x.ndim):
            gy = numpy.ones_like(self.x.min(axis=i)) * self.gy
            self.check_backward(self.x, gy, axis=i)

    def test_backward_negative_axis_cpu(self):
        gy = numpy.ones_like(self.x.min(axis=-1)) * self.gy
        self.check_backward(self.x, gy, axis=-1)

    def test_backward_multi_axis_cpu(self):
        gy = numpy.ones_like(self.x.min(axis=(0, 1))) * self.gy
        self.check_backward(self.x, gy, axis=(0, 1))

    def test_backward_multi_axis_invert_cpu(self):
        gy = numpy.ones_like(self.x.min(axis=(1, 0))) * self.gy
        self.check_backward(self.x, gy, axis=(1, 0))

    def test_backward_negative_multi_axis_cpu(self):
        gy = numpy.ones_like(self.x.min(axis=(0, -1))) * self.gy
        self.check_backward(self.x, gy, axis=(0, -1))

    def test_backward_negative_multi_axis_invert_cpu(self):
        gy = numpy.ones_like(self.x.min(axis=(-2, 0))) * self.gy
        self.check_backward(self.x, gy, axis=(-2, 0))

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    @attr.gpu
    def test_backward_axis_gpu(self):
        for i in range(self.x.ndim):
            gy = numpy.ones_like(self.x.sum(axis=i)) * self.gy
            self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(gy), axis=i)

    @attr.gpu
    def test_backward_negative_axis_gpu(self):
        for i in range(self.x.ndim):
            gy = numpy.ones_like(self.x.sum(axis=-1)) * self.gy
            self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(gy), axis=-1)

    @attr.gpu
    def test_backward_multi_axis_gpu(self):
        gy = numpy.ones_like(self.x.sum(axis=(0, 1))) * self.gy
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(gy), axis=(0, 1))

    @attr.gpu
    def test_backward_multi_axis_invert_gpu(self):
        gy = numpy.ones_like(self.x.sum(axis=(1, 0))) * self.gy
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(gy), axis=(1, 0))

    @attr.gpu
    def test_backward_negative_multi_axis_gpu(self):
        gy = numpy.ones_like(self.x.sum(axis=(0, -1))) * self.gy
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(gy), axis=(0, -1))

    @attr.gpu
    def test_backward_negative_multi_axis_invert_gpu(self):
        gy = numpy.ones_like(self.x.sum(axis=(-2, 0))) * self.gy
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(gy), axis=(-2, 0))

    def check_double_backward(
            self, x_data, y_grad, x_grad_grad, axis=None, keepdims=False):
        def f(x):
            x = functions.max(x, axis, keepdims)
            return x * x

        gradient_check.check_double_backward(
            f, x_data, y_grad, x_grad_grad, dtype='d',
            **self.check_double_backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(self.x, self.gy, self.ggx)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.gy), cuda.to_gpu(self.ggx))

    def test_invalid_axis_type(self):
        with self.assertRaises(TypeError):
            functions.Min([0])

    def test_invalid_axis_type_in_tuple(self):
        with self.assertRaises(TypeError):
            functions.Min((1, 'x'))

    def test_duplicate_axis(self):
        with self.assertRaises(ValueError):
            functions.Min((0, 0))

    def test_pos_neg_duplicate_axis(self):
        with self.assertRaises(ValueError):
            self.x.min(axis=(1, -2))


@testing.parameterize(*testing.product_dict(
    [
        {'function_name': 'argmax', 'function_class': functions.ArgMax},
        {'function_name': 'argmin', 'function_class': functions.ArgMin},
    ],
    [
        {'axis': None},
        {'axis': 0},
        {'axis': 1},
        {'axis': 2},
        {'axis': -1},
        {'axis': -2},
        {'axis': -3},
    ],
    [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64},
    ]
))
class TestArgMinMax(unittest.TestCase):

    def setUp(self):
        self.function = getattr(functions, self.function_name)
        self.expect = getattr(numpy, self.function_name)

        self.x = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(self.dtype)

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = self.function(x, axis=self.axis)
        self.assertEqual(y.data.dtype, numpy.int32)
        y_expect = self.expect(self.x, axis=self.axis)
        self.assertEqual(y.data.shape, y_expect.shape)
        testing.assert_allclose(y_expect, y.data)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data):
        x = chainer.Variable(x_data)
        y = self.function(x, axis=self.axis)
        y.backward()
        self.assertIsNone(x.grad)

    def test_backward_cpu(self):
        self.check_backward(self.x)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x))

    def test_invalid_axis_type(self):
        with self.assertRaises(TypeError):
            self.function_class([0])

    def test_invalid_axis_type_in_tuple(self):
        with self.assertRaises(TypeError):
            self.function_class((1, 'x'))


testing.run_module(__name__, __file__)
