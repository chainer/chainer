import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
import chainerx


@testing.parameterize(*testing.product({
    'axis': [
        None,
        0, 1, 2,  # axis
        -1,  # negative_axis
        (0, 1),  # multi_axis
        (1, 0),  # multi_axis_invert
        (0, -1),  # negative_multi_axis
        (-2, 0),  # negative_multi_axis_invert
    ],
    'keepdims': [True, False],
}))
class TestMax(unittest.TestCase):

    def setUp(self):
        eps = 1e-5

        # Sample x with single maximum value
        while True:
            x = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(numpy.float32)
            y = x.max(axis=self.axis, keepdims=True)
            if numpy.all((x > y - 2 * eps).sum(axis=self.axis) == 1):
                self.x = x
                if not self.keepdims:
                    y = y.squeeze(axis=self.axis)
                self.y_expect = y
                break

        self.gy = numpy.full(self.y_expect.shape, 2, dtype=numpy.float32)
        self.ggx = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(numpy.float32)

        self.check_backward_options = {
            'eps': eps, 'atol': 1e-3, 'rtol': 1e-2}
        self.check_double_backward_options = {
            'eps': eps, 'atol': 1e-3, 'rtol': 1e-2}

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = functions.max(x, axis=self.axis, keepdims=self.keepdims)
        self.assertEqual(y.data.dtype, numpy.float32)
        y_expect = self.y_expect
        self.assertEqual(y.data.shape, y_expect.shape)
        testing.assert_allclose(y_expect, y.data)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    @attr.chainerx
    def test_forward_chainerx(self):
        self.check_forward(chainerx.array(self.x))

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            lambda x: functions.max(x, self.axis, self.keepdims),
            x_data, y_grad, dtype='d',
            **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    @attr.chainerx
    def test_backward_chainerx(self):
        self.check_backward(chainerx.array(self.x), chainerx.array(self.gy))

    def check_double_backward(
            self, x_data, y_grad, x_grad_grad):
        def f(x):
            return functions.max(x, self.axis, self.keepdims)

        gradient_check.check_double_backward(
            f, x_data, y_grad, x_grad_grad, dtype='d',
            **self.check_double_backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(self.x, self.gy, self.ggx)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.gy), cuda.to_gpu(self.ggx))

    @attr.chainerx
    def test_double_backward_chainerx(self):
        self.check_double_backward(
            chainerx.array(self.x),
            chainerx.array(self.gy),
            chainerx.array(self.ggx))


class TestMaxInvalid(unittest.TestCase):

    def setUp(self):
        self.x = numpy.array([1], dtype=numpy.float32)

    def test_invalid_axis_type(self):
        with self.assertRaises(TypeError):
            functions.max(self.x, [0])

    def test_invalid_axis_type_in_tuple(self):
        with self.assertRaises(TypeError):
            functions.max(self.x, (1, 'x'))

    def test_duplicate_axis(self):
        with self.assertRaises(ValueError):
            functions.max(self.x, (0, 0))

    def test_pos_neg_duplicate_axis(self):
        x_data = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(numpy.float32)
        x = chainer.Variable(x_data)
        with self.assertRaises(ValueError):
            functions.max(x, axis=(1, -2))


@testing.parameterize(*testing.product({
    'axis': [
        None,
        0, 1, 2,  # axis
        -1,  # negative_axis
        (0, 1),  # multi_axis
        (1, 0),  # multi_axis_invert
        (0, -1),  # negative_multi_axis
        (-2, 0),  # negative_multi_axis_invert
    ],
    'keepdims': [True, False],
}))
class TestMin(unittest.TestCase):

    def setUp(self):
        eps = 1e-5

        # Sample x with single minimum value
        while True:
            x = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(numpy.float32)
            y = x.min(axis=self.axis, keepdims=True)
            if numpy.all((x < y + 2 * eps).sum(axis=self.axis) == 1):
                self.x = x
                if not self.keepdims:
                    y = y.squeeze(axis=self.axis)
                self.y_expect = y
                break

        self.gy = numpy.full(self.y_expect.shape, 2, dtype=numpy.float32)
        self.ggx = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(numpy.float32)

        self.check_backward_options = {
            'eps': eps, 'atol': 1e-3, 'rtol': 1e-2}
        self.check_double_backward_options = {
            'eps': eps, 'atol': 1e-3, 'rtol': 1e-2}

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = functions.min(x, axis=self.axis, keepdims=self.keepdims)
        self.assertEqual(y.data.dtype, numpy.float32)
        y_expect = self.y_expect
        self.assertEqual(y.data.shape, y_expect.shape)
        testing.assert_allclose(y_expect, y.data)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            lambda x: functions.min(x, self.axis, self.keepdims),
            x_data, y_grad, dtype='d',
            **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    def check_double_backward(
            self, x_data, y_grad, x_grad_grad):
        def f(x):
            return functions.min(x, self.axis, self.keepdims)

        gradient_check.check_double_backward(
            f, x_data, y_grad, x_grad_grad, dtype='d',
            **self.check_double_backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(self.x, self.gy, self.ggx)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.gy), cuda.to_gpu(self.ggx))


class TestMinInvalid(unittest.TestCase):

    def setUp(self):
        self.x = numpy.array([1], dtype=numpy.float32)

    def test_invalid_axis_type(self):
        with self.assertRaises(TypeError):
            functions.min(self.x, [0])

    def test_invalid_axis_type_in_tuple(self):
        with self.assertRaises(TypeError):
            functions.min(self.x, (1, 'x'))

    def test_duplicate_axis(self):
        with self.assertRaises(ValueError):
            functions.min(self.x, (0, 0))

    def test_pos_neg_duplicate_axis(self):
        x_data = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(numpy.float32)
        x = chainer.Variable(x_data)
        with self.assertRaises(ValueError):
            functions.min(x, axis=(1, -2))


@testing.parameterize(*testing.product_dict(
    [
        {'function_name': 'argmax'},
        {'function_name': 'argmin'},
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

    @attr.chainerx
    def test_forward_chainerx(self):
        self.check_forward(chainerx.array(self.x))

    def check_backward(self, x_data):
        x = chainer.Variable(x_data)
        y = self.function(x, axis=self.axis)
        if isinstance(x_data, chainerx.ndarray):
            with self.assertRaises(chainerx.ChainerxError):
                y.backward()
        else:
            y.backward()
            self.assertIsNone(x.grad)

    def test_backward_cpu(self):
        self.check_backward(self.x)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x))

    @attr.chainerx
    def test_backward_chainerx(self):
        self.check_backward(chainerx.array(self.x))

    def test_invalid_axis_type(self):
        with self.assertRaises(TypeError):
            self.function(self.x, [0])

    def test_invalid_axis_type_in_tuple(self):
        with self.assertRaises(TypeError):
            self.function(self.x, (1, 'x'))


testing.run_module(__name__, __file__)
