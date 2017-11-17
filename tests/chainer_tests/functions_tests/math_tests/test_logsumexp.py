import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(
    {'dtype': numpy.float16},
    {'dtype': numpy.float32},
    {'dtype': numpy.float64},
)
class TestLogSumExp(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1, ()).astype(self.dtype)
        self.ggx = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(self.dtype)
        self.check_forward_option = {}
        self.check_backward_option = {
            'eps': 2.0 ** -5, 'rtol': 1e-4, 'atol': 1e-4}
        self.check_double_backward_option = {
            'eps': 2.0 ** -5, 'rtol': 1e-4, 'atol': 1e-4}
        if self.dtype == numpy.float16:
            self.check_forward_option = {'rtol': 1e-2, 'atol': 1e-2}
            self.check_backward_option = {
                'eps': 2.0 ** -3, 'rtol': 1e-1, 'atol': 1e-1}
            self.check_double_backward_option = {
                'eps': 2.0 ** -3, 'rtol': 1e-1, 'atol': 1e-1}

    def check_forward(self, x_data, axis=None):
        x = chainer.Variable(x_data)
        y = functions.logsumexp(x, axis=axis)
        self.assertEqual(y.data.dtype, self.dtype)
        y_expect = numpy.log(numpy.exp(self.x).sum(axis=axis))
        testing.assert_allclose(
            y_expect, y.data, **self.check_forward_option)

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

    def check_backward(self, x_data, y_grad, axis=None):
        gradient_check.check_backward(
            lambda x: functions.logsumexp(x, axis), x_data, y_grad,
            **self.check_backward_option)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    def test_backward_axis_cpu(self):
        for i in range(self.x.ndim):
            gy = numpy.ones_like(self.x.sum(axis=i)) * self.gy
            self.check_backward(self.x, gy, axis=i)

    def test_backward_negative_axis_cpu(self):
        gy = numpy.ones_like(self.x.sum(axis=-1)) * self.gy
        self.check_backward(self.x, gy, axis=-1)

    def test_backward_multi_axis_cpu(self):
        gy = numpy.ones_like(self.x.sum(axis=(0, 1))) * self.gy
        self.check_backward(self.x, gy, axis=(0, 1))

    def test_backward_multi_axis_invert_cpu(self):
        gy = numpy.ones_like(self.x.sum(axis=(1, 0))) * self.gy
        self.check_backward(self.x, gy, axis=(1, 0))

    def test_backward_negative_multi_axis_cpu(self):
        gy = numpy.ones_like(self.x.sum(axis=(0, -1))) * self.gy
        self.check_backward(self.x, gy, axis=(0, -1))

    def test_backward_negative_multi_axis_invert_cpu(self):
        gy = numpy.ones_like(self.x.sum(axis=(-2, 0))) * self.gy
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

    def check_double_backward(self, x_data, y_grad, x_grad_grad, axis=None):
        gradient_check.check_double_backward(
            lambda x: functions.logsumexp(x, axis), x_data, y_grad,
            x_grad_grad, dtype=numpy.float64,
            **self.check_double_backward_option)

    def test_double_backward_cpu(self):
        self.check_double_backward(self.x, self.gy, self.ggx)

    def test_double_backward_axis_cpu(self):
        for i in range(self.x.ndim):
            gy = numpy.ones_like(self.x.sum(axis=i)) * self.gy
            self.check_double_backward(self.x, gy, self.ggx, axis=i)

    def test_double_backward_negative_axis_cpu(self):
        gy = numpy.ones_like(self.x.sum(axis=-1)) * self.gy
        self.check_double_backward(self.x, gy, self.ggx, axis=-1)

    def test_double_backward_multi_axis_cpu(self):
        gy = numpy.ones_like(self.x.sum(axis=(0, 1))) * self.gy
        self.check_double_backward(self.x, gy, self.ggx, axis=(0, 1))

    def test_double_backward_multi_axis_invert_cpu(self):
        gy = numpy.ones_like(self.x.sum(axis=(1, 0))) * self.gy
        self.check_double_backward(self.x, gy, self.ggx, axis=(1, 0))

    def test_double_backward_negative_multi_axis_cpu(self):
        gy = numpy.ones_like(self.x.sum(axis=(0, -1))) * self.gy
        self.check_double_backward(self.x, gy, self.ggx, axis=(0, -1))

    def test_double_backward_negative_multi_axis_invert_cpu(self):
        gy = numpy.ones_like(self.x.sum(axis=(-2, 0))) * self.gy
        self.check_double_backward(self.x, gy, self.ggx, axis=(-2, 0))

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.gy), cuda.to_gpu(self.ggx))

    @attr.gpu
    def test_double_backward_axis_gpu(self):
        for i in range(self.x.ndim):
            gy = numpy.ones_like(self.x.sum(axis=i)) * self.gy
            self.check_double_backward(
                cuda.to_gpu(self.x), cuda.to_gpu(gy), cuda.to_gpu(self.ggx),
                axis=i)

    @attr.gpu
    def test_double_backward_negative_axis_gpu(self):
        for i in range(self.x.ndim):
            gy = numpy.ones_like(self.x.sum(axis=-1)) * self.gy
            self.check_double_backward(
                cuda.to_gpu(self.x), cuda.to_gpu(gy), cuda.to_gpu(self.ggx),
                axis=-1)

    @attr.gpu
    def test_double_backward_multi_axis_gpu(self):
        gy = numpy.ones_like(self.x.sum(axis=(0, 1))) * self.gy
        self.check_double_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(gy), cuda.to_gpu(self.ggx),
            axis=(0, 1))

    @attr.gpu
    def test_double_backward_multi_axis_invert_gpu(self):
        gy = numpy.ones_like(self.x.sum(axis=(1, 0))) * self.gy
        self.check_double_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(gy), cuda.to_gpu(self.ggx),
            axis=(1, 0))

    @attr.gpu
    def test_double_backward_negative_multi_axis_gpu(self):
        gy = numpy.ones_like(self.x.sum(axis=(0, -1))) * self.gy
        self.check_double_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(gy), cuda.to_gpu(self.ggx),
            axis=(0, -1))

    @attr.gpu
    def test_double_backward_negative_multi_axis_invert_gpu(self):
        gy = numpy.ones_like(self.x.sum(axis=(-2, 0))) * self.gy
        self.check_double_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(gy), cuda.to_gpu(self.ggx),
            axis=(-2, 0))

    def test_invalid_axis_type(self):
        with self.assertRaises(TypeError):
            functions.Sum([0])

    def test_invalid_axis_type_in_tuple(self):
        with self.assertRaises(TypeError):
            functions.Sum((1, 'x'))

    def test_duplicate_axis(self):
        with self.assertRaises(ValueError):
            functions.Sum((0, 0))

    def test_pos_neg_duplicate_axis(self):
        with self.assertRaises(ValueError):
            self.x.sum(axis=(1, -2))


testing.run_module(__name__, __file__)
