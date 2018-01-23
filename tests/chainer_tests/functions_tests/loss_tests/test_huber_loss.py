import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(
    {'reduce': 'no'},
    {'reduce': 'sum_along_second_axis'}
)
class TestHuberLoss(unittest.TestCase):

    def setUp(self):
        self.shape = (4, 10)
        self.x = (numpy.random.random(self.shape) - 0.5) * 20
        self.x = self.x.astype(numpy.float32)
        self.t = numpy.random.random(self.shape).astype(numpy.float32)
        if self.reduce == 'sum_along_second_axis':
            gy_shape = self.shape[0]
        else:
            gy_shape = self.shape
        self.gy = numpy.random.random(gy_shape).astype(numpy.float32)
        self.ggx = numpy.random.uniform(-1, 1, self.x.shape)
        self.ggx = self.ggx.astype(numpy.float32)
        self.ggt = numpy.random.uniform(-1, 1, self.t.shape)
        self.ggt = self.ggt.astype(numpy.float32)

        self.check_backward_options = {'atol': 1e-2, 'rtol': 1e-2}
        self.check_double_backward_options = {'atol': 5e-2, 'rtol': 5e-2}

    def check_forward(self, x_data, t_data):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)
        loss = functions.huber_loss(x, t, delta=1, reduce=self.reduce)
        self.assertEqual(loss.data.dtype, numpy.float32)
        loss_value = cuda.to_cpu(loss.data)

        diff_data = cuda.to_cpu(x_data) - cuda.to_cpu(t_data)
        loss_expect = numpy.zeros(self.shape)
        mask = numpy.abs(diff_data) < 1
        loss_expect[mask] = 0.5 * diff_data[mask] ** 2
        loss_expect[~mask] = numpy.abs(diff_data[~mask]) - 0.5
        if self.reduce == 'sum_along_second_axis':
            loss_expect = numpy.sum(loss_expect, axis=1)
        testing.assert_allclose(loss_value, loss_expect)

    def test_forward_cpu(self):
        self.check_forward(self.x, self.t)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.t))

    def check_backward(self, x_data, t_data, y_grad):
        def f(x, t):
            return functions.huber_loss(x, t, delta=1, reduce=self.reduce)

        gradient_check.check_backward(
            f, (x_data, t_data), y_grad, eps=1e-2,
            **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.t, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.t),
                            cuda.to_gpu(self.gy))

    def check_double_backward(self, x_data, t_data, y_grad, x_grad_grad,
                              t_grad_grad):
        def f(x, t):
            return functions.huber_loss(x, t, delta=1, reduce=self.reduce)

        gradient_check.check_double_backward(
            f, (x_data, t_data), y_grad, (x_grad_grad, t_grad_grad), eps=1e-2,
            **self.check_double_backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(self.x, self.t, self.gy, self.ggx, self.ggt)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.t), cuda.to_gpu(self.gy),
            cuda.to_gpu(self.ggx), cuda.to_gpu(self.ggt))


class TestHuberLossInvalidReductionOption(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (4, 10)).astype(numpy.float32)
        self.t = numpy.random.uniform(-1, 1, (4, 10)).astype(numpy.float32)

    def check_invalid_option(self, xp):
        x = xp.asarray(self.x)
        t = xp.asarray(self.t)
        with self.assertRaises(ValueError):
            functions.huber_loss(x, t, 1, 'invalid_option')

    def test_invalid_option_cpu(self):
        self.check_invalid_option(numpy)

    @attr.gpu
    def test_invalid_option_gpu(self):
        self.check_invalid_option(cuda.cupy)


testing.run_module(__name__, __file__)
