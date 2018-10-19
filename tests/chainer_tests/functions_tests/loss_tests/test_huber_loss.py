import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product_dict(
    [{'dtype': numpy.float16,
      'forward_options': {'rtol': 5e-3, 'atol': 5e-3},
      'backward_options': {'eps': 1e-1, 'rtol': 1e0, 'atol': 1e0},
      'double_backward_options': {'eps': 1e-1, 'rtol': 1e0, 'atol': 1e0}},
     {'dtype': numpy.float32,
      'forward_options': {},
      'backward_options': {'eps': 1e-2, 'rtol': 1e-2, 'atol': 1e-2},
      'double_backward_options': {'eps': 1e-2, 'rtol': 1e-1, 'atol': 1e-1}},
     {'dtype': numpy.float64,
      'forward_options': {},
      'backward_options': {'eps': 1e-2, 'rtol': 1e-2, 'atol': 1e-2},
      'double_backward_options': {'eps': 1e-2, 'rtol': 1e-1, 'atol': 1e-1}},
     ],
    testing.product({
        'shape': [(4, 10), (2, 5, 3, 3)],
        'reduce': ['no', 'sum_along_second_axis'],
    }),
))
class TestHuberLoss(unittest.TestCase):

    def setUp(self):
        self._config_user = chainer.using_config('dtype', self.dtype)
        self._config_user.__enter__()

        self.x = (numpy.random.random(self.shape) - 0.5) * 20
        self.x = self.x.astype(self.dtype)
        self.t = numpy.random.random(self.shape).astype(self.dtype)
        if self.reduce == 'sum_along_second_axis':
            gy_shape = self.shape[:1] + self.shape[2:]
        else:
            gy_shape = self.shape
        self.gy = numpy.random.random(gy_shape).astype(self.dtype)
        self.ggx = numpy.random.uniform(-1, 1, self.x.shape)
        self.ggx = self.ggx.astype(self.dtype)
        self.ggt = numpy.random.uniform(-1, 1, self.t.shape)
        self.ggt = self.ggt.astype(self.dtype)

    def tearDown(self):
        self._config_user.__exit__(None, None, None)

    def check_forward(self, x_data, t_data):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)
        loss = functions.huber_loss(x, t, delta=1, reduce=self.reduce)
        self.assertEqual(loss.data.dtype, self.dtype)
        loss_value = cuda.to_cpu(loss.data)

        diff_data = cuda.to_cpu(x_data) - cuda.to_cpu(t_data)
        loss_expect = numpy.zeros(self.shape)
        mask = numpy.abs(diff_data) < 1
        loss_expect[mask] = 0.5 * diff_data[mask] ** 2
        loss_expect[~mask] = numpy.abs(diff_data[~mask]) - 0.5
        if self.reduce == 'sum_along_second_axis':
            loss_expect = numpy.sum(loss_expect, axis=1)
        testing.assert_allclose(
            loss_value, loss_expect, **self.forward_options)

    def test_forward_cpu(self):
        self.check_forward(self.x, self.t)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.t))

    def check_backward(self, x_data, t_data, y_grad):
        def f(x, t):
            return functions.huber_loss(x, t, delta=1, reduce=self.reduce)

        gradient_check.check_backward(
            f, (x_data, t_data), y_grad, **self.backward_options)

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
            f, (x_data, t_data), y_grad, (x_grad_grad, t_grad_grad),
            **self.double_backward_options)

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
