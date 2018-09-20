import math
import unittest

import numpy
import six

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product_dict(
    [{'dtype': numpy.float16,
      'forward_options': {'rtol': 1e-2, 'atol': 1e-2},
      'backward_options': {'rtol': 1e-2, 'atol': 1e-3},
      'double_backward_options': {'rtol': 3e-1, 'atol': 3e-1}},
     {'dtype': numpy.float32,
      'forward_options': {'rtol': 1e-2},
      'backward_options': {'rtol': 1e-2, 'atol': 1e-3},
      'double_backward_options': {'rtol': 1e-2, 'atol': 1e-3}},
     {'dtype': numpy.float64,
      'forward_options': {'rtol': 1e-2},
      'backward_options': {'rtol': 1e-2, 'atol': 1e-3},
      'double_backward_options': {'rtol': 1e-2, 'atol': 1e-3}},
     ],
    testing.product({
        'batchsize': [5, 10],
        'input_dim': [2, 3],
        'margin': [1, 2],
        'reduce': ['mean', 'no'],
        'label_dtype': [numpy.int32, numpy.int64]
    })
))
class TestContrastive(unittest.TestCase):

    def setUp(self):
        x_shape = (self.batchsize, self.input_dim)
        retry = 0
        while True:
            self.x0 = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)
            self.x1 = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)
            d = numpy.sqrt(numpy.sum((self.x0 - self.x1) ** 2, axis=1))
            if (numpy.abs(d - self.margin) > 1e-2).all():
                break
            retry += 1
            assert retry <= 10, 'Too many retries to generate inputs'
        self.t = numpy.random.randint(
            0, 2, (self.batchsize,)).astype(self.label_dtype)
        if self.reduce == 'mean':
            self.gy = numpy.random.uniform(-1, 1, ()).astype(self.dtype)
        else:
            self.gy = numpy.random.uniform(
                -1, 1, (self.batchsize,)).astype(self.dtype)
        self.gx0 = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)
        self.gx1 = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)

    def check_forward(self, x0_data, x1_data, t_data):
        x0_val = chainer.Variable(x0_data)
        x1_val = chainer.Variable(x1_data)
        t_val = chainer.Variable(t_data)
        loss = functions.contrastive(
            x0_val, x1_val, t_val, self.margin, self.reduce)
        self.assertEqual(loss.data.dtype, self.dtype)
        if self.reduce == 'mean':
            self.assertEqual(loss.data.shape, ())
        else:
            self.assertEqual(loss.data.shape, (self.batchsize,))
        loss_value = cuda.to_cpu(loss.data)

        # Compute expected value
        loss_expect = numpy.empty((self.batchsize,), self.dtype)
        for i in six.moves.range(self.x0.shape[0]):
            x0d, x1d, td = self.x0[i], self.x1[i], self.t[i]
            d = numpy.sum((x0d - x1d) ** 2)
            if td == 1:  # similar pair
                loss_expect[i] = d
            elif td == 0:  # dissimilar pair
                loss_expect[i] = max(self.margin - math.sqrt(d), 0) ** 2
            loss_expect[i] /= 2.
        if self.reduce == 'mean':
            loss_expect = numpy.sum(loss_expect) / self.t.shape[0]
        numpy.testing.assert_allclose(
            loss_expect, loss_value, **self.forward_options)

    def test_negative_margin(self):
        self.margin = -1
        self.assertRaises(ValueError, self.check_forward,
                          self.x0, self.x1, self.t)
        self.assertRaises(ValueError, self.check_backward,
                          self.x0, self.x1, self.t, self.gy)

    def test_forward_cpu(self):
        self.check_forward(self.x0, self.x1, self.t)

    @attr.gpu
    def test_forward_gpu_no_cudnn(self):
        self.check_forward(cuda.to_gpu(self.x0), cuda.to_gpu(self.x1),
                           cuda.to_gpu(self.t))

    def check_backward(self, x0_data, x1_data, t_data, gy_data):
        def f(x0, x1, t):
            return functions.contrastive(x0, x1, t, self.margin, self.reduce)

        gradient_check.check_backward(
            f, (x0_data, x1_data, t_data), gy_data, dtype='d',
            **self.backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x0, self.x1, self.t, self.gy)

    @attr.gpu
    def test_backward_gpu_no_cudnn(self):
        self.check_backward(cuda.to_gpu(self.x0), cuda.to_gpu(self.x1),
                            cuda.to_gpu(self.t), cuda.to_gpu(self.gy))

    def test_backward_zero_dist_cpu(self):
        self.check_backward(self.x0, self.x0, self.t, self.gy)

    @attr.gpu
    def test_backward_zero_dist_gpu_no_cudnn(self):
        self.check_backward(cuda.to_gpu(self.x0), cuda.to_gpu(self.x0),
                            cuda.to_gpu(self.t), cuda.to_gpu(self.gy))

    def check_double_backward(
            self, x0_data, x1_data, t_data, gy_data, gx0_data, gx1_data):
        def f(x0, x1):
            return functions.contrastive(
                x0, x1, t_data, self.margin, self.reduce)

        gradient_check.check_double_backward(
            f, (x0_data, x1_data), gy_data,
            (gx0_data, gx1_data),
            dtype='f', **self.double_backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(
            self.x0, self.x1, self.t, self.gy, self.gx0, self.gx1)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.x0), cuda.to_gpu(self.x1),
            cuda.to_gpu(self.t), cuda.to_gpu(self.gy),
            cuda.to_gpu(self.gx0), cuda.to_gpu(self.gx1))


class TestContrastiveInvalidReductionOption(unittest.TestCase):

    def setUp(self):
        self.x0 = numpy.random.uniform(-1, 1, (5, 10)).astype(numpy.float32)
        self.x1 = numpy.random.uniform(-1, 1, (5, 10)).astype(numpy.float32)
        self.t = numpy.random.randint(0, 2, (5,)).astype(numpy.int32)

    def check_invalid_option(self, xp):
        x0 = xp.asarray(self.x0)
        x1 = xp.asarray(self.x1)
        t = xp.asarray(self.t)

        with self.assertRaises(ValueError):
            functions.contrastive(x0, x1, t, 1, 'invalid_option')

    def test_invalid_option_cpu(self):
        self.check_invalid_option(numpy)

    @attr.gpu
    def test_invalid_option_gpu(self):
        self.check_invalid_option(cuda.cupy)


testing.run_module(__name__, __file__)
