import unittest

import numpy
import six.moves.cPickle as pickle

import chainer
from chainer.backends import cuda
from chainer import links
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


@testing.parameterize(*testing.product({
    'x_dtype': [numpy.float16, numpy.float32, numpy.float64],
    'W_dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestSpectralNormalizationConvolution2D(unittest.TestCase):

    def setUp(self):
        self.initial_u = numpy.random.normal(size=(1, 2)).astype(
            dtype=self.W_dtype)
        self.link = links.SpectralNormalizationConvolution2D(
            3, 2, 3, stride=2, pad=1,
            initialW=chainer.initializers.Normal(1, self.W_dtype),
            initial_bias=chainer.initializers.Normal(1, self.x_dtype))
        self.link.cleargrads()

        self.x = numpy.random.uniform(-1, 1,
                                      (2, 3, 4, 3)).astype(self.x_dtype)
        self.gy = numpy.random.uniform(-1, 1,
                                       (2, 2, 2, 2)).astype(self.x_dtype)
        self.check_backward_options = {}
        if (self.x_dtype in (numpy.float16, numpy.float32) or
                self.W_dtype in (numpy.float16, numpy.float32)):
            self.check_backward_options = {'atol': 3e-2, 'rtol': 5e-2}

    def reset_u(self):
        xp = cuda.get_array_module(self.link.u)
        if xp is numpy:
            self.link.u = numpy.copy(self.initial_u)
        else:
            self.link.u = xp.copy(cuda.to_gpu(self.initial_u))

    def check_forward_consistency(self):
        self.reset_u()
        x_cpu = chainer.Variable(self.x)
        y_cpu = self.link(x_cpu)
        self.assertEqual(y_cpu.data.dtype, self.x_dtype)

        self.reset_u()
        self.link.to_gpu()
        x_gpu = chainer.Variable(cuda.to_gpu(self.x))
        y_gpu = self.link(x_gpu)
        self.assertEqual(y_gpu.data.dtype, self.x_dtype)

        testing.assert_allclose(y_cpu.data, y_gpu.data.get())

    @attr.gpu
    @condition.retry(3)
    def test_forward_consistency(self):
        self.check_forward_consistency()

    @attr.gpu
    @condition.retry(3)
    def test_forward_consistency_im2col(self):
        with chainer.using_config('use_cudnn', 'never'):
            self.check_forward_consistency()

    def check_pickling(self, x_data):
        self.reset_u()
        x = chainer.Variable(x_data)
        y = self.link(x)
        y_data1 = y.data

        del x, y

        pickled = pickle.dumps(self.link, -1)
        del self.link
        self.link = pickle.loads(pickled)

        self.reset_u()
        x = chainer.Variable(x_data)
        y = self.link(x)
        y_data2 = y.data

        testing.assert_allclose(y_data1, y_data2, atol=0, rtol=0)

    def test_pickling_cpu(self):
        self.check_pickling(self.x)

    @attr.gpu
    def test_pickling_gpu(self):
        self.link.to_gpu()
        self.check_pickling(cuda.to_gpu(self.x))


testing.run_module(__name__, __file__)
