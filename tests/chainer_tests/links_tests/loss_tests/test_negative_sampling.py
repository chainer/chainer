import unittest

import numpy

import chainer
from chainer import cuda
from chainer.functions.loss import negative_sampling
from chainer import gradient_check
from chainer import links
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


class TestNegativeSampling(unittest.TestCase):

    reduce = 'sum'

    def setUp(self):
        self.link = links.NegativeSampling(3, [10, 5, 2, 5, 2], 2)
        self.link.cleargrads()
        self.x = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)
        self.t = numpy.array([0, 2]).astype(numpy.int32)

        if self.reduce == 'none':
            g_shape = self.t.shape
        elif self.reduce == 'sum':
            g_shape = ()
        self.gy = numpy.random.uniform(-1, 1, g_shape).astype(numpy.float32)

    def check_forward(self, x_data, t_data):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)
        y = self.link(x, t)
        self.assertEqual(y.shape, self.gy.shape)

        W = cuda.to_cpu(self.link.W.data)
        samples = cuda.to_cpu(y.creator.samples)

        loss = numpy.empty((len(self.x),), numpy.float32)
        for i in range(len(self.x)):
            ix = self.x[i]
            it = self.t[i]
            if it == -1:
                loss[i] = 0
            else:
                w = W[samples[i]]
                f = w.dot(ix)
                # first one is positive example
                f[0] *= -1
                loss[i] = numpy.logaddexp(f, 0).sum()

        if self.reduce == 'sum':
            loss = loss.sum()

        testing.assert_allclose(y.data, loss)

    def test_forward_cpu(self):
        self.check_forward(self.x, self.t)

    @attr.gpu
    def test_forward_gpu(self):
        self.link.to_gpu()
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.t))

    def check_backward(self, x_data, t_data, w_data, sample, y_grad):
        t = chainer.Variable(t_data)
        ns = negative_sampling.NegativeSamplingFunction(
            sample, self.link.sample_size, self.reduce)

        def f(x, w):
            return ns(x, t, w)

        gradient_check.check_backward(
            f, (x_data, w_data), y_grad, eps=1e-2, atol=1e-4, rtol=1e-4)

    def test_backward_cpu(self):
        self.check_backward(
            self.x, self.t, self.link.W.data, self.link.sampler.sample, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.link.to_gpu()
        self.check_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.t),
            self.link.W.data, self.link.sampler.sample, cuda.to_gpu(self.gy))

    @attr.gpu
    def test_to_cpu(self):
        self.link.to_gpu()
        self.assertTrue(self.link.sampler.use_gpu)
        self.link.to_cpu()
        self.assertFalse(self.link.sampler.use_gpu)


class TestNegativeSamplingIgnoreMask(TestNegativeSampling):

    def setUp(self):
        # Create two identical datasets except that 2nd dataset has the
        # negative targets explicitly removed. Both cases should have identical
        # outcomes.
        self.link = links.NegativeSampling(3, [10, 5, 2, 5, 2], 2)
        self.link.cleargrads()
        self.x = numpy.random.uniform(-1, 1, (3, 3)).astype(numpy.float32)
        self.t = numpy.array([-1, 1, 2]).astype(numpy.int32)

        if self.reduce == 'none':
            g_shape = self.t.shape
        elif self.reduce == 'sum':
            g_shape = ()
        self.gy = numpy.random.uniform(-1, 1, g_shape).astype(numpy.float32)
        self.idx = self.t > -1
        self.x0 = self.x.copy()[self.idx]
        self.t0 = self.t.copy()[self.idx]
        if self.reduce == 'none':
            self.gy0 = self.gy.copy()[self.idx]
        else:
            self.gy0 = self.gy.copy()


testing.run_module(__name__, __file__)
