import unittest

import numpy as np

import chainer
from chainer import backend
from chainer import cuda
import chainer.initializers as I
from chainer import optimizer_hooks
from chainer.optimizer_hooks import gradient_clipping
from chainer import optimizers
from chainer import testing
from chainer.testing import attr


class SimpleLink(chainer.Link):

    def __init__(self, w, g):
        super(SimpleLink, self).__init__()
        with self.init_scope():
            self.param = chainer.Parameter(I.Zero(), w.shape)
        self.param.data = w
        self.param.grad = g


class TestOptimizerUtility(unittest.TestCase):

    def setUp(self):
        self.x = np.linspace(-1.0, 1.5, num=6).astype(np.float32).reshape(2, 3)
        self.a = np.array(2.0)

    def test_sqnorm_cpu(self):
        # \Sum_{n=0}^{5} (-1.0+0.5n)**2 = 4.75
        self.assertAlmostEqual(gradient_clipping._sum_sqnorm([self.x]), 4.75)

    def test_sqnorm_scalar_cpu(self):
        self.assertAlmostEqual(gradient_clipping._sum_sqnorm([self.a]), 4)

    @attr.gpu
    def test_sqnorm_gpu(self):
        x = cuda.to_gpu(self.x)
        self.assertAlmostEqual(gradient_clipping._sum_sqnorm([x]), 4.75)

    @attr.gpu
    def test_sqnorm_scalar_gpu(self):
        a = cuda.to_gpu(self.a)
        self.assertAlmostEqual(gradient_clipping._sum_sqnorm([a]), 4)

    @attr.gpu
    def test_sqnorm_array(self):
        x = cuda.to_gpu(self.x)
        a = cuda.to_gpu(self.a)
        self.assertAlmostEqual(gradient_clipping._sum_sqnorm(
            [self.x, self.a, x, a]), 8.75 * 2)

    @attr.multi_gpu(2)
    def test_sqnorm_array_multi_gpu(self):
        x0 = cuda.to_gpu(self.x, device=0)
        x1 = cuda.to_gpu(self.x, device=1)
        a0 = cuda.to_gpu(self.a, device=0)
        a1 = cuda.to_gpu(self.a, device=1)
        self.assertAlmostEqual(gradient_clipping._sum_sqnorm(
            [self.x, self.a, x0, a0, x1, a1]), 8.75 * 3)


class TestGradientClipping(unittest.TestCase):

    def setUp(self):
        self.target = SimpleLink(
            np.arange(6, dtype=np.float32).reshape(2, 3),
            np.arange(3, -3, -1, dtype=np.float32).reshape(2, 3))

    def check_clipping(self, multiplier):
        w = self.target.param.data
        g = self.target.param.grad
        xp = backend.get_array_module(w)

        norm = xp.sqrt(gradient_clipping._sum_sqnorm(g))
        threshold = norm * multiplier
        if multiplier < 1:
            expect = w - g * multiplier
        else:
            expect = w - g

        opt = optimizers.SGD(lr=1)
        opt.setup(self.target)
        opt.add_hook(
            optimizer_hooks.GradientClipping(threshold))
        opt.update()

        testing.assert_allclose(expect, w)

    def test_clipping_cpu(self):
        self.check_clipping(0.5)

    @attr.gpu
    def test_clipping_gpu(self):
        self.target.to_gpu()
        self.check_clipping(0.5)

    def test_clipping_2_cpu(self):
        self.check_clipping(2.0)

    @attr.gpu
    def test_clipping_2_gpu(self):
        self.target.to_gpu()
        self.check_clipping(2.0)


testing.run_module(__name__, __file__)
