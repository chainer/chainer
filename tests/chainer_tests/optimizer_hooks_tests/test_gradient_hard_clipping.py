import unittest

import numpy as np

import chainer
from chainer import backend
import chainer.initializers as I
from chainer import optimizer_hooks
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


class TestGradientHardClipping(unittest.TestCase):

    def setUp(self):
        self.target = SimpleLink(
            np.arange(6, dtype=np.float32).reshape(2, 3),
            np.arange(3, -3, -1, dtype=np.float32).reshape(2, 3))

    def check_hardclipping(self):
        w = self.target.param.data
        g = self.target.param.grad
        xp = backend.get_array_module(w)
        lower_bound = -0.9
        upper_bound = 1.1
        expect = w - xp.clip(g, lower_bound, upper_bound)

        opt = optimizers.SGD(lr=1)
        opt.setup(self.target)
        opt.add_hook(
            optimizer_hooks.GradientHardClipping(lower_bound, upper_bound))
        opt.update()

        testing.assert_allclose(expect, w)

    def test_hardclipping_cpu(self):
        self.check_hardclipping()

    @attr.gpu
    def test_hardclipping_gpu(self):
        self.target.to_gpu()
        self.check_hardclipping()


testing.run_module(__name__, __file__)
