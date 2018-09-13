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


class TestLasso(unittest.TestCase):
    def setUp(self):
        self.target = SimpleLink(
            np.arange(6, dtype=np.float32).reshape(2, 3),
            np.arange(3, -3, -1, dtype=np.float32).reshape(2, 3))

    def check_lasso(self):
        w = self.target.param.data
        g = self.target.param.grad
        xp = backend.get_array_module(w)
        decay = 0.2
        expect = w - g - decay * xp.sign(w)

        opt = optimizers.SGD(lr=1)
        opt.setup(self.target)
        opt.add_hook(optimizer_hooks.Lasso(decay))
        opt.update()

        testing.assert_allclose(expect, w)

    def test_lasso_cpu(self):
        self.check_lasso()

    @attr.gpu
    def test_lasso_gpu(self):
        self.target.to_gpu()
        self.check_lasso()


testing.run_module(__name__, __file__)
