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


class TestGradientLARS(unittest.TestCase):

    def setUp(self):
        self.target = chainer.ChainList(
            SimpleLink(np.arange(6).astype(np.float32).reshape(2, 3),
                       np.arange(3, -3, -1).astype(np.float32).reshape(2, 3)),
            SimpleLink(np.arange(6).astype(np.float32).reshape(2, 3) * 0.0001,
                       np.arange(3, -3, -1).astype(np.float32).reshape(2, 3))
        )

    def check_LARS(self):
        w0 = self.target[0].param.data
        g0 = self.target[0].param.grad
        w1 = self.target[1].param.data
        g1 = self.target[1].param.grad
        xp = backend.get_array_module(w0)
        threshold = 1e-2
        weight_decay = 0.2
        eps = 1e-9

        p0_norm = xp.linalg.norm(w0)
        g0_norm = xp.linalg.norm(g0)
        clip_rate = p0_norm / (eps + g0_norm + weight_decay * p0_norm)
        expect0 = w0 - clip_rate * (g0 + weight_decay * w0)
        expect1 = w1 - 1.0 * (g1 + weight_decay * w1)

        opt = optimizers.SGD(lr=1)
        opt.setup(self.target)
        opt.add_hook(optimizer_hooks.GradientLARS(threshold=threshold,
                                                  weight_decay=weight_decay,
                                                  eps=eps))
        opt.update()

        testing.assert_allclose(expect0, w0)
        testing.assert_allclose(expect1, w1)

    def test_LARS_cpu(self):
        self.check_LARS()

    @attr.gpu
    def test_LARS_gpu(self):
        self.target.to_gpu()
        self.check_LARS()


testing.run_module(__name__, __file__)
