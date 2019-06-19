import unittest

import numpy as np

import chainer
import chainer.functions as F
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


class TestWeightDecay(unittest.TestCase):

    def setUp(self):
        self.target = SimpleLink(
            np.arange(6, dtype=np.float32).reshape(2, 3),
            np.arange(3, -3, -1, dtype=np.float32).reshape(2, 3))

    def check_weight_decay(self):
        w = self.target.param.data
        g = self.target.param.grad

        decay = 0.2
        expect = w - g - decay * w

        opt = optimizers.SGD(lr=1)
        opt.setup(self.target)
        opt.add_hook(optimizer_hooks.WeightDecay(decay))
        opt.update()

        testing.assert_allclose(expect, w)

    def test_weight_decay_cpu(self):
        self.check_weight_decay()

    @attr.gpu
    def test_weight_decay_gpu(self):
        self.target.to_gpu()
        self.check_weight_decay()


@testing.inject_backend_tests(
    None,
    # CPU tests
    [{}]
    # GPU tests
    + testing.product({
        'use_cuda': [True],
    })
)
class TestWeightDecayLossScale(unittest.TestCase):

    def test_weight_decay_loss_scale(self, backend_config):
        a = self._updated_array(backend_config, None)
        b = self._updated_array(backend_config, loss_scale=4.)
        testing.assert_allclose(a, b)

    def _updated_array(self, backend_config, loss_scale):
        arr = np.arange(3, dtype=np.float32)
        param = chainer.Parameter(arr)
        link = chainer.Link()
        with link.init_scope():
            link.p = param
        link.to_device(backend_config.device)
        opt = optimizers.SGD(lr=1)
        opt.setup(link)
        opt.add_hook(optimizer_hooks.WeightDecay(1/8.))
        loss = F.sum(link.p ** 3)
        loss.backward(loss_scale=loss_scale)
        opt.update()
        return link.p.array


testing.run_module(__name__, __file__)
