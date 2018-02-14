import unittest

import six

import numpy as np

import chainer
from chainer import optimizers
from chainer import testing


@testing.parameterize(*testing.product({
    'impl': [
        optimizers.AdaDelta,
        optimizers.AdaGrad,
        optimizers.Adam,
        optimizers.MomentumSGD,
        optimizers.NesterovAG,
        optimizers.RMSprop,
        optimizers.RMSpropGraves,
        optimizers.SGD,
        optimizers.SMORMS3,
    ]
}))
class TestOptimizerHyperparameter(unittest.TestCase):

    def setUp(self):
        self.target = chainer.Link()
        with self.target.init_scope():
            self.target.w = chainer.Parameter()

    def create(self, *args, **kwargs):
        self.optimizer = self.impl(*args, **kwargs)
        self.optimizer.setup(self.target)

    def get_hyperparam(self, name):
        return getattr(self.target.w.update_rule.hyperparam, name)

    def test_hyperparams(self):
        self.create()
        default = self.optimizer.hyperparam.get_dict()
        for name, default_value in six.iteritems(default):
            self.create()
            self.assertEqual(self.get_hyperparam(name), default_value)
            new_value = default_value + 0.1
            self.create(**{name: new_value})
            self.assertEqual(self.get_hyperparam(name), new_value)


class WeightSaveHook(object):
    name = 'WeightSaveHook'
    call_for_each_param = True

    def __init__(self):
        self.value = None

    def __call__(self, rule, param):
        p, g = param.data, param.grad
        if p is None or g is None:
            return
        self.value = np.copy(p)


class SimpleChain(chainer.Chain):

    def __init__(self):
        super(SimpleChain, self).__init__()
        with self.init_scope():
            self.w = chainer.Parameter(42, (), 'w')

    def __call__(self, x):
        return (x - self.w) ** 2


@testing.parameterize(*testing.product({
    'impl': [
        optimizers.AdaDelta,
        optimizers.AdaGrad,
        optimizers.Adam,
        optimizers.MomentumSGD,
        optimizers.NesterovAG,
        optimizers.RMSprop,
        optimizers.RMSpropGraves,
        optimizers.SGD,
        optimizers.SMORMS3,
    ]
}))
class TestOptimizerHooks(unittest.TestCase):

    def setUp(self):
        self.target = SimpleChain()

    def create(self, *args, **kwargs):
        self.optimizer = self.impl(*args, **kwargs)
        self.optimizer.setup(self.target)

    def get_hyperparam(self, name):
        return getattr(self.target.w.update_rule.hyperparam, name)

    def test_hooks(self):
        w_pre = np.copy(self.target.w.data)
        h_pre = WeightSaveHook()
        h_post = WeightSaveHook()
        self.create()
        self.optimizer.add_hook(h_pre, timing='pre')
        self.optimizer.add_hook(h_post, name='WeightSaveHookPost',
                                timing='post')

        x = chainer.Variable(np.array(5., dtype=np.float32))
        self.optimizer.update(self.target, x)
        w_post = np.copy(self.target.w.data)

        self.assertEqual(w_pre, h_pre.value)
        self.assertEqual(w_post, h_post.value)
        self.assertNotEqual(h_pre.value, h_post.value)

    def test_hooks_auto(self):
        w_pre = np.copy(self.target.w.data)
        h_pre = WeightSaveHook()
        h_pre.timing = 'pre'
        h_post = WeightSaveHook()
        h_post.timing = 'post'
        self.create()
        self.optimizer.add_hook(h_pre, timing='auto')
        self.optimizer.add_hook(h_post, name='WeightSaveHookPost',
                                timing='auto')

        x = chainer.Variable(np.array(5., dtype=np.float32))
        self.optimizer.update(self.target, x)
        w_post = np.copy(self.target.w.data)

        self.assertEqual(w_pre, h_pre.value)
        self.assertEqual(w_post, h_post.value)
        self.assertNotEqual(h_pre.value, h_post.value)


class TestGradientLARS(unittest.TestCase):

    def setUp(self):
        self.target = chainer.ChainList(
            SimpleLink(np.arange(3).astype(np.float32),
                       np.arange(3).astype(np.float32)),
            SimpleLink(np.arange(3).astype(np.float32) * 0.0001,
                       np.arange(3).astype(np.float32) * 0.0001))

    def check_LARS(self):
        w0 = self.target[0].param.data
        g0 = self.target[0].param.grad
        w1 = self.target[1].param.data
        g1 = self.target[1].param.grad
        xp = cuda.get_array_module(w0)
        threshold = 1e-2
        weight_decay = 0.2
        eps = 1e-9

        p0_norm = xp.linalg.norm(w0)
        g0_norm = xp.linalg.norm(g0)
        local_rate = p0_norm / (eps + g0_norm + weight_decay * p0_norm)
        expect0 = w0 - local_rate * (g0 + weight_decay * w0)
        expect1 = w1 - 1.0 * (g1 + weight_decay * w1)

        opt = optimizers.SGD(lr=1)
        opt.setup(self.target)
        opt.add_hook(optimizer.GradientLARS(threshold=threshold,
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

    def test_call_hooks_uninitialized_param(self):
        target = UninitializedChain()
        opt = optimizers.MomentumSGD()
        opt.setup(target)
        opt.add_hook(optimizer.GradientLARS(threshold=1))
        target(np.ones((4, 10), dtype=np.float32))
        opt.call_hooks()

        # This test is for asserting that calling the hook on a chain
        # with uninitialized parameters does not crash, so if we reach
        # here, the test has passed.
