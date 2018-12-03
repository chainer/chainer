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
        optimizers.CorrectedMomentumSGD,
        optimizers.MomentumSGD,
        optimizers.MSVAG,
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
        optimizers.CorrectedMomentumSGD,
        optimizers.MomentumSGD,
        optimizers.MSVAG,
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


testing.run_module(__name__, __file__)
