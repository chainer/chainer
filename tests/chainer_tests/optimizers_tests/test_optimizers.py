import unittest

import six

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


testing.run_module(__name__, __file__)
