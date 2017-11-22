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
class TestGradientMethodHyperparameter(unittest.TestCase):

    def create_target(self):
        target = chainer.Link()
        with target.init_scope():
            target.w = chainer.Parameter()
        return target

    def get_hyperparam(self, target, name):
        return getattr(target.w.update_rule.hyperparam, name)

    def check_hyperparams(self, create):
        # Retrieve the default hyperparameters of the optimizer.
        target = self.create_target()
        optimizer = create(target)
        default = optimizer.hyperparam.get_dict()

        for name, default_value in six.iteritems(default):
            # Without explicit values, hyperparam of the target link must be
            # initialized with the default value.
            target = self.create_target()
            optimizer = create(target)
            assert self.get_hyperparam(target, name) == default_value

            # With explicit values, hyperparam of the target link must be
            # initialized with that value.
            target = self.create_target()
            new_value = default_value + 0.1
            optimizer = create(target, **{name: new_value})
            assert self.get_hyperparam(target, name) == new_value

    def test_hyperparams_setup_with_init(self):
        # Test hyperparameters, using an optimizer whose model is set up by
        # __init__ argument.
        def create(target, *args, **kwargs):
            optimizer = self.impl(*args, link=target, **kwargs)
            return optimizer
        self.check_hyperparams(create)

    def test_hyperparams_separate_setup(self):
        # Test hyperparameters, using an optimizer whose model is set up by
        # setup() method.
        def create(target, *args, **kwargs):
            optimizer = self.impl(*args, **kwargs)
            optimizer.setup(target)
            return optimizer
        self.check_hyperparams(create)

    def test_link_keyword_only_argument(self):
        # Link argument must be specified with keyword (link=).
        # This test assumes all the optimizers have the first argument as
        # a hyperparameter, thus the link argument is rejected.
        target = self.create_target()
        with self.assertRaises(TypeError):
            self.impl(target)


testing.run_module(__name__, __file__)
