import unittest
import chainer
from chainer import testing
import chainer.link
import chainer.links
import chainer.functions
import chainer.optimizers


class TestDataParallel(unittest.TestCase):

    def setUp(self):

        # creating a really simple model to test dataparallel behavior
        class SimpleModel(chainer.link.Chain):
            def __init__(self):
                super(SimpleModel, self).__init__()

                with self.init_scope():
                    self.dense_1 = chainer.links.Linear(3, 32)
                    self.dense_2 = chainer.links.Linear(32, 2)

            def forward(self, x):
                return self.dense_2(chainer.functions.relu(self.dense_1(x)))

        self.model = chainer.links.DataParallel(SimpleModel(),
                                                devices=["@numpy", "@numpy"])

        self.optimizer = \
            chainer.optimizers.DataParallelOptimizer.from_optimizer_class(
                chainer.optimizers.Adam
            )
        self.optimizer.setup(self.model)

    def test_update(self):
        import numpy as np

        input_tensor = np.random.rand(10, 3).astype(np.float32)
        label_tensor = np.random.rand(10, 2).astype(np.float)

        model_copy = self.model.copy()

        preds = self.model(input_tensor)

        loss = chainer.functions.sum(preds-label_tensor)

        self.model.cleargrads()
        loss.backward()
        self.optimizer.update()

        # check if param was updated
        for orig_param, updated_param in zip(model_copy.params(),
                                             self.model.params()):

            self.assertFalse(np.array_equal(orig_param.array,
                                            updated_param.array))

        # check if all grads were cleared
        self.model.cleargrads()
        for module in self.model.modules:
            for updated_param in module.params():
                self.assertIsNone(updated_param.grad_var)


testing.run_module(__name__, __file__)
