import unittest

import numpy

import chainer
from chainer import functions as F


# TODO(niboshi): Write test: observe_value
# TODO(niboshi): Write test: observe_lr


class TestObserveOptimizer(unittest.TestCase):

    def test_observe_optimizer(self):

        # TODO(niboshi): Finish writing this test

        class Model(chainer.Link):
            def forward(self, x):
                return F.identity(x)

        dataset = [
            numpy.random.randn(2, 2).astype(numpy.float32)
            for _ in range(5)]
        model = Model()
        iterator = chainer.iterators.SerialIterator(
            dataset,
            batch_size=1)
        optimizer = chainer.optimizers.Adam(
            weight_decay_rate=0.3)
        optimizer.setup(model)
        updater = chainer.training.StandardUpdater(iterator, optimizer)
        trainer = chainer.training.Trainer(
            updater,
            stop_trigger=(3, 'epoch'))
        ext = chainer.training.extensions.observe_optimizer(
            'weight_decay_rate')
        trainer.extend(ext)

        trainer.run()

        # TODO(niboshi): Write check logic
        assert True
