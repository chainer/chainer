import unittest

import chainer
import numpy

from chainer import testing
from chainer import training
from chainer.training import triggers
from chainer.training import util


class DummyUpdater(training.Updater):

    def __init__(self):
        self.iteration = 0
        self.epoch = 0

    def finalize(self):
        pass

    def get_all_optimizers(self):
        return {}

    def update(self):
        self.iteration += 1
        self.epoch += 1

    def epoch(self):
        return self.epoch

    @property
    def is_new_epoch(self):
        return False

    @property
    def epoch_detail(self):
        return self.epoch


def _test_trigger(self, trigger, key, accuracies, expected):
    updater = DummyUpdater()
    trainer = training.Trainer(updater)
    for accuracy, expected in zip(accuracies, expected):
        updater.update()
        trainer.observation = {key: accuracy}
        self.assertEqual(trigger(trainer), expected)


class TestEarlyStoppingTrigger(unittest.TestCase):

    def test_early_stopping_trigger_with_accuracy(self):
        key = 'main/accuracy'
        trigger = triggers.EarlyStoppingTrigger(monitor=key, patients=3,
                                                trigger=(1, 'iteration'),
                                                verbose=False)
        trigger = util.get_trigger(trigger)

        accuracies = [0.5, 0.5, 0.6, 0.7, 0.6, 0.4, 0.3, 0.2]
        accuracies = numpy.asarray([
            chainer.Variable(numpy.asarray(acc, dtype=numpy.float32))
            for acc in accuracies])

        expected = [False, False, False, False, False, False, False, True]
        _test_trigger(self, trigger, key, accuracies, expected)

    def test_early_stopping_trigger_with_loss(self):
        key = 'main/loss'
        trigger = triggers.EarlyStoppingTrigger(monitor=key, patients=3,
                                                trigger=(1, 'iteration'),
                                                verbose=True)
        trigger = util.get_trigger(trigger)

        accuracies = [100, 80, 30, 10, 20, 24, 30, 35]
        accuracies = numpy.asarray([
            chainer.Variable(numpy.asarray(acc, dtype=numpy.float32))
            for acc in accuracies])

        expected = [False, False, False, False, False, False, False, True]
        _test_trigger(self, trigger, key, accuracies, expected)

    def test_early_stopping_trigger_with_max_epoch(self):
        key = 'main/loss'
        trigger = triggers.EarlyStoppingTrigger(monitor=key, patients=3,
                                                trigger=(1, 'epoch'),
                                                verbose=True, max_epoch=3)
        trigger = util.get_trigger(trigger)

        accuracies = [100, 80, 30, 10]
        accuracies = numpy.asarray([
            chainer.Variable(numpy.asarray(acc, dtype=numpy.float32))
            for acc in accuracies])

        expected = [False, False, True, True]
        _test_trigger(self, trigger, key, accuracies, expected)


testing.run_module(__name__, __file__)
