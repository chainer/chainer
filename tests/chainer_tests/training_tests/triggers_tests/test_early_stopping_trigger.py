import unittest

import chainer
import numpy

from chainer import testing
from chainer.training import triggers
from chainer.training import util


def _test_trigger(self, trigger, key, accuracies, expected):
    trainer = testing.training.get_trainer_with_mock_updater(
        stop_trigger=None, iter_per_epoch=1)

    for accuracy, expected in zip(accuracies, expected):
        trainer.updater.update()
        trainer.observation = {key: accuracy}
        self.assertEqual(trigger(trainer), expected)


class TestEarlyStoppingTrigger(unittest.TestCase):

    def test_early_stopping_trigger_with_accuracy(self):
        key = 'main/accuracy'
        trigger = triggers.EarlyStoppingTrigger(monitor=key, patients=3,
                                                check_trigger=(1, 'epoch'),
                                                verbose=False)
        trigger = util.get_trigger(trigger)

        accuracies = [0.5, 0.5, 0.6, 0.7, 0.6, 0.4, 0.3, 0.2]
        accuracies = numpy.asarray([
            chainer.Variable(numpy.asarray(acc, dtype=numpy.float32))
            for acc in accuracies])

        expected = [False, False, False, False, False, False, True, True]
        _test_trigger(self, trigger, key, accuracies, expected)

    def test_early_stopping_trigger_with_loss(self):
        key = 'main/loss'
        trigger = triggers.EarlyStoppingTrigger(monitor=key, patients=3,
                                                check_trigger=(1, 'epoch'))
        trigger = util.get_trigger(trigger)

        accuracies = [100, 80, 30, 10, 20, 24, 30, 35]
        accuracies = numpy.asarray([
            chainer.Variable(numpy.asarray(acc, dtype=numpy.float32))
            for acc in accuracies])

        expected = [False, False, False, False, False, False, True, True]
        _test_trigger(self, trigger, key, accuracies, expected)

    def test_early_stopping_trigger_with_max_epoch(self):
        key = 'main/loss'
        trigger = triggers.EarlyStoppingTrigger(monitor=key, patients=3,
                                                check_trigger=(1, 'epoch'),
                                                max_trigger=(3, 'epoch'))
        trigger = util.get_trigger(trigger)

        accuracies = [100, 80, 30]
        accuracies = numpy.asarray([
            chainer.Variable(numpy.asarray(acc, dtype=numpy.float32))
            for acc in accuracies])

        expected = [False, False, True]
        _test_trigger(self, trigger, key, accuracies, expected)

    def test_early_stopping_trigger_with_max_iteration(self):
        key = 'main/loss'
        trigger = triggers.EarlyStoppingTrigger(monitor=key, patients=3,
                                                check_trigger=(1, 'epoch'),
                                                max_trigger=(3, 'iteration'))
        trigger = util.get_trigger(trigger)

        accuracies = [100, 80, 30]
        accuracies = numpy.asarray([
            chainer.Variable(numpy.asarray(acc, dtype=numpy.float32))
            for acc in accuracies])

        expected = [False, False, True]
        _test_trigger(self, trigger, key, accuracies, expected)


testing.run_module(__name__, __file__)
