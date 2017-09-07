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

    def finalize(self):
        pass

    def get_all_optimizers(self):
        return {}

    def update(self):
        self.iteration += 1

    @property
    def epoch(self):
        return 1

    @property
    def is_new_epoch(self):
        return False


def _test_trigger(self, trigger, key, accuracies, expected):
    updater = DummyUpdater()
    trainer = training.Trainer(updater)
    for accuracy, expected in zip(accuracies, expected):
        updater.update()
        trainer.observation = {key: accuracy}
        self.assertEqual(trigger(trainer), expected)


class TestEarlyStoppingTrigger(unittest.TestCase):

    def test_early_stopping_trigger(self):
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


testing.run_module(__name__, __file__)
