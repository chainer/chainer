from __future__ import division

import unittest

import numpy
import pytest

import chainer
from chainer import testing
import chainermn
from chainermn.extensions import MultiNodeEarlyStoppingTrigger
from chainer.training import util


def _test_trigger(self, trigger, key, accuracies, expected):
    trainer = testing.training.get_trainer_with_mock_updater(
        stop_trigger=None, iter_per_epoch=2)

    for accuracy, expected in zip(accuracies, expected):
        trainer.updater.update()
        trainer.observation = {key: accuracy}
        self.assertEqual(trigger(trainer), expected)


class TestMultiNodeEarlyStoppingTrigger(unittest.TestCase):

    def setUp(self):
        self.communicator = chainermn.create_communicator('naive')

    def test_early_stopping_trigger_with_accuracy(self):
        comm = self.communicator
        key = 'main/accuracy'
        trigger = MultiNodeEarlyStoppingTrigger(monitor=key, patience=3,
                                                check_trigger=(1, 'epoch'),
                                                verbose=False)
        trigger = util.get_trigger(trigger)

        accuracies = [0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.7, 0.7, 0.6, 0.6, 0.4, 0.4, 0.3, 0.3, 0.2, 0.2]
        accuracies = [x * (1 - comm.rank / comm.size) for x in accuracies]
        accuracies = numpy.asarray([
            chainer.Variable(numpy.asarray(acc, dtype=numpy.float32))
            for acc in accuracies])

        expected = [False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True]
        _test_trigger(self, trigger, key, accuracies, expected)

