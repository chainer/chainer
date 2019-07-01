from __future__ import division

import unittest

import numpy as np

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


@chainer.testing.parameterize(*chainer.testing.product({
    'use_cupy': [True, False],
}))
class TestMultiNodeEarlyStoppingTrigger(unittest.TestCase):

    def setUp(self):
        if self.use_cupy:
            comm_name = 'pure_nccl'
            import cupy
            self.xp = cupy
        else:
            comm_name = 'naive'
            self.xp = np
        self.communicator = chainermn.create_communicator(comm_name)

        if self.use_cupy:
            cupy.cuda.Device(self.communicator.intra_rank).use()

    def test_early_stopping_trigger_with_accuracy(self):
        comm = self.communicator
        key = 'main/accuracy'
        trigger = MultiNodeEarlyStoppingTrigger(comm, monitor=key, patience=3,
                                                check_trigger=(1, 'epoch'),
                                                verbose=False)
        trigger = util.get_trigger(trigger)

        accuracies = [0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.7,
                      0.7, 0.6, 0.6, 0.4, 0.4, 0.3, 0.3, 0.2, 0.2]
        accuracies = [x * (1 - comm.rank / comm.size) for x in accuracies]
        accuracies = [
            chainer.Variable(self.xp.asarray(acc, dtype=np.float32))
            for acc in accuracies]

        expected = [False, False, False, False, False, False,
                    False, False, False, False, False, False,
                    False, True, False, True]
        _test_trigger(self, trigger, key, accuracies, expected)
