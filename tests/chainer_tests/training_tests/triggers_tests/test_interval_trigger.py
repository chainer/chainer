from __future__ import division

import mock
import random
import tempfile
import unittest

from chainer import serializers
from chainer import testing
from chainer import training


def get_trainer_with_mock_updater(iter_per_epoch):
    updater = mock.Mock()
    updater.get_all_optimizers.return_value = {}
    updater.iteration = 0
    updater.epoch = 0
    updater.epoch_detail = 0
    updater.is_new_epoch = True

    def update():
        updater.iteration += 1
        updater.epoch = updater.iteration // iter_per_epoch
        updater.epoch_detail = updater.iteration / iter_per_epoch
        updater.is_new_epoch = updater.epoch == updater.epoch_detail

    updater.update = update
    trainer = training.Trainer(updater)
    return trainer


@testing.parameterize(
    # iteration
    {
        'iter_per_epoch': 5, 'interval': (2, 'iteration'), 'resume': 4,
        'expected': [False, True, False, True, False, True, False]},
    # basic epoch
    {
        'iter_per_epoch': 1, 'interval': (3, 'epoch'), 'resume': 4,
        'expected': [False, False, True, False, False, True, False]},
    # fractional epoch
    {
        'iter_per_epoch': 2, 'interval': (1.5, 'epoch'), 'resume': 4,
        'expected': [False, False, True, False, False, True, False]},
    # unaligned epoch
    {
        'iter_per_epoch': 2.5, 'interval': (1, 'epoch'), 'resume': 3,
        'expected': [False, False, True, False, True, False, False]},
    # tiny epoch
    {
        'iter_per_epoch': 0.5, 'interval': (1, 'epoch'), 'resume': 4,
        'expected': [True, True, True, True, True, True, True]},
)
class TestIntervalTrigger(unittest.TestCase):

    def test_trigger(self):
        trainer = get_trainer_with_mock_updater(self.iter_per_epoch)
        trigger = training.trigger.IntervalTrigger(*self.interval)
        # before the first iteration, trigger should be False
        for expected in [False] + self.expected:
            self.assertEqual(trigger(trainer), expected)
            trainer.updater.update()

    def test_resumed_trigger(self):
        trainer = get_trainer_with_mock_updater(self.iter_per_epoch)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            trigger = training.trigger.IntervalTrigger(*self.interval)
            for expected in self.expected[:self.resume]:
                trainer.updater.update()
                self.assertEqual(trigger(trainer), expected)
            serializers.save_npz(f.name, trigger)

            trigger = training.trigger.IntervalTrigger(*self.interval)
            serializers.load_npz(f.name, trigger)
            for expected in self.expected[self.resume:]:
                trainer.updater.update()
                self.assertEqual(trigger(trainer), expected)

    @testing.condition.repeat(10)
    def test_trigger_sparse_call(self):
        trainer = get_trainer_with_mock_updater(self.iter_per_epoch)
        trigger = training.trigger.IntervalTrigger(*self.interval)
        accumulated = False
        # before the first iteration, trigger should be False
        for expected in [False] + self.expected:
            accumulated = accumulated or expected
            if random.randrange(2):
                self.assertEqual(trigger(trainer), accumulated)
                accumulated = False
            trainer.updater.update()

    @testing.condition.repeat(10)
    def test_resumed_trigger_sparse_call(self):
        trainer = get_trainer_with_mock_updater(self.iter_per_epoch)
        accumulated = False
        with tempfile.NamedTemporaryFile(delete=False) as f:
            trigger = training.trigger.IntervalTrigger(*self.interval)
            for expected in self.expected[:self.resume]:
                trainer.updater.update()
                accumulated = accumulated or expected
                if random.randrange(2):
                    self.assertEqual(trigger(trainer), accumulated)
                    accumulated = False
            serializers.save_npz(f.name, trigger)

            trigger = training.trigger.IntervalTrigger(*self.interval)
            serializers.load_npz(f.name, trigger)
            for expected in self.expected[self.resume:]:
                trainer.updater.update()
                accumulated = accumulated or expected
                if random.randrange(2):
                    self.assertEqual(trigger(trainer), accumulated)
                    accumulated = False


testing.run_module(__name__, __file__)
