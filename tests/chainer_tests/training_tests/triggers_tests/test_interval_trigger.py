from __future__ import division

import numpy as np
import random
import tempfile
import unittest

from chainer import serializers
from chainer import testing
from chainer.testing import condition
from chainer import training


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
        trainer = testing.get_trainer_with_mock_updater(
            stop_trigger=None, iter_per_epoch=self.iter_per_epoch)
        trigger = training.triggers.IntervalTrigger(*self.interval)
        # before the first iteration, trigger should be False
        for expected in [False] + self.expected:
            self.assertEqual(trigger(trainer), expected)
            trainer.updater.update()

    def test_resumed_trigger(self):
        trainer = testing.get_trainer_with_mock_updater(
            stop_trigger=None, iter_per_epoch=self.iter_per_epoch)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            trigger = training.triggers.IntervalTrigger(*self.interval)
            for expected in self.expected[:self.resume]:
                trainer.updater.update()
                self.assertEqual(trigger(trainer), expected)
            serializers.save_npz(f.name, trigger)

            trigger = training.triggers.IntervalTrigger(*self.interval)
            serializers.load_npz(f.name, trigger)
            for expected in self.expected[self.resume:]:
                trainer.updater.update()
                self.assertEqual(trigger(trainer), expected)

    @condition.repeat(10)
    def test_trigger_sparse_call(self):
        trainer = testing.get_trainer_with_mock_updater(
            stop_trigger=None, iter_per_epoch=self.iter_per_epoch)
        trigger = training.triggers.IntervalTrigger(*self.interval)
        accumulated = False
        # before the first iteration, trigger should be False
        for expected in [False] + self.expected:
            accumulated = accumulated or expected
            if random.randrange(2):
                self.assertEqual(trigger(trainer), accumulated)
                accumulated = False
            trainer.updater.update()

    @condition.repeat(10)
    def test_resumed_trigger_sparse_call(self):
        trainer = testing.get_trainer_with_mock_updater(
            stop_trigger=None, iter_per_epoch=self.iter_per_epoch)
        accumulated = False
        with tempfile.NamedTemporaryFile(delete=False) as f:
            trigger = training.triggers.IntervalTrigger(*self.interval)
            for expected in self.expected[:self.resume]:
                trainer.updater.update()
                accumulated = accumulated or expected
                if random.randrange(2):
                    self.assertEqual(trigger(trainer), accumulated)
                    accumulated = False
            serializers.save_npz(f.name, trigger)

            trigger = training.triggers.IntervalTrigger(*self.interval)
            serializers.load_npz(f.name, trigger)
            for expected in self.expected[self.resume:]:
                trainer.updater.update()
                accumulated = accumulated or expected
                if random.randrange(2):
                    self.assertEqual(trigger(trainer), accumulated)
                    accumulated = False

    def test_resumed_trigger_backward_compat(self):
        trainer = testing.get_trainer_with_mock_updater(
            stop_trigger=None, iter_per_epoch=self.iter_per_epoch)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            trigger = training.triggers.IntervalTrigger(*self.interval)
            for expected in self.expected[:self.resume]:
                trainer.updater.update()
                self.assertEqual(trigger(trainer), expected)
            # old version does not save anything
            np.savez(f, dummy=0)

            trigger = training.triggers.IntervalTrigger(*self.interval)
            serializers.load_npz(f.name, trigger)
            for expected in self.expected[self.resume:]:
                trainer.updater.update()
                self.assertEqual(trigger(trainer), expected)


testing.run_module(__name__, __file__)
