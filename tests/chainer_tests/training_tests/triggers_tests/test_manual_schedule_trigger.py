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
    # single iteration
    {
        'iter_per_epoch': 2, 'schedule': (2, 'iteration'), 'resume': 3,
        'expected': [False, True, False, False, False, False, False]},
    # multiple iteration
    {
        'iter_per_epoch': 2, 'schedule': ([2, 4], 'iteration'), 'resume': 3,
        'expected': [False, True, False, True, False, False, False]},
    # single epoch
    {
        'iter_per_epoch': 3, 'schedule': (1, 'epoch'), 'resume': 3,
        'expected': [False, False, True, False, False, False, False]},
    # multiple epoch
    {
        'iter_per_epoch': 3, 'schedule': ([1, 2], 'epoch'), 'resume': 4,
        'expected': [False, False, True, False, False, True, False]},
    # single fractional epoch
    {
        'iter_per_epoch': 2, 'schedule': (1.5, 'epoch'), 'resume': 4,
        'expected': [False, False, True, False, False, False, False]},
    # multiple fractional epoch
    {
        'iter_per_epoch': 2, 'schedule': ([1.5, 2.5], 'epoch'), 'resume': 4,
        'expected': [False, False, True, False, True, False, False]},
    # single unaligned epoch
    {
        'iter_per_epoch': 2.5, 'schedule': (1, 'epoch'), 'resume': 4,
        'expected': [False, False, True, False, False, False, False]},
    # multiple unaligned epoch
    {
        'iter_per_epoch': 2.5, 'schedule': ([1, 2], 'epoch'), 'resume': 4,
        'expected': [False, False, True, False, True, False, False]},
    # single tiny epoch
    {
        'iter_per_epoch': 0.5, 'schedule': (1, 'epoch'), 'resume': 4,
        'expected': [True, False, False, False, False, False, False]},
    # multiple tiny epoch
    {
        'iter_per_epoch': 0.5, 'schedule': ([1, 2], 'epoch'), 'resume': 4,
        'expected': [True, False, False, False, False, False, False]},
)
class TestTrigger(unittest.TestCase):

    def test_trigger(self):
        trainer = testing.get_trainer_with_mock_updater(
            stop_trigger=None, iter_per_epoch=self.iter_per_epoch)
        trigger = training.triggers.ManualScheduleTrigger(*self.schedule)
        for expected in self.expected:
            trainer.updater.update()
            self.assertEqual(trigger(trainer), expected)

    def test_resumed_trigger(self):
        trainer = testing.get_trainer_with_mock_updater(
            stop_trigger=None, iter_per_epoch=self.iter_per_epoch)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            trigger = training.triggers.ManualScheduleTrigger(*self.schedule)
            for expected in self.expected[:self.resume]:
                trainer.updater.update()
                self.assertEqual(trigger(trainer), expected)
            serializers.save_npz(f.name, trigger)

            trigger = training.triggers.ManualScheduleTrigger(*self.schedule)
            serializers.load_npz(f.name, trigger)
            for expected in self.expected[self.resume:]:
                trainer.updater.update()
                self.assertEqual(trigger(trainer), expected)

    @condition.repeat(10)
    def test_trigger_sparse_call(self):
        trainer = testing.get_trainer_with_mock_updater(
            stop_trigger=None, iter_per_epoch=self.iter_per_epoch)
        trigger = training.triggers.ManualScheduleTrigger(*self.schedule)
        accumulated = False
        for expected in self.expected:
            trainer.updater.update()
            accumulated = accumulated or expected
            if random.randrange(2):
                self.assertEqual(trigger(trainer), accumulated)
                accumulated = False

    @condition.repeat(10)
    def test_resumed_trigger_sparse_call(self):
        trainer = testing.get_trainer_with_mock_updater(
            stop_trigger=None, iter_per_epoch=self.iter_per_epoch)
        accumulated = False
        with tempfile.NamedTemporaryFile(delete=False) as f:
            trigger = training.triggers.ManualScheduleTrigger(*self.schedule)
            for expected in self.expected[:self.resume]:
                trainer.updater.update()
                accumulated = accumulated or expected
                if random.randrange(2):
                    self.assertEqual(trigger(trainer), accumulated)
                    accumulated = False
            serializers.save_npz(f.name, trigger)

            trigger = training.triggers.ManualScheduleTrigger(*self.schedule)
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
            trigger = training.triggers.ManualScheduleTrigger(*self.schedule)
            for expected in self.expected[:self.resume]:
                trainer.updater.update()
                self.assertEqual(trigger(trainer), expected)
            # old version does not save anything
            np.savez(f, dummy=0)

            trigger = training.triggers.ManualScheduleTrigger(*self.schedule)
            serializers.load_npz(f.name, trigger)
            for expected in self.expected[self.resume:]:
                trainer.updater.update()
                self.assertEqual(trigger(trainer), expected)


testing.run_module(__name__, __file__)
