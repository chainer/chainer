from __future__ import division

import random
import tempfile
import unittest

from chainer import serializers
from chainer import testing
from chainer import training


class DummyUpdater(training.Updater):

    def __init__(self, iters_per_epoch):
        self.iteration = 0
        self.iters_per_epoch = iters_per_epoch

    def finalize(self):
        pass

    def get_all_optimizers(self):
        return {}

    def update(self):
        self.iteration += 1

    @property
    def epoch(self):
        return self.iteration // self.iters_per_epoch

    @property
    def epoch_detail(self):
        return self.iteration / self.iters_per_epoch

    @property
    def is_new_epoch(self):
        return 0 <= self.iteration % self.iters_per_epoch < 1


@testing.parameterize(
    # iteration
    {
        'iters_per_epoch': 5, 'interval': (2, 'iteration'), 'resume': 4,
        'expected': [False, True, False, True, False, True, False]},
    # basic epoch
    {
        'iters_per_epoch': 1, 'interval': (3, 'epoch'), 'resume': 4,
        'expected': [False, False, True, False, False, True, False]},
    # fractional epoch
    {
        'iters_per_epoch': 2, 'interval': (1.5, 'epoch'), 'resume': 4,
        'expected': [False, False, True, False, False, True, False]},
    # unaligned epoch
    {
        'iters_per_epoch': 2.5, 'interval': (1, 'epoch'), 'resume': 3,
        'expected': [False, False, True, False, True, False, False]},
    # tiny epoch
    {
        'iters_per_epoch': 0.5, 'interval': (1, 'epoch'), 'resume': 4,
        'expected': [True, True, True, True, True, True, True]},
)
class TestIntervalTrigger(unittest.TestCase):

    def test_trigger(self):
        trigger = training.trigger.IntervalTrigger(*self.interval)
        updater = DummyUpdater(self.iters_per_epoch)
        trainer = training.Trainer(updater)
        # before the first iteration, trigger should be False
        for expected in [False] + self.expected:
            self.assertEqual(trigger(trainer), expected)
            updater.update()

    def test_resumed_trigger(self):
        updater = DummyUpdater(self.iters_per_epoch)
        trainer = training.Trainer(updater)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            trigger = training.trigger.IntervalTrigger(*self.interval)
            for expected in self.expected[:self.resume]:
                updater.update()
                self.assertEqual(trigger(trainer), expected)
            serializers.save_npz(f.name, trigger)

            trigger = training.trigger.IntervalTrigger(*self.interval)
            serializers.load_npz(f.name, trigger)
            for expected in self.expected[self.resume:]:
                updater.update()
                self.assertEqual(trigger(trainer), expected)

    @testing.condition.repeat(10)
    def test_trigger_sparse_call(self):
        trigger = training.trigger.IntervalTrigger(*self.interval)
        updater = DummyUpdater(self.iters_per_epoch)
        trainer = training.Trainer(updater)
        accumulated = False
        # before the first iteration, trigger should be False
        for expected in [False] + self.expected:
            accumulated = accumulated or expected
            if random.randrange(2):
                self.assertEqual(trigger(trainer), accumulated)
                accumulated = False
            updater.update()

    @testing.condition.repeat(10)
    def test_resumed_trigger_sparse_call(self):
        updater = DummyUpdater(self.iters_per_epoch)
        trainer = training.Trainer(updater)
        accumulated = False
        with tempfile.NamedTemporaryFile(delete=False) as f:
            trigger = training.trigger.IntervalTrigger(*self.interval)
            for expected in self.expected[:self.resume]:
                updater.update()
                accumulated = accumulated or expected
                if random.randrange(2):
                    self.assertEqual(trigger(trainer), accumulated)
                    accumulated = False
            serializers.save_npz(f.name, trigger)

            trigger = training.trigger.IntervalTrigger(*self.interval)
            serializers.load_npz(f.name, trigger)
            for expected in self.expected[self.resume:]:
                updater.update()
                accumulated = accumulated or expected
                if random.randrange(2):
                    self.assertEqual(trigger(trainer), accumulated)
                    accumulated = False


testing.run_module(__name__, __file__)
