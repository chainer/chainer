from __future__ import division

import os
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
    def previous_epoch_detail(self):
        if self.iteration == 0:
            return None
        return (self.iteration - 1) / self.iters_per_epoch

    @property
    def is_new_epoch(self):
        return 0 <= self.iteration % self.iters_per_epoch < 1

    def serialize(self, serializer):
        self.iteration = serializer('iteration', self.iteration)


@testing.parameterize(*testing.product_dict([
    # iteration
    {
        'iters_per_epoch': 5, 'interval': (2, 'iteration'),
        'expected': [False, True, False, True, False, True, False]},
    # basic epoch
    {
        'iters_per_epoch': 1, 'interval': (3, 'epoch'),
        'expected': [False, False, True, False, False, True, False]},
    # fractional epoch
    {
        'iters_per_epoch': 2, 'interval': (1.5, 'epoch'),
        'expected': [False, False, True, False, False, True, False]},
    # unaligned epoch
    {
        'iters_per_epoch': 2.5, 'interval': (1, 'epoch'),
        'expected': [False, False, True, False, True, False, False]},
    # tiny epoch
    {
        'iters_per_epoch': 0.5, 'interval': (1, 'epoch'),
        'expected': [True, True, True, True, True, True, True]},
]))
class TestTrigger(unittest.TestCase):

    def setUp(self):
        self.trigger = training.trigger.IntervalTrigger(*self.interval)

    def test_trigger(self):
        updater = DummyUpdater(self.iters_per_epoch)
        trainer = training.Trainer(updater)
        for expected in self.expected:
            updater.update()
            self.assertEqual(self.trigger(trainer), expected)

    def test_stop_trigger(self):
        updater = DummyUpdater(self.iters_per_epoch)
        trainer = training.Trainer(updater)
        for expected in [False] + self.expected:
            self.assertEqual(self.trigger(trainer), expected)
            if expected:
                break
            updater.update()

    def test_resumed_trigger(self):
        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, 'temp.npz')

        for resume in range(len(self.expected)):
            updater = DummyUpdater(self.iters_per_epoch)
            trainer = training.Trainer(updater)
            for expected in self.expected[:resume]:
                updater.update()
                self.assertEqual(self.trigger(trainer), expected)
            serializers.save_npz(temp_file, updater)

            updater = DummyUpdater(self.iters_per_epoch)
            serializers.load_npz(temp_file, updater)
            trainer = training.Trainer(updater)
            for expected in self.expected[resume:]:
                updater.update()
                self.assertEqual(self.trigger(trainer), expected)


testing.run_module(__name__, __file__)
