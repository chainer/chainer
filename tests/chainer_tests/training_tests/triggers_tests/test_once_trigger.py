from __future__ import division

import numpy as np
import random
import six
import tempfile
import unittest

from chainer import serializers
from chainer import testing
from chainer.testing import condition
from chainer import training


@testing.parameterize(
    # basic
    {
        'iter_per_epoch': 5, 'call_on_resume': False, 'resume': 4},
    # call on resume
    {
        'iter_per_epoch': 5, 'call_on_resume': True, 'resume': 4},
    # unaligned epoch
    {
        'iter_per_epoch': 2.5, 'call_on_resume': False, 'resume': 3},
    # unaligned epoch, call on resume
    {
        'iter_per_epoch': 2.5, 'call_on_resume': True, 'resume': 3},
    # tiny epoch
    {
        'iter_per_epoch': 0.5, 'call_on_resume': False, 'resume': 4},
    # tiny epoch, call on resume
    {
        'iter_per_epoch': 0.5, 'call_on_resume': True, 'resume': 4},
)
class TestOnceTrigger(unittest.TestCase):

    def setUp(self):
        self.expected = [1] + [0] * 6
        self.finished = [0] + [1] * 6
        if self.call_on_resume:
            self.expected[self.resume] = 1
            self.finished[self.resume] = 0

    def test_trigger(self):
        trainer = testing.get_trainer_with_mock_updater(
            stop_trigger=None, iter_per_epoch=self.iter_per_epoch)
        trigger = training.triggers.OnceTrigger(self.call_on_resume)
        for expected, finished in zip(self.expected, self.finished):
            self.assertEqual(trigger.finished, finished)
            self.assertEqual(trigger(trainer), expected)
            trainer.updater.update()

    def test_resumed_trigger(self):
        trainer = testing.get_trainer_with_mock_updater(
            stop_trigger=None, iter_per_epoch=self.iter_per_epoch)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            trigger = training.triggers.OnceTrigger(self.call_on_resume)
            for expected, finished in zip(self.expected, self.finished):
                trainer.updater.update()
                self.assertEqual(trigger.finished, finished)
                self.assertEqual(trigger(trainer), expected)
            serializers.save_npz(f.name, trigger)

            trigger = training.triggers.OnceTrigger(self.call_on_resume)
            serializers.load_npz(f.name, trigger)
            for expected, finished in zip(self.expected[self.resume:],
                                          self.finished[self.resume:]):
                trainer.updater.update()
                self.assertEqual(trigger.finished, finished)
                self.assertEqual(trigger(trainer), expected)

    @condition.repeat(10)
    def test_trigger_sparse_call(self):
        trainer = testing.get_trainer_with_mock_updater(
            stop_trigger=None, iter_per_epoch=self.iter_per_epoch)
        trigger = training.triggers.OnceTrigger(self.call_on_resume)
        accumulated = False
        for expected, finished in zip(self.expected, self.finished):
            accumulated = accumulated or expected
            if random.randrange(2):
                self.assertEqual(trigger.finished, finished)
                self.assertEqual(trigger(trainer), accumulated)
                accumulated = False
            trainer.updater.update()

    @condition.repeat(10)
    def test_resumed_trigger_sparse_call(self):
        trainer = testing.get_trainer_with_mock_updater(
            stop_trigger=None, iter_per_epoch=self.iter_per_epoch)
        accumulated = False
        with tempfile.NamedTemporaryFile(delete=False) as f:
            trigger = training.triggers.OnceTrigger(self.call_on_resume)
            for expected, finished in zip(self.expected, self.finished):
                trainer.updater.update()
                accumulated = accumulated or expected
                if random.randrange(2):
                    self.assertEqual(trigger.finished, finished)
                    self.assertEqual(trigger(trainer), accumulated)
                    accumulated = False
            serializers.save_npz(f.name, trigger)

            trigger = training.triggers.OnceTrigger(self.call_on_resume)
            serializers.load_npz(f.name, trigger)
            for expected, finished in zip(self.expected[self.resume:],
                                          self.finished[self.resume:]):
                trainer.updater.update()
                accumulated = accumulated or expected
                if random.randrange(2):
                    self.assertEqual(trigger.finished, finished)
                    self.assertEqual(trigger(trainer), accumulated)
                    accumulated = False

    def test_resumed_trigger_backward_compat(self):
        trainer = testing.get_trainer_with_mock_updater(
            stop_trigger=None, iter_per_epoch=self.iter_per_epoch)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            trigger = training.triggers.OnceTrigger(self.call_on_resume)
            for expected, finished in zip(self.expected, self.finished):
                trainer.updater.update()
                self.assertEqual(trigger.finished, finished)
                self.assertEqual(trigger(trainer), expected)
            # old version does not save anything
            np.savez(f, dummy=0)

            trigger = training.triggers.OnceTrigger(self.call_on_resume)
            with testing.assert_warns(UserWarning):
                serializers.load_npz(f.name, trigger)
            for expected, finished in zip(self.expected[self.resume:],
                                          self.finished[self.resume:]):
                trainer.updater.update()
                self.assertEqual(trigger.finished, finished)
                self.assertEqual(trigger(trainer), expected)


testing.run_module(__name__, __file__)
