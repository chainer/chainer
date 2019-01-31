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
        'iter_per_epoch': 5, 'call_on_resume': True, 'resume': 4,},
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

    expected = [1] + [0] * 6
    finished = [0] + [1] * 6

    def test_trigger(self):
        trainer = testing.get_trainer_with_mock_updater(
            stop_trigger=None, iter_per_epoch=self.iter_per_epoch)
        trigger = training.triggers.OnceTrigger(self.call_on_resume)
        for expected, finished in zip(self.expected, self.finished):
            self.assertEqual(trigger(trainer), expected)
            self.assertEqual(trigger.finished, finished)
            trainer.updater.update()

    def test_resumed_trigger(self):
        if self.call_on_resume:
            self.expected[self.resume] = 1
            self.finished[self.resume] = 0
        trainer = testing.get_trainer_with_mock_updater(
            stop_trigger=None, iter_per_epoch=self.iter_per_epoch)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            trigger = training.triggers.OnceTrigger(self.call_on_resume)
            for expected, finished in zip(self.expected, self.finished):
                trainer.updater.update()
                self.assertEqual(trigger(trainer), expected)
                self.assertEqual(trigger.finished, finished)
            serializers.save_npz(f.name, trigger)

            trigger = training.triggers.OnceTrigger(self.call_on_resume)
            serializers.load_npz(f.name, trigger)
            for expected, finished in zip(self.expected[self.resume:],
                                          self.finished[self.resume:]):
                trainer.updater.update()
                self.assertEqual(trigger(trainer), expected)
                self.assertEqual(trigger.finished, finished)

    @condition.repeat(10)
    def test_trigger_sparse_call(self):
        trainer = testing.get_trainer_with_mock_updater(
            stop_trigger=None, iter_per_epoch=self.iter_per_epoch)
        trigger = training.triggers.OnceTrigger(self.call_on_resume)
        accumulated = False
        for expected, finished in zip(self.expected, self.finished):
            accumulated = accumulated or expected
            if random.randrange(2):
                self.assertEqual(trigger(trainer), accumulated)
                self.assertEqual(trigger.finished, finished)
                accumulated = False
            trainer.updater.update()

    @condition.repeat(10)
    def test_resumed_trigger_sparse_call(self):
        if self.call_on_resume:
            self.expected[self.resume] = 1
            self.finished[self.resume] = 0
        trainer = testing.get_trainer_with_mock_updater(
            stop_trigger=None, iter_per_epoch=self.iter_per_epoch)
        accumulated = False
        with tempfile.NamedTemporaryFile(delete=False) as f:
            trigger = training.triggers.OnceTrigger(self.call_on_resume)
            for expected, finished in zip(self.expected, self.finished):
                trainer.updater.update()
                accumulated = accumulated or expected
                if random.randrange(2):
                    self.assertEqual(trigger(trainer), accumulated)
                    self.assertEqual(trigger.finished, finished)
                    accumulated = False
            serializers.save_npz(f.name, trigger)

            trigger = training.triggers.OnceTrigger(self.call_on_resume)
            serializers.load_npz(f.name, trigger)
            for expected, finished in zip(self.expected[self.resume:],
                                          self.finished[self.resume:]):
                trainer.updater.update()
                accumulated = accumulated or expected
                if random.randrange(2):
                    self.assertEqual(trigger(trainer), accumulated)
                    self.assertEqual(trigger.finished, finished)
                    accumulated = False

    def test_resumed_trigger_backward_compat(self):
        if self.call_on_resume:
            self.expected[self.resume] = 1
            self.finished[self.resume] = 0
        trainer = testing.get_trainer_with_mock_updater(
            stop_trigger=None, iter_per_epoch=self.iter_per_epoch)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            trigger = training.triggers.OnceTrigger(self.call_on_resume)
            for expected, finished in zip(self.expected, self.finished):
                trainer.updater.update()
                self.assertEqual(trigger(trainer), expected)
                self.assertEqual(trigger.finished, finished)
            # old version does not save anything
            np.savez(f, dummy=0)

            trigger = training.triggers.OnceTrigger(self.call_on_resume)
            with testing.assert_warns(UserWarning):
                serializers.load_npz(f.name, trigger)
            for expected, finished in zip(self.expected[self.resume:],
                                          self.finished[self.resume:]):
                trainer.updater.update()
                self.assertEqual(trigger(trainer), expected)
                self.assertEqual(trigger.finished, finished)


testing.run_module(__name__, __file__)
