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

    expected = [True] + [False] * 6
    finished = [False] + [True] * 6

    def setUp(self):
        self.resumed_expected = [True] + [False] * 6
        self.resumed_finished = [False] + [True] * 6
        if self.call_on_resume:
            self.resumed_expected[self.resume] = True
            self.resumed_finished[self.resume] = False

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
            for expected, finished in zip(self.resumed_expected[:self.resume],
                                          self.resumed_finished[:self.resume]):
                trainer.updater.update()
                self.assertEqual(trigger.finished, finished)
                self.assertEqual(trigger(trainer), expected)
            serializers.save_npz(f.name, trigger)

            trigger = training.triggers.OnceTrigger(self.call_on_resume)
            serializers.load_npz(f.name, trigger)
            for expected, finished in zip(self.resumed_expected[self.resume:],
                                          self.resumed_finished[self.resume:]):
                trainer.updater.update()
                self.assertEqual(trigger.finished, finished)
                self.assertEqual(trigger(trainer), expected)

    @condition.repeat(10)
    def test_trigger_sparse_call(self):
        trainer = testing.get_trainer_with_mock_updater(
            stop_trigger=None, iter_per_epoch=self.iter_per_epoch)
        trigger = training.triggers.OnceTrigger(self.call_on_resume)
        accumulated = False
        accumulated_finished = True
        for expected, finished in zip(self.expected, self.finished):
            accumulated = accumulated or expected
            accumulated_finished = accumulated_finished and finished
            if random.randrange(2):
                self.assertEqual(trigger.finished, accumulated_finished)
                self.assertEqual(trigger(trainer), accumulated)
                accumulated = False
                accumulated_finished = True
            trainer.updater.update()

    @condition.repeat(10)
    def test_resumed_trigger_sparse_call(self):
        trainer = testing.get_trainer_with_mock_updater(
            stop_trigger=None, iter_per_epoch=self.iter_per_epoch)
        accumulated = False
        accumulated_finished = True
        with tempfile.NamedTemporaryFile(delete=False) as f:
            trigger = training.triggers.OnceTrigger(self.call_on_resume)
            for expected, finished in zip(self.resumed_expected[:self.resume],
                                          self.resumed_finished[:self.resume]):
                trainer.updater.update()
                accumulated = accumulated or expected
                accumulated_finished = accumulated_finished and finished
                if random.randrange(2):
                    self.assertEqual(trigger.finished, accumulated_finished)
                    self.assertEqual(trigger(trainer), accumulated)
                    accumulated = False
                    accumulated_finished = True
            serializers.save_npz(f.name, trigger)

            trigger = training.triggers.OnceTrigger(self.call_on_resume)
            serializers.load_npz(f.name, trigger)
            for expected, finished in zip(self.resumed_expected[self.resume:],
                                          self.resumed_finished[self.resume:]):
                trainer.updater.update()
                accumulated = accumulated or expected
                accumulated_finished = accumulated_finished and finished
                if random.randrange(2):
                    self.assertEqual(trigger.finished, accumulated_finished)
                    self.assertEqual(trigger(trainer), accumulated)
                    accumulated = False
                    accumulated_finished = True

    def test_resumed_trigger_backward_compat(self):
        trainer = testing.get_trainer_with_mock_updater(
            stop_trigger=None, iter_per_epoch=self.iter_per_epoch)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            trigger = training.triggers.OnceTrigger(self.call_on_resume)
            for expected, finished in zip(self.resumed_expected[:self.resume],
                                          self.resumed_finished[:self.resume]):
                trainer.updater.update()
                self.assertEqual(trigger.finished, finished)
                self.assertEqual(trigger(trainer), expected)
            # old version does not save anything
            np.savez(f, dummy=0)

            trigger = training.triggers.OnceTrigger(self.call_on_resume)
            with testing.assert_warns(UserWarning):
                serializers.load_npz(f.name, trigger)
            for expected, finished in zip(self.resumed_expected[self.resume:],
                                          self.resumed_finished[self.resume:]):
                trainer.updater.update()
                self.assertEqual(trigger.finished, finished)
                self.assertEqual(trigger(trainer), expected)


testing.run_module(__name__, __file__)
