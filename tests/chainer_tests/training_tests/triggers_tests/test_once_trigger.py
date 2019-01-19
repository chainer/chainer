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


def get_expected(num, call_on_first=True):
    return [i == 0 and call_on_first for i in six.moves.range(num)]


def get_finished(num, call_on_first=True):
    return [i > 0 or not call_on_first for i in six.moves.range(num)]


@testing.parameterize(
    # basic
    {
        'iter_per_epoch': 5, 'call_on_resume': False, 'resume': 4,
        'expected': get_expected(7),
        'expected_resume': get_expected(7, False),
        'finished': get_finished(7),
        'finished_resume': get_finished(7, False)},
    # call on resume
    {
        'iter_per_epoch': 5, 'call_on_resume': True, 'resume': 4,
        'expected': get_expected(7),
        'expected_resume': get_expected(7, True),
        'finished': get_finished(7),
        'finished_resume': get_finished(7, True)},
    # unaligned epoch
    {
        'iter_per_epoch': 2.5, 'call_on_resume': False, 'resume': 3,
        'expected': get_expected(7),
        'expected_resume': get_expected(7, False),
        'finished': get_finished(7),
        'finished_resume': get_finished(7, False)},
    # unaligned epoch, call on resume
    {
        'iter_per_epoch': 2.5, 'call_on_resume': True, 'resume': 3,
        'expected': get_expected(7),
        'expected_resume': get_expected(7, True),
        'finished': get_finished(7),
        'finished_resume': get_finished(7, True)},
    # tiny epoch
    {
        'iter_per_epoch': 0.5, 'call_on_resume': False, 'resume': 4,
        'expected': get_expected(7),
        'expected_resume': get_expected(7, False),
        'finished': get_finished(7),
        'finished_resume': get_finished(7, False)},
    # tiny epoch, call on resume
    {
        'iter_per_epoch': 0.5, 'call_on_resume': True, 'resume': 4,
        'expected': get_expected(7),
        'expected_resume': get_expected(7, True),
        'finished': get_finished(7),
        'finished_resume': get_finished(7, True)},
)
class TestOnceTrigger(unittest.TestCase):

    def test_trigger(self):
        trainer = testing.get_trainer_with_mock_updater(
            stop_trigger=None, iter_per_epoch=self.iter_per_epoch)
        trigger = training.triggers.OnceTrigger(self.call_on_resume)
        for expected, finished in zip(self.expected, self.finished):
            self.assertEqual(trigger(trainer), expected)
            self.assertEqual(trigger.finished, finished)
            trainer.updater.update()

    def test_resumed_trigger(self):
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
            for expected, finished in zip(self.expected_resume, self.finished_resume):
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
            for expected, finished in zip(self.expected_resume, self.finished_resume):
                trainer.updater.update()
                accumulated = accumulated or expected
                if random.randrange(2):
                    self.assertEqual(trigger(trainer), accumulated)
                    self.assertEqual(trigger.finished, finished)
                    accumulated = False

    def test_resumed_trigger_backward_compat(self):
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
            for expected, finished in zip(self.expected_resume, self.finished_resume):
                trainer.updater.update()
                self.assertEqual(trigger(trainer), expected)
                self.assertEqual(trigger.finished, finished)


testing.run_module(__name__, __file__)
