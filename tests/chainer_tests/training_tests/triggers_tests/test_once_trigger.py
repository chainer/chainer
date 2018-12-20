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


@testing.parameterize(
    # basic
    {
        'iter_per_epoch': 5, 'call_on_resume': False, 'resume': 4,
        'expected': get_expected(7),
        'expected_resume': get_expected(7, False)},
    # call on resume
    {
        'iter_per_epoch': 5, 'call_on_resume': True, 'resume': 4,
        'expected': get_expected(7),
        'expected_resume': get_expected(7, True)},
    # unaligned epoch
    {
        'iter_per_epoch': 2.5, 'call_on_resume': False, 'resume': 3,
        'expected': get_expected(7),
        'expected_resume': get_expected(7, False)},
    # unaligned epoch, call on resume
    {
        'iter_per_epoch': 2.5, 'call_on_resume': True, 'resume': 3,
        'expected': get_expected(7),
        'expected_resume': get_expected(7, True)},
    # tiny epoch
    {
        'iter_per_epoch': 0.5, 'call_on_resume': False, 'resume': 4,
        'expected': get_expected(7),
        'expected_resume': get_expected(7, False)},
    # tiny epoch, call on resume
    {
        'iter_per_epoch': 0.5, 'call_on_resume': True, 'resume': 4,
        'expected': get_expected(7),
        'expected_resume': get_expected(7, True)},
)
class TestOnceTrigger(unittest.TestCase):

    def test_trigger(self):
        trainer = testing.get_trainer_with_mock_updater(
            stop_trigger=None, iter_per_epoch=self.iter_per_epoch)
        trigger = training.triggers.OnceTrigger(self.call_on_resume)
        for expected in self.expected:
            self.assertEqual(trigger(trainer), expected)
            trainer.updater.update()

    def test_resumed_trigger(self):
        trainer = testing.get_trainer_with_mock_updater(
            stop_trigger=None, iter_per_epoch=self.iter_per_epoch)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            trigger = training.triggers.OnceTrigger(self.call_on_resume)
            for expected in self.expected:
                trainer.updater.update()
                self.assertEqual(trigger(trainer), expected)
            serializers.save_npz(f.name, trigger)

            trigger = training.triggers.OnceTrigger(self.call_on_resume)
            serializers.load_npz(f.name, trigger)
            for expected in self.expected_resume:
                trainer.updater.update()
                self.assertEqual(trigger(trainer), expected)

    @condition.repeat(10)
    def test_trigger_sparse_call(self):
        trainer = testing.get_trainer_with_mock_updater(
            stop_trigger=None, iter_per_epoch=self.iter_per_epoch)
        trigger = training.triggers.OnceTrigger(self.call_on_resume)
        accumulated = False
        for expected in self.expected:
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
            trigger = training.triggers.OnceTrigger(self.call_on_resume)
            for expected in self.expected:
                trainer.updater.update()
                accumulated = accumulated or expected
                if random.randrange(2):
                    self.assertEqual(trigger(trainer), accumulated)
                    accumulated = False
            serializers.save_npz(f.name, trigger)

            trigger = training.triggers.OnceTrigger(self.call_on_resume)
            serializers.load_npz(f.name, trigger)
            for expected in self.expected_resume:
                trainer.updater.update()
                accumulated = accumulated or expected
                if random.randrange(2):
                    self.assertEqual(trigger(trainer), accumulated)
                    accumulated = False

    def test_resumed_trigger_backward_compat(self):
        trainer = testing.get_trainer_with_mock_updater(
            stop_trigger=None, iter_per_epoch=self.iter_per_epoch)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            trigger = training.triggers.OnceTrigger(self.call_on_resume)
            for expected in self.expected:
                trainer.updater.update()
                self.assertEqual(trigger(trainer), expected)
            # old version does not save anything
            np.savez(f, dummy=0)

            trigger = training.triggers.OnceTrigger(self.call_on_resume)
            with testing.assert_warns(UserWarning):
                serializers.load_npz(f.name, trigger)
            for expected in self.expected_resume:
                trainer.updater.update()
                self.assertEqual(trigger(trainer), expected)


testing.run_module(__name__, __file__)
