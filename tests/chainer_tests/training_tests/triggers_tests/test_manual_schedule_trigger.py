from __future__ import division

import unittest

from chainer import testing
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

        trigger = training.triggers.ManualScheduleTrigger(*self.schedule)
        for expected in self.expected[:self.resume]:
            trainer.updater.update()
            self.assertEqual(trigger(trainer), expected)

        trigger = training.triggers.ManualScheduleTrigger(*self.schedule)
        for expected in self.expected[self.resume:]:
            trainer.updater.update()
            self.assertEqual(trigger(trainer), expected)


testing.run_module(__name__, __file__)
