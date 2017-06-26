import unittest

import mock

from chainer import testing
from chainer import training
from chainer.training import extensions


class TestLinearShift(unittest.TestCase):

    value_range = (2.0, 6.0)
    time_range = (1, 3)
    expect = [2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 6.0, 6.0, 6.0, 6.0]

    def setUp(self):
        self.optimizer = mock.MagicMock()
        self.extension = extensions.LinearShift(
            'x', self.value_range, self.time_range, self.optimizer)

        self.interval = 2
        self.trigger = training.get_trigger((self.interval, 'iteration'))

        self.trainer = testing.get_trainer_with_mock_updater(self.trigger)
        self.trainer.updater.get_optimizer.return_value = self.optimizer

    def _run_trainer(self, extension, expect, optimizer=None):
        if optimizer is None:
            optimizer = self.optimizer
        extension.initialize(self.trainer)

        actual = []
        for _ in expect:
            self.trainer.updater.update()
            actual.append(optimizer.x)
            if self.trigger(self.trainer):
                extension(self.trainer)

        self.assertEqual(actual, expect)

    def test_basic(self):
        self.optimizer.x = 0
        extension = extensions.LinearShift(
            'x', self.value_range, self.time_range)
        self._run_trainer(extension, self.expect)

    def test_with_optimizer(self):
        optimizer = mock.Mock()
        optimizer.x = 0
        extension = extensions.LinearShift(
            'x', self.value_range, self.time_range, optimizer)
        self._run_trainer(extension, self.expect, optimizer)

    def test_resume(self):
        new_optimizer = mock.Mock()
        new_extension = extensions.LinearShift(
            'x', self.value_range, self.time_range, new_optimizer)

        self.trainer.extend(self.extension)
        self.trainer.run()

        new_trainer = testing.get_trainer_with_mock_updater((5, 'iteration'))
        new_trainer.extend(new_extension)
        testing.save_and_load_npz(self.trainer, new_trainer)

        new_extension.initialize(new_trainer)
        self.assertEqual(new_optimizer.x, self.optimizer.x)
        self.assertIsInstance(new_optimizer.x, float)


testing.run_module(__name__, __file__)
