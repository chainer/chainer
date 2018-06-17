import unittest

import mock

from chainer import testing
from chainer.training import extensions
from chainer.training import util


@testing.parameterize(
    {'init': 3.0, 'gamma': 1.0, 'power': 1.0, 'target': None,
     'expect': [3.0, 1.5, 1.0]},
    {'init': 3.0, 'gamma': 1.0, 'power': 1.0, 'target': 1.8,
     'expect': [3.0, 1.8, 1.8]},
    {'init': -3.0, 'gamma': 1.0, 'power': 1.0, 'target': -1.8,
     'expect': [-3.0, -1.8, -1.8]},
    {'init': 3.0, 'gamma': 1.0, 'power': -2.0, 'target': None,
     'expect': [3.0, 12.0, 27.0]},
    {'init': 3.0, 'gamma': 1.0, 'power': -2.0, 'target': 4.0,
     'expect': [3.0, 4.0, 4.0]},
    {'init': -3.0, 'gamma': 1.0, 'power': -2.0, 'target': -4.0,
     'expect': [-3.0, -4.0, -4.0]},
)
class TestInverseShift(unittest.TestCase):

    def setUp(self):
        self.optimizer = mock.MagicMock()
        self.extension = extensions.InverseShift(
            'x', self.gamma, self.power, self.init, self.target,
            self.optimizer)

        self.interval = 4
        self.expect = [e for e in self.expect for _ in range(self.interval)]
        self.trigger = util.get_trigger((self.interval, 'iteration'))

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
        extension = extensions.InverseShift(
            'x', self.gamma, self.power, init=self.init, target=self.target)
        self._run_trainer(extension, self.expect)

    def test_without_init(self):
        self.optimizer.x = self.init
        extension = extensions.InverseShift(
            'x', self.gamma, self.power, target=self.target)
        self._run_trainer(extension, self.expect)

    def test_with_optimizer(self):
        optimizer = mock.Mock()
        optimizer.x = 0
        extension = extensions.InverseShift(
            'x', self.gamma, self.power, init=self.init, target=self.target,
            optimizer=optimizer)
        self._run_trainer(extension, self.expect, optimizer)

    def test_resume(self):
        new_optimizer = mock.Mock()
        new_extension = extensions.InverseShift(
            'x', self.gamma, self.power, self.init, self.target, new_optimizer)

        self.trainer.extend(self.extension)
        self.trainer.run()

        new_trainer = testing.get_trainer_with_mock_updater((3, 'iteration'))
        new_trainer.extend(new_extension)
        testing.save_and_load_npz(self.trainer, new_trainer)

        new_extension.initialize(new_trainer)
        self.assertEqual(new_optimizer.x, self.optimizer.x)
        self.assertIsInstance(new_optimizer.x, float)


class TestInverseShiftInvalidArgument(unittest.TestCase):

    def test_negative_rate(self):
        with self.assertRaises(ValueError):
            extensions.InverseShift('x', -1.0, 1.0)


testing.run_module(__name__, __file__)
