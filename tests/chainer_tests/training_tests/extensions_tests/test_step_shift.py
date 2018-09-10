import unittest

import mock

from chainer import testing
from chainer import training
from chainer.training import extensions


@testing.parameterize(
    {'init': 2.0, 'gamma': 0.5, 'step': 2, 'target': None,
     'expect': [2.0, 2.0, 1.0, 1.0, 0.5, 0.5]},
    {'init': 2.0, 'gamma': 0.5, 'step': 2, 'target': 1.2,
     'expect': [2.0, 2.0, 1.2, 1.2, 1.2, 1.2]},
    {'init': -2.0, 'gamma': 0.5, 'step': 2, 'target': -1.2,
     'expect': [-2.0, -2.0, -1.2, -1.2, -1.2, -1.2]},
    {'init': 2.0, 'gamma': 2.0, 'step': 2, 'target': None,
     'expect': [2.0, 2.0, 4.0, 4.0, 8.0, 8.0]},
    {'init': 2.0, 'gamma': 2.0, 'step': 2, 'target': 3.0,
     'expect': [2.0, 2.0, 3.0, 3.0, 3.0, 3.0]},
    {'init': -2.0, 'gamma': 2.0, 'step': 2, 'target': -3.0,
     'expect': [-2.0, -2.0, -3.0, -3.0, -3.0, -3.0]},
)
class TestStepShift(unittest.TestCase):

    def setUp(self):
        self.optimizer = mock.MagicMock()
        self.extension = extensions.StepShift(
            'x', self.gamma, self.step, self.init, self.target, self.optimizer)

        self.interval = 1
        self.expect = [e for e in self.expect for _ in range(self.interval)]
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
        extension = extensions.StepShift(
            'x', self.gamma, self.step, init=self.init, target=self.target)
        self._run_trainer(extension, self.expect)

    def test_without_init(self):
        self.optimizer.x = self.init
        extension = extensions.StepShift(
            'x', self.gamma, self.step, init=self.init, target=self.target)
        self._run_trainer(extension, self.expect)

    def test_with_optimizer(self):
        optimizer = mock.Mock()
        optimizer.x = 0
        extension = extensions.StepShift(
            'x', self.gamma, self.step, self.init, self.target, optimizer)
        self._run_trainer(extension, self.expect, optimizer)

    def test_resume(self):
        new_optimizer = mock.Mock()
        new_extension = extensions.StepShift(
            'x', self.gamma, self.step, self.init, self.target, new_optimizer)

        self.trainer.extend(self.extension)
        self.trainer.run()

        new_trainer = testing.get_trainer_with_mock_updater((5, 'iteration'))
        new_trainer.extend(new_extension)
        testing.save_and_load_npz(self.trainer, new_trainer)

        new_extension.initialize(new_trainer)
        self.assertEqual(new_optimizer.x, self.optimizer.x)
        self.assertIsInstance(new_optimizer.x, float)


testing.run_module(__name__, __file__)
