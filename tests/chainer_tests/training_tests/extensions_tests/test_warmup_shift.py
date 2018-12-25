import unittest

import mock

from chainer import testing
from chainer.training import extensions
from chainer.training import util


@testing.parameterize(
    {'init': 1, 'warmup_start': 0.1,
     'warmup_iter': 100, 'expect': [0.1, 0.991, 1, 1]},
    {'init': 0.1, 'warmup_start': 1,
     'warmup_iter': 10, 'expect': [1, 0.19, 0.1, 0.1]},
    {'init': 1, 'warmup_start': -1,
     'warmup_iter': 10, 'expect': [-1, 0.8, 1, 1]},
    {'init': 1, 'warmup_start': -1,
     'warmup_iter': 2, 'expect': [-1, 0, 1, 1]},
    {'init': 0.1, 'warmup_start': 1,
     'warmup_iter': 2, 'expect': [1, 0.55, 0.1, 0.1]},
)
class TestWarmupShift(unittest.TestCase):

    def setUp(self):
        self.optimizer = mock.MagicMock()
        self.extension = extensions.WarmupShift(
            'x', self.warmup_start, self.warmup_iter,
            self.init, self.optimizer)

        self.interval = 1
        self.expect = [e for e in self.expect for _ in range(self.interval)]
        self.trigger = util.get_trigger((self.interval, 'iteration'))

        self.trainer = testing.get_trainer_with_mock_updater(self.trigger)
        self.trainer.updater.get_optimizer.return_value = self.optimizer

    def _run_trainer(self, extension, expect, optimizer=None):
        if optimizer is None:
            optimizer = self.optimizer
        extension.initialize(self.trainer)
        actual = []
        for _ in range(self.warmup_iter + 2):
            self.trainer.updater.update()
            actual.append(optimizer.x)
            if self.trigger(self.trainer):
                extension(self.trainer)

        testing.assert_allclose(actual[0], expect[0])
        testing.assert_allclose(actual[self.warmup_iter-1], expect[1])
        testing.assert_allclose(actual[self.warmup_iter], expect[2])
        testing.assert_allclose(actual[self.warmup_iter+1], expect[3])

    def test_basic(self):
        self.optimizer.x = 0
        extension = extensions.WarmupShift(
            'x', self.warmup_start, self.warmup_iter,
            self.init, self.optimizer)
        self._run_trainer(extension, self.expect)

    def test_without_init(self):
        self.optimizer.x = self.warmup_start
        extension = extensions.WarmupShift(
            'x', self.warmup_start, self.warmup_iter,
            self.init, self.optimizer)
        self._run_trainer(extension, self.expect)

    def test_with_optimizer(self):
        optimizer = mock.Mock()
        optimizer.x = 0
        extension = extensions.WarmupShift(
            'x', self.warmup_start, self.warmup_iter,
            self.init, optimizer)
        self._run_trainer(extension, self.expect, optimizer)


testing.run_module(__name__, __file__)
