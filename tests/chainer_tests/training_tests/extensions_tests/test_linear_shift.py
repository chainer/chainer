import unittest

import mock

from chainer import testing
from chainer.training import extensions


class TestLinearShift(unittest.TestCase):

    value_range = (2.0, 6.0)
    time_range = (1, 3)
    expect = [2.0, 2.0, 4.0, 6.0, 6.0]

    def setUp(self):
        self.optimizer = mock.MagicMock()
        self.trainer = testing.get_trainer_with_mock_updater((3, 'iteration'))
        self.extension = extensions.LinearShift(
            'x', self.value_range, self.time_range, self.optimizer)

    def test_call(self):
        for e in self.expect:
            self.extension(self.trainer)
            self.assertEqual(self.optimizer.x, e)

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


testing.run_module(__name__, __file__)
