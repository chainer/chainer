from __future__ import division

import math
import unittest

from chainer import testing


@testing.parameterize(*testing.product({
    'stop_trigger': [(5, 'iteration'), (5, 'epoch')],
    'iter_per_epoch': [0.5, 1, 1.5, 5],
}))
class TestGetTrainerWithMockUpdater(unittest.TestCase):

    def setUp(self):
        self.trainer = testing.get_trainer_with_mock_updater(
            self.stop_trigger, self.iter_per_epoch)

    def test_run(self):
        iteration = [0]

        def check(trainer):
            iteration[0] += 1

            self.assertEqual(trainer.updater.iteration, iteration[0])
            self.assertEqual(
                trainer.updater.epoch, iteration[0] // self.iter_per_epoch)
            self.assertEqual(
                trainer.updater.epoch_detail,
                iteration[0] / self.iter_per_epoch)
            self.assertEqual(
                trainer.updater.is_new_epoch,
                (iteration[0] - 1) // self.iter_per_epoch !=
                iteration[0] // self.iter_per_epoch)
            self.assertEqual(
                trainer.updater.previous_epoch_detail,
                (iteration[0] - 1) / self.iter_per_epoch)

        self.trainer.extend(check)
        self.trainer.run()

        if self.stop_trigger[1] == 'iteration':
            self.assertEqual(iteration[0], self.stop_trigger[0])
        elif self.stop_trigger[1] == 'epoch':
            self.assertEqual(
                iteration[0],
                math.ceil(self.stop_trigger[0] * self.iter_per_epoch))


testing.run_module(__name__, __file__)
