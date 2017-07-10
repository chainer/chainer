import unittest

from chainer import testing


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

        def check_count(trainer):
            count[0] += 1
            self.assertEqual(trainer.updater.iteration, count[0])

        self.trainer.extend(check_count)
        self.trainer.run()
        self.assertEqual(count[0], 5)


testing.run_module(__name__, __file__)
