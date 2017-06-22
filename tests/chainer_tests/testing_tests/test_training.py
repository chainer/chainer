import unittest

from chainer import testing


class TestGetTrainerWithMockUpdater(unittest.TestCase):

    def setUp(self):
        self.trainer = testing.get_trainer_with_mock_updater((5, 'iteration'))

    def test_update_count(self):
        count = [0]

        def check_count(trainer):
            count[0] += 1
            self.assertEqual(trainer.updater.iteration, count[0])

        self.trainer.extend(check_count)
        self.trainer.run()
        self.assertEqual(count[0], 5)


testing.run_module(__name__, __file__)
