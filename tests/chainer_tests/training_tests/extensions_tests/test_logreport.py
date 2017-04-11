import unittest

import mock

from chainer import testing
from chainer import training
from chainer.training import extensions


class TestLogReport(unittest.TestCase):

    def test_trigger(self):
        trainer = _get_mocked_trainer((10, 'iteration'))
        with mock.patch.object(extensions.LogReport, '_write') as mocked:
            log_report = extensions.LogReport(trigger=(1, 'iteration'))
            trainer.extend(log_report)
            trainer.run()
            mocked.assert_called()
            self.assertEqual(mocked.call_count, 10)
            self.assertEqual(len(log_report._log), 10)

    def test_write_trigger(self):
        trainer = _get_mocked_trainer((10, 'iteration'))
        with mock.patch.object(extensions.LogReport, '_write') as mocked:
            log_report = extensions.LogReport(trigger=(1, 'iteration'),
                                              write_trigger=(10, 'iteration'))
            trainer.extend(log_report)
            trainer.run()
            mocked.assert_called()
            self.assertEqual(mocked.call_count, 1)
            self.assertEqual(len(log_report._log), 10)


def _get_mocked_trainer(stop_trigger=(10, 'iteration')):
    updater = mock.Mock()
    updater.get_all_optimizers.return_value = {}
    updater.iteration = 0
    updater.epoch = 0
    updater.epoch_detail = 0
    updater.is_new_epoch = True
    iter_per_epoch = 10

    def update():
        updater.iteration += 1
        updater.epoch = updater.iteration // iter_per_epoch
        updater.epoch_detail = updater.iteration / iter_per_epoch
        updater.is_new_epoch = updater.epoch == updater.epoch_detail

    updater.update = update
    return training.Trainer(updater, stop_trigger)


testing.run_module(__name__, __file__)
