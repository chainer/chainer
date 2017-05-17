import json
import os.path
import shutil
import tempfile
import unittest

import mock

from chainer import testing
from chainer import training
from chainer.training import extensions


class TestLogReport(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_trigger(self):
        trainer = _get_mocked_trainer((10, 'iteration'), self.temp_dir)
        log_report = extensions.LogReport(trigger=(1, 'iteration'))
        trainer.extend(log_report)
        trainer.run()
        with open(os.path.join(self.temp_dir, 'log')) as f:
            self.assertEqual(log_report.log, json.load(f))

    def test_write_trigger(self):
        stops, aggregates, writes = 10, 2, 5
        trainer = _get_mocked_trainer((stops, 'iteration'), self.temp_dir)
        with mock.patch.object(extensions.LogReport, '_write') as mocked:
            log_report = extensions.LogReport(
                trigger=(aggregates, 'iteration'),
                write_trigger=(writes, 'iteration')
            )
            trainer.extend(log_report)
            trainer.run()
            mocked.assert_called_with(log_report.log[-1], self.temp_dir)
            self.assertEqual(mocked.call_count, stops // writes)
            self.assertEqual(len(log_report.log), stops // aggregates)


def _get_mocked_trainer(stop_trigger=(10, 'iteration'), out='result'):
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
    return training.Trainer(updater, stop_trigger, out)


testing.run_module(__name__, __file__)
