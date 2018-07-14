import os
import unittest

import mock

from chainer import testing
from chainer.training import extensions


class TestSnapshotObject(unittest.TestCase):

    def test_trigger(self):
        target = mock.MagicMock()
        snapshot_object = extensions.snapshot_object(target, 'myfile.dat')
        self.assertEqual(snapshot_object.trigger, (1, 'epoch',))


class TestSnapshot(unittest.TestCase):

    def test_trigger(self):
        snapshot = extensions.snapshot()
        self.assertEqual(snapshot.trigger, (1, 'epoch'))


class TestSnapshotSaveFile(unittest.TestCase):

    def setUp(self):
        self.trainer = testing.get_trainer_with_mock_updater()
        self.trainer.out = '.'
        self.trainer._done = True

    def tearDown(self):
        if os.path.exists('myfile.dat'):
            os.remove('myfile.dat')

    def test_save_file(self):
        snapshot = extensions.snapshot_object(self.trainer, 'myfile.dat')
        snapshot(self.trainer)

        self.assertTrue(os.path.exists('myfile.dat'))

    def test_clean_up_tempdir(self):
        snapshot = extensions.snapshot_object(self.trainer, 'myfile.dat')
        snapshot(self.trainer)

        left_tmps = [fn for fn in os.listdir('.')
                     if fn.startswith('tmpmyfile.dat')]
        self.assertEqual(len(left_tmps), 0)


testing.run_module(__name__, __file__)
