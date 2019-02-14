import os
import unittest

import mock
import pytest

from chainer import testing
from chainer import training
from chainer.training import extensions


class TestSnapshot(unittest.TestCase):

    def test_call(self):
        t = mock.MagicMock()
        c = mock.MagicMock(side_effect=[True, False])
        w = mock.MagicMock()
        snapshot = extensions.snapshot(target=t, condition=c, writer=w)
        trainer = mock.MagicMock()
        snapshot(trainer)
        snapshot(trainer)

        assert c.call_count == 2
        assert w.call_count == 1

    def test_savefun_and_writer_exclusive(self):
        # savefun and writer arguments cannot be specified together.
        def savefun(*args, **kwargs):
            assert False
        writer = extensions.snapshot_writers.SimpleWriter()
        with pytest.raises(TypeError):
            extensions.snapshot(savefun=savefun, writer=writer)


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


class TestSnapshotOnError(unittest.TestCase):

    def setUp(self):
        self.trainer = testing.get_trainer_with_mock_updater()
        self.trainer.out = '.'
        self.filename = 'myfile-deadbeef.dat'

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def test_on_error(self):

        class TheOnlyError(Exception):
            pass

        @training.make_extension(trigger=(1, 'iteration'), priority=100)
        def exception_raiser(trainer):
            raise TheOnlyError()
        self.trainer.extend(exception_raiser)

        snapshot = extensions.snapshot_object(self.trainer, self.filename,
                                              snapshot_on_error=True)
        self.trainer.extend(snapshot)

        self.assertFalse(os.path.exists(self.filename))

        with self.assertRaises(TheOnlyError):
            self.trainer.run()

        self.assertTrue(os.path.exists(self.filename))


testing.run_module(__name__, __file__)
