import ctypes
import io
import multiprocessing
import os
import pickle
import platform
import sys
import unittest

import mock

from chainer import datasets
from chainer.datasets import pickle_dataset
from chainer import testing
from chainer import utils


class ReaderMock(object):
    def __init__(self, io_):
        self.io = io_
        self._lock = multiprocessing.RLock()
        self._hook_called = multiprocessing.Value(ctypes.c_int, 0, lock=False)
        self._last_caller_pid = multiprocessing.Value(
            ctypes.c_int, -1, lock=False)

    @property
    def n_hook_called(self):
        with self._lock:
            return self._hook_called.value

    @property
    def last_caller_pid(self):
        with self._lock:
            return self._last_caller_pid.value

    def __getattr__(self, name):
        return getattr(self.io, name)

    def after_fork(self):
        with self._lock:
            self._hook_called.value += 1
            self._last_caller_pid.value = os.getpid()


class TestPickleDataset(unittest.TestCase):

    def setUp(self):
        self.io = io.BytesIO()

    def test_write_read(self):
        writer = datasets.PickleDatasetWriter(self.io)
        writer.write(1)
        writer.write('hello')
        writer.write(1.5)
        writer.flush()

        dataset = datasets.PickleDataset(self.io)
        assert len(dataset) == 3
        assert dataset[0] == 1
        assert dataset[2] == 1.5
        assert dataset[1] == 'hello'

    def test_picklable(self):
        writer = datasets.PickleDatasetWriter(self.io)
        writer.write(1)
        writer.flush()

        dataset = datasets.PickleDataset(self.io)
        dataset = pickle.loads(pickle.dumps(dataset))
        assert len(dataset) == 1
        assert dataset[0] == 1

    @unittest.skipIf(platform.system() == 'Windows',
                     'Windows does not support `fork` method')
    def test_after_fork(self):
        writer = datasets.PickleDatasetWriter(self.io)
        writer.write(1)
        writer.flush()

        reader = ReaderMock(self.io)
        # Assign to avoid destruction of the instance
        # before creation a child process
        dataset = datasets.PickleDataset(reader)

        assert reader.n_hook_called == 0
        ctx = multiprocessing.get_context('fork')
        p = ctx.Process()
        p.start()
        p.join()
        assert reader.n_hook_called == 1
        assert reader.last_caller_pid == p.pid

        # Touch to suppress "unused variable' warning
        del dataset


class TestPickleDatasetHelper(unittest.TestCase):

    def setUp(self):
        self.tempdir = utils.tempdir()
        dirpath = self.tempdir.__enter__()
        self.path = os.path.join(dirpath, 'test.pkl')

    def tearDown(self):
        self.tempdir.__exit__(*sys.exc_info())

    def test_write_read(self):
        with datasets.open_pickle_dataset_writer(self.path) as writer:
            writer.write(1)

        with datasets.open_pickle_dataset(self.path) as dataset:
            assert dataset[0] == 1

    def test_file_reader_after_fork(self):
        m = mock.mock_open()
        with mock.patch('chainer.datasets.pickle_dataset.open', m):
            r = pickle_dataset._FileReader(self.path)

            m.assert_called_once_with(self.path, 'rb')
            m().close.assert_not_called()

            m.reset_mock()
            r.after_fork()

            m.assert_called_once_with(self.path, 'rb')
            m().close.assert_called_once_with()

    def test_file_reader_picklable(self):
        m = mock.mock_open()
        with mock.patch('chainer.datasets.pickle_dataset.open', m):
            r = pickle_dataset._FileReader(self.path)

            m.assert_called_once_with(self.path, 'rb')

            m.reset_mock()
            pickle.loads(pickle.dumps(r))

            m.assert_called_once_with(self.path, 'rb')


testing.run_module(__name__, __file__)
