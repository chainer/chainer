import unittest

import mock
import multiprocessing
import threading

from chainer import testing
from chainer.training.extensions import snapshot_writers
from chainer import utils


snapshot_writers_path = 'chainer.training.extensions.snapshot_writers'


class TestSimpleWriter(unittest.TestCase):

    @staticmethod
    def _touch(filename, target, **kwargs):
        open(filename, 'w')

    def test_call_without_kwargs(self):
        target = mock.MagicMock()
        savefun = mock.MagicMock()
        savefun.side_effect = self._touch
        w = snapshot_writers.SimpleWriter(savefun)
        with utils.tempdir() as tempd:
            w('myfile.dat', tempd, target)

        # check if called as `savefun('filename', target)`
        assert savefun.call_count == 1
        args, kwargs = savefun.call_args_list[0]
        assert len(args) == 2
        assert isinstance(args[0], str)
        assert args[1] is target
        assert kwargs == {}

    def test_call_with_kwargs(self):
        target = mock.MagicMock()
        savefun = mock.MagicMock()
        savefun.side_effect = self._touch
        w = snapshot_writers.SimpleWriter(savefun, spam='ham', egg=1)
        with utils.tempdir() as tempd:
            w('myfile.dat', tempd, target)

        # check if called as `savefun('filename', target, spam='ham', egg=1)`
        assert savefun.call_count == 1
        args, kwargs = savefun.call_args_list[0]
        assert len(args) == 2
        assert isinstance(args[0], str)
        assert args[1] is target
        assert kwargs == { 'spam': 'ham', 'egg': 1 }


class TestStandardWriter(unittest.TestCase):

    def test_call(self):
        target = mock.MagicMock()
        w = snapshot_writers.StandardWriter()
        worker = mock.MagicMock()
        name = snapshot_writers_path + '.StandardWriter.create_worker'
        with mock.patch(name, return_value=worker):
            with utils.tempdir() as tempd:
                w('myfile.dat', tempd, target)
                w('myfile.dat', tempd, target)
                w.finalize()

            assert worker.start.call_count == 2
            assert worker.join.call_count == 2


class TestThreadWriter(unittest.TestCase):

    def test_create_worker(self):
        target = mock.MagicMock()
        w = snapshot_writers.ThreadWriter()
        with utils.tempdir() as tempd:
            worker = w.create_worker('myfile.dat', tempd, target)
            assert isinstance(worker, threading.Thread)


class TestProcessWriter(unittest.TestCase):

    def test_create_worker(self):
        target = mock.MagicMock()
        w = snapshot_writers.ProcessWriter()
        with utils.tempdir() as tempd:
            worker = w.create_worker('myfile.dat', tempd, target)
            assert isinstance(worker, multiprocessing.Process)


class TestQueueWriter(unittest.TestCase):

    def test_call(self):
        target = mock.MagicMock()
        q = mock.MagicMock()
        consumer = mock.MagicMock()
        names = [snapshot_writers_path + '.QueueWriter.create_queue',
                 snapshot_writers_path + '.QueueWriter.create_consumer']
        with mock.patch(names[0], return_value=q):
            with mock.patch(names[1], return_value=consumer):
                w = snapshot_writers.QueueWriter()

                with utils.tempdir() as tempd:
                    w('myfile.dat', tempd, target)
                    w('myfile.dat', tempd, target)
                    w.finalize()

                assert consumer.start.call_count == 1
                assert q.put.call_count == 3
                assert q.join.call_count, 1
                assert consumer.join.call_count == 1

    def test_consume(self):
        names = [snapshot_writers_path + '.QueueWriter.create_queue',
                 snapshot_writers_path + '.QueueWriter.create_consumer']
        with mock.patch(names[0]):
            with mock.patch(names[1]):
                task = mock.MagicMock()
                q = mock.MagicMock()
                q.get = mock.MagicMock(side_effect=[task, task, None])
                w = snapshot_writers.QueueWriter()
                w.consume(q)

                assert q.get.call_count == 3
                assert task[0].call_count == 2
                assert q.task_done.call_count == 3


testing.run_module(__name__, __file__)
