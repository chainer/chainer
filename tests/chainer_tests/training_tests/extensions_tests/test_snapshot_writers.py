import unittest

import mock
import multiprocessing
import threading

from chainer import testing
from chainer.training.extensions import snapshot_writers


snapshot_writers_path = 'chainer.training.extensions.snapshot_writers'


class TestSimpleWriter(unittest.TestCase):

    def setUp(self):
        self.filename = 'myfile.dat'
        self.outdir = 'mydir'
        self.target = mock.MagicMock()

    def test_call(self):
        w = snapshot_writers.SimpleWriter()
        w.save = mock.MagicMock()
        w(self.filename, self.outdir, self.target)

        assert w.save.call_count == 1


class TestStandardWriter(unittest.TestCase):

    def setUp(self):
        self.filename = 'myfile.dat'
        self.outdir = 'mydir'
        self.target = mock.MagicMock()

    def test_call(self):
        w = snapshot_writers.StandardWriter()
        worker = mock.MagicMock()
        name = snapshot_writers_path + '.StandardWriter.create_worker'
        with mock.patch(name, return_value=worker):
            w(self.filename, self.outdir, self.target)
            w(self.filename, self.outdir, self.target)
            w.finalize()

            assert worker.start.call_count == 2
            assert worker.join.call_count == 2


class TestThreadWriter(unittest.TestCase):

    def setUp(self):
        self.filename = 'myfile.dat'
        self.outdir = 'mydir'
        self.target = mock.MagicMock()

    def test_create_worker(self):
        w = snapshot_writers.ThreadWriter()
        worker = w.create_worker(self.filename, self.outdir, self.target)

        assert isinstance(worker, threading.Thread)


class TestProcessWriter(unittest.TestCase):

    def setUp(self):
        self.filename = 'myfile.dat'
        self.outdir = 'mydir'
        self.target = mock.MagicMock()

    def test_create_worker(self):
        w = snapshot_writers.ProcessWriter()
        worker = w.create_worker(self.filename, self.outdir, self.target)

        assert isinstance(worker, multiprocessing.Process)


class TestQueueWriter(unittest.TestCase):

    def setUp(self):
        self.filename = 'myfile.dat'
        self.outdir = 'mydir'
        self.target = mock.MagicMock()

    def test_call(self):
        q = mock.MagicMock()
        consumer = mock.MagicMock()
        names = [snapshot_writers_path + '.QueueWriter.create_queue',
                 snapshot_writers_path + '.QueueWriter.create_consumer']
        with mock.patch(names[0], return_value=q):
            with mock.patch(names[1], return_value=consumer):
                w = snapshot_writers.QueueWriter()
                w(self.filename, self.outdir, self.target)
                w(self.filename, self.outdir, self.target)
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
