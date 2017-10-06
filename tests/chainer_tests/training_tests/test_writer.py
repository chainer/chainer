import unittest

import mock
import multiprocessing
import threading

from chainer import testing
from chainer.training import writer


class TestSimpleWriter(unittest.TestCase):

    def setUp(self):
        self.filename = 'myfile.dat'
        self.outdir = 'mydir'
        self.target = mock.MagicMock()

    def test_call(self):
        w = writer.SimpleWriter()
        w.save = mock.MagicMock()
        w(self.filename, self.outdir, self.target)

        self.assertEqual(w.save.call_count, 1)


class TestStandardWriter(unittest.TestCase):

    def setUp(self):
        self.filename = 'myfile.dat'
        self.outdir = 'mydir'
        self.target = mock.MagicMock()

    def test_call(self):
        w = writer.StandardWriter()
        worker = mock.MagicMock()
        name = 'chainer.training.writer.StandardWriter.create_worker'
        with mock.patch(name, return_value=worker):
            w(self.filename, self.outdir, self.target)
            w(self.filename, self.outdir, self.target)
            w.finalize()

            self.assertEqual(worker.start.call_count, 2)
            self.assertEqual(worker.join.call_count, 2)


class TestThreadWriter(unittest.TestCase):

    def setUp(self):
        self.filename = 'myfile.dat'
        self.outdir = 'mydir'
        self.target = mock.MagicMock()

    def test_create_worker(self):
        w = writer.ThreadWriter()
        worker = w.create_worker(self.filename, self.outdir, self.target)

        self.assertIsInstance(worker, threading.Thread)


class TestProcessWriter(unittest.TestCase):

    def setUp(self):
        self.filename = 'myfile.dat'
        self.outdir = 'mydir'
        self.target = mock.MagicMock()

    def test_create_worker(self):
        w = writer.ProcessWriter()
        worker = w.create_worker(self.filename, self.outdir, self.target)

        self.assertIsInstance(worker, multiprocessing.Process)


class TestQueueWriter(unittest.TestCase):

    def setUp(self):
        self.filename = 'myfile.dat'
        self.outdir = 'mydir'
        self.target = mock.MagicMock()

    def test_call(self):
        q = mock.MagicMock()
        consumer = mock.MagicMock()
        names = ['chainer.training.writer.QueueWriter.create_queue',
                 'chainer.training.writer.QueueWriter.create_consumer']
        with mock.patch(names[0], return_value=q):
            with mock.patch(names[1], return_value=consumer):
                w = writer.QueueWriter()
                w(self.filename, self.outdir, self.target)
                w(self.filename, self.outdir, self.target)
                w.finalize()

                self.assertEqual(consumer.start.call_count, 1)
                self.assertEqual(q.put.call_count, 3)
                self.assertEqual(q.join.call_count, 1)
                self.assertEqual(consumer.join.call_count, 1)

    def test_consume(self):
        names = ['chainer.training.writer.QueueWriter.create_queue',
                 'chainer.training.writer.QueueWriter.create_consumer']
        with mock.patch(names[0]):
            with mock.patch(names[1]):
                task = mock.MagicMock()
                q = mock.MagicMock()
                q.get = mock.MagicMock(side_effect=[task, task, None])
                w = writer.QueueWriter()
                w.consume(q)

                self.assertEqual(q.get.call_count, 3)
                self.assertEqual(task[0].call_count, 2)
                self.assertEqual(q.task_done.call_count, 3)


testing.run_module(__name__, __file__)
