import time
import unittest

import numpy

from chainer import cuda
from chainer import testing
from chainer.testing import attr
from chainer import utils


class TestGetTimer(unittest.TestCase):

    def test_cpu(self):
        timer = utils.get_timer(numpy)
        self.assertIs(timer.xp, numpy)

    @attr.gpu
    def test_gpu(self):
        timer = utils.get_timer(cuda.cupy)
        selfassertIs(timer.xp, cuda.cupy)


class TestCPUTimer(unittest.TestCase):

    def setUp(self):
        self.timer = utils.CPUTimer()

    def test_interface(self):
        self.timer.xp
        self.timer.reset()
        self.timer.start()
        self.timer.stop()
        with self.timer:
            pass
        self.timer.total_time()
        self.timer.count()
        self.timer.mean()

    def test_normal_start_stop_times(self):
        self.assertEqual(len(self.timer.start_times), 0)
        self.assertEqual(len(self.timer.stop_times), 0)
        self.timer.start()
        self.timer.stop()
        self.assertEqual(len(self.timer.start_times), 1)
        self.assertEqual(len(self.timer.stop_times), 1)
        self.timer.start()
        self.timer.stop()
        self.assertEqual(len(self.timer.start_times), 2)
        self.assertEqual(len(self.timer.stop_times), 2)

    def test_running(self):
        self.assertFalse(self.timer.running)
        self.timer.start()
        self.assertTrue(self.timer.running)
        self.timer.stop()
        self.assertFalse(self.timer.running)
        self.timer.start()
        self.assertTrue(self.timer.running)
        self.timer.stop()
        self.assertFalse(self.timer.running)

    def test_duplicate_start(self):
        self.timer.start()
        self.timer.start()
        self.timer.stop()
        self.assertEqual(len(self.timer.start_times), 1)

    def test_duplicate_stop(self):
        self.timer.start()
        self.timer.stop()
        self.timer.stop()
        self.assertEqual(len(self.timer.start_times), 1)

    def count(self):
        self.assertEqual(self.timer.count(), 0)
        self.timer.start()
        self.assertEqual(self.timer.count(), 0)
        self.timer.stop()
        self.assertEqual(self.timer.count(), 1)
        self.timer.start()
        self.assertEqual(self.timer.count(), 1)
        self.timer.stop()
        self.assertEqual(self.timer.count(), 2)

    def test_elapsed_times(self):
        self.timer.start()
        time.sleep(0.1)
        self.timer.stop()
        numpy.testing.assert_allclose(self.timer.total_time(), 0.1, atol=0.01, rtol=1.5)
        self.timer.start()
        time.sleep(0.1)
        self.timer.stop()
        numpy.testing.assert_allclose(self.timer.total_time(), 0.2, atol=0.02, rtol=1.5)


@attr.gpu
class TestGPUTimer(unittest.TestCase):

    def setUp(self):
        self.timer = utils.GPUTimer()

    def test_interface(self):
        self.timer.xp
        self.timer.reset()
        self.timer.start()
        self.timer.stop()
        with self.timer:
            pass
        self.timer.total_time()
        self.timer.count()
        self.timer.mean()
