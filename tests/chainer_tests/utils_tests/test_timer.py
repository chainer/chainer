import mock
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
        self.assertIs(timer.xp, cuda.cupy)


class TimerTestBase(object):

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

    def test_count(self):
        self.assertEqual(self.timer.count(), 0)
        self.timer.start()
        self.assertEqual(self.timer.count(), 0)
        self.timer.stop()
        self.assertEqual(self.timer.count(), 1)
        self.timer.start()
        self.assertEqual(self.timer.count(), 1)
        self.timer.stop()
        self.assertEqual(self.timer.count(), 2)

    def test_normal_start_stop_times(self):
        self.assertEqual(len(self.start_count), 0)
        self.assertEqual(len(self.stop_count), 0)
        self.timer.start()
        self.timer.stop()
        self.assertEqual(len(self.start_count), 1)
        self.assertEqual(len(self.stop_count), 1)
        self.timer.start()
        self.timer.stop()
        self.assertEqual(len(self.start_count), 2)
        self.assertEqual(len(self.stop_count), 2)

    def test_duplicate_start(self):
        self.timer.start()
        self.timer.start()
        self.timer.stop()
        self.assertEqual(len(self.start_count), 1)

    def test_duplicate_stop(self):
        self.timer.start()
        self.timer.stop()
        self.timer.stop()
        self.assertEqual(len(self.start_count), 1)

    def test_total_time(self):
        self.timer.start()
        time.sleep(0.1)
        self.timer.stop()
        numpy.testing.assert_allclose(self.timer.total_time(), 0.1, atol=0.01, rtol=1.5)


class TestCPUTimer(TimerTestBase, unittest.TestCase):

    def setUp(self):
        self.timer = utils.CPUTimer()
        self.start_count = self.timer.start_times
        self.stop_count = self.timer.stop_times

    def test_xp(self):
        self.assertIs(numpy, self.timer.xp)

    def test_total_time_twice(self):
        self.timer.start()
        time.sleep(0.1)
        self.timer.stop()
        numpy.testing.assert_allclose(self.timer.total_time(), 0.1, atol=0.01, rtol=1.5)
        self.timer.start()
        time.sleep(0.1)
        self.timer.stop()
        numpy.testing.assert_allclose(self.timer.total_time(), 0.2, atol=0.02, rtol=1.5)


@testing.parameterize(
    {'blocking_method': 'non_block'},
    {'blocking_method': 'block_first_time'},
    {'blocking_method': 'block_every_time'}
)
@attr.gpu
class TestGPUTimer(TimerTestBase, unittest.TestCase):

    def setUp(self):
        self.timer = utils.GPUTimer(blocking_method=self.blocking_method)
        self.start_count = self.timer.start_events
        self.stop_count = self.timer.stop_events

    def test_xp(self):
        self.assertIs(cuda.cupy, self.timer.xp)

    def test_forbid_measurement_after_synchronization(self):
        self.timer.synchronize()
        with self.assertRaises(RuntimeError):
            self.timer.start()


@attr.gpu
class TestGPUInvalid(unittest.TestCase):

    def test_invalid_blocking(self):
        with self.assertRaises(ValueError):
            self.timer = utils.GPUTimer(blocking_method='invalid_argument')


@testing.parameterize(
    {'blocking_method': 'non_block'},
    {'blocking_method': 'block_first_time'},
    {'blocking_method': 'block_every_time'}
)
@attr.gpu
class TestGPUTimerSynchronization(unittest.TestCase):

    def setUp(self):
        self.timer = utils.GPUTimer(blocking_method=self.blocking_method)
        self.mock = mock.Mock()
        self.timer._need_synchronization_before_measurement = self.mock

    def test_synchronization(self):
        self.timer.start()
        if (self.blocking_method == 'block_first_time'
            or self.blocking_method == 'block_every_time'):
            self.mock.assert_called_with()

        self.mock.reset_mock()
        self.timer.stop()
        self.timer.start()
        if self.blocking_method == 'block_every_time':
            self.mock.assert_called_with()


@testing.parameterize(
    {'blocking_method': 'non_block'},
    {'blocking_method': 'block_first_time'},
    {'blocking_method': 'block_every_time'}
)
@attr.gpu
class TestGPUTimerSynchronization(unittest.TestCase):

    def setUp(self):
        self.timer = utils.GPUTimer(blocking_method=self.blocking_method)
        self.mock = mock.Mock()
        self.timer._need_synchronization_before_measurement = self.mock

    def test_synchronization(self):
        self.timer.start()
        if (self.blocking_method == 'block_first_time'
            or self.blocking_method == 'block_every_time'):
            self.mock.assert_called_with()

        self.mock.reset_mock()
        self.timer.stop()
        self.timer.start()
        if self.blocking_method == 'block_every_time':
            self.mock.assert_called_with()


@testing.parameterize(
    {'blocking_method': 'non_block'},
    {'blocking_method': 'block_first_time'},
    {'blocking_method': 'block_every_time'}
)
@attr.gpu
class TestGPUTimerSynchronization(unittest.TestCase):

    def setUp(self):
        self.timer = utils.GPUTimer(blocking_method=self.blocking_method)
        self.mock = mock.Mock()
        self.timer.synchronize = self.mock

    def test_synchronization(self):
        self.timer.total_time()
        self.mock.assert_called_with()


testing.run_module(__name__, __file__)
