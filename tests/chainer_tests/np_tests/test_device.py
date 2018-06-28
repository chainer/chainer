import mock
import numpy
import unittest

import chainer
from chainer.backends import cuda
from chainer import np
from chainer import testing
from chainer.testing import attr


class TestDevice(unittest.TestCase):

    def setUp(self):
        self.orig_device = chainer.config.default_device

    def tearDown(self):
        chainer.config.default_device = self.orig_device

    @mock.patch.object(np.Device, '__exit__')
    @mock.patch.object(np.Device, '__enter__')
    def test_device_init(self, enter_mock, exit_mock):
        index = 1
        if self.orig_device.backend == 'native':
            index = 1 if self.orig_device.index == 0 else 0

        device = np.Device('mydevice', index, numpy, cuda.DummyDeviceType())
        assert str(device) == 'mydevice:1'
        assert repr(device) == 'Device("mydevice", 1)'

        with device:
            assert chainer.config.default_device is device
            enter_mock.assert_called_once()

        assert chainer.config.default_device is self.orig_device
        exit_mock.assert_called_once()


class TestGetDevice(unittest.TestCase):

    def check_device(self, device, backend, index, xp):
        assert device.backend == backend
        assert device.index == index
        assert device.xp is xp
        assert device is np.get_device(backend, index)

    def test_native(self):
        self.check_device(np.get_device('native'), 'native', 0, numpy)

    def test_native_2(self):
        self.check_device(np.get_device('native:1'), 'native', 1, numpy)

    @attr.gpu
    def test_cuda(self):
        self.check_device(np.get_device('cuda'), 'cuda', 0, cuda.cupy)

    @attr.multi_gpu(2)
    def test_cuda_2(self):
        self.check_device(np.get_device('cuda:1'), 'cuda', 1, cuda.cupy)

    def test_invalid_device(self):
        with self.assertRaises(ValueError):
            np.get_device('')
        with self.assertRaises(ValueError):
            np.get_device('foo')
        with self.assertRaises(ValueError):
            np.get_device('native:0', 1)
        with self.assertRaises(ValueError):
            np.get_device('native:0:')


testing.run_module(__name__, __file__)
