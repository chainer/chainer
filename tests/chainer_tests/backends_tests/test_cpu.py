import unittest

import numpy

from chainer import backend
from chainer import testing


class TestCpuDevice(unittest.TestCase):

    def test_hashable(self):
        assert isinstance(hash(backend.CpuDevice()), int)


class TestCpuDeviceFromArray(unittest.TestCase):

    def check_device(self, device):
        assert device.xp is numpy
        assert device.supported_array_types == (numpy.ndarray,)
        assert device.name == '@numpy'
        assert str(device) == '@numpy'

    def test_init(self):
        device = backend.CpuDevice()
        self.check_device(device)

    def test_from_array(self):
        arr = numpy.ndarray((2,), numpy.float32)
        expected_device = backend.CpuDevice()

        device = backend.CpuDevice.from_array(arr)
        self.check_device(device)
        assert device == expected_device

        device = backend.get_device_from_array(arr)
        self.check_device(device)
        assert device == expected_device


@testing.backend.inject_backend_tests(
    None,
    [
        {'use_cuda': True},
        {'use_ideep': 'always'},
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
    ])
class TestCpuDeviceFromArrayInvalidArray(unittest.TestCase):

    def test_from_array(self, backend_config):
        arr = backend_config.get_array(numpy.ndarray((2,), numpy.float32))
        device = backend.CpuDevice.from_array(arr)
        assert device is None


@testing.parameterize(*testing.product(
    {
        'value': [None, 1, (), numpy.float32(1)],
    }))
class TestCpuDeviceFromArrayInvalidValue(unittest.TestCase):

    def test_from_array(self):
        device = backend.CpuDevice.from_array(self.value)
        assert device is None


@testing.backend.inject_backend_tests(  # backend_config2
    None,
    [
        {},
        {'use_cuda': True},
        {'use_ideep': 'always'},
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
    ])
@testing.backend.inject_backend_tests(  # backend_config1
    None,
    [
        {},
    ])
class TestCpuIsArraySupported(unittest.TestCase):

    def test_is_array_supported(self, backend_config1, backend_config2):
        target = backend_config1.device  # backend.CpuDevice

        arr = backend_config2.get_array(numpy.ndarray((2,), numpy.float32))
        device = backend_config2.device

        if isinstance(device, backend.CpuDevice):
            assert target.is_array_supported(arr)
        else:
            assert not target.is_array_supported(arr)


testing.run_module(__name__, __file__)
