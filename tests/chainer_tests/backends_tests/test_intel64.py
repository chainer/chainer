import unittest

import numpy

from chainer import backend
from chainer.backends import intel64
from chainer import testing


@testing.backend.inject_backend_tests(
    None,
    [
        {'use_ideep': 'always'},
    ])
class TestIntel64DeviceFromArray(unittest.TestCase):

    def test_from_array(self, backend_config):
        arr = backend_config.get_array(numpy.ndarray((2,), numpy.float32))
        # Test precondition check
        assert isinstance(arr, intel64.mdarray)

        expected_device = backend.Intel64Device()

        device = backend.Intel64Device.from_array(arr)
        assert device == expected_device

        device = backend.get_device_from_array(arr)
        assert device == expected_device


@testing.backend.inject_backend_tests(
    None,
    [
        {},
        {'use_cuda': True},
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
    ])
class TestIntel64DeviceFromArrayInvalidArray(unittest.TestCase):

    def test_from_array(self, backend_config):
        arr = backend_config.get_array(numpy.ndarray((2,), numpy.float32))
        device = backend.Intel64Device.from_array(arr)
        assert device is None


@testing.parameterize(*testing.product(
    {
        'value': [None, 1, ()],
    }))
class TestIntel64DeviceFromArrayInvalidValue(unittest.TestCase):

    def test_from_array(self):
        device = backend.Intel64Device.from_array(self.value)
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
        {'use_ideep': 'always'},
    ])
class TestIntel64DeviceIsArraySupported(unittest.TestCase):

    def test_is_array_supported(self, backend_config1, backend_config2):
        target = backend_config1.device  # backend.Intel64Device

        arr = backend_config2.get_array(numpy.ndarray((2,), numpy.float32))
        device = backend_config2.device

        if isinstance(device, (backend.CpuDevice, backend.Intel64Device)):
            assert target.is_array_supported(arr)
        else:
            assert not target.is_array_supported(arr)


testing.run_module(__name__, __file__)
