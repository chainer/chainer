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


testing.run_module(__name__, __file__)
