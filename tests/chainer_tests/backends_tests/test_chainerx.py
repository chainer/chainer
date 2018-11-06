import unittest

import numpy

from chainer import backend
from chainer import testing
import chainer.testing.backend  # NOQA


@testing.backend.inject_backend_tests(
    None,
    [
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:1'},
    ])
class TestChainerxDeviceFromArray(unittest.TestCase):

    def test_from_array(self, backend_config):
        arr = backend_config.get_array(numpy.ndarray((2,), numpy.float32))
        # Test precondition check
        assert arr.device.name == backend_config.chainerx_device

        expected_device = backend_config.device

        device = backend.ChainerxDevice.from_array(arr)
        assert device == expected_device

        device = backend.get_device_from_array(arr)
        assert device == expected_device


@testing.backend.inject_backend_tests(
    None,
    [
        {},
        {'use_cuda': True},
    ])
class TestChainerxDeviceFromArrayInvalidArray(unittest.TestCase):

    def test_from_array(self, backend_config):
        arr = backend_config.get_array(numpy.ndarray((2,), numpy.float32))
        device = backend.ChainerxDevice.from_array(arr)
        assert device is None


@testing.parameterize(*testing.product(
    {
        'value': [None, 1, ()],
    }))
class TestChainerxDeviceFromArrayInvalidValue(unittest.TestCase):

    def test_from_array(self):
        device = backend.ChainerxDevice.from_array(self.value)
        assert device is None


testing.run_module(__name__, __file__)
