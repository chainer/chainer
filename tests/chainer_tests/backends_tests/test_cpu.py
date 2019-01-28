import unittest

import numpy

from chainer import backend
from chainer import testing
import chainer.testing.backend  # NOQA


class TestCpuDeviceFromArray(unittest.TestCase):

    def test_from_array(self):
        arr = numpy.ndarray((2,), numpy.float32)
        expected_device = backend.CpuDevice()

        device = backend.CpuDevice.from_array(arr)
        assert device == expected_device

        device = backend.get_device_from_array(arr)
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


testing.run_module(__name__, __file__)
