import unittest

import numpy

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import testing
from chainer.testing import attr
import chainerx


@testing.inject_backend_tests(
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


@testing.inject_backend_tests(
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


@testing.inject_backend_tests(
    None,
    [
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:1'},
    ])
class TestChainerxDeviceUse(unittest.TestCase):

    def test_use(self, backend_config):
        device = chainer.get_device(backend_config.chainerx_device)
        with chainerx.using_device('native:1'):
            device.use()
            assert device.device is chainerx.get_default_device()


@chainer.testing.inject_backend_tests(
    None,
    [
        # NumPy
        {},
        # CuPy
        {'use_cuda': True, 'cuda_device': 0},
        {'use_cuda': True, 'cuda_device': 1},
        # ChainerX
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:1'},
    ])
@attr.chainerx
class TestFromToChainerx(unittest.TestCase):

    def check_equal_memory_shared(self, arr1, arr2):
        # Check that the two arrays share the internal memory.
        numpy.testing.assert_array_equal(
            backend.CpuDevice().send(arr1), backend.CpuDevice().send(arr2))
        with chainer.using_device(backend.get_device_from_array(arr1)):
            arr1 += 2
        numpy.testing.assert_array_equal(
            backend.CpuDevice().send(arr1), backend.CpuDevice().send(arr2))
        with chainer.using_device(backend.get_device_from_array(arr1)):
            arr1 -= 2

    def test_from_chx(self, backend_config):
        arr = backend_config.get_array(numpy.ones((2, 3), numpy.float32))
        arr_converted = backend.from_chx(arr)

        src_device = backend_config.device
        if src_device.xp is chainerx:
            dst_xp = src_device.fallback_device.xp
            assert isinstance(arr_converted, dst_xp.ndarray)
            if dst_xp is cuda.cupy:
                assert arr_converted.device.id == src_device.device.index
        else:
            assert arr is arr_converted

        with backend_config:
            self.check_equal_memory_shared(arr, arr_converted)

    def test_to_chx(self, backend_config):
        arr = backend_config.get_array(numpy.ones((2, 3), numpy.float32))
        arr_converted = backend.to_chx(arr)

        src_device = backend_config.device
        assert isinstance(arr_converted, chainerx.ndarray)
        if src_device.xp is chainerx:
            assert arr is arr_converted
        elif src_device.xp is cuda.cupy:
            assert arr.device.id == arr_converted.device.index

        self.check_equal_memory_shared(arr, arr_converted)


testing.run_module(__name__, __file__)
