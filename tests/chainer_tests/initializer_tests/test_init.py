import os
import unittest

import chainer
from chainer.backends import cuda
from chainer import initializers
from chainer import testing
from chainer.testing import attr
import chainerx

import numpy


class TestGenerateArray(unittest.TestCase):

    def _generate_array(self, xp, dtype=None, device=None):
        initializer = initializers.Zero(dtype)
        return initializers.generate_array(initializer, (), xp, device=device)

    def test_default_init(self):
        default_dtype = os.environ.get('CHAINER_DTYPE', 'float32')
        array = self._generate_array(numpy)
        self.assertEqual(default_dtype, array.dtype)

    def test_custom_init(self):
        with chainer.using_config('dtype', 'float16'):
            array = self._generate_array(numpy)
        self.assertEqual('float16', array.dtype)

    def test_init_with_initializer_dtype(self):
        with chainer.using_config('dtype', 'float16'):
            array = self._generate_array(numpy, 'float64')
        self.assertEqual('float64', array.dtype)

    @attr.gpu(2)
    def test_init_gpu_with_device(self):
        device = cuda.Device(1)
        array = self._generate_array(cuda.cupy, 'float64', device)
        assert array.device == device

    @attr.gpu(2)
    def test_init_gpu_with_current_device(self):
        device_id = 1
        with cuda.get_device_from_id(device_id):
            array = self._generate_array(cuda.cupy, 'float64')
        assert array.device.id == device_id

    @attr.chainerx
    def test_init_chainerx_with_device(self):
        device = chainerx.get_device('native:1')
        array = self._generate_array(chainerx, 'float64', device)
        assert array.device is device

    @attr.chainerx
    def test_init_chainerx_with_device_string(self):
        device = 'native:1'
        array = self._generate_array(chainerx, 'float64', device)
        assert array.device.name == device

    @attr.chainerx
    def test_init_chainerx_with_default_device(self):
        device = chainerx.get_device('native:1')
        with chainerx.device_scope(device):
            array = self._generate_array(chainerx, 'float64')
        assert array.device is device

    @attr.chainerx
    @attr.gpu
    def test_init_chainerx_with_cuda(self):
        device = chainerx.get_device('cuda:0')
        with chainerx.device_scope(device):
            array = self._generate_array(chainerx, 'float64')
        assert array.device is device


class TestGetInitializer(unittest.TestCase):

    def test_scalar(self):
        init = initializers._get_initializer(10)
        self.assertIsInstance(init, initializers.Constant)

        x = numpy.empty((2, 3), dtype=numpy.int32)
        init(x)

        expected = numpy.full((2, 3), 10, dtype=numpy.int32)
        numpy.testing.assert_array_equal(x, expected)

    def test_numpy_array(self):
        c = numpy.array([1, 2, 3])
        init = initializers._get_initializer(c)

        self.assertIsInstance(init, initializers.Constant)

        x = numpy.empty((3,), dtype=numpy.int32)
        init(x)

        expected = numpy.array([1, 2, 3], dtype=numpy.int32)
        numpy.testing.assert_array_equal(x, expected)

    def test_callable(self):

        def initializer(arr):
            arr[...] = 100

        init = initializers._get_initializer(initializer)
        self.assertTrue(callable(init))

        x = numpy.empty((2, 3), dtype=numpy.int32)
        init(x)

        expected = numpy.full((2, 3), 100, dtype=numpy.int32)
        numpy.testing.assert_array_equal(x, expected)


testing.run_module(__name__, __file__)
