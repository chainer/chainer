import unittest
import warnings

import numpy
import six

import chainer
from chainer import cuda
from chainer import testing
from chainer.testing import attr


class TestDummyDeviceType(unittest.TestCase):

    def test_int(self):
        self.assertEqual(int(cuda.DummyDeviceType()), -1)

    def test_eq(self):
        self.assertEqual(cuda.DummyDeviceType(), cuda.DummyDeviceType())

    def test_ne(self):
        self.assertNotEqual(cuda.DummyDeviceType(), 1)


_builtins_available = False
try:
    import builtins
    _builtins_available = True
except ImportError:
    pass


class TestCuda(unittest.TestCase):

    def test_get_dummy_device(self):
        self.assertIs(cuda.get_device_from_id(None), cuda.DummyDevice)

    @attr.gpu
    def test_get_device_from_id_for_numpy_int(self):
        self.assertEqual(
            cuda.get_device_from_id(numpy.int64(0)), cuda.Device(0))

    def test_get_device_from_array_for_numpy_int(self):
        self.assertIs(
            cuda.get_device_from_array(numpy.int64(0)), cuda.DummyDevice)

    @attr.gpu
    def test_get_dummy_device_for_empty_array(self):
        x = cuda.cupy.array([]).reshape((0, 10))
        self.assertIs(cuda.get_device_from_array(x), cuda.DummyDevice)

    @attr.gpu
    @unittest.skipUnless(
        six.PY3, 'Python2.7 has a bug in catch_warnings, so this test is '
                 'skipped for Python2.7')
    def test_get_device_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            cuda.get_device(cuda.cupy.array([1]))

        self.assertEqual(len(w), 1)
        self.assertIs(w[0].category, DeprecationWarning)
        self.assertIn(
            'get_device is deprecated. Please use get_device_from_id'
            ' or get_device_from_array instead.', str(w[0].message))

    @attr.gpu
    def test_get_device_from_id(self):
        self.assertEqual(cuda.get_device_from_id(0), cuda.Device(0))

    @attr.gpu
    def test_get_device_from_array(self):
        self.assertEqual(cuda.get_device_from_array(cuda.cupy.array([0])),
                         cuda.Device(0))

    @attr.gpu
    def test_get_device_for_int(self):
        with testing.assert_warns(DeprecationWarning):
            device = cuda.get_device(0)
        self.assertEqual(device, cuda.Device(0))

    @attr.gpu
    @unittest.skipUnless(_builtins_available,
                         'builtins module is not available')
    def test_get_device_from_id_for_builtin_int(self):
        # builtins.int is from future package and it is different
        # from builtin int/long on Python 2.
        self.assertEqual(
            cuda.get_device_from_id(builtins.int(0)), cuda.Device(0))

    @attr.gpu
    @unittest.skipUnless(_builtins_available,
                         'builtins module is not available')
    def test_get_device_for_builtin_int(self):
        # builtins.int is from future package and it is different
        # from builtin int/long on Python 2.
        with testing.assert_warns(DeprecationWarning):
            device = cuda.get_device(builtins.int(0))
        self.assertEqual(device, cuda.Device(0))

    @attr.gpu
    def test_get_device_for_device(self):
        device = cuda.get_device_from_id(0)
        with testing.assert_warns(DeprecationWarning):
            self.assertIs(cuda.get_device(device), device)

    def test_to_gpu_unavailable(self):
        x = numpy.array([1])
        if not cuda.available:
            with self.assertRaises(RuntimeError):
                cuda.to_gpu(x)

    def test_get_array_module_for_numpy_array(self):
        xp = cuda.get_array_module(numpy.array([]))
        self.assertIs(xp, numpy)
        self.assertIsNot(xp, cuda.cupy)

    def test_get_array_module_for_numpy_variable(self):
        xp = cuda.get_array_module(chainer.Variable(numpy.array([])))
        self.assertIs(xp, numpy)
        self.assertIsNot(xp, cuda.cupy)

    @attr.gpu
    def test_get_array_module_for_cupy_array(self):
        xp = cuda.get_array_module(cuda.cupy.array([]))
        self.assertIs(xp, cuda.cupy)
        self.assertIsNot(xp, numpy)

    @attr.gpu
    def test_get_array_module_for_cupy_variable(self):
        xp = cuda.get_array_module(chainer.Variable(cuda.cupy.array([])))
        self.assertIs(xp, cuda.cupy)
        self.assertIsNot(xp, numpy)

    def test_cupy_is_not_none(self):
        self.assertIsNotNone(cuda.cupy)


@testing.parameterize(
    {'c_contiguous': True},
    {'c_contiguous': False},
)
class TestToCPU(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 3))

    def test_numpy_array(self):
        y = cuda.to_cpu(self.x)
        self.assertIs(self.x, y)  # Do not copy

    @attr.gpu
    def test_cupy_array(self):
        x = cuda.to_gpu(self.x)
        if not self.c_contiguous:
            x = cuda.cupy.asfortranarray(x)
        y = cuda.to_cpu(x)
        self.assertIsInstance(y, numpy.ndarray)
        numpy.testing.assert_array_equal(self.x, y)

    @attr.multi_gpu(2)
    def test_cupy_array2(self):
        with cuda.Device(0):
            x = cuda.to_gpu(self.x)
            if not self.c_contiguous:
                x = cuda.cupy.asfortranarray(x)
        with cuda.Device(1):
            y = cuda.to_cpu(x)
        self.assertIsInstance(y, numpy.ndarray)
        numpy.testing.assert_array_equal(self.x, y)

    @attr.gpu
    def test_numpy_array_async(self):
        y = cuda.to_cpu(self.x, stream=cuda.Stream())
        self.assertIsInstance(y, numpy.ndarray)
        self.assertIs(self.x, y)  # Do not copy

    @attr.gpu
    def test_cupy_array_async1(self):
        x = cuda.to_gpu(self.x)
        if not self.c_contiguous:
            x = cuda.cupy.asfortranarray(x)
        y = cuda.to_cpu(x, stream=cuda.Stream.null)
        self.assertIsInstance(y, numpy.ndarray)
        cuda.cupy.testing.assert_array_equal(self.x, y)

    @attr.multi_gpu(2)
    def test_cupy_array_async2(self):
        x = cuda.to_gpu(self.x, device=1)
        with x.device:
            if not self.c_contiguous:
                x = cuda.cupy.asfortranarray(x)
        y = cuda.to_cpu(x, stream=cuda.Stream.null)
        self.assertIsInstance(y, numpy.ndarray)
        cuda.cupy.testing.assert_array_equal(self.x, y)

    def test_variable(self):
        x = chainer.Variable(self.x)
        with self.assertRaises(TypeError):
            cuda.to_cpu(x)


@testing.parameterize(*testing.product({
    'dtype': [
        numpy.bool_, numpy.uint8, numpy.int8, numpy.uint16,
        numpy.int16, numpy.uint32, numpy.int32, numpy.uint64,
        numpy.int64, numpy.float16, numpy.float32, numpy.float64,
        numpy.complex_],
}))
class TestToCPUScalar(unittest.TestCase):

    def test_numpy_scalar(self):
        dtype = self.dtype
        if dtype is numpy.bool_:
            x = dtype(True)
        elif issubclass(dtype, numpy.complex_):
            x = dtype(3.2 - 2.4j)
        elif issubclass(dtype, numpy.integer):
            x = dtype(3)
        elif issubclass(dtype, numpy.floating):
            x = dtype(3.2)
        else:
            assert False

        y = cuda.to_cpu(x)
        assert isinstance(y, numpy.ndarray)
        assert y.shape == ()
        assert y.dtype == dtype
        assert y == x


class TestWorkspace(unittest.TestCase):

    def setUp(self):
        self.space = cuda.get_max_workspace_size()

    def tearDown(self):
        cuda.set_max_workspace_size(self.space)

    def test_size(self):
        size = 1024
        cuda.set_max_workspace_size(size)
        self.assertEqual(size, cuda.get_max_workspace_size())


@testing.parameterize(
    {'c_contiguous': True},
    {'c_contiguous': False},
)
class TestToGPU(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 3))
        if not self.c_contiguous:
            self.x = self.x.T

    @attr.gpu
    def test_numpy_array(self):
        y = cuda.to_gpu(self.x)
        self.assertIsInstance(y, cuda.ndarray)
        cuda.cupy.testing.assert_array_equal(self.x, y)

    @attr.gpu
    def test_cupy_array1(self):
        x = cuda.to_gpu(self.x)
        y = cuda.to_gpu(x)
        self.assertIsInstance(y, cuda.ndarray)
        self.assertIs(x, y)  # Do not copy

    @attr.multi_gpu(2)
    def test_cupy_array2(self):
        x = cuda.to_gpu(self.x, device=0)
        with x.device:
            if not self.c_contiguous:
                x = cuda.cupy.asfortranarray(x)
        y = cuda.to_gpu(x, device=1)
        self.assertIsInstance(y, cuda.ndarray)
        self.assertEqual(int(y.device), 1)

    @attr.gpu
    def test_numpy_array_async(self):
        with testing.assert_warns(DeprecationWarning):
            y = cuda.to_gpu(self.x, stream=cuda.Stream.null)
        self.assertIsInstance(y, cuda.ndarray)
        cuda.cupy.testing.assert_array_equal(self.x, y)

    @attr.multi_gpu(2)
    def test_numpy_array_async2(self):
        with testing.assert_warns(DeprecationWarning):
            y = cuda.to_gpu(self.x, device=1, stream=cuda.Stream.null)
        self.assertIsInstance(y, cuda.ndarray)
        cuda.cupy.testing.assert_array_equal(self.x, y)
        self.assertEqual(int(y.device), 1)

    @attr.multi_gpu(2)
    def test_numpy_array_async3(self):
        with cuda.Device(1):
            with testing.assert_warns(DeprecationWarning):
                y = cuda.to_gpu(self.x, stream=cuda.Stream.null)
        self.assertIsInstance(y, cuda.ndarray)
        cuda.cupy.testing.assert_array_equal(self.x, y)
        self.assertEqual(int(y.device), 1)

    @attr.gpu
    def test_cupy_array_async1(self):
        x = cuda.to_gpu(self.x)
        if not self.c_contiguous:
            x = cuda.cupy.asfortranarray(x)
        with testing.assert_warns(DeprecationWarning):
            y = cuda.to_gpu(x, stream=cuda.Stream())
        self.assertIsInstance(y, cuda.ndarray)
        self.assertIs(x, y)  # Do not copy
        cuda.cupy.testing.assert_array_equal(x, y)

    @attr.multi_gpu(2)
    def test_cupy_array_async2(self):
        x = cuda.to_gpu(self.x, device=0)
        with x.device:
            if not self.c_contiguous:
                x = cuda.cupy.asfortranarray(x)
        with testing.assert_warns(DeprecationWarning):
            y = cuda.to_gpu(x, device=1, stream=cuda.Stream.null)
        self.assertIsInstance(y, cuda.ndarray)
        self.assertIsNot(x, y)  # Do copy
        cuda.cupy.testing.assert_array_equal(x, y)

    @attr.multi_gpu(2)
    def test_cupy_array_async3(self):
        with cuda.Device(0):
            x = cuda.to_gpu(self.x)
            if not self.c_contiguous:
                x = cuda.cupy.asfortranarray(x)
        with cuda.Device(1):
            with testing.assert_warns(DeprecationWarning):
                y = cuda.to_gpu(x, stream=cuda.Stream.null)
        self.assertIsInstance(y, cuda.ndarray)
        self.assertIsNot(x, y)  # Do copy
        cuda.cupy.testing.assert_array_equal(x, y)

    @attr.gpu
    def test_variable_gpu(self):
        x = chainer.Variable(self.x)
        with self.assertRaises(TypeError):
            cuda.to_gpu(x)


@testing.parameterize(*testing.product({
    'dtype': [
        numpy.bool_, numpy.uint8, numpy.int8, numpy.uint16,
        numpy.int16, numpy.uint32, numpy.int32, numpy.uint64,
        numpy.int64, numpy.float16, numpy.float32, numpy.float64,
        numpy.complex_],
}))
class TestToGPUScalar(unittest.TestCase):

    @attr.gpu
    def test_numpy_scalar(self):
        dtype = self.dtype
        if dtype is numpy.bool_:
            x = dtype(True)
        elif issubclass(dtype, numpy.complex_):
            x = dtype(3.2 - 2.4j)
        elif issubclass(dtype, numpy.integer):
            x = dtype(3)
        elif issubclass(dtype, numpy.floating):
            x = dtype(3.2)
        else:
            assert False

        y = cuda.to_gpu(x)
        assert isinstance(y, cuda.ndarray)
        assert y.shape == ()
        assert y.dtype == dtype
        assert y == x


testing.run_module(__name__, __file__)
