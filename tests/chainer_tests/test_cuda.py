import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings

import numpy
import pytest
import six

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import testing
from chainer.testing import attr
import chainer.testing.backend


class TestDummyDeviceType(unittest.TestCase):

    def test_int(self):
        assert int(cuda.DummyDeviceType()) == -1

    def test_eq(self):
        assert cuda.DummyDeviceType() == cuda.DummyDeviceType()

    def test_ne(self):
        assert cuda.DummyDeviceType() != 1


_builtins_available = False
try:
    import builtins
    _builtins_available = True
except ImportError:
    pass


class TestCudaModuleAliasForBackwardCompatibility(unittest.TestCase):

    def _check(self, code):
        temp_dir = tempfile.mkdtemp()
        try:
            script_path = os.path.join(temp_dir, 'script.py')
            with open(script_path, 'w') as f:
                f.write(code)
            proc = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
            stdoutdata, stderrdata = proc.communicate()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        ret = proc.returncode
        assert ret == 0, (
            'Import test failed.\n'
            '[code]:\n{}\n'
            '[stdout]:{!r}\n'
            '[stderr]:{!r}'.format(
                code, stdoutdata, stderrdata))

    def test_import1(self):
        self._check('from chainer import cuda; cuda.get_device_from_id')

    def test_import2(self):
        self._check('import chainer.cuda; chainer.cuda.get_device_from_id')

    def test_import3(self):
        self._check('import chainer; chainer.cuda.get_device_from_id')


class TestCuda(unittest.TestCase):

    def test_get_dummy_device(self):
        assert cuda.get_device_from_id(None) is cuda.DummyDevice

    @attr.gpu
    def test_get_device_from_id_for_numpy_int(self):
        assert cuda.get_device_from_id(numpy.int64(0)) == cuda.Device(0)

    def test_get_device_from_array_for_numpy_int(self):
        assert cuda.get_device_from_array(numpy.int64(0)) is cuda.DummyDevice

    @attr.gpu
    def test_get_device_for_empty_array(self):
        x = cuda.get_device_from_array(cuda.cupy.array([]).reshape((0, 10)))
        # TODO(okuta): Only check `assert x == cuda.Device(0)`
        #              when cupy/cupy#946 is merged
        assert x == cuda.Device(0) or x == cuda.DummyDevice

    @attr.gpu
    @unittest.skipUnless(
        six.PY3, 'Python2.7 has a bug in catch_warnings, so this test is '
                 'skipped for Python2.7')
    def test_get_device_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            cuda.get_device(cuda.cupy.array([1]))

        assert len(w) == 1
        assert w[0].category is DeprecationWarning
        assert ('get_device is deprecated. Please use get_device_from_id'
                ' or get_device_from_array instead.' in str(w[0].message))

    @attr.gpu
    def test_get_device_from_id(self):
        assert cuda.get_device_from_id(0) == cuda.Device(0)

    @attr.gpu
    def test_get_device_from_array(self):
        arr = cuda.cupy.array([0])
        assert cuda.get_device_from_array(arr) == cuda.Device(0)

    @attr.gpu
    def test_get_device_for_int(self):
        with testing.assert_warns(DeprecationWarning):
            device = cuda.get_device(0)
        assert device == cuda.Device(0)

    @attr.gpu
    @unittest.skipUnless(_builtins_available,
                         'builtins module is not available')
    def test_get_device_from_id_for_builtin_int(self):
        # builtins.int is from future package and it is different
        # from builtin int/long on Python 2.
        assert cuda.get_device_from_id(builtins.int(0)) == cuda.Device(0)

    @attr.gpu
    @unittest.skipUnless(_builtins_available,
                         'builtins module is not available')
    def test_get_device_for_builtin_int(self):
        # builtins.int is from future package and it is different
        # from builtin int/long on Python 2.
        with testing.assert_warns(DeprecationWarning):
            device = cuda.get_device(builtins.int(0))
        assert device == cuda.Device(0)

    @attr.gpu
    def test_get_device_for_device(self):
        device = cuda.get_device_from_id(0)
        with testing.assert_warns(DeprecationWarning):
            assert cuda.get_device(device) is device

    def test_to_gpu_unavailable(self):
        x = numpy.array([1])
        if not cuda.available:
            with self.assertRaises(RuntimeError):
                cuda.to_gpu(x)

    def test_cupy_is_not_none(self):
        assert cuda.cupy is not None


@testing.parameterize(
    {'c_contiguous': True},
    {'c_contiguous': False},
)
class TestToCPU(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 3))

    def test_numpy_array(self):
        y = cuda.to_cpu(self.x)
        assert self.x is y  # Do not copy

    @attr.gpu
    def test_cupy_array(self):
        x = cuda.to_gpu(self.x)
        if not self.c_contiguous:
            x = cuda.cupy.asfortranarray(x)
        y = cuda.to_cpu(x)
        assert isinstance(y, numpy.ndarray)
        numpy.testing.assert_array_equal(self.x, y)

    @attr.multi_gpu(2)
    def test_cupy_array2(self):
        with cuda.Device(0):
            x = cuda.to_gpu(self.x)
            if not self.c_contiguous:
                x = cuda.cupy.asfortranarray(x)
        with cuda.Device(1):
            y = cuda.to_cpu(x)
        assert isinstance(y, numpy.ndarray)
        numpy.testing.assert_array_equal(self.x, y)

    @attr.gpu
    def test_numpy_array_async(self):
        y = cuda.to_cpu(self.x, stream=cuda.Stream())
        assert isinstance(y, numpy.ndarray)
        assert self.x is y  # Do not copy

    @attr.gpu
    def test_cupy_array_async1(self):
        x = cuda.to_gpu(self.x)
        if not self.c_contiguous:
            x = cuda.cupy.asfortranarray(x)
        y = cuda.to_cpu(x, stream=cuda.Stream.null)
        assert isinstance(y, numpy.ndarray)
        cuda.cupy.testing.assert_array_equal(self.x, y)

    @attr.multi_gpu(2)
    def test_cupy_array_async2(self):
        x = cuda.to_gpu(self.x, device=1)
        with x.device:
            if not self.c_contiguous:
                x = cuda.cupy.asfortranarray(x)
        y = cuda.to_cpu(x, stream=cuda.Stream.null)
        assert isinstance(y, numpy.ndarray)
        cuda.cupy.testing.assert_array_equal(self.x, y)

    def test_single_none(self):
        assert cuda.to_cpu(None) is None

    def _check_list_tuple(self, typ):
        assert typ in (list, tuple)
        a = numpy.random.uniform(-1, 1, (0,))
        b = numpy.random.uniform(-1, 1, (2, 3))
        c = cuda.cupy.random.uniform(-1, 1, (0,))
        d = cuda.cupy.random.uniform(-1, 1, (2, 2))
        xs = typ([a, b, c, d, None, a, b, None, c, d])
        xs_cpu = cuda.to_cpu(xs)

        assert isinstance(xs_cpu, typ)
        assert len(xs) == len(xs_cpu)
        for i in (0, 1, 2, 3, 5, 6, 8, 9):
            assert isinstance(xs_cpu[i], numpy.ndarray)
            cuda.cupy.testing.assert_array_equal(xs[i], xs_cpu[i])
        assert xs_cpu[0] is a
        assert xs_cpu[1] is b
        assert xs_cpu[2] is xs_cpu[8]
        assert xs_cpu[3] is xs_cpu[9]
        assert xs_cpu[4] is None
        assert xs_cpu[5] is a
        assert xs_cpu[6] is b
        assert xs_cpu[7] is None

    @attr.gpu
    def test_list(self):
        self._check_list_tuple(list)

    @attr.gpu
    def test_tuple(self):
        self._check_list_tuple(tuple)

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


@attr.cudnn
class TestWorkspace(unittest.TestCase):

    def setUp(self):
        self.space = cuda.get_max_workspace_size()

    def tearDown(self):
        cuda.set_max_workspace_size(self.space)

    def test_size(self):
        size = 1024
        cuda.set_max_workspace_size(size)
        assert size == cuda.get_max_workspace_size()


@testing.parameterize(*(testing.product({
    'c_contiguous': [True],
    'device_dtype': [int, numpy.uint8, numpy.int8, numpy.uint16,
                     numpy.int16, numpy.uint32, numpy.int32, numpy.uint64,
                     numpy.int64]
}) + testing.product({
    'c_contiguous': [False],
    'device_dtype': [int]
}))
)
class TestToGPU(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 3))
        if not self.c_contiguous:
            self.x = self.x.T

    @attr.gpu
    def test_numpy_array(self):
        y = cuda.to_gpu(self.x)
        assert isinstance(y, cuda.ndarray)
        cuda.cupy.testing.assert_array_equal(self.x, y)

    @attr.gpu
    def test_cupy_array1(self):
        x = cuda.to_gpu(self.x)
        y = cuda.to_gpu(x)
        assert isinstance(y, cuda.ndarray)
        assert x is y  # Do not copy

    @attr.multi_gpu(2)
    def test_cupy_array2(self):
        x = cuda.to_gpu(self.x, device=self.device_dtype(0))
        with x.device:
            if not self.c_contiguous:
                x = cuda.cupy.asfortranarray(x)
        y = cuda.to_gpu(x, device=self.device_dtype(1))
        assert isinstance(y, cuda.ndarray)
        assert int(y.device) == 1

    @attr.gpu
    def test_numpy_array_async(self):
        with testing.assert_warns(DeprecationWarning):
            y = cuda.to_gpu(self.x, stream=cuda.Stream.null)
        assert isinstance(y, cuda.ndarray)
        cuda.cupy.testing.assert_array_equal(self.x, y)

    @attr.multi_gpu(2)
    def test_numpy_array_async2(self):
        with testing.assert_warns(DeprecationWarning):
            y = cuda.to_gpu(self.x, device=self.device_dtype(1),
                            stream=cuda.Stream.null)
        assert isinstance(y, cuda.ndarray)
        cuda.cupy.testing.assert_array_equal(self.x, y)
        assert int(y.device) == 1

    @attr.multi_gpu(2)
    def test_numpy_array_async3(self):
        with cuda.Device(1):
            with testing.assert_warns(DeprecationWarning):
                y = cuda.to_gpu(self.x, stream=cuda.Stream.null)
        assert isinstance(y, cuda.ndarray)
        cuda.cupy.testing.assert_array_equal(self.x, y)
        assert int(y.device) == 1

    @attr.gpu
    def test_cupy_array_async1(self):
        x = cuda.to_gpu(self.x)
        if not self.c_contiguous:
            x = cuda.cupy.asfortranarray(x)
        with testing.assert_warns(DeprecationWarning):
            y = cuda.to_gpu(x, stream=cuda.Stream())
        assert isinstance(y, cuda.ndarray)
        assert x is y  # Do not copy
        cuda.cupy.testing.assert_array_equal(x, y)

    @attr.multi_gpu(2)
    def test_cupy_array_async2(self):
        x = cuda.to_gpu(self.x, device=self.device_dtype(0))
        with x.device:
            if not self.c_contiguous:
                x = cuda.cupy.asfortranarray(x)
        with testing.assert_warns(DeprecationWarning):
            y = cuda.to_gpu(x, device=self.device_dtype(1),
                            stream=cuda.Stream.null)
        assert isinstance(y, cuda.ndarray)
        assert x is not y  # Do copy
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
        assert isinstance(y, cuda.ndarray)
        assert x is not y  # Do copy
        cuda.cupy.testing.assert_array_equal(x, y)

    @attr.gpu
    def test_single_none(self):
        assert cuda.to_gpu(None) is None

    def _check_list_tuple(self, typ):
        assert typ in (list, tuple)
        a = numpy.random.uniform(-1, 1, (0,))
        b = numpy.random.uniform(-1, 1, (2, 3))
        c = cuda.cupy.random.uniform(-1, 1, (0,))
        d = cuda.cupy.random.uniform(-1, 1, (2, 2))
        xs = typ([a, b, c, d, None, a, b, None, c, d])
        xs_gpu = cuda.to_gpu(xs)

        assert isinstance(xs_gpu, typ)
        assert len(xs) == len(xs_gpu)
        for i in (0, 1, 2, 3, 5, 6, 8, 9):
            assert isinstance(xs_gpu[i], cuda.cupy.ndarray)
            cuda.cupy.testing.assert_array_equal(xs[i], xs_gpu[i])
        assert xs_gpu[0] is xs_gpu[5]
        assert xs_gpu[1] is xs_gpu[6]
        assert xs_gpu[2] is c
        assert xs_gpu[3] is d
        assert xs_gpu[4] is None
        assert xs_gpu[7] is None
        assert xs_gpu[8] is c
        assert xs_gpu[9] is d

    @attr.gpu
    def test_list(self):
        self._check_list_tuple(list)

    @attr.gpu
    def test_tuple(self):
        self._check_list_tuple(tuple)

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


@testing.backend.inject_backend_tests(
    None,
    [
        {'use_cuda': True, 'cuda_device': 0},
        {'use_cuda': True, 'cuda_device': 1},
    ])
class TestGpuDeviceFromArray(unittest.TestCase):

    def test_from_array(self, backend_config):
        with cuda.Device(backend_config.cuda_device):
            arr = cuda.ndarray((), numpy.float32)
        # Test precondition check
        assert arr.device.id == backend_config.cuda_device

        device = backend.GpuDevice.from_array(arr)
        assert isinstance(device, backend.GpuDevice)
        assert (device
                == chainer.get_device((cuda.cupy, backend_config.cuda_device)))

    def test_get_device_from_array(self, backend_config):
        with cuda.Device(backend_config.cuda_device):
            arr = cuda.ndarray((), numpy.float32)
        # Test precondition check
        assert arr.device.id == backend_config.cuda_device

        expected_device = backend_config.device

        device = backend.GpuDevice.from_array(arr)
        assert device == expected_device

        device = backend.get_device_from_array(arr)
        assert device == expected_device


@testing.backend.inject_backend_tests(
    None,
    [
        {},
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
    ])
@attr.gpu
class TestGpuDeviceFromArrayInvalidArray(unittest.TestCase):

    def test_from_array(self, backend_config):
        arr = backend_config.get_array(numpy.ndarray((2,), numpy.float32))
        device = backend.GpuDevice.from_array(arr)
        assert device is None


@testing.parameterize(*testing.product(
    {
        'value': [None, 1, ()],
    }))
@attr.gpu
class TestGpuDeviceFromArrayInvalidValue(unittest.TestCase):

    def test_from_array(self):
        device = backend.GpuDevice.from_array(self.value)
        assert device is None


@testing.parameterize(*testing.product(
    {
        'device_id': [0, 1, 99999, numpy.int32(1)],
    }))
@attr.gpu
class TestGpuDeviceFromDeviceId(unittest.TestCase):

    def test_from_device_id(self):
        device = backend.GpuDevice.from_device_id(self.device_id)
        assert isinstance(device, backend.GpuDevice)
        assert device == chainer.get_device((cuda.cupy, self.device_id))
        assert device.device.id == int(self.device_id)


@testing.parameterize(*testing.product(
    {
        'device_id': [None, -1, (), 0.0, numpy.float32(0)],
    }))
@attr.gpu
class TestGpuDeviceFromDeviceIdInvalid(unittest.TestCase):

    def test_from_device_id(self):
        with pytest.raises(ValueError):
            backend.GpuDevice.from_device_id(self.device_id)


@testing.backend.inject_backend_tests(
    None,
    [
        {'use_cuda': True, 'cuda_device': 0},
        {'use_cuda': True, 'cuda_device': 1},
    ])
class TestGpuDeviceUse(unittest.TestCase):

    def test_use(self, backend_config):
        device = chainer.get_device((cuda.cupy, backend_config.cuda_device))
        with cuda.Device(0):
            device.use()
            assert device.device == cuda.Device()


testing.run_module(__name__, __file__)
