import unittest

import numpy
import pytest

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer.backends import intel64
from chainer import testing
from chainer.testing import attr
import chainerx
import chainerx.testing


class _TestCopyToBase(object):

    src_data = numpy.arange(1, 5, dtype=numpy.float32)
    dst_data = numpy.zeros_like(src_data)

    def _get_dst(self):
        raise NotImplementedError

    def test_from_cpu(self):
        src = self.src_data
        dst = self._get_dst()
        backend.copyto(dst, src)
        numpy.testing.assert_array_equal(cuda.to_cpu(dst), self.src_data)

    @attr.gpu
    def test_from_gpu(self):
        src = cuda.cupy.array(self.src_data)
        dst = self._get_dst()
        backend.copyto(dst, src)
        numpy.testing.assert_array_equal(cuda.to_cpu(dst), self.src_data)

    @attr.ideep
    def test_from_ideep(self):
        src = intel64.ideep.array(self.src_data)
        dst = self._get_dst()
        assert isinstance(src, intel64.mdarray)
        backend.copyto(dst, src)
        numpy.testing.assert_array_equal(cuda.to_cpu(dst), self.src_data)


class TestCopyToCPU(_TestCopyToBase, unittest.TestCase):
    def _get_dst(self):
        return self.dst_data


@attr.gpu
class TestCopyToGPU(_TestCopyToBase, unittest.TestCase):
    def _get_dst(self):
        return cuda.cupy.array(self.dst_data)

    @attr.multi_gpu(2)
    def test_gpu_to_another_gpu(self):
        src = cuda.cupy.array(self.src_data)
        with cuda.get_device_from_id(1):
            dst = self._get_dst()
        backend.copyto(dst, src)
        cuda.cupy.testing.assert_array_equal(dst, src)


@attr.ideep
class TestCopyToIDeep(_TestCopyToBase, unittest.TestCase):
    def _get_dst(self):
        dst = intel64.ideep.array(self.src_data)
        assert isinstance(dst, intel64.mdarray)
        return dst


class TestCopyToError(unittest.TestCase):
    def test_fail_on_invalid_src(self):
        src = None
        dst = numpy.zeros(1)
        with self.assertRaises(TypeError):
            backend.copyto(dst, src)

    def test_fail_on_invalid_dst(self):
        src = numpy.zeros(1)
        dst = None
        with self.assertRaises(TypeError):
            backend.copyto(dst, src)


class TestGetArrayModule(unittest.TestCase):

    def test_get_array_module_for_numpy_array(self):
        xp = backend.get_array_module(numpy.array([]))
        self.assertIs(xp, numpy)
        assert xp is not cuda.cupy
        assert xp is not chainerx

    def test_get_array_module_for_numpy_variable(self):
        xp = backend.get_array_module(chainer.Variable(numpy.array([])))
        assert xp is numpy
        assert xp is not cuda.cupy
        assert xp is not chainerx

    @attr.gpu
    def test_get_array_module_for_cupy_array(self):
        xp = backend.get_array_module(cuda.cupy.array([]))
        assert xp is cuda.cupy
        assert xp is not numpy
        assert xp is not chainerx

    @attr.gpu
    def test_get_array_module_for_cupy_variable(self):
        xp = backend.get_array_module(chainer.Variable(cuda.cupy.array([])))
        assert xp is cuda.cupy
        assert xp is not numpy
        assert xp is not chainerx

    @attr.chainerx
    def test_get_array_module_for_chainerx_array(self):
        xp = backend.get_array_module(chainerx.array([]))
        assert xp is chainerx
        assert xp is not numpy
        assert xp is not cuda.cupy

    @attr.chainerx
    def test_get_array_module_for_chainerx_variable(self):
        xp = backend.get_array_module(chainer.Variable(chainerx.array([])))
        assert xp is chainerx
        assert xp is not numpy
        assert xp is not cuda.cupy


class TestGetDeviceFromArray(unittest.TestCase):

    def test_numpy_int(self):
        device = chainer.backend.get_device_from_array(numpy.int64(0))
        assert isinstance(device, chainer.backends.cpu.CpuDevice)
        assert device.xp is numpy

    def test_numpy_array(self):
        device = chainer.backend.get_device_from_array(numpy.array([0]))
        assert isinstance(device, chainer.backends.cpu.CpuDevice)
        assert device.xp is numpy

    @attr.gpu
    def test_empty_cupy_array(self):
        arr = cuda.cupy.array([]).reshape((0, 10))
        device = chainer.backend.get_device_from_array(arr)
        assert isinstance(device, chainer.backends.cuda.GpuDevice)
        assert device.xp is cuda.cupy
        assert device.device == cuda.Device(0)

    @attr.gpu
    def test_cupy_array(self):
        device = chainer.backend.get_device_from_array(cuda.cupy.array([0]))
        assert isinstance(device, chainer.backends.cuda.GpuDevice)
        assert device.xp is cuda.cupy
        assert device.device == cuda.Device(0)

    @attr.chainerx
    def test_chainerx_cpu_array(self):
        arr = chainer.backend.to_chainerx(numpy.array([0]))
        device = chainer.backend.get_device_from_array(arr)
        assert isinstance(device, chainer.backends.chainerx.ChainerxDevice)
        assert device.xp is chainerx
        assert device.device == chainerx.get_device('native:0')

    @attr.chainerx
    @attr.gpu
    def test_chainerx_gpu_array(self):
        arr = chainer.backend.to_chainerx(cuda.cupy.array([0]))
        device = chainer.backend.get_device_from_array(arr)
        assert isinstance(device, chainer.backends.chainerx.ChainerxDevice)
        assert device.xp is chainerx
        assert device.device == chainerx.get_device('cuda:0')


class TestToBackend(unittest.TestCase):

    def orig_numpy(self):
        return numpy.ones((2, 3), numpy.float32)

    def orig_cupy(self):
        arr = cuda.to_gpu(numpy.ones((2, 3), numpy.float32))
        assert isinstance(arr, cuda.ndarray)
        return arr

    def orig_chainerx(self, device_name):
        return chainerx.ones((2, 3), numpy.float32, device=device_name)

    def assert_array_equal(self, array1, array2):
        # Convert CuPy to NumPy
        if isinstance(array1, cuda.ndarray):
            array1 = array1.get()
        if isinstance(array2, cuda.ndarray):
            array1 = array2.get()

        # At this moment arrays are either NumPy or ChainerX

        if (isinstance(array1, numpy.ndarray)
                and isinstance(array2, numpy.ndarray)):
            numpy.testing.assert_array_equal(array1, array2)
        else:
            chainerx.testing.assert_array_equal(array1, array2)

    def to_numpy_check_equal(self, orig):
        converted = backend.to_numpy(orig)
        assert isinstance(converted, numpy.ndarray)
        self.assert_array_equal(orig, converted)
        return converted

    def to_chainerx_check_equal(self, orig):
        converted = backend.to_chainerx(orig)
        assert isinstance(converted, chainerx.ndarray)
        self.assert_array_equal(orig, converted)
        return converted

    def test_numpy_to_numpy(self):
        orig = self.orig_numpy()
        converted = self.to_numpy_check_equal(orig)
        assert converted is orig

    @attr.gpu
    def test_cupy_to_numpy(self):
        orig = self.orig_cupy()
        self.to_numpy_check_equal(orig)

    @attr.chainerx
    def test_numpy_to_chainerx(self):
        orig = self.orig_numpy()
        converted = self.to_chainerx_check_equal(orig)
        assert converted.device is chainerx.get_device('native:0')

        # memory must be shared
        orig[:] *= 2
        self.assert_array_equal(orig, converted)

    @attr.chainerx
    @attr.gpu
    def test_cupy_to_chainerx(self):
        orig = self.orig_cupy()
        converted = self.to_chainerx_check_equal(orig)
        assert converted.device == chainerx.get_device('cuda:0')

        # memory must be shared
        orig[:] *= 2
        self.assert_array_equal(orig, converted)

    @attr.chainerx
    def test_chainerx_to_chainerx(self):
        orig = self.orig_chainerx('native:0')
        converted = self.to_chainerx_check_equal(orig)
        assert converted is orig

    # TODO(niboshi): Add more test variants


class TestGetDevice(unittest.TestCase):
    """Test for backend.Device.__init__() and backend.get_device()"""

    def check_device_spec_numpy(self, device_spec):
        device = backend.get_device(device_spec)
        assert isinstance(device, chainer.backends.cpu.CpuDevice)
        assert device.xp is numpy

    def check_device_spec_cupy(self, device_spec, expected_device_id):
        device = backend.get_device(device_spec)
        assert isinstance(device, cuda.GpuDevice)
        assert isinstance(device.device, cuda.Device)
        assert device.xp is cuda.cupy
        assert device.device.id == expected_device_id

    def check_device_spec_chainerx(self, device_spec, expected_device_name):
        device = backend.get_device(device_spec)
        assert isinstance(device, chainer.backends.chainerx.ChainerxDevice)
        assert device.xp is chainerx
        assert isinstance(device.device, chainerx.Device)
        assert device.device.name == expected_device_name

    @attr.ideep
    def check_device_spec_intel64(self, device_spec):
        device = backend.get_device(device_spec)
        assert isinstance(device, chainer.backends.intel64.Intel64Device)
        assert device.xp is numpy

    def check_invalid(self, device_spec):
        with pytest.raises(Exception):
            backend.get_device(device_spec)

    def test_module_numpy(self):
        self.check_device_spec_numpy(numpy)

    def test_module_numpy_device(self):
        self.check_device_spec_numpy(chainer.backends.cpu.CpuDevice())

    @attr.chainerx
    def test_str_chainerx_backend(self):
        self.check_device_spec_chainerx('native', 'native:0')

    @attr.chainerx
    def test_str_chainerx_device(self):
        self.check_device_spec_chainerx('native:0', 'native:0')

    @attr.chainerx
    def test_tuple_chainerx_device(self):
        self.check_device_spec_chainerx(('native', 0), 'native:0')

    @attr.gpu
    def test_tuple_cupy_device(self):
        self.check_device_spec_cupy((cuda.cupy, 0), 0)

    @attr.chainerx
    def test_chainerx_device(self):
        chainerx_device = chainerx.get_device('native:0')
        self.check_device_spec_chainerx(chainerx_device, 'native:0')

    @attr.gpu
    def test_cuda_device(self):
        cupy_device = cuda.Device(0)
        self.check_device_spec_cupy(cupy_device, 0)

    @attr.ideep
    def test_intel64(self):
        self.check_device_spec_intel64(intel64)

    def test_str_chainerx_invalid(self):
        self.check_invalid('native:foo')

    def test_tuple_chainerx_invalid(self):
        self.check_invalid(('native', 'foo'))

    def test_tuple_cupy_invalid_device(self):
        self.check_invalid((cuda.cupy, 'foo'))


class TestDevice(unittest.TestCase):

    def test_repr_module_numpy(self):
        device = chainer.get_device(numpy)
        assert str(device) == '<CpuDevice (numpy)>'

    @attr.chainerx
    def test_repr_tuple_chainerx_device(self):
        device = chainer.get_device(('native', 0))
        assert str(device) == '<ChainerxDevice native:0>'

    @attr.gpu
    def test_repr_tuple_cupy_device(self):
        device = chainer.get_device((cuda.cupy, 0))
        assert str(device) == '<GpuDevice (cupy):0>'

    @attr.ideep
    def test_repr_tuple_intel64_device(self):
        device = chainer.get_device(intel64)
        assert str(device) == '<Intel64Device>'

    def test_eq_numpy(self):
        assert backend.get_device(numpy) == backend.get_device(numpy)
        assert (backend.get_device(chainer.backends.cpu.CpuDevice())
                == backend.get_device(numpy))

    @attr.gpu
    def test_eq_cupy(self):
        assert (backend.get_device((cuda.cupy, 0))
                != backend.get_device(numpy))
        assert (backend.get_device((cuda.cupy, 0))
                == backend.get_device((cuda.cupy, 0)))
        assert (backend.get_device((cuda.cupy, 0))
                != backend.get_device((cuda.cupy, 1)))

    @attr.chainerx
    def test_eq_chainerx(self):
        assert backend.get_device('native:0') == backend.get_device('native:0')
        assert backend.get_device('native:0') != backend.get_device('native:1')

    @attr.chainerx
    @attr.gpu
    def test_eq_chainerx_cupy(self):
        assert (
            backend.get_device('native:0')
            != backend.get_device((cuda.cupy, 0)))


class TestDeviceSend(unittest.TestCase):

    def orig_numpy(self):
        return numpy.ones((2, 3), numpy.float32)

    def orig_cupy(self):
        arr = cuda.to_gpu(numpy.ones((2, 3), numpy.float32))
        assert isinstance(arr, cuda.ndarray)
        return arr

    def orig_chainerx(self, device_name):
        return chainerx.ones((2, 3), numpy.float32, device=device_name)

    def send_check_equal(self, orig, device_spec):
        device = backend.get_device(device_spec)
        converted = device.send(orig)
        numpy.testing.assert_array_equal(
            backend.to_numpy(orig),
            backend.to_numpy(converted))
        return converted

    def test_numpy_to_numpy(self):
        orig = self.orig_numpy()
        converted = self.send_check_equal(orig, numpy)
        assert converted is orig

    @attr.gpu
    def test_numpy_to_cupy(self):
        orig = self.orig_numpy()
        converted = self.send_check_equal(orig, (cuda.cupy, 0))
        assert isinstance(converted, cuda.ndarray)
        assert converted.device == cuda.Device(0)

    @attr.chainerx
    def test_numpy_to_chainerx(self):
        orig = self.orig_numpy()
        converted = self.send_check_equal(orig, 'native:0')
        assert isinstance(converted, chainerx.ndarray)
        assert converted.device.name == 'native:0'

        # memory must be shared
        orig[:] *= 2
        numpy.testing.assert_array_equal(
            orig, backend.to_numpy(converted))

    @attr.chainerx
    @attr.gpu
    def test_numpy_to_chainerx_cuda(self):
        orig = self.orig_numpy()
        converted = self.send_check_equal(orig, 'cuda:0')
        assert isinstance(converted, chainerx.ndarray)
        assert converted.device.name == 'cuda:0'

    @attr.gpu
    def test_cupy_to_numpy(self):
        orig = self.orig_cupy()
        converted = self.send_check_equal(orig, numpy)
        assert isinstance(converted, numpy.ndarray)

    @attr.gpu
    def test_cupy_to_cupy(self):
        orig = self.orig_cupy()
        converted = self.send_check_equal(orig, (cuda.cupy, 0))
        assert isinstance(converted, cuda.ndarray)
        assert converted.device == orig.device

        # memory must be shared
        orig[:] *= 2
        numpy.testing.assert_array_equal(
            backend.to_numpy(orig), backend.to_numpy(converted))

    @attr.chainerx
    @attr.gpu
    def test_cupy_to_chainerx(self):
        orig = self.orig_cupy()
        converted = self.send_check_equal(orig, 'cuda:0')
        assert isinstance(converted, chainerx.ndarray)
        assert converted.device.name == 'cuda:0'

        # memory must be shared
        orig[:] *= 2
        numpy.testing.assert_array_equal(
            backend.to_numpy(orig), backend.to_numpy(converted))

    @attr.multi_gpu(2)
    def test_cupy_to_cupy_multigpu(self):
        orig = self.orig_cupy()
        converted = self.send_check_equal(orig, (cuda.cupy, 1))
        assert isinstance(converted, cuda.ndarray)
        assert converted.device.id == 1

    @attr.chainerx
    def test_cupy_to_chainerx_native(self):
        orig = self.orig_cupy()
        converted = self.send_check_equal(orig, 'native:0')
        assert isinstance(converted, chainerx.ndarray)
        assert converted.device.name == 'native:0'

    @attr.chainerx
    @attr.multi_gpu(2)
    def test_cupy_to_chainerx_multigpu(self):
        orig = self.orig_cupy()
        converted = self.send_check_equal(orig, 'cuda:1')
        assert isinstance(converted, chainerx.ndarray)
        assert converted.device.name == 'cuda:1'

    @attr.chainerx
    def test_chainerx_native_to_numpy(self):
        orig = self.orig_chainerx('native:0')
        converted = self.send_check_equal(orig, numpy)
        assert isinstance(converted, numpy.ndarray)

        # memory must be shared
        converted[:] *= 2
        numpy.testing.assert_array_equal(
            backend.to_numpy(orig), backend.to_numpy(converted))

    @attr.chainerx
    @attr.gpu
    def test_chainerx_cuda_to_cupy(self):
        orig = self.orig_chainerx('cuda:0')
        converted = self.send_check_equal(orig, (cuda.cupy, 0))
        assert isinstance(converted, cuda.ndarray)
        assert converted.device.id == 0

        # memory must be shared
        converted[:] *= 2
        numpy.testing.assert_array_equal(
            backend.to_numpy(orig), backend.to_numpy(converted))

    @attr.chainerx
    @attr.multi_gpu(2)
    def test_chainerx_cuda_to_cupy_multigpu(self):
        orig = self.orig_chainerx('cuda:0')
        converted = self.send_check_equal(orig, (cuda.cupy, 1))
        assert isinstance(converted, cuda.ndarray)
        assert converted.device.id == 1

        # memory must not be shared
        converted_copy = converted.copy()
        with cuda.Device(1):
            converted[:] *= 2
        numpy.testing.assert_array_equal(
            backend.to_numpy(orig), backend.to_numpy(converted_copy))

    @attr.chainerx
    @attr.gpu
    def test_chainerx_cuda_to_numpy(self):
        orig = self.orig_chainerx('cuda:0')
        converted = self.send_check_equal(orig, numpy)
        assert isinstance(converted, numpy.ndarray)

    def test_numpy_to_numpy_with_device(self):
        orig = self.orig_numpy()
        self.send_check_equal(
            orig, chainer.get_device(chainer.backends.cpu.CpuDevice()))


testing.run_module(__name__, __file__)
