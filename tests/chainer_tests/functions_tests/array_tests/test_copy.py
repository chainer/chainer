import unittest

import numpy
import pytest

import chainer
from chainer import backend
from chainer.backends import _cpu
from chainer.backends import cuda
from chainer import functions
from chainer import testing
import chainerx


def _to_gpu(x, device_id):
    if device_id >= 0:
        return cuda.to_gpu(x, device_id)
    else:
        return x


_nonchainerx_backend_configs = (
    [
        # NumPy
        {},
        # CuPy
        {'use_cuda': True, 'cuda_device': 0},
        {'use_cuda': True, 'cuda_device': 1},
    ])


_chainerx_backend_configs = (
    [
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:1'},
    ])

_numpy_device = chainer.get_device('@numpy')


class CopyTestBase(object):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (10, 5)).astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1, (10, 5)).astype(self.dtype)
        self.ggx = numpy.random.uniform(-1, 1, (10, 5)).astype(self.dtype)
        self.check_double_backward_options = {}
        if self.dtype == numpy.float16:
            self.check_double_backward_options = {'atol': 5e-3, 'rtol': 5e-2}

    def check_forward(self, dst_device_spec, src_device, dst_device):
        x = src_device.send(self.x)

        x_var = chainer.Variable(x)
        y = functions.copy(x_var, dst_device_spec)

        assert y.device == dst_device
        assert backend.get_device_from_array(y.array) == dst_device
        assert y.dtype == self.dtype
        numpy.testing.assert_array_equal(_numpy_device.send(y.array), self.x)

    def test_forward(self, src_backend_config, dst_backend_config):
        self.check_forward(
            dst_backend_config.device,
            src_backend_config.device,
            dst_backend_config.device)

    def test_backward(self, src_backend_config, dst_backend_config):
        x = src_backend_config.get_array(self.x)
        gy = dst_backend_config.get_array(self.gy)
        src_device = src_backend_config.device
        dst_device = dst_backend_config.device

        x_var = chainer.Variable(x, requires_grad=True)

        y_var = functions.copy(x_var, dst_device)
        y_var.grad = gy

        y_var.backward()

        x_grad = x_var.grad
        assert x_var.grad_var.device == src_device
        assert backend.get_device_from_array(x_grad) == src_device
        numpy.testing.assert_array_equal(_numpy_device.send(x_grad), self.gy)

    def test_double_backward(self, src_backend_config, dst_backend_config):
        x = src_backend_config.get_array(self.x)
        gy = dst_backend_config.get_array(self.gy)
        ggx = src_backend_config.get_array(self.ggx)
        dst_device = dst_backend_config.device

        x_var = chainer.Variable(x, requires_grad=True)

        y_var = functions.copy(x_var, dst_device)

        y_var.grad = gy

        gy_var = y_var.grad_var
        y_var.backward(enable_double_backprop=True)

        assert x_var.grad_var.requires_grad is True

        x_var.grad_var.grad = ggx
        x_var.grad_var.backward()

        assert gy_var.grad_var.device == dst_device
        assert (
            backend.get_device_from_array(gy_var.grad_var.array)
            == dst_device)
        numpy.testing.assert_array_equal(
            _numpy_device.send(gy_var.grad_var.array), self.ggx)


@testing.inject_backend_tests(None, _nonchainerx_backend_configs)
@testing.inject_backend_tests(None, _nonchainerx_backend_configs)
@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestCopyNonChainerx(CopyTestBase, unittest.TestCase):

    def test_forward_int(self, src_backend_config, dst_backend_config):
        src_device = src_backend_config.device
        dst_device = dst_backend_config.device
        if dst_device.xp is numpy:
            dst_device_spec = -1
        elif dst_device.xp is chainer.backends.cuda.cupy:
            dst_device_spec = dst_device.device.id
        else:
            assert False, dst_device

        self.check_forward(
            dst_device_spec,
            src_device,
            dst_device)

    def test_forward_str(self, src_backend_config, dst_backend_config):
        src_device = src_backend_config.device
        dst_device = dst_backend_config.device
        if dst_device.xp is numpy:
            dst_device_spec = '@numpy'
        elif dst_device.xp is chainer.backends.cuda.cupy:
            dst_device_spec = '@cupy:{}'.format(dst_device.device.id)
        else:
            assert False, dst_device

        self.check_forward(
            dst_device_spec,
            src_device,
            dst_device)


@testing.inject_backend_tests(None, _chainerx_backend_configs)
@testing.inject_backend_tests(None, _chainerx_backend_configs)
@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestCopyChainerx(CopyTestBase, unittest.TestCase):

    def test_forward_str(self, src_backend_config, dst_backend_config):
        src_device = src_backend_config.device
        dst_device = dst_backend_config.device
        dst_device_spec = dst_device.device.name

        self.check_forward(
            dst_device_spec,
            src_device,
            dst_device)


@testing.inject_backend_tests(None, _chainerx_backend_configs)
@testing.inject_backend_tests(None, _nonchainerx_backend_configs)
class TestCopyBetweenChainerxAndNonChainerx(unittest.TestCase):
    # Copy between non-ChainerX and ChainerX devices are not supported.

    dtype = numpy.float32

    def check_invalid(self, src_device, dst_device_spec):
        x = src_device.send(
            numpy.random.uniform(-1, 1, (10, 5)).astype(self.dtype))

        x_var = chainer.Variable(x)

        with pytest.raises(RuntimeError):
            functions.copy(x_var, dst_device_spec)

    def test_invalid(self, nonchx_backend_config, chx_backend_config):
        assert nonchx_backend_config.xp is not chainerx
        assert chx_backend_config.xp is chainerx

        self.check_invalid(
            nonchx_backend_config.device, chx_backend_config.device)
        self.check_invalid(
            chx_backend_config.device, nonchx_backend_config.device)
        # cuda.DummyDevice is not supported either.
        self.check_invalid(
            chx_backend_config.device, cuda.DummyDevice)


@testing.inject_backend_tests(None, _nonchainerx_backend_configs)
@testing.inject_backend_tests(None, _nonchainerx_backend_configs)
class TestCopyCudaDummyDevice(unittest.TestCase):

    def test_dummy_device(self, src_backend_config, current_backend_config):

        x_arr = src_backend_config.get_array(numpy.zeros((2, 3)))

        with current_backend_config:
            y = functions.copy(x_arr, cuda.DummyDevice)

        # Always transferred to NumPy device, regardless of the current CUDA
        # device.
        assert isinstance(y.device, _cpu.CpuDevice)


testing.run_module(__name__, __file__)
