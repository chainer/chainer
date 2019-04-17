import copy
import os
import unittest

import numpy
import pytest

import chainer
from chainer.backends import _cpu
from chainer.link_hooks.spectral_normalization import SpectralNormalization
import chainer.links as L
from chainer import serializers
from chainer import testing
from chainer.testing import attr
from chainer.testing.backend import BackendConfig
from chainer import utils


class TestExceptions(unittest.TestCase):

    def setUp(self):
        self.x = chainer.Variable(numpy.ones((10, 5), dtype=numpy.float32))
        self.layer = L.Linear(5, 20)

    def test_wrong_weight_name(self):
        wrong_Weight_name = 'w'
        hook = SpectralNormalization(weight_name=wrong_Weight_name)
        with pytest.raises(ValueError):
            self.layer.add_hook(hook)

    def test_raises(self):
        with pytest.raises(NotImplementedError):
            with SpectralNormalization():
                self.layer(self.x)

    def test_invalid_shaped_weight(self):
        with pytest.raises(ValueError):
            L.Linear(10, 0).add_hook(SpectralNormalization())


class BaseTest(object):

    def test_add_sn_hook(self):
        layer, hook = self.layer, self.hook
        layer.add_hook(hook)
        if self.lazy_init:
            assert not hasattr(layer, hook.vector_name)
            if self.use_gamma:
                assert not hasattr(layer, 'gamma')
            with chainer.using_config('train', False):
                layer(self.x)
        assert hasattr(layer, hook.vector_name)
        assert (self.out_size,) == getattr(layer, hook.vector_name).shape
        if not self.use_gamma:
            assert not hasattr(layer, 'gamma')
        else:  # Use gamma parameter
            assert hasattr(layer, 'gamma')
            assert layer.gamma.ndim == 0 and layer.gamma.size == 1

    def _init_layer(self):
        hook = SpectralNormalization(use_gamma=self.use_gamma)
        layer = self.layer
        layer.add_hook(hook)
        if self.lazy_init:
            # Initialize weight and bias.
            with chainer.using_config('train', False):
                layer(self.x)
        return layer, hook

    def check_weight_is_parameter(self, backend_config):
        layer, hook = self._init_layer()
        layer.to_device(backend_config.device)
        source_weight = getattr(layer, hook.weight_name)
        x = backend_config.get_array(self.x)
        layer(x)
        assert getattr(layer, hook.weight_name) is source_weight

    def test_weight_is_parameter(self, backend_config):
        if not self.lazy_init:
            self.check_weight_is_parameter(backend_config)

    def check_in_recomputing(self, backend_config):
        layer, hook = self.layer, self.hook
        layer.add_hook(hook)
        layer.to_device(backend_config.device)
        x = backend_config.get_array(self.x)

        y1 = layer(x).array
        u1 = getattr(layer, hook.vector_name).copy()
        v1 = hook.v.copy()
        with chainer.using_config('in_recomputing', True):
            y2 = layer(x).array
        u2 = getattr(layer, hook.vector_name)
        v2 = hook.v

        u1, u2 = _cpu._to_cpu(u1), _cpu._to_cpu(u2)
        v1, v2 = _cpu._to_cpu(v1), _cpu._to_cpu(v2)
        numpy.testing.assert_array_equal(u1, u2)
        numpy.testing.assert_array_equal(v1, v2)
        testing.assert_allclose(y1, y2)

    def test_in_recomputing(self, backend_config):
        if not self.lazy_init:
            self.check_in_recomputing(backend_config)

    def check_deleted(self, backend_config):
        layer, hook = self.layer, self.hook
        layer.add_hook(hook)
        layer.to_device(backend_config.device)
        x = backend_config.get_array(self.x)

        with chainer.using_device(backend_config.device):
            y1 = layer(x).array
            with chainer.using_config('train', False):
                y2 = layer(x).array
            layer.delete_hook(hook.name)
            assert not hasattr(layer, hook.vector_name)
            y3 = layer(x).array
        y1, y2, y3 = _cpu._to_cpu(y1), _cpu._to_cpu(y2), _cpu._to_cpu(y3)
        assert not numpy.array_equal(y1, y3)
        assert not numpy.array_equal(y2, y3)

    def test_deleted(self, backend_config):
        self.check_deleted(backend_config)

    def check_u_updated_in_train(self, backend_config):
        layer, hook = self.layer, self.hook
        layer.add_hook(hook)
        layer.to_device(backend_config.device)
        x = backend_config.get_array(self.x)

        y1 = layer(x).array
        u1 = getattr(layer, hook.vector_name).copy()
        y2 = layer(x).array
        u2 = getattr(layer, hook.vector_name)
        y1, y2 = _cpu._to_cpu(y1), _cpu._to_cpu(y2)
        u1, u2 = _cpu._to_cpu(u1), _cpu._to_cpu(u2)
        assert not numpy.array_equal(u1, u2)
        assert not numpy.array_equal(y1, y2)

    def test_u_updated_in_train(self, backend_config):
        if not self.lazy_init:
            self.check_u_updated_in_train(backend_config)

    def check_u_not_updated_in_test(self, backend_config):
        layer, hook = self.layer, self.hook
        layer.add_hook(hook)
        layer.to_device(backend_config.device)
        x = backend_config.get_array(self.x)

        with chainer.using_config('train', False):
            y1 = layer(x).array
            u1 = getattr(layer, hook.vector_name).copy()
            v1 = hook.v.copy()
            y2 = layer(x).array
            u2 = getattr(layer, hook.vector_name)
            v2 = hook.v.copy()

        u1, u2 = _cpu._to_cpu(u1), _cpu._to_cpu(u2)
        v1, v2 = _cpu._to_cpu(v1), _cpu._to_cpu(v2)
        numpy.testing.assert_array_equal(u1, u2)
        numpy.testing.assert_array_equal(v1, v2)
        testing.assert_allclose(y1, y2)

    def test_u_not_updated_in_test(self, backend_config):
        if not self.lazy_init:
            self.check_u_not_updated_in_test(backend_config)

    def check_multi_devices_forward(self, device_0, device_1):
        layer, hook = self.layer, self.hook
        layer.add_hook(hook)
        layer.to_device(device_1)
        x = device_1.send(self.x)

        msg = None
        with chainer.using_device(device_0):
            try:
                layer(x)
            except Exception as e:
                msg = e
        assert msg is None

    @attr.chainerx
    @attr.multi_gpu(2)
    def test_forward_chx_on_multi_devices(self):
        if not self.lazy_init:
            device_0 = BackendConfig(
                {'use_chainerx': True, 'chainerx_device': 'cuda:0'}).device
            device_1 = BackendConfig(
                {'use_chainerx': True, 'chainerx_device': 'cuda:1'}).device
            self.check_multi_devices_forward(device_0, device_1)

    @attr.multi_gpu(2)
    def test_forward_multi_gpus(self):
        if not self.lazy_init:
            device_0 = BackendConfig(
                {'use_cuda': True, 'cuda_device': 0}).device
            device_1 = BackendConfig(
                {'use_cuda': True, 'cuda_device': 1}).device
            self.check_multi_devices_forward(device_0, device_1)

    def check_serialization(self, backend_config):
        with utils.tempdir() as root:
            filename = os.path.join(root, 'tmp.npz')

            layer1 = self.layer.copy('copy')
            hook1 = copy.deepcopy(self.hook)
            layer1.add_hook(hook1)

            layer1.to_device(backend_config.device)
            x = backend_config.get_array(self.x)
            with backend_config:
                layer1(x)
                with chainer.using_config('train', False):
                    y1 = layer1(x)
            serializers.save_npz(filename, layer1)

            layer2 = self.layer.copy('copy')
            hook2 = copy.deepcopy(self.hook)
            layer2.add_hook(hook2)

            # Test loading is nice.
            msg = None
            try:
                serializers.load_npz(filename, layer2)
            except Exception as e:
                msg = e
            assert msg is None

            with chainer.using_config('train', False):
                y2 = layer2(self.x.copy())

            # Test attributes are the same.
            orig_weight = _cpu._to_cpu(
                getattr(layer1, hook1.weight_name).array)
            orig_vector = _cpu._to_cpu(getattr(layer1, hook1.vector_name))
            numpy.testing.assert_array_equal(
                orig_weight, getattr(layer2, hook2.weight_name).array)
            numpy.testing.assert_array_equal(
                orig_vector, getattr(layer2, hook2.vector_name))
            testing.assert_allclose(y1.array, y2.array)

    def test_serialization(self, backend_config):
        if not self.lazy_init:
            self.check_serialization(backend_config)


_inject_backend_tests = testing.inject_backend_tests(
    ['test_weight_is_parameter', 'test_in_recomputing', 'test_deleted',
     'test_u_updated_in_train', 'test_u_not_updated_in_test',
     'test_serialization'],
    # CPU tests
    testing.product({
        'use_ideep': ['always', 'never'],
    })
    # GPU tests
    + testing.product({
        'use_cuda': [True],
    })
    # ChainerX tests
    + testing.product({
        'use_chainerx': [True],
        'chainerx_device': ['native:0', 'cuda:0'],
    })
)


@testing.parameterize(*testing.product({
    'use_gamma': [True, False],
}))
@_inject_backend_tests
class TestEmbedID(unittest.TestCase, BaseTest):

    def setUp(self):
        self.lazy_init = False  # For convenience.
        self.bs, self.in_size, self.out_size = 5, 10, 20
        self.x = numpy.arange(self.in_size, dtype=numpy.int32)
        self.layer = L.EmbedID(self.in_size, self.out_size)
        self.hook = SpectralNormalization(use_gamma=self.use_gamma)

    def test_add_sn_hook(self):
        hook = SpectralNormalization(use_gamma=self.use_gamma)
        layer = self.layer
        layer.add_hook(hook)
        if self.lazy_init:
            assert not hasattr(layer, hook.vector_name)
            if self.use_gamma:
                assert not hasattr(layer, 'gamma')
            with chainer.using_config('train', False):
                layer(self.x)
        assert hasattr(layer, hook.vector_name)
        assert (self.in_size,) == getattr(layer, hook.vector_name).shape
        if not self.use_gamma:
            assert not hasattr(layer, 'gamma')
        else:  # Use gamma parameter
            assert hasattr(layer, 'gamma')
            assert layer.gamma.ndim == 0 and layer.gamma.size == 1


@testing.parameterize(*testing.product({
    'lazy_init': [True, False],
    'use_gamma': [True, False],
}))
@_inject_backend_tests
class TestLinear(unittest.TestCase, BaseTest):

    def setUp(self):
        self.bs, self.in_size, self.out_size = 10, 20, 30
        self.x = numpy.random.normal(
            size=(self.bs, self.in_size)).astype(numpy.float32)
        self.layer = L.Linear(self.out_size)  # Lazy initialization
        in_size = None if self.lazy_init else self.in_size
        self.layer = L.Linear(in_size, self.out_size)
        self.hook = SpectralNormalization(use_gamma=self.use_gamma)


@testing.parameterize(*testing.product({
    'use_gamma': [True, False],
    'lazy_init': [True, False],
    'link': [L.Convolution1D, L.Deconvolution1D],
}))
@_inject_backend_tests
class TestConvolution1D(unittest.TestCase, BaseTest):

    def setUp(self):
        self.in_channels, self.out_channels = 3, 10
        in_channels = None if self.lazy_init else self.in_channels
        conv_init_args = {'ksize': 3, 'stride': 1, 'pad': 1}
        self.layer = self.link(
            in_channels, self.out_channels, **conv_init_args)
        self.x = numpy.random.normal(
            size=(5, self.in_channels, 4)).astype(numpy.float32)
        self.hook = SpectralNormalization(use_gamma=self.use_gamma)
        self.out_size = self.out_channels  # For compatibility


@testing.parameterize(*testing.product({
    'use_gamma': [True, False],
    'lazy_init': [True, False],
    'link': [L.Convolution2D, L.Deconvolution2D],
}))
@_inject_backend_tests
class TestConvolution2D(unittest.TestCase, BaseTest):

    def setUp(self):
        self.in_channels, self.out_channels = 3, 10
        in_channels = None if self.lazy_init else self.in_channels
        conv_init_args = {'ksize': 3, 'stride': 1, 'pad': 1}
        self.layer = self.link(
            in_channels, self.out_channels, **conv_init_args)
        self.x = numpy.random.normal(
            size=(5, self.in_channels, 4, 4)).astype(numpy.float32)
        self.hook = SpectralNormalization(use_gamma=self.use_gamma)
        self.out_size = self.out_channels  # For compatibility


@testing.parameterize(*testing.product({
    'use_gamma': [True, False],
    'lazy_init': [True, False],
    'link': [L.Convolution3D, L.Deconvolution3D],
}))
@_inject_backend_tests
class TestConvolution3D(unittest.TestCase, BaseTest):

    def setUp(self):
        self.in_channels, self.out_channels = 3, 10
        in_channels = None if self.lazy_init else self.in_channels
        conv_init_args = {'ksize': 3, 'stride': 1, 'pad': 1}
        self.layer = self.link(
            in_channels, self.out_channels, **conv_init_args)
        self.x = numpy.random.normal(
            size=(5, self.in_channels, 4, 4, 4)).astype(numpy.float32)
        self.hook = SpectralNormalization(use_gamma=self.use_gamma)
        self.out_size = self.out_channels  # For compatibility


testing.run_module(__name__, __file__)
