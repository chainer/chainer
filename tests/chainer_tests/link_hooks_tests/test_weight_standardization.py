import unittest

import numpy
import pytest

import chainer
from chainer.backends import cuda
from chainer.link_hooks.weight_standardization import WeightStandardization
import chainer.links as L
from chainer import testing
from chainer.testing import attr


class TestExceptions(unittest.TestCase):

    def setUp(self):
        self.x = chainer.Variable(numpy.ones((10, 5), dtype=numpy.float32))
        self.layer = L.Linear(5, 20)

    def test_wrong_weight_name(self):
        wrong_Weight_name = 'w'
        hook = WeightStandardization(weight_name=wrong_Weight_name)
        with pytest.raises(ValueError):
            self.layer.add_hook(hook)

    def test_raises(self):
        with pytest.raises(NotImplementedError):
            with WeightStandardization():
                self.layer(self.x)


class BaseTest(object):

    def test_add_ws_hook(self):
        layer, hook = self.layer, self.hook
        layer.add_hook(hook)
        if self.lazy_init:
            with chainer.using_config('train', False):
                layer(self.x)

    def _init_layer(self):
        hook = WeightStandardization()
        layer = self.layer
        layer.add_hook(hook)
        if self.lazy_init:
            # Initialize weight and bias.
            with chainer.using_config('train', False):
                layer(self.x)
        return layer, hook

    def check_weight_is_parameter(self, gpu):
        layer, hook = self._init_layer()
        if gpu:
            layer = layer.to_gpu()
        source_weight = getattr(layer, hook.weight_name)
        x = cuda.to_gpu(self.x) if gpu else self.x
        layer(x)
        assert getattr(layer, hook.weight_name) is source_weight

    def test_weight_is_parameter_cpu(self):
        if not self.lazy_init:
            self.check_weight_is_parameter(False)

    @attr.gpu
    def test_weight_is_parameter_gpu(self):
        if not self.lazy_init:
            self.check_weight_is_parameter(True)

    def check_deleted(self, gpu):
        layer, hook = self.layer, self.hook
        layer.add_hook(hook)
        if gpu:
            layer = layer.to_gpu()
        x = cuda.to_gpu(self.x) if gpu else self.x

        y1 = layer(x).array
        with chainer.using_config('train', False):
            y2 = layer(x).array
        layer.delete_hook(hook.name)
        y3 = layer(x).array
        if gpu:
            y1, y2, y3 = cuda.to_cpu(y1), cuda.to_cpu(y2), cuda.to_cpu(y3)
        assert not numpy.array_equal(y1, y3)
        assert not numpy.array_equal(y2, y3)

    def test_deleted_cpu(self):
        self.check_deleted(False)

    @attr.gpu
    def test_deleted_gpu(self):
        self.check_deleted(True)


class TestEmbedID(unittest.TestCase, BaseTest):

    def setUp(self):
        self.lazy_init = False  # For convenience.
        self.bs, self.in_size, self.out_size = 5, 10, 20
        self.x = numpy.arange(self.in_size, dtype=numpy.int32)
        self.layer = L.EmbedID(self.in_size, self.out_size)
        self.hook = WeightStandardization()

    def test_add_ws_hook(self):
        hook = WeightStandardization()
        layer = self.layer
        layer.add_hook(hook)
        if self.lazy_init:
            with chainer.using_config('train', False):
                layer(self.x)


@testing.parameterize(*testing.product({
    'lazy_init': [True, False],
}))
class TestLinear(unittest.TestCase, BaseTest):

    def setUp(self):
        self.bs, self.in_size, self.out_size = 10, 20, 30
        self.x = numpy.random.normal(
            size=(self.bs, self.in_size)).astype(numpy.float32)
        self.layer = L.Linear(self.out_size)  # Lazy initialization
        in_size = None if self.lazy_init else self.in_size
        self.layer = L.Linear(in_size, self.out_size)
        self.hook = WeightStandardization()


@testing.parameterize(*testing.product({
    'lazy_init': [True, False],
    'link': [L.Convolution1D],
}))
class TestConvolution1D(unittest.TestCase, BaseTest):

    def setUp(self):
        self.in_channels, self.out_channels = 3, 10
        in_channels = None if self.lazy_init else self.in_channels
        conv_init_args = {'ksize': 3, 'stride': 1, 'pad': 1}
        self.layer = self.link(
            in_channels, self.out_channels, **conv_init_args)
        self.x = numpy.random.normal(
            size=(5, self.in_channels, 4)).astype(numpy.float32)
        self.hook = WeightStandardization()
        self.out_size = self.out_channels  # For compatibility


@testing.parameterize(*testing.product({
    'lazy_init': [True, False],
    'link': [L.Convolution2D],
}))
class TestConvolution2D(unittest.TestCase, BaseTest):

    def setUp(self):
        self.in_channels, self.out_channels = 3, 10
        in_channels = None if self.lazy_init else self.in_channels
        conv_init_args = {'ksize': 3, 'stride': 1, 'pad': 1}
        self.layer = self.link(
            in_channels, self.out_channels, **conv_init_args)
        self.x = numpy.random.normal(
            size=(5, self.in_channels, 4, 4)).astype(numpy.float32)
        self.hook = WeightStandardization()
        self.out_size = self.out_channels  # For compatibility


@testing.parameterize(*testing.product({
    'lazy_init': [True, False],
    'link': [L.Convolution3D],
}))
class TestConvolution3D(unittest.TestCase, BaseTest):

    def setUp(self):
        self.in_channels, self.out_channels = 3, 10
        in_channels = None if self.lazy_init else self.in_channels
        conv_init_args = {'ksize': 3, 'stride': 1, 'pad': 1}
        self.layer = self.link(
            in_channels, self.out_channels, **conv_init_args)
        self.x = numpy.random.normal(
            size=(5, self.in_channels, 4, 4, 4)).astype(numpy.float32)
        self.hook = WeightStandardization()
        self.out_size = self.out_channels  # For compatibility


testing.run_module(__name__, __file__)
