import os
import tempfile
import unittest

import numpy

import chainer
from chainer import backend
import chainer.links as L
from chainer.link_hooks.spectral_normalization import SpectralNormalization
from chainer.serializers import npz
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product({
    'lazy_init': [True, False],
    'use_gamma': [True, False],
}))
class TestSpectralNormalizationLinkHook(unittest.TestCase):

    """Test initialization and serialization with Linear."""

    def setUp(self):
        self.in_size, self.out_size = 10, 20
        if self.lazy_init:
            self.layer = L.Linear(self.out_size)
        else:
            self.layer = L.Linear(self.in_size, self.out_size)
        self.x = numpy.random.uniform(
            size=(5, self.in_size)).astype(numpy.float32)

    def test_vector_initialization(self):
        hook = SpectralNormalization(use_gamma=self.use_gamma)
        layer = self.layer
        layer.add_hook(hook)
        if self.lazy_init:
            assert not hasattr(layer, hook.vector_name)
            layer(self.x)
        self.assertTupleEqual((1, self.out_size),
                              tuple(getattr(layer, hook.vector_name).shape))
        self.assertTrue(isinstance(getattr(layer, hook.weight_name),
                                   chainer.variable.Parameter))
        if self.use_gamma:
            assert hasattr(layer, 'gamma')

    def test_remove_hook(self):
        hook = SpectralNormalization(use_gamma=self.use_gamma)
        layer = self.layer
        layer.add_hook(hook)
        layer(self.x)

        prev_weight = getattr(layer, hook.weight_name)
        layer.delete_hook(hook.name)
        cur_weight = getattr(layer, hook.weight_name)

        self.assertTrue((prev_weight.array == cur_weight.array).all())
        assert not hasattr(layer, hook.vector_name)
        if self.use_gamma:
            assert not hasattr(layer, 'gamma')

    def test_serialization(self):
        hook = SpectralNormalization(use_gamma=self.use_gamma)
        snlayer = self.layer
        snlayer.add_hook(hook)
        if self.lazy_init:
            snlayer(self.x)
        fd, temp_file_path = tempfile.mkstemp()
        os.close(fd)
        npz.save_npz(temp_file_path, snlayer)
        lin2 = L.Linear(self.out_size)
        lin2.add_hook(hook)
        npz.load_npz(temp_file_path, lin2)
        self.assertTrue((snlayer.W.array == lin2.W.array).all())


@testing.parameterize(*testing.product({
    'link': [L.Convolution2D, L.Deconvolution2D],
}))
class TestSNConv(unittest.TestCase):

    def setUp(self):
        self.in_channels, self.out_channels, self.ksize = 3, 10, 3
        layer = self.link(self.in_channels, self.out_channels, self.ksize)
        hook = SpectralNormalization()
        layer.add_hook(hook)
        self.layer = layer
        self.x = numpy.random.normal(
            size=(5, self.in_channels, 4, 4)).astype(numpy.float32)
        self.vector_name = hook.vector_name

    def test_forward(self):
        for _ in range(3):
            self.layer(self.x)
        y_train = self.layer(self.x)
        with chainer.using_config('train', False):
            y_test = self.layer(self.x)
        numpy.testing.assert_equal(y_train.array, y_test.array)


@testing.parameterize(*testing.product({
    'link': [L.Convolution3D, L.DeconvolutionND],
}))
class TestSNConvND(unittest.TestCase):

    def setUp(self):
        self.in_channels, self.out_channels, self.ksize = 3, 10, 3
        layer = self.link(self.in_channels, self.out_channels, self.ksize)
        hook = SpectralNormalization()
        layer.add_hook(hook)
        self.layer = layer
        self.x = numpy.random.normal(
            size=(5, self.in_channels, 4, 4, 4)).astype(numpy.float32)
        self.vector_name = hook.vector_name

    def test_forward(self):
        for _ in range(3):
            self.layer(self.x)
        y_train = self.layer(self.x)
        with chainer.using_config('train', False):
            y_test = self.layer(self.x)
        numpy.testing.assert_equal(y_train.array, y_test.array)


testing.run_module(__name__, __file__)
