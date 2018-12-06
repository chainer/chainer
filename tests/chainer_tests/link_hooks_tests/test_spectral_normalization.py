import unittest

import numpy

import chainer
import chainer.links as L
from chainer.link_hooks.spectral_normalization import SpectralNormalization
from chainer import testing


class TestExceptions(unittest.TestCase):

    def setUp(self):
        self.x = chainer.Variable(numpy.ones((10, 5), dtype=numpy.float32))
        self.layer = L.Linear(5, 20)

    def test_wrong_weight_name(self):
        wrong_Weight_name = 'w'
        hook = SpectralNormalization(weight_name=wrong_Weight_name)
        with self.assertRaises(ValueError):
            self.layer.add_hook(hook)

    def test_raises(self):
        with self.assertRaises(NotImplementedError):
            with SpectralNormalization():
                self.layer(self.x)


@testing.parameterize(*testing.product({
    'use_gamma': [True, False],
}))
class TestEmbedID(unittest.TestCase):

    def setUp(self):
        self.bs, self.in_size, self.out_size = 5, 10, 20
        self.x = numpy.arange(self.in_size, dtype=numpy.int32)
        self.layer = L.EmbedID(self.in_size, self.out_size)
        self.hook = SpectralNormalization(use_gamma=self.use_gamma)

    def test_add_sn_hook(self):
        layer, hook = self.layer, self.hook
        layer.add_hook(hook)

        self.assertTrue(hook._initialied)
        if self.use_gamma:
            self.assertTrue(hasattr(layer, 'gamma'))
            self.assertIsInstance(getattr(layer, 'gamma'), chainer.Parameter)
        self.assertTupleEqual(
            getattr(layer, hook.vector_name).shape,
            (1, self.in_size)
        )

    def test_weight_is_parameter(self):
        layer, hook = self.layer, self.hook
        layer.add_hook(hook)
        source_weight = getattr(layer, hook.weight_name)
        layer(self.x)
        self.assertIs(getattr(layer, hook.weight_name), source_weight)

    def test_u_updated_in_train(self):
        layer, hook = self.layer, self.hook
        layer.add_hook(hook)

        y1 = layer(self.x).array
        u1 = numpy.copy(getattr(layer, hook.vector_name))
        y2 = layer(self.x).array
        u2 = getattr(layer, hook.vector_name)
        self.assertFalse((u1 == u2).all())
        self.assertFalse((y1 == y2).all())

    def test_u_not_updated_in_test(self):
        layer, hook = self.layer, self.hook
        layer.add_hook(hook)

        with chainer.using_config('train', False):
            y1 = layer(self.x).array
            u1 = numpy.copy(getattr(layer, hook.vector_name))
            y2 = layer(self.x).array
            u2 = getattr(layer, hook.vector_name)

        numpy.testing.assert_equal(u1, u2)
        self.assertTrue((y1 == y2).all())

    def test_deleted(self):
        layer, hook = self.layer, self.hook
        layer.add_hook(hook)

        y1 = layer(self.x).array
        with chainer.using_config('train', False):
            y2 = layer(self.x).array
        layer.delete_hook(hook.name)
        self.assertFalse(hasattr(layer, hook.vector_name))
        y3 = layer(self.x).array
        self.assertFalse((y1 == y3).all())
        self.assertFalse((y2 == y3).all())


@testing.parameterize(*testing.product({
    'lazy_init': [True, False],
    'use_gamma': [True, False],
}))
class TestLinear(unittest.TestCase):

    def setUp(self):
        self.bs, self.in_size, self.out_size = 10, 20, 30
        self.x = numpy.random.normal(
            size=(self.bs, self.in_size)).astype(numpy.float32)
        self.layer = L.Linear(self.out_size)  # Lazy initialization
        in_size = None if self.lazy_init else self.in_size
        self.layer = L.Linear(in_size, self.out_size)

    def test_add_sn_hook(self):
        hook = SpectralNormalization(use_gamma=self.use_gamma)
        layer = self.layer
        layer.add_hook(hook)
        if self.lazy_init:
            self.assertFalse(hasattr(layer, hook.vector_name))
            if self.use_gamma:
                self.assertFalse(hasattr(layer, 'gamma'))
            with chainer.using_config('train', False), chainer.no_backprop_mode():
                layer(self.x)
        self.assertTrue(hasattr(layer, hook.vector_name))
        self.assertTupleEqual(
            (1, self.out_size), getattr(layer, hook.vector_name).shape)
        if not self.use_gamma:
            self.assertFalse(hasattr(layer, 'gamma'))
        else:  # Use gamma parameter
            self.assertTrue(hasattr(layer, 'gamma'))
            self.assertEqual(
                getattr(layer, 'gamma').ndim,
                getattr(layer, hook.weight_name).ndim
            )

    def _init_layer(self):
        hook = SpectralNormalization(use_gamma=self.use_gamma)
        layer = self.layer
        layer.add_hook(hook)
        if self.lazy_init:
            with chainer.using_config('train', False), chainer.no_backprop_mode():
                layer(self.x)
        return layer, hook

    def test_weight_is_parameter(self):
        if not self.lazy_init:
            layer, hook = self._init_layer()
            source_weight = getattr(layer, hook.weight_name)
            layer(self.x)
            self.assertIs(getattr(layer, hook.weight_name), source_weight)
        else:
            pass

    def test_deleted(self):
        if not self.lazy_init:
            layer, hook = self._init_layer()
            layer.delete_hook(hook.name)

            self.assertFalse(hasattr(layer, hook.vector_name))
            if self.use_gamma:
                self.assertFalse(hasattr(layer, 'gamma'))
        else:
            pass

    def test_u_updated_in_train(self):
        if not self.lazy_init:
            layer, hook = self._init_layer()
            u1 = numpy.copy(getattr(layer, hook.vector_name))
            layer(self.x)
            u2 = getattr(layer, hook.vector_name)
            self.assertFalse((u1 == u2).all())
        else:
            pass

    def test_u_not_updated_in_test(self):
        if not self.lazy_init:
            layer, hook = self._init_layer()
            u = getattr(layer, hook.vector_name)
            with chainer.using_config('train', False):
                layer(self.x)
            self.assertTrue((u == getattr(layer, hook.vector_name)).all())
        else:
            pass


@testing.parameterize(*testing.product({
    'use_gamma': [True, False],
    'lazy_init': [True, False],
    'link': [L.Convolution1D, L.Deconvolution1D],
}))
class TestConvolution1D(unittest.TestCase):

    def setUp(self):
        self.in_channels, self.out_channels = 3, 10
        in_channels = None if self.lazy_init else self.in_channels
        conv_init_args = {'ksize': 3, 'stride': 1, 'pad': 1}
        self.layer = self.link(in_channels, self.out_channels, **conv_init_args)
        self.x = numpy.random.normal(
            size=(5, self.in_channels, 4)).astype(numpy.float32)
        self.hook = SpectralNormalization(use_gamma=self.use_gamma)

    def _init_layer(self):
        layer, hook = self.layer, self.hook
        layer.add_hook(hook)
        return layer, hook

    def test_add_sn_hook(self):
        hook = SpectralNormalization(use_gamma=self.use_gamma)
        layer = self.layer
        layer.add_hook(hook)
        if self.lazy_init:
            self.assertFalse(hasattr(layer, hook.vector_name))
            if self.use_gamma:
                self.assertFalse(hasattr(layer, 'gamma'))
            with chainer.using_config('train', False), chainer.no_backprop_mode():
                layer(self.x)
        self.assertTrue(hasattr(layer, hook.vector_name))
        self.assertTupleEqual(
            (1, self.out_channels), getattr(layer, hook.vector_name).shape)
        if not self.use_gamma:
            self.assertFalse(hasattr(layer, 'gamma'))
        else:  # Use gamma parameter
            self.assertTrue(hasattr(layer, 'gamma'))
            self.assertEqual(
                getattr(layer, 'gamma').ndim,
                getattr(layer, hook.weight_name).ndim
            )

    def test_deleted(self):
        if not self.lazy_init:
            layer, hook = self._init_layer()
            layer.delete_hook(hook.name)

            self.assertFalse(hasattr(layer, hook.vector_name))
            if self.use_gamma:
                self.assertFalse(hasattr(layer, 'gamma'))
        else:
            pass

    def test_u_updated_in_train(self):
        if not self.lazy_init:
            layer, hook = self._init_layer()
            u1 = numpy.copy(getattr(layer, hook.vector_name))
            layer(self.x)
            u2 = getattr(layer, hook.vector_name)
            self.assertFalse((u1 == u2).all())
        else:
            pass

    def test_u_not_updated_in_test(self):
        if not self.lazy_init:
            layer, hook = self._init_layer()
            u = getattr(layer, hook.vector_name)
            with chainer.using_config('train', False):
                layer(self.x)
            self.assertTrue((u == getattr(layer, hook.vector_name)).all())
        else:
            pass


@testing.parameterize(*testing.product({
    'use_gamma': [True, False],
    'lazy_init': [True, False],
    'link': [L.Convolution2D, L.Deconvolution2D],
}))
class TestConvolution2D(unittest.TestCase):

    def setUp(self):
        self.in_channels, self.out_channels = 3, 10
        in_channels = None if self.lazy_init else self.in_channels
        conv_init_args = {'ksize': 3, 'stride': 1, 'pad': 1}
        self.layer = self.link(in_channels, self.out_channels, **conv_init_args)
        self.x = numpy.random.normal(
            size=(5, self.in_channels, 4, 4)).astype(numpy.float32)
        self.hook = SpectralNormalization(use_gamma=self.use_gamma)

    def _init_layer(self):
        layer, hook = self.layer, self.hook
        layer.add_hook(hook)
        return layer, hook

    def test_add_sn_hook(self):
        hook = SpectralNormalization(use_gamma=self.use_gamma)
        layer = self.layer
        layer.add_hook(hook)
        if self.lazy_init:
            self.assertFalse(hasattr(layer, hook.vector_name))
            if self.use_gamma:
                self.assertFalse(hasattr(layer, 'gamma'))
            with chainer.using_config('train', False), chainer.no_backprop_mode():
                layer(self.x)
        self.assertTrue(hasattr(layer, hook.vector_name))
        self.assertTupleEqual(
            (1, self.out_channels), getattr(layer, hook.vector_name).shape)
        if not self.use_gamma:
            self.assertFalse(hasattr(layer, 'gamma'))
        else:  # Use gamma parameter
            self.assertTrue(hasattr(layer, 'gamma'))
            self.assertEqual(
                getattr(layer, 'gamma').ndim,
                getattr(layer, hook.weight_name).ndim
            )

    def test_deleted(self):
        if not self.lazy_init:
            layer, hook = self._init_layer()
            layer.delete_hook(hook.name)

            self.assertFalse(hasattr(layer, hook.vector_name))
            if self.use_gamma:
                self.assertFalse(hasattr(layer, 'gamma'))
        else:
            pass

    def test_u_updated_in_train(self):
        if not self.lazy_init:
            layer, hook = self._init_layer()
            u1 = numpy.copy(getattr(layer, hook.vector_name))
            layer(self.x)
            u2 = getattr(layer, hook.vector_name)
            self.assertFalse((u1 == u2).all())
        else:
            pass

    def test_u_not_updated_in_test(self):
        if not self.lazy_init:
            layer, hook = self._init_layer()
            u = getattr(layer, hook.vector_name)
            with chainer.using_config('train', False):
                layer(self.x)
            self.assertTrue((u == getattr(layer, hook.vector_name)).all())
        else:
            pass


@testing.parameterize(*testing.product({
    'use_gamma': [True, False],
    'lazy_init': [True, False],
    'link': [L.Convolution3D, L.Deconvolution3D],
}))
class TestConvolution3D(unittest.TestCase):

    def setUp(self):
        self.in_channels, self.out_channels = 3, 10
        in_channels = None if self.lazy_init else self.in_channels
        conv_init_args = {'ksize': 3, 'stride': 1, 'pad': 1}
        self.layer = self.link(in_channels, self.out_channels, **conv_init_args)
        self.x = numpy.random.normal(
            size=(5, self.in_channels, 4, 4, 4)).astype(numpy.float32)
        self.hook = SpectralNormalization(use_gamma=self.use_gamma)

    def _init_layer(self):
        layer = self.layer
        hook = SpectralNormalization()
        layer.add_hook(hook)
        return layer, hook

    def test_add_sn_hook(self):
        hook = SpectralNormalization(use_gamma=self.use_gamma)
        layer = self.layer
        layer.add_hook(hook)
        if self.lazy_init:
            self.assertFalse(hasattr(layer, hook.vector_name))
            if self.use_gamma:
                self.assertFalse(hasattr(layer, 'gamma'))
            with chainer.using_config('train', False), chainer.no_backprop_mode():
                layer(self.x)
        self.assertTrue(hasattr(layer, hook.vector_name))
        self.assertTupleEqual(
            (1, self.out_channels), getattr(layer, hook.vector_name).shape)
        if not self.use_gamma:
            self.assertFalse(hasattr(layer, 'gamma'))
        else:  # Use gamma parameter
            self.assertTrue(hasattr(layer, 'gamma'))
            self.assertEqual(
                getattr(layer, 'gamma').ndim,
                getattr(layer, hook.weight_name).ndim
            )

    def test_deleted(self):
        if not self.lazy_init:
            layer, hook = self._init_layer()
            layer.delete_hook(hook.name)

            self.assertFalse(hasattr(layer, hook.vector_name))
            if self.use_gamma:
                self.assertFalse(hasattr(layer, 'gamma'))
        else:
            pass

    def test_u_updated_in_train(self):
        if not self.lazy_init:
            layer, hook = self._init_layer()
            u1 = numpy.copy(getattr(layer, hook.vector_name))
            layer(self.x)
            u2 = getattr(layer, hook.vector_name)
            self.assertFalse((u1 == u2).all())
        else:
            pass

    def test_u_not_updated_in_test(self):
        if not self.lazy_init:
            layer, hook = self._init_layer()
            u = getattr(layer, hook.vector_name)
            with chainer.using_config('train', False):
                layer(self.x)
            self.assertTrue((u == getattr(layer, hook.vector_name)).all())
        else:
            pass


testing.run_module(__name__, __file__)
