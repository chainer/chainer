import unittest

import chainer
from chainer import backend
from chainer import initializers
from chainer import testing
import numpy


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
@testing.backend.inject_backend_tests(
    None,
    [
        {},
        {'use_ideep': 'always'},
        {'use_cuda': True, 'cuda_device': 0},
        {'use_cuda': True, 'cuda_device': 1},
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:1'},
    ]
)
class TestIdentity(unittest.TestCase):

    scale = 0.1
    shape = (2, 2)

    def setUp(self):
        self.check_options = {}
        if self.dtype == numpy.float16:
            self.check_options = {'atol': 1e-4, 'rtol': 1e-3}

    def check_initializer(self, w):
        initializer = initializers.Identity(scale=self.scale)
        initializer(w)
        testing.assert_allclose(
            w, self.scale * numpy.identity(len(self.shape)),
            **self.check_options)

    def test_initializer(self, backend_config):
        w = numpy.empty(self.shape, dtype=self.dtype)
        w = backend_config.get_array(w)
        with chainer.using_device(backend_config.device):
            self.check_initializer(w)

    def check_shaped_initializer(self, backend_config):
        initializer = initializers.Identity(
            scale=self.scale, dtype=self.dtype)
        xp = backend_config.xp
        w = initializers.generate_array(initializer, self.shape, xp)
        self.assertIs(backend.get_array_module(w), xp)
        self.assertTupleEqual(w.shape, self.shape)
        self.assertEqual(w.dtype, self.dtype)
        testing.assert_allclose(
            w, self.scale * numpy.identity(len(self.shape)),
            **self.check_options)

    def test_shaped_initializer(self, backend_config):
        with chainer.using_device(backend_config.device):
            self.check_shaped_initializer(backend_config)


@testing.parameterize(
    {'shape': (2, 3)},
    {'shape': (2, 2, 4)},
    {'shape': ()},
    {'shape': 0})
class TestIdentityInvalid(unittest.TestCase):

    def setUp(self):
        self.initializer = initializers.Identity()

    def test_invalid_shape(self):
        w = numpy.empty(self.shape, dtype=numpy.float32)
        with self.assertRaises(ValueError):
            self.initializer(w)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
@testing.backend.inject_backend_tests(
    None,
    [
        {},
        {'use_ideep': 'always'},
        {'use_cuda': True, 'cuda_device': 0},
        {'use_cuda': True, 'cuda_device': 1},
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:1'},
    ]
)
class TestConstant(unittest.TestCase):

    fill_value = 0.1
    shape = (2, 3)

    def setUp(self):
        self.check_options = {}
        if self.dtype == numpy.float16:
            self.check_options = {'atol': 1e-4, 'rtol': 1e-3}

    def check_initializer(self, w):
        initializer = initializers.Constant(fill_value=self.fill_value)
        initializer(w)
        testing.assert_allclose(
            w, numpy.full(self.shape, self.fill_value),
            **self.check_options)

    def test_initializer(self, backend_config):
        w = numpy.empty(self.shape, dtype=self.dtype)
        w = backend_config.get_array(w)
        with chainer.using_device(backend_config.device):
            self.check_initializer(w)

    def check_shaped_initializer(self, backend_config):
        initializer = initializers.Constant(
            fill_value=self.fill_value, dtype=self.dtype)
        xp = backend_config.xp
        w = initializers.generate_array(initializer, self.shape, xp)
        self.assertIs(backend.get_array_module(w), xp)
        self.assertTupleEqual(w.shape, self.shape)
        self.assertEqual(w.dtype, self.dtype)
        testing.assert_allclose(
            w, numpy.full(self.shape, self.fill_value),
            **self.check_options)

    def test_shaped_initializer(self, backend_config):
        with chainer.using_device(backend_config.device):
            self.check_shaped_initializer(backend_config)


testing.run_module(__name__, __file__)
