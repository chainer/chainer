import unittest

import numpy

from chainer import backend
from chainer.backends import cuda
from chainer import initializers
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product_dict(
    [
        {'shape': (), 'dim_out': 1},
        {'shape': (1,), 'dim_out': 1},
        {'shape': (3, 4), 'dim_out': 3},
        {'shape': (3, 4, 5), 'dim_out': 3}
    ],
    [
        {'scale': 2., 'dtype': numpy.float16}
    ] + testing.product({
        'scale': [None, 7.3],
        'dtype': [numpy.float32, numpy.float64],
    })
))
class OrthogonalBase(unittest.TestCase):

    def setUp(self):
        kwargs = {}
        if self.scale is not None:
            kwargs['scale'] = self.scale
        self.target_kwargs = kwargs

        self.check_options = {}
        if self.dtype == numpy.float16:
            self.check_options = {'atol': 5e-3, 'rtol': 5e-2}

    def check_initializer(self, w):
        initializer = initializers.Orthogonal(**self.target_kwargs)
        initializer(w)
        self.assertTupleEqual(w.shape, self.shape)
        self.assertEqual(w.dtype, self.dtype)

    def test_initializer_cpu(self):
        w = numpy.empty(self.shape, dtype=self.dtype)
        self.check_initializer(w)

    @attr.gpu
    def test_initializer_gpu(self):
        w = cuda.cupy.empty(self.shape, dtype=self.dtype)
        self.check_initializer(w)

    def check_shaped_initializer(self, xp):
        initializer = initializers.Orthogonal(
            dtype=self.dtype, **self.target_kwargs)
        w = initializers.generate_array(initializer, self.shape, xp)
        self.assertIs(backend.get_array_module(w), xp)
        self.assertTupleEqual(w.shape, self.shape)
        self.assertEqual(w.dtype, self.dtype)

    def test_shaped_initializer_cpu(self):
        self.check_shaped_initializer(numpy)

    @attr.gpu
    def test_shaped_initializer_gpu(self):
        self.check_shaped_initializer(cuda.cupy)

    def check_orthogonality(self, w):
        initializer = initializers.Orthogonal(**self.target_kwargs)
        initializer(w)
        n = self.dim_out
        w = w.astype(numpy.float64).reshape(n, -1)
        dots = w.dot(w.T)
        expected_scale = self.scale or 1.1
        testing.assert_allclose(
            dots, numpy.identity(n) * expected_scale**2,
            **self.check_options)

    def test_orthogonality_cpu(self):
        w = numpy.empty(self.shape, dtype=self.dtype)
        self.check_orthogonality(w)

    @attr.gpu
    def test_orthogonality_gpu(self):
        w = cuda.cupy.empty(self.shape, dtype=self.dtype)
        self.check_orthogonality(w)


class TestEmpty(unittest.TestCase):

    def setUp(self):
        self.w = numpy.empty(0, dtype=numpy.float32)
        self.initializer = initializers.Orthogonal()

    def check_assert(self, w):
        with self.assertRaises(ValueError):
            self.initializer(w)

    def test_cpu(self):
        self.check_assert(self.w)

    @attr.gpu
    def test_gpu(self):
        self.check_assert(cuda.to_gpu(self.w))


@testing.parameterize(
    {'shape': (4, 3)},
    {'shape': (21, 4, 5)})
class TestOverComplete(unittest.TestCase):

    def setUp(self):
        self.w = numpy.empty(self.shape, dtype=numpy.float32)
        self.initializer = initializers.Orthogonal(scale=1.0)

    def check_invalid(self, w):
        with self.assertRaises(ValueError):
            self.initializer(w)

    def test_invalid_cpu(self):
        self.check_invalid(self.w)

    @attr.gpu
    def test_invalid_gpu(self):
        self.check_invalid(cuda.to_gpu(self.w))


testing.run_module(__name__, __file__)
