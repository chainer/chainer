import unittest

import numpy

from chainer.backends import cuda
from chainer import initializers
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product({
    'target': [
        initializers.UpsamplingDeconvFilter,
        initializers.DownsamplingConvFilter,
    ],
    'interpolation': ['linear'],
    'shape': [(5, 5, 3, 3), (5, 5, 6, 6), (5, 1, 3, 3), (5, 1, 4, 4)],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestUpsamplingDeconvFilter(unittest.TestCase):

    def check_initializer(self, w):
        initializer = self.target(
            interpolation=self.interpolation, dtype=self.dtype)
        initializer(w)
        if self.target == initializers.UpsamplingDeconvFilter:
            w_sum = ((numpy.array(self.shape[2:]) + 1) // 2).prod()
            w_sum = w_sum * self.shape[0]
            self.assertAlmostEqual(cuda.to_cpu(w).sum(), w_sum)
        else:
            self.assertAlmostEqual(cuda.to_cpu(w).sum(), 1.0 * self.shape[0])
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
        initializer = self.target(dtype=self.dtype)
        w = initializers.generate_array(initializer, self.shape, xp)
        self.assertIs(cuda.get_array_module(w), xp)
        self.assertTupleEqual(w.shape, self.shape)
        self.assertEqual(w.dtype, self.dtype)

    def test_shaped_initializer_cpu(self):
        self.check_shaped_initializer(numpy)

    @attr.gpu
    def test_shaped_initializer_gpu(self):
        self.check_shaped_initializer(cuda.cupy)


testing.run_module(__name__, __file__)
