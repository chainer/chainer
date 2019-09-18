import unittest

import numpy

from chainer import testing


@testing.parameterize(*testing.product({
    'dtype': [
        numpy.float16, numpy.float32, numpy.float64,
        numpy.complex64, numpy.complex128,
    ],
    'x_s_shapes': [
        ((2, 2), (2,)),
        ((2, 3), (2,)),
        ((3, 2), (2,)),
        ((2, 3, 4), (2, 3)),
        ((2, 4, 3), (2, 3)),
        ((0, 2, 3), (0, 2)),
        # broadcast
        ((2, 2), ()),
        ((2, 3, 4), (2, 1)),
    ],
}))
class TestGenerateMatrix(unittest.TestCase):

    def test_generate_matrix(self):
        dtype = self.dtype
        x_shape, s_shape = self.x_s_shapes
        sv = 0.5 + numpy.random.random(s_shape).astype(dtype().real.dtype)
        x = testing.generate_matrix(x_shape, dtype=dtype, singular_values=sv)
        assert x.shape == x_shape

        s = numpy.linalg.svd(
            x.astype(numpy.complex128), full_matrices=False, compute_uv=False,
        )
        sv = numpy.broadcast_to(sv, s.shape)
        sv_sorted = numpy.sort(sv, axis=-1)[..., ::-1]

        rtol = 1e-3 if dtype == numpy.float16 else 1e-7
        numpy.testing.assert_allclose(s, sv_sorted, rtol=rtol)


class TestGenerateMatrixInvalid(unittest.TestCase):

    def test_no_singular_values(self):
        with self.assertRaises(TypeError):
            testing.generate_matrix((2, 2))

    def test_invalid_shape(self):
        with self.assertRaises(ValueError):
            testing.generate_matrix((2,), singular_values=1)

    def test_invalid_dtype(self):
        with self.assertRaises(ValueError):
            testing.generate_matrix(
                (2, 2), dtype=numpy.int32, singular_values=1)

    def test_shape_mismatch(self):
        with self.assertRaises(ValueError):
            testing.generate_matrix(
                (2, 2), singular_values=numpy.ones(3))


testing.run_module(__name__, __file__)
