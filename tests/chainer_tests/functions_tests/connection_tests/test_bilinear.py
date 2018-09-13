import unittest

import numpy

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


def _uniform(*shape):
    return numpy.random.uniform(-1, 1, shape).astype(numpy.float32)


@testing.parameterize(*testing.product({
    'in_shapes': [((2,), (4,)), ((2, 1), (4, 2))],
    'out_size': [3],
    'batch_size': [2]
}))
class TestBilinearFunction(unittest.TestCase):

    def setUp(self):
        e1_shape = (self.batch_size,) + self.in_shapes[0]
        e2_shape = (self.batch_size,) + self.in_shapes[1]
        e1_size = numpy.prod(self.in_shapes[0])
        e2_size = numpy.prod(self.in_shapes[1])

        self.e1 = _uniform(*e1_shape)
        self.e2 = _uniform(*e2_shape)
        self.W = _uniform(e1_size, e2_size, self.out_size)
        self.V1 = _uniform(e1_size, self.out_size)
        self.V2 = _uniform(e2_size, self.out_size)
        self.b = _uniform(self.out_size)

        self.gy = _uniform(self.batch_size, self.out_size)

        self.gge1 = _uniform(*self.e1.shape)
        self.gge2 = _uniform(*self.e2.shape)
        self.ggW = _uniform(*self.W.shape)
        self.ggV1 = _uniform(*self.V1.shape)
        self.ggV2 = _uniform(*self.V2.shape)
        self.ggb = _uniform(*self.b.shape)

        self.check_backward_options = {
            'atol': 1e-5, 'rtol': 1e-4, 'dtype': numpy.float64}
        self.check_double_backward_options = {
            'atol': 1e-4, 'rtol': 1e-3, 'dtype': numpy.float64}

    def check_forward(self, e1_data, e2_data, W_data, V1_data, V2_data,
                      b_data):
        e1 = chainer.Variable(e1_data)
        e2 = chainer.Variable(e2_data)
        W = chainer.Variable(W_data)

        e1_data = e1_data.reshape(e1_data.shape[0], -1)
        e2_data = e2_data.reshape(e2_data.shape[0], -1)
        xp = backend.get_array_module(e1)
        y_expect = xp.einsum('ij,ik,jkl->il', e1_data, e2_data, W_data)

        flags = V1_data is None, V2_data is None, b_data is None
        if any(flags):
            if not all(flags):
                raise ValueError(
                    'Test either all or none of the optional parameters.')
            y = functions.bilinear(e1, e2, W)
        else:
            V1 = chainer.Variable(V1_data)
            V2 = chainer.Variable(V2_data)
            b = chainer.Variable(b_data)
            y = functions.bilinear(e1, e2, W, V1, V2, b)

            y_expect = xp.einsum('ij,ik,jkl->il', e1_data, e2_data, W_data)
            y_expect += e1_data.dot(V1_data)
            y_expect += e2_data.dot(V2_data)
            y_expect += b_data

        testing.assert_allclose(y_expect, cuda.to_cpu(y.data))
        assert y.data.dtype == e1_data.dtype

    def test_forward_cpu(self):
        self.check_forward(self.e1, self.e2, self.W, self.V1, self.V2, self.b)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(
            cuda.to_gpu(self.e1), cuda.to_gpu(self.e2), cuda.to_gpu(self.W),
            cuda.to_gpu(self.V1), cuda.to_gpu(self.V2), cuda.to_gpu(self.b))

    def test_partial_backward_cpu(self):
        gradient_check.check_backward(
            functions.bilinear, (self.e1, self.e2, self.W), self.gy,
            **self.check_backward_options)

    @attr.gpu
    def test_partial_backward_gpu(self):
        gradient_check.check_backward(
            functions.bilinear,
            (cuda.to_gpu(self.e1), cuda.to_gpu(self.e2), cuda.to_gpu(self.W)),
            cuda.to_gpu(self.gy), **self.check_backward_options)

    def test_full_backward_cpu(self):
        gradient_check.check_backward(
            functions.bilinear,
            (self.e1, self.e2, self.W, self.V1, self.V2, self.b), self.gy,
            **self.check_backward_options)

    @attr.gpu
    def test_full_backward_gpu(self):
        gradient_check.check_backward(
            functions.bilinear,
            (cuda.to_gpu(self.e1), cuda.to_gpu(self.e2), cuda.to_gpu(self.W),
             cuda.to_gpu(self.V1), cuda.to_gpu(self.V2), cuda.to_gpu(self.b)),
            cuda.to_gpu(self.gy), **self.check_backward_options)

    def test_partial_double_backward_cpu(self):
        gradient_check.check_double_backward(
            functions.bilinear, (self.e1, self.e2, self.W), self.gy,
            (self.gge1, self.gge2, self.ggW), **self.check_backward_options)

    @attr.gpu
    def test_partial_double_backward_gpu(self):
        gradient_check.check_double_backward(
            functions.bilinear,
            (cuda.to_gpu(self.e1), cuda.to_gpu(self.e2), cuda.to_gpu(self.W)),
            cuda.to_gpu(self.gy),
            (cuda.to_gpu(self.gge1), cuda.to_gpu(self.gge2),
             cuda.to_gpu(self.ggW)), **self.check_backward_options)

    def test_full_double_backward_cpu(self):
        gradient_check.check_double_backward(
            functions.bilinear,
            (self.e1, self.e2, self.W, self.V1, self.V2, self.b),
            self.gy,
            (self.gge1, self.gge2, self.ggW, self.ggV1, self.ggV2, self.ggb),
            **self.check_double_backward_options)

    @attr.gpu
    def test_full_double_backward_gpu(self):
        gradient_check.check_double_backward(
            functions.bilinear,
            (cuda.to_gpu(self.e1), cuda.to_gpu(self.e2), cuda.to_gpu(self.W),
             cuda.to_gpu(self.V1), cuda.to_gpu(self.V2), cuda.to_gpu(self.b)),
            cuda.to_gpu(self.gy),
            (cuda.to_gpu(self.gge1), cuda.to_gpu(self.gge2),
             cuda.to_gpu(self.ggW), cuda.to_gpu(self.V1), cuda.to_gpu(self.V2),
             cuda.to_gpu(self.ggb)), **self.check_double_backward_options)


@attr.slow
class TestBilinearFunctionLarge(unittest.TestCase):

    def setUp(self):
        self.e1 = _uniform(256, 256)
        self.e2 = _uniform(256, 256)
        self.w = _uniform(256, 256, 256)
        self.v1 = _uniform(256, 256)
        self.v2 = _uniform(256, 256)
        self.b = _uniform(256)

    def test_cpu(self):
        chainer.functions.bilinear(
            self.e1, self.e2, self.w, self.v1, self.v2, self.b)

    @attr.gpu
    def test_gpu(self):
        chainer.functions.bilinear(*map(cuda.to_gpu, (
            self.e1, self.e2, self.w, self.v1, self.v2, self.b)))


class TestBilinearFunctionInvalidArgument(unittest.TestCase):

    def setUp(self):
        e1 = _uniform(3, 2)
        e2 = _uniform(3, 4)
        W = _uniform(2, 4, 5)
        V1 = _uniform(2, 5)

        self.e1 = chainer.Variable(e1)
        self.e2 = chainer.Variable(e2)
        self.W = chainer.Variable(W)
        self.V1 = chainer.Variable(V1)

    def test_invalid_full_partial_ambiguous(self):
        with self.assertRaises(ValueError):
            functions.bilinear(self.e1, self.e2, self.W, self.V1)


testing.run_module(__name__, __file__)
