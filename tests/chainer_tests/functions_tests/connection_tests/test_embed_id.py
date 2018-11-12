import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer.functions.connection import embed_id
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product_dict(
    [{'x_data': [0, 1, 0], 'ignore_label': None},
     {'x_data': [[0, 1, 0], [1, 0, 1]], 'ignore_label': None},
     {'x_data': [0, 1, -1], 'ignore_label': -1},
     {'x_data': [[0, 1, -1], [-1, 0, 1]], 'ignore_label': -1}],
    [{'label_dtype': numpy.int8},
     {'label_dtype': numpy.int16},
     {'label_dtype': numpy.int32},
     {'label_dtype': numpy.int64}]
))
class TestEmbedID(unittest.TestCase):

    def setUp(self):
        self.x = numpy.array(self.x_data, dtype=self.label_dtype)
        self.W = numpy.random.uniform(-1, 1, (3, 2)).astype('f')
        y_shape = self.x.shape + (2,)
        self.gy = numpy.random.uniform(-1, 1, y_shape).astype(numpy.float32)
        self.ggW = numpy.random.uniform(-1, 1, (3, 2)).astype('f')

        self.check_backward_options = {'atol': 1e-2, 'rtol': 1e-2}
        self.check_double_backward_options = {'atol': 1e-2, 'rtol': 1e-2}

    def check_forward(self, x_data, W_data):
        x = chainer.Variable(x_data)
        W = chainer.Variable(W_data)
        y = chainer.functions.embed_id(x, W, self.ignore_label)
        self.assertEqual(y.data.dtype, numpy.float32)

        y_expect = numpy.empty_like(self.gy)
        for i in numpy.ndindex(self.x.shape):
            if self.x[i] == -1:
                y_expect[i] = 0
            else:
                y_expect[i] = self.W[int(self.x[i])]

        testing.assert_allclose(y_expect, y.data, atol=0, rtol=0)

    def test_forward_cpu(self):
        self.check_forward(self.x, self.W)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.W))

    def check_backward(self, x_data, W_data, y_grad):
        def f(x, W):
            return chainer.functions.embed_id(x, W, self.ignore_label)

        gradient_check.check_backward(
            f, (x_data, W_data), y_grad, dtype=numpy.float64,
            **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.W, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.W), cuda.to_gpu(self.gy))

    def check_double_backward(self, x_data, W_data, gy_data, ggW_data):
        def f(W):
            return chainer.functions.embed_id(
                x_data, W, self.ignore_label)

        gradient_check.check_double_backward(
            f, W_data, gy_data, ggW_data,
            **self.check_double_backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(self.x, self.W, self.gy, self.ggW)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.W), cuda.to_gpu(self.gy),
            cuda.to_gpu(self.ggW))


@testing.parameterize(
    {'x_data': [0, 1, 0], 'ignore_label': None},
    {'x_data': [[0, 1, 0], [1, 0, 1]], 'ignore_label': None},
    {'x_data': [0, 1, -1], 'ignore_label': -1},
    {'x_data': [[0, 1, -1], [-1, 0, 1]], 'ignore_label': -1},
    {'x_data': [0, 1, 2], 'ignore_label': 2},
    {'x_data': [[0, 1, 0], [1, 0, 1]], 'ignore_label': 1},
)
class TestEmbedIdGrad(unittest.TestCase):

    n_unit = (4,)
    w_shape = (4, 2)

    def setUp(self):
        self.x = numpy.array(self.x_data, dtype='i')
        self.gy = numpy.random.uniform(
            -1, 1, self.x.shape + (2,)).astype('f')
        self.ggW = numpy.random.uniform(-1, 1, self.w_shape).astype('f')

    def check_backward(self, x, gy, ggW):
        return

        def f(x, gy):
            emb = embed_id.EmbedIDGrad(
                self.w_shape, self.ignore_label)
            return emb.apply((x, numpy.zeros(()), gy))[0]

        gradient_check.check_backward(f, (x, gy), (ggW,))

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy, self.ggW)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.gy), cuda.to_gpu(self.ggW))


testing.run_module(__name__, __file__)
