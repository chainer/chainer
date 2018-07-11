import os
import tempfile
import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import gradient_check
from chainer import links
from chainer.serializers import npz
from chainer import testing
from chainer.testing import attr
from chainer.utils import type_check


def gen_mask(ratio, shape):
    return numpy.random.rand(*shape) >= ratio


@testing.parameterize(*testing.product({
    'in_shape': [(3,), (3, 2, 2)],
    'x_dtype': [numpy.float16, numpy.float32, numpy.float64],
    'W_dtype': [numpy.float16, numpy.float32, numpy.float64],
    'use_batchwise_mask': [True, False],
}))
class TestSimplifiedDropconnect(unittest.TestCase):

    out_size = 2
    ratio = 0.5

    def setUp(self):
        in_size = numpy.prod(self.in_shape)

        self.link = links.SimplifiedDropconnect(
            in_size, self.out_size,
            initialW=chainer.initializers.Normal(1, self.W_dtype),
            initial_bias=chainer.initializers.Normal(1, self.x_dtype))
        self.link.cleargrads()

        x_shape = (4,) + self.in_shape
        self.x = numpy.random.uniform(-1, 1, x_shape).astype(self.x_dtype)
        self.gy = numpy.random.uniform(
            -1, 1, (4, self.out_size)).astype(self.x_dtype)
        W = self.link.W.data
        b = self.link.b.data

        if self.use_batchwise_mask:
            mask_shape = (4,) + self.link.W.shape
        else:
            mask_shape = self.link.W.shape
        self.mask = gen_mask(self.ratio, mask_shape)

        W = (W * self.mask) * (1. / (1 - self.ratio))
        x = self.x.reshape(4, -1)

        # numpy 1.9 does not support matmul.
        # So we use numpy.einsum instead of numpy.matmul.
        if self.use_batchwise_mask:
            self.y_expect = numpy.einsum('ijk,ikl->ijl',
                                         W, x[:, :, None]).reshape(4, -1) + b
        else:
            self.y_expect = numpy.einsum('jk,ikl->ijl',
                                         W, x[:, :, None]).reshape(4, -1) + b

        self.check_forward_options = {}
        self.check_backward_options = {'atol': 1e-4, 'rtol': 1e-3}
        if self.x_dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-3, 'rtol': 1e-2}
            self.check_backward_options = {'atol': 1e-2, 'rtol': 5e-2}
        elif self.W_dtype == numpy.float16:
            self.check_backward_options = {'atol': 1e-3, 'rtol': 1e-2}

    def check_forward(self, x_data, mask):
        x = chainer.Variable(x_data)
        y = self.link(x, train=True, mask=mask,
                      use_batchwise_mask=self.use_batchwise_mask)
        self.assertEqual(y.data.dtype, self.x_dtype)
        testing.assert_allclose(self.y_expect, y.data,
                                **self.check_forward_options)

    def test_forward_cpu(self):
        self.check_forward(self.x, self.mask)

    @attr.gpu
    def test_forward_gpu(self):
        self.link.to_gpu()
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.mask))

    def link_wrapper(self, *data):
        return self.link(x=data[0], train=True, mask=data[1],
                         use_batchwise_mask=self.use_batchwise_mask)

    def check_backward(self, x_data, y_grad, mask):
        gradient_check.check_backward(
            self.link_wrapper, (x_data, mask), y_grad,
            (self.link.W, self.link.b),
            dtype='d', **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy, self.mask)

    @attr.gpu
    def test_backward_gpu(self):
        self.link.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy),
                            cuda.to_gpu(self.mask))


class TestSimplifiedDropconnectParameterShapePlaceholder(unittest.TestCase):

    in_size = 3
    in_shape = (in_size,)
    out_size = 2
    in_size_or_none = None
    ratio = 0.5

    def setUp(self):
        self.link = links.SimplifiedDropconnect(self.in_size_or_none,
                                                self.out_size)
        temp_x = numpy.random.uniform(-1, 1,
                                      (4, self.in_size)).astype(numpy.float32)
        self.link(chainer.Variable(temp_x))
        W = self.link.W.data
        W[...] = numpy.random.uniform(-1, 1, W.shape)
        b = self.link.b.data
        b[...] = numpy.random.uniform(-1, 1, b.shape)
        self.link.cleargrads()
        mask_shape = (4, self.out_size, self.in_size)
        self.mask = gen_mask(self.ratio, mask_shape)

        x_shape = (4,) + self.in_shape
        self.x = numpy.random.uniform(-1, 1, x_shape).astype(numpy.float32)
        self.gy = numpy.random.uniform(
            -1, 1, (4, self.out_size)).astype(numpy.float32)
        W = (W * self.mask) * (1. / (1 - self.ratio))

        # numpy 1.9 does not support matmul.
        # So we use numpy.einsum instead of numpy.matmul.
        self.y_expect = numpy.einsum('ijk,ikl->ijl',
                                     W, self.x[:, :, None]).reshape(4, -1) + b

    def check_forward(self, x_data, mask):
        x = chainer.Variable(x_data)
        y = self.link(x, train=True, mask=mask, use_batchwise_mask=True)
        self.assertEqual(y.data.dtype, numpy.float32)
        testing.assert_allclose(self.y_expect, y.data)

    def test_forward_cpu(self):
        self.check_forward(self.x, self.mask)

    @attr.gpu
    def test_forward_gpu(self):
        self.link.to_gpu()
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.mask))

    def link_wrapper(self, *data):
        return self.link(x=data[0], train=True, mask=data[1],
                         use_batchwise_mask=True)

    def check_backward(self, x_data, y_grad, mask):
        gradient_check.check_backward(
            self.link_wrapper, (x_data, mask), y_grad,
            (self.link.W, self.link.b), dtype='d',
            atol=1e-4, rtol=1e-3)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy, self.mask)

    @attr.gpu
    def test_backward_gpu(self):
        self.link.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy),
                            cuda.to_gpu(self.mask))

    def test_serialization(self):
        lin1 = links.SimplifiedDropconnect(None, self.out_size)
        x = chainer.Variable(self.x)
        # Must call the link to initialize weights.
        lin1(x)
        w1 = lin1.W.data
        fd, temp_file_path = tempfile.mkstemp()
        os.close(fd)
        npz.save_npz(temp_file_path, lin1)
        lin2 = links.SimplifiedDropconnect(None, self.out_size)
        npz.load_npz(temp_file_path, lin2)
        w2 = lin2.W.data
        self.assertEqual((w1 == w2).all(), True)


class TestSimplifiedDropconnectNotBatchwiseMask(unittest.TestCase):

    in_shape = (3,)
    out_size = 2
    ratio = 0.5

    def setUp(self):
        in_size = numpy.prod(self.in_shape)

        self.link = links.SimplifiedDropconnect(
            in_size, self.out_size,
            initialW=chainer.initializers.Normal(1, numpy.float32),
            initial_bias=chainer.initializers.Normal(1, numpy.float32))
        self.link.cleargrads()

        x_shape = (4,) + self.in_shape
        self.x = numpy.ones(x_shape).astype(numpy.float32)
        self.W = self.link.W.data
        self.b = self.link.b.data

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = self.link(x, train=True, use_batchwise_mask=False)

        # check mask equality here.
        testing.assert_allclose(y.data[0], y.data[1])
        testing.assert_allclose(y.data[0], y.data[2])
        testing.assert_allclose(y.data[0], y.data[3])

        mask = y.creator.mask
        mask = cuda.to_cpu(mask)

        y_expect = self.x.dot(self.W.T * mask.T) * (1. / (1 - self.ratio))
        y_expect += self.b
        testing.assert_allclose(y_expect, y.data)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.link.to_gpu()
        self.check_forward(cuda.to_gpu(self.x))


class TestInvalidSimplifiedDropconnect(unittest.TestCase):

    def test_invalid_input_size(self):
        link = links.SimplifiedDropconnect(3, 2)
        x = numpy.random.uniform(-1, 1, (4, 1, 2)).astype(numpy.float32)
        with self.assertRaises(type_check.InvalidType):
            link(chainer.Variable(x))

    def test_invalid_mask_size(self):
        link = links.SimplifiedDropconnect(3, 2)
        x = numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32)
        mask = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)
        with self.assertRaises(type_check.InvalidType):
            link(chainer.Variable(x), use_batchwise_mask=True, mask=mask)

    def test_invalid_mask_size2(self):
        link = links.SimplifiedDropconnect(3, 2)
        x = numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32)
        mask = numpy.random.uniform(-1, 1, (4, 3, 2)).astype(numpy.float32)
        with self.assertRaises(type_check.InvalidType):
            link(chainer.Variable(x), use_batchwise_mask=False, mask=mask)


testing.run_module(__name__, __file__)
