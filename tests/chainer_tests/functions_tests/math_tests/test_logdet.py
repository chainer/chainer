import unittest

import numpy
import six

import chainer
from chainer.backends import cuda
from chainer import functions as F
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.utils import type_check


@testing.parameterize(*[
    {'batched': True},
    {'batched': False}
])
class LogdetFunctionTest(unittest.TestCase):

    def setUp(self):
        if self.batched:
            while True:
                self.x = numpy.random.uniform(
                    .5, 1, (6, 3, 3)).astype(numpy.float32)
                # Avoid backward/double_backward instability.
                if not numpy.any(numpy.isclose(
                        numpy.linalg.det(self.x), 0, atol=1e-2, rtol=1e-2)):
                    break
            self.x = numpy.stack([x_.T.dot(x_) for x_ in self.x])
            self.y = numpy.random.uniform(
                .5, 1, (6, 3, 3)).astype(numpy.float32)
            self.y = numpy.stack([y_.T.dot(y_) for y_ in self.y])
            self.gy = numpy.random.uniform(-1, 1, (6,)).astype(numpy.float32)
            self.ggx = numpy.random.uniform(
                .5, 1, (6, 3, 3)).astype(numpy.float32)
            self.ct = self.x.transpose(0, 2, 1)
            self.logdet = F.batch_logdet
            self.matmul = F.matmul
        else:
            while True:
                self.x = numpy.random.uniform(
                    .5, 1, (5, 5)).astype(numpy.float32)
                if not numpy.isclose(
                        numpy.linalg.det(self.x), 0, atol=1e-2, rtol=1e-2):
                    break
            self.x = self.x.T.dot(self.x)
            self.y = numpy.random.uniform(.5, 1, (5, 5)).astype(numpy.float32)
            self.y = self.y.T.dot(self.y)
            self.gy = numpy.random.uniform(-1, 1, ()).astype(numpy.float32)
            self.ggx = numpy.random.uniform(
                .5, 1, (5, 5)).astype(numpy.float32)
            self.ct = self.x.transpose()
            self.logdet = F.logdet
            self.matmul = F.matmul

        self.check_backward_options = {
            'atol': 1e-2, 'rtol': 1e-2, 'eps': 1e-3}
        self.check_double_backward_options = {
            'atol': 1e-1, 'rtol': 1e-1, 'eps': 1e-4}

    def logdet_transpose(self, gpu=False):
        if gpu:
            cx = cuda.to_gpu(self.x)
            ct = cuda.to_gpu(self.ct)
        else:
            cx = self.x
            ct = self.ct
        xn = chainer.Variable(cx)
        xt = chainer.Variable(ct)
        yn = self.logdet(xn)
        yt = self.logdet(xt)
        testing.assert_allclose(yn.data, yt.data, rtol=1e-4, atol=1)

    @attr.gpu
    def test_logdet_transpose_gpu(self):
        self.logdet_transpose(gpu=True)

    def test_logdet_transpose_cpu(self):
        self.logdet_transpose(gpu=False)

    def logdet_scaling(self, gpu=False):
        scaling = numpy.random.randn(1).astype('float32')
        scaling = numpy.abs(scaling)
        if gpu:
            cx = cuda.to_gpu(self.x)
            sx = cuda.to_gpu(scaling * self.x)
        else:
            cx = self.x
            sx = scaling * self.x
        c = float(scaling ** self.x.shape[1])
        cxv = chainer.Variable(cx)
        sxv = chainer.Variable(sx)
        cxd = self.logdet(cxv)
        sxd = self.logdet(sxv)
        testing.assert_allclose(cxd.data + numpy.log(c), sxd.data,
                                atol=1e-3, rtol=1e-3)

    @attr.gpu
    def test_logdet_scaling_gpu(self):
        self.logdet_scaling(gpu=True)

    def test_logdet_scaling_cpu(self):
        self.logdet_scaling(gpu=False)

    def logdet_identity(self, gpu=False):
        if self.batched:
            chk = numpy.zeros(len(self.x), dtype=numpy.float32)
            dt = numpy.identity(self.x.shape[1], dtype=numpy.float32)
            idt = numpy.repeat(dt[None], len(self.x), axis=0)
        else:
            idt = numpy.identity(self.x.shape[1], dtype=numpy.float32)
            chk = numpy.zeros(1, dtype=numpy.float32)
        if gpu:
            chk = cuda.to_gpu(chk)
            idt = cuda.to_gpu(idt)
        idtv = chainer.Variable(idt)
        idtd = self.logdet(idtv)
        testing.assert_allclose(idtd.data, chk, rtol=1e-4, atol=1e-4)

    @attr.gpu
    def test_logdet_identity_gpu(self):
        self.logdet_identity(gpu=True)

    def test_logdet_identity_cpu(self):
        self.logdet_identity(gpu=False)

    def logdet_product(self, gpu=False):
        if gpu:
            cx = cuda.to_gpu(self.x)
            cy = cuda.to_gpu(self.y)
        else:
            cx = self.x
            cy = self.y
        vx = chainer.Variable(cx)
        vy = chainer.Variable(cy)
        dxy1 = self.logdet(self.matmul(vx, vy))
        dxy2 = self.logdet(vx) + self.logdet(vy)
        testing.assert_allclose(
            dxy1.data, dxy2.data, rtol=1e-1, atol=1e-1)

    def test_logdet_product_cpu(self):
        self.logdet_product(gpu=False)

    @attr.gpu
    def test_logdet_product_gpu(self):
        self.logdet_product(gpu=True)

    @attr.gpu
    def test_batch_backward_gpu(self):
        x_data = cuda.to_gpu(self.x)
        y_grad = cuda.to_gpu(self.gy)
        gradient_check.check_backward(
            self.logdet, x_data, y_grad,
            **self.check_backward_options)

    def test_batch_backward_cpu(self):
        x_data, y_grad = self.x, self.gy
        gradient_check.check_backward(
            self.logdet, x_data, y_grad,
            **self.check_backward_options)

    @attr.gpu
    def test_batch_double_backward_gpu(self):
        x_data = cuda.to_gpu(self.x)
        y_grad = cuda.to_gpu(self.gy)
        x_grad_grad = cuda.to_gpu(self.ggx)
        gradient_check.check_double_backward(
            self.logdet, x_data, y_grad, x_grad_grad,
            **self.check_double_backward_options)

    def test_batch_double_backward_cpu(self):
        x_data, y_grad, x_grad_grad = self.x, self.gy, self.ggx
        gradient_check.check_double_backward(
            self.logdet, x_data, y_grad, x_grad_grad,
            **self.check_double_backward_options)

    def check_single_matrix(self, x):
        x = chainer.Variable(x)
        y = self.logdet(x)
        if self.batched:
            self.assertEqual(y.data.ndim, 1)
        else:
            self.assertEqual(y.data.ndim, 0)

    def test_single_matrix_cpu(self):
        self.check_single_matrix(self.x)

    @attr.gpu
    def test_expect_scalar_gpu(self):
        self.check_single_matrix(cuda.to_gpu(self.x))

    def check_zero_logdet(self, x, gy, err):
        if self.batched:
            x[0, ...] = 0.0
        else:
            x[...] = 0.0
        with self.assertRaises(err):
            gradient_check.check_backward(
                self.logdet, x, gy,
                **self.check_backward_options)

    def test_zero_logdet_cpu(self):
        self.check_zero_logdet(self.x, self.gy, ValueError)

    @attr.gpu
    def test_zero_logdet_gpu(self):
        with chainer.using_config('debug', True):
            self.check_zero_logdet(
                cuda.to_gpu(self.x), cuda.to_gpu(self.gy), ValueError)


class TestLogdetSmallCase(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(.5, 1, (2, 2)).astype(numpy.float32)
        if self.x[0, 0] * self.x[1, 1] - self.x[0, 1] * self.x[1, 0] < 0:
            self.x[0, 0], self.x[1, 0] = self.x[1, 0], self.x[0, 0]
            self.x[0, 1], self.x[1, 1] = self.x[1, 1], self.x[0, 1]

    def check_by_definition(self, x):
        ans = F.logdet(chainer.Variable(x)).data
        y = x[0, 0] * x[1, 1] - x[0, 1] * x[1, 0]
        xp = cuda.get_array_module(x)
        y = xp.log(xp.abs(y))
        testing.assert_allclose(ans, y)

    def test_answer_cpu(self):
        self.check_by_definition(self.x)

    @attr.gpu
    def test_answer_gpu(self):
        self.check_by_definition(cuda.to_gpu(self.x))


@testing.parameterize(
    *testing.product({
        'shape': [(s, s) for s in six.moves.range(1, 5)],
    }))
class TestLogdetGPUCPUConsistency(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(.5, 1, self.shape).astype(numpy.float32)
        self.x = self.x.T.dot(self.x)

    @attr.gpu
    def test_answer_gpu_cpu(self):
        x = cuda.to_gpu(self.x)
        y = F.logdet(chainer.Variable(x))
        gpu = cuda.to_cpu(y.data)
        cpu = numpy.linalg.slogdet(self.x)[1]
        testing.assert_allclose(gpu, cpu, atol=1e-2, rtol=1e-2)


@testing.parameterize(
    *testing.product({
        'shape': [(w, s, s) for s in six.moves.range(1, 5)
                  for w in six.moves.range(1, 5)],
    }))
class TestBatchLogdetGPUCPUConsistency(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(.5, 1, self.shape).astype(numpy.float32)
        self.x = numpy.stack([x_.T.dot(x_) for x_ in self.x])

    @attr.gpu
    def test_answer_gpu_cpu(self):
        x = cuda.to_gpu(self.x)
        y = F.batch_logdet(chainer.Variable(x))
        gpu = cuda.to_cpu(y.data)
        cpu = numpy.linalg.slogdet(self.x)[1]
        testing.assert_allclose(gpu, cpu, atol=1e-2, rtol=1e-2)


class LogdetFunctionRaiseTest(unittest.TestCase):

    def test_invalid_ndim(self):
        with self.assertRaises(type_check.InvalidType):
            F.batch_logdet(chainer.Variable(numpy.zeros((2, 2))))

    def test_invalid_shape(self):
        with self.assertRaises(type_check.InvalidType):
            F.batch_logdet(chainer.Variable(numpy.zeros((1, 2))))


testing.run_module(__name__, __file__)
