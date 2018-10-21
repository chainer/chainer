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


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'batched': [True, False],
}))
class DetFunctionTest(unittest.TestCase):

    def setUp(self):
        if self.batched:
            while True:
                x = numpy.random.uniform(
                    .5, 1, (6, 3, 3)).astype(numpy.float32)
                # Avoid backward/double_backward instability.
                if not numpy.any(numpy.isclose(
                        numpy.linalg.det(x), 0, atol=1e-2, rtol=1e-2)):
                    self.x = x.astype(self.dtype, copy=False)
                    break
            self.y = numpy.random.uniform(
                .5, 1, (6, 3, 3)).astype(self.dtype)
            self.gy = numpy.random.uniform(-1, 1, (6,)).astype(self.dtype)
            self.ggx = numpy.random.uniform(
                .5, 1, (6, 3, 3)).astype(self.dtype)
            self.ct = self.x.transpose(0, 2, 1)
            self.det = F.batch_det
            self.matmul = F.matmul
        else:
            while True:
                x = numpy.random.uniform(.5, 1, (5, 5)).astype(numpy.float32)
                if not numpy.isclose(
                        numpy.linalg.det(x), 0, atol=1e-2, rtol=1e-2):
                    self.x = x.astype(self.dtype, copy=False)
                    break
            self.y = numpy.random.uniform(.5, 1, (5, 5)).astype(self.dtype)
            self.gy = numpy.random.uniform(-1, 1, ()).astype(self.dtype)
            self.ggx = numpy.random.uniform(.5, 1, (5, 5)).astype(self.dtype)
            self.ct = self.x.transpose()
            self.det = F.det
            self.matmul = F.matmul

        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 5e-3, 'rtol': 5e-3}
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 5e-3, 'rtol': 1e-3}
            self.check_double_backward_options = {
                'dtype': numpy.float64, 'atol': 1e-2, 'rtol': 1e-2}
        else:
            self.check_forward_options = {}
            self.check_backward_options = {'atol': 5e-3, 'rtol': 1e-3}
            self.check_double_backward_options = {'atol': 1e-2, 'rtol': 1e-2}

    def det_transpose(self, gpu=False):
        if gpu:
            cx = cuda.to_gpu(self.x)
            ct = cuda.to_gpu(self.ct)
        else:
            cx = self.x
            ct = self.ct
        xn = chainer.Variable(cx)
        xt = chainer.Variable(ct)
        yn = self.det(xn)
        yt = self.det(xt)
        testing.assert_allclose(yn.data, yt.data, rtol=1e-4, atol=1)

    @attr.gpu
    def test_det_transpose_gpu(self):
        self.det_transpose(gpu=True)

    def test_det_transpose_cpu(self):
        self.det_transpose(gpu=False)

    def det_scaling(self, gpu=False):
        scaling = numpy.random.randn(1).astype(self.dtype)
        if gpu:
            cx = cuda.to_gpu(self.x)
            sx = cuda.to_gpu(scaling * self.x)
        else:
            cx = self.x
            sx = scaling * self.x
        c = float(scaling ** self.x.shape[1])
        cxv = chainer.Variable(cx)
        sxv = chainer.Variable(sx)
        cxd = self.det(cxv)
        sxd = self.det(sxv)
        testing.assert_allclose(
            cxd.data * c, sxd.data, **self.check_forward_options)

    @attr.gpu
    def test_det_scaling_gpu(self):
        self.det_scaling(gpu=True)

    def test_det_scaling_cpu(self):
        self.det_scaling(gpu=False)

    def det_identity(self, gpu=False):
        if self.batched:
            chk = numpy.ones(len(self.x), dtype=self.dtype)
            dt = numpy.identity(self.x.shape[1], dtype=self.dtype)
            idt = numpy.repeat(dt[None], len(self.x), axis=0)
        else:
            idt = numpy.identity(self.x.shape[1], dtype=self.dtype)
            chk = numpy.ones(1, dtype=self.dtype)
        if gpu:
            chk = cuda.to_gpu(chk)
            idt = cuda.to_gpu(idt)
        idtv = chainer.Variable(idt)
        idtd = self.det(idtv)
        testing.assert_allclose(idtd.data, chk, **self.check_forward_options)

    @attr.gpu
    def test_det_identity_gpu(self):
        self.det_identity(gpu=True)

    def test_det_identity_cpu(self):
        self.det_identity(gpu=False)

    def det_product(self, gpu=False):
        if gpu:
            cx = cuda.to_gpu(self.x)
            cy = cuda.to_gpu(self.y)
        else:
            cx = self.x
            cy = self.y
        vx = chainer.Variable(cx)
        vy = chainer.Variable(cy)
        dxy1 = self.det(self.matmul(vx, vy))
        dxy2 = self.det(vx) * self.det(vy)
        testing.assert_allclose(
            dxy1.data, dxy2.data, **self.check_forward_options)

    def test_det_product_cpu(self):
        self.det_product(gpu=False)

    @attr.gpu
    def test_det_product_gpu(self):
        self.det_product(gpu=True)

    @attr.gpu
    def test_batch_backward_gpu(self):
        x_data = cuda.to_gpu(self.x)
        y_grad = cuda.to_gpu(self.gy)
        gradient_check.check_backward(
            self.det, x_data, y_grad, **self.check_backward_options)

    def test_batch_backward_cpu(self):
        x_data, y_grad = self.x, self.gy
        gradient_check.check_backward(
            self.det, x_data, y_grad, **self.check_backward_options)

    @attr.gpu
    def test_batch_double_backward_gpu(self):
        x_data = cuda.to_gpu(self.x)
        y_grad = cuda.to_gpu(self.gy)
        x_grad_grad = cuda.to_gpu(self.ggx)
        gradient_check.check_double_backward(
            self.det, x_data, y_grad, x_grad_grad,
            **self.check_double_backward_options)

    def test_batch_double_backward_cpu(self):
        x_data, y_grad, x_grad_grad = self.x, self.gy, self.ggx
        gradient_check.check_double_backward(
            self.det, x_data, y_grad, x_grad_grad,
            **self.check_double_backward_options)

    def check_single_matrix(self, x):
        x = chainer.Variable(x)
        y = self.det(x)
        if self.batched:
            self.assertEqual(y.data.ndim, 1)
        else:
            self.assertEqual(y.data.ndim, 0)

    def test_single_matrix_cpu(self):
        self.check_single_matrix(self.x)

    @attr.gpu
    def test_expect_scalar_gpu(self):
        self.check_single_matrix(cuda.to_gpu(self.x))

    def check_singular_matrix(self, x):
        if self.batched:
            x[0, ...] = 0.0
        else:
            x[...] = 0.0
        x = chainer.Variable(x)
        # it checks no errors are raised
        self.det(x)

    def test_singular_matrix_cpu(self):
        self.check_singular_matrix(self.x)

    @attr.gpu
    def test_singular_matrix_gpu(self):
        self.check_singular_matrix(cuda.to_gpu(self.x))

    def check_zero_det(self, x, gy, err):
        if self.batched:
            x[0, ...] = 0.0
        else:
            x[...] = 0.0
        with self.assertRaises(err):
            gradient_check.check_backward(
                self.det, x, gy,
                **self.check_backward_options)

    def test_zero_det_cpu(self):
        self.check_zero_det(self.x, self.gy, ValueError)

    @attr.gpu
    def test_zero_det_gpu(self):
        with chainer.using_config('debug', True):
            self.check_zero_det(
                cuda.to_gpu(self.x), cuda.to_gpu(self.gy), ValueError)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestDetSmallCase(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(.5, 1, (2, 2)).astype(self.dtype)
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 5e-3, 'rtol': 5e-3}
        else:
            self.check_forward_options = {}

    def check_by_definition(self, x):
        ans = F.det(chainer.Variable(x)).data
        y = x[0, 0] * x[1, 1] - x[0, 1] * x[1, 0]
        testing.assert_allclose(ans, y, **self.check_forward_options)

    def test_answer_cpu(self):
        self.check_by_definition(self.x)

    @attr.gpu
    def test_answer_gpu(self):
        self.check_by_definition(cuda.to_gpu(self.x))


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'shape': [(s, s) for s in six.moves.range(1, 5)],
}))
class TestDetGPUCPUConsistency(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(.5, 1, self.shape).astype(self.dtype)

    @attr.gpu
    def test_answer_gpu_cpu(self):
        x = cuda.to_gpu(self.x)
        y = F.det(chainer.Variable(x))
        gpu = cuda.to_cpu(y.data)
        if self.dtype == numpy.float16:
            cpu = numpy.linalg.det(
                self.x.astype(numpy.float32)).astype(numpy.float16)
            testing.assert_allclose(gpu, cpu, atol=5e-3, rtol=5e-3)
        else:
            cpu = numpy.linalg.det(self.x)
            testing.assert_allclose(gpu, cpu)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'shape': [(w, s, s) for s in six.moves.range(1, 5)
              for w in six.moves.range(1, 5)],
}))
class TestBatchDetGPUCPUConsistency(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(.5, 1, self.shape).astype(self.dtype)

    @attr.gpu
    def test_answer_gpu_cpu(self):
        x = cuda.to_gpu(self.x)
        y = F.batch_det(chainer.Variable(x))
        gpu = cuda.to_cpu(y.data)
        if self.dtype == numpy.float16:
            cpu = numpy.linalg.det(
                self.x.astype(numpy.float32)).astype(numpy.float16)
            testing.assert_allclose(gpu, cpu, atol=5e-3, rtol=5e-3)
        else:
            cpu = numpy.linalg.det(self.x)
            testing.assert_allclose(gpu, cpu)


class DetFunctionRaiseTest(unittest.TestCase):

    def test_invalid_ndim(self):
        with self.assertRaises(type_check.InvalidType):
            F.batch_det(chainer.Variable(numpy.zeros((2, 2))))

    def test_invalid_shape(self):
        with self.assertRaises(type_check.InvalidType):
            F.batch_det(chainer.Variable(numpy.zeros((1, 2))))


testing.run_module(__name__, __file__)
