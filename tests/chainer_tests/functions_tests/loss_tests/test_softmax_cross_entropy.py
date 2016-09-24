import unittest

import mock
import numpy
import six

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


@testing.parameterize(*(testing.product({
    'shape': [None, (2, 3), (2, 3, 2), (2, 3, 2, 2)],
    'cache_score': [True, False],
    'normalize': [True, False],
    'ignore_index': [None, (slice(None),), (0,), (0, 1), (0, 1, 0)],
    'dtype': [numpy.float32],
}) + testing.product({
    'shape': [None, (2, 3), (2, 3, 2), (2, 3, 2, 2)],
    'cache_score': [False],
    'normalize': [True],
    'ignore_index': [(0, 1)],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
})))
class TestSoftmaxCrossEntropy(unittest.TestCase):

    def setUp(self):
        if self.shape is None:
            if self.dtype == numpy.float16:
                self.x = numpy.array([[-5, 1]], dtype=self.dtype)
            else:
                self.x = numpy.array([[-1000, 1]], dtype=self.dtype)
            self.t = numpy.array([0], dtype=numpy.int32)
        else:
            self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
            out_shape = (self.shape[0],) + self.shape[2:]
            self.t = numpy.random.randint(0, 3, out_shape).astype(numpy.int32)
            if (self.ignore_index is not None and
                    len(self.ignore_index) <= self.t.ndim):
                self.t[self.ignore_index] = -1
        self.check_forward_options = {}
        self.check_backward_options = {'dtype': numpy.float64}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 5e-4, 'rtol': 5e-3}
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 5e-4, 'rtol': 5e-3}

    def check_forward(self, x_data, t_data, use_cudnn=True):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)
        loss = functions.softmax_cross_entropy(
            x, t, use_cudnn=use_cudnn, normalize=self.normalize,
            cache_score=self.cache_score)
        self.assertEqual(loss.data.shape, ())
        self.assertEqual(loss.data.dtype, self.dtype)
        self.assertEqual(hasattr(loss.creator, 'y'), self.cache_score)
        loss_value = float(cuda.to_cpu(loss.data))

        # Compute expected value
        loss_expect = 0.0
        count = 0
        x = numpy.rollaxis(self.x, 1, self.x.ndim).reshape(
            (self.t.size, self.x.shape[1]))
        t = self.t.ravel()
        for xi, ti in six.moves.zip(x, t):
            if ti == -1:
                continue
            log_z = numpy.ufunc.reduce(numpy.logaddexp, xi)
            loss_expect -= (xi - log_z)[ti]
            count += 1

        if self.normalize:
            if count == 0:
                loss_expect = 0.0
            else:
                loss_expect /= count
        else:
            loss_expect /= len(t_data)

        testing.assert_allclose(
            loss_expect, loss_value, **self.check_forward_options)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x, self.t)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.t))

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_no_cudnn(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.t), False)

    def check_backward(self, x_data, t_data, use_cudnn=True):
        gradient_check.check_backward(
            functions.SoftmaxCrossEntropy(
                use_cudnn=use_cudnn, cache_score=self.cache_score),
            (x_data, t_data), None, eps=0.02, **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.t)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.t))

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_no_cudnn(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.t), False)


@testing.parameterize(
    {'t_value': -2, 'valid': False},
    {'t_value': 3,  'valid': False},
    {'t_value': -1, 'valid': True},  # -1 is ignore_label
)
class TestSoftmaxCrossEntropyValueCheck(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 2)).astype(numpy.float32)
        # `0` is required to avoid NaN
        self.t = numpy.array([self.t_value, 0], dtype=numpy.int32)
        self.original_debug = chainer.is_debug()
        chainer.set_debug(True)

    def tearDown(self):
        chainer.set_debug(self.original_debug)

    def check_value_check(self, x_data, t_data, use_cudnn):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)

        if self.valid:
            # Check if it throws nothing
            functions.softmax_cross_entropy(x, t, use_cudnn)
        else:
            with self.assertRaises(ValueError):
                functions.softmax_cross_entropy(x, t, use_cudnn)

    def test_value_check_cpu(self):
        self.check_value_check(self.x, self.t, False)

    @attr.gpu
    def test_value_check_gpu(self):
        self.check_value_check(self.x, self.t, False)

    @attr.gpu
    def test_value_check_gpu_cudnn(self):
        self.check_value_check(cuda.to_gpu(self.x), cuda.to_gpu(self.t), True)


@testing.parameterize(*testing.product({
    'use_cudnn': [True, False],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
@attr.cudnn
class TestSoftmaxCrossEntropyCudnnCall(unittest.TestCase):

    def setUp(self):
        self.x = cuda.cupy.random.uniform(-1, 1, (4, 3)).astype(self.dtype)
        self.t = cuda.cupy.random.randint(0, 3, (4,)).astype(numpy.int32)

    def forward(self):
        x = chainer.Variable(self.x)
        t = chainer.Variable(self.t)
        return functions.softmax_cross_entropy(x, t, self.use_cudnn)

    @unittest.skipIf(cuda.cudnn_enabled and
                     cuda.cudnn.cudnn.getVersion() < 3000,
                     'Only cudnn ver>=3 supports softmax-log')
    def test_call_cudnn_forward(self):
        with mock.patch('cupy.cudnn.cudnn.softmaxForward') as func:
            self.forward()
            self.assertEqual(func.called, self.use_cudnn)

    # Note that SoftmaxCrossEntropy does not use cudnn on backward


testing.run_module(__name__, __file__)
