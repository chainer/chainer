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


def _batch_normalization(expander, gamma, beta, x, mean, var):
    mean = mean[expander]
    std = numpy.sqrt(var)[expander]
    y_expect = (gamma[expander] * (x - mean) / std + beta[expander])
    return y_expect


@testing.parameterize(*(testing.product({
    'param_shape': [(3,), (3, 4), (3, 2, 3)],
    'ndim': [0, 1, 2],
    'dtype': [numpy.float32],
}) + testing.product({
    'param_shape': [(3,)],
    'ndim': [1],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
})))
class TestBatchNormalization(unittest.TestCase):

    def setUp(self):
        self.expander = (None, Ellipsis) + (None,) * self.ndim
        self.eps = 2e-5
        self.decay = 0.9

        self.gamma = numpy.random.uniform(.5, 1,
                                          self.param_shape).astype(self.dtype)
        self.beta = numpy.random.uniform(-1, 1,
                                         self.param_shape).astype(self.dtype)
        head_ndim = self.gamma.ndim + 1
        shape = (5,) + self.param_shape + (2,) * self.ndim
        self.x = numpy.random.uniform(-1, 1, shape).astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1, shape).astype(self.dtype)
        self.ggx = numpy.random.uniform(-1, 1, shape).astype(self.dtype)
        self.gggamma = numpy.random.uniform(-1, 1, self.param_shape).astype(
            self.dtype)
        self.ggbeta = numpy.random.uniform(-1, 1, self.param_shape).astype(
            self.dtype)

        self.args = [self.x, self.gamma, self.beta]
        self.ggargs = [self.ggx, self.gggamma, self.ggbeta]
        self.aggr_axes = (0,) + tuple(six.moves.range(head_ndim, self.x.ndim))
        self.mean = self.x.mean(axis=self.aggr_axes)
        self.var = self.x.var(axis=self.aggr_axes) + self.eps
        self.train = True
        self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-3}
        self.check_backward_options = {'dtype': numpy.float64}
        self.check_double_backward_options = {
            'dtype': numpy.float64, 'atol': 1e-3, 'rtol': 1e-2}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-2, 'rtol': 1e-2}
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 1e-2, 'rtol': 1e-2}

    def batch_normalization(self, *args):
        return functions.batch_normalization(
            *args, decay=self.decay, eps=self.eps)

    def check_forward(self, args, use_cudnn='always'):
        with chainer.using_config('use_cudnn', use_cudnn):
            y = functions.batch_normalization(
                *[chainer.Variable(i) for i in args], running_mean=None,
                running_var=None, decay=self.decay, eps=self.eps)
        self.assertEqual(y.data.dtype, self.dtype)

        y_expect = _batch_normalization(
            self.expander, self.gamma, self.beta, self.x, self.mean, self.var)

        testing.assert_allclose(
            y_expect, y.data, **self.check_forward_options)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.args)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward([cuda.to_gpu(i) for i in self.args])

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_no_cudnn(self):
        self.check_forward([cuda.to_gpu(i) for i in self.args], 'never')

    @attr.cudnn
    @condition.retry(3)
    def test_forward_gpu_non_contiguous(self):
        self.check_forward([cuda.cupy.asfortranarray(cuda.to_gpu(i))
                            for i in self.args])

    def check_backward(self, args, y_grad, use_cudnn='always'):
        with chainer.using_config('use_cudnn', use_cudnn), \
                chainer.using_config('train', self.train):
            gradient_check.check_backward(
                self.batch_normalization, args, y_grad,
                **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.args, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(
            [cuda.to_gpu(i) for i in self.args], cuda.to_gpu(self.gy))

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_no_cudnn(self):
        self.check_backward(
            [cuda.to_gpu(i) for i in self.args], cuda.to_gpu(self.gy), 'never')

    @attr.cudnn
    @condition.retry(3)
    def test_backward_gpu_non_contiguous(self):
        self.check_backward(
            [cuda.cupy.asfortranarray(cuda.to_gpu(i)) for i in self.args],
            cuda.cupy.asfortranarray(cuda.to_gpu(self.gy)))

    def check_double_backward(self, args, y_grad, x_grad_grad,
                              use_cudnn='always'):
        def f(*args):
            y = self.batch_normalization(*args)
            return y * y  # make nonlinear against beta
        with chainer.using_config('use_cudnn', use_cudnn), \
                chainer.using_config('train', self.train):
            gradient_check.check_double_backward(
                f, args, y_grad, x_grad_grad,
                **self.check_double_backward_options)

    @condition.retry(3)
    def test_double_backward_cpu(self):
        self.check_double_backward(self.args, self.gy, self.ggargs)

    @attr.gpu
    @condition.retry(3)
    def test_double_backward_gpu(self):
        self.check_double_backward(
            [cuda.to_gpu(x) for x in self.args], cuda.to_gpu(self.gy),
            [cuda.to_gpu(ggx) for ggx in self.ggargs])

    @attr.cudnn
    @condition.retry(3)
    def test_double_backward_gpu_non_contiguous(self):
        self.check_double_backward(
            [cuda.cupy.asfortranarray(cuda.to_gpu(x)) for x in self.args],
            cuda.cupy.asfortranarray(cuda.to_gpu(self.gy)),
            [cuda.cupy.asfortranarray(cuda.to_gpu(ggx)) for ggx in self.ggargs]
        )


@testing.parameterize(*(testing.product({
    'param_shape': [(3,), (3, 4), (3, 2, 3)],
    'ndim': [0, 1, 2],
    'dtype': [numpy.float32],
}) + testing.product({
    'param_shape': [(3,)],
    'ndim': [1],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
})))
class TestFixedBatchNormalization(unittest.TestCase):

    def setUp(self):
        self.gamma = numpy.random.uniform(.5, 1,
                                          self.param_shape).astype(self.dtype)
        self.beta = numpy.random.uniform(-1, 1,
                                         self.param_shape).astype(self.dtype)
        self.expander = (None, Ellipsis) + (None,) * self.ndim
        shape = (5,) + self.param_shape + (2,) * self.ndim
        self.x = numpy.random.uniform(-1, 1, shape).astype(self.dtype)
        self.eps = 2e-5
        self.decay = 0.0
        head_ndim = self.gamma.ndim + 1
        self.aggr_axes = (0,) + tuple(six.moves.range(head_ndim, self.x.ndim))
        self.mean = numpy.random.uniform(-1, 1,
                                         self.param_shape).astype(self.dtype)
        self.var = numpy.random.uniform(
            0.5, 1, self.param_shape).astype(self.dtype)
        self.args = [self.x, self.gamma, self.beta, self.mean, self.var]

        self.gy = numpy.random.uniform(-1, 1, shape).astype(self.dtype)
        self.ggx = numpy.random.uniform(-1, 1, shape).astype(self.dtype)
        self.gggamma = numpy.random.uniform(-1, 1, self.param_shape).astype(
            self.dtype)
        self.ggbeta = numpy.random.uniform(-1, 1, self.param_shape).astype(
            self.dtype)
        self.ggmean = numpy.random.uniform(-1, 1, self.param_shape).astype(
            self.dtype)
        self.ggvar = numpy.random.uniform(-1, 1, self.param_shape).astype(
            self.dtype)
        self.ggargs = [self.ggx, self.gggamma, self.ggbeta, self.ggmean,
                       self.ggvar]

        self.train = False
        self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-3}
        self.check_backward_options = {'dtype': numpy.float64}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-2, 'rtol': 1e-2}
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 1e-2, 'rtol': 1e-2}

    def batch_normalization(self, *args):
        return functions.fixed_batch_normalization(*args, eps=self.eps)

    def check_forward(self, args, use_cudnn='always'):
        with chainer.using_config('use_cudnn', use_cudnn):
            y = self.batch_normalization(*[chainer.Variable(x) for x in args])
        self.assertEqual(y.data.dtype, self.dtype)

        y_expect = _batch_normalization(
            self.expander, self.gamma, self.beta, self.x, self.mean, self.var)

        testing.assert_allclose(
            y_expect, y.data, **self.check_forward_options)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.args)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward([cuda.to_gpu(i) for i in self.args])

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_no_cudnn(self):
        self.check_forward([cuda.to_gpu(i) for i in self.args], 'never')

    @attr.cudnn
    @condition.retry(3)
    def test_forward_gpu_non_contiguous(self):
        self.check_forward([cuda.cupy.asfortranarray(cuda.to_gpu(i))
                            for i in self.args])

    def check_backward(self, args, y_grad, use_cudnn='always'):
        with chainer.using_config('use_cudnn', use_cudnn), \
                chainer.using_config('train', self.train):
            gradient_check.check_backward(
                self.batch_normalization, args, y_grad,
                **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.args, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(
            [cuda.to_gpu(i) for i in self.args], cuda.to_gpu(self.gy))

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_no_cudnn(self):
        self.check_backward(
            [cuda.to_gpu(i) for i in self.args], cuda.to_gpu(self.gy), 'never')

    @attr.cudnn
    @condition.retry(3)
    def test_backward_gpu_no_contiguous(self):
        self.check_backward(
            [cuda.cupy.asfortranarray(cuda.to_gpu(i)) for i in self.args],
            cuda.cupy.asfortranarray(cuda.to_gpu(self.gy)))

    def check_double_backward(self, args, y_grad, x_grad_grad,
                              use_cudnn='always'):
        def f(*args):
            y = self.batch_normalization(*args)
            return y * y  # make nonlinear against some beta
        with chainer.using_config('use_cudnn', use_cudnn):
            gradient_check.check_double_backward(
                f, args, y_grad, x_grad_grad,
                **self.check_backward_options)

    @condition.retry(3)
    def test_double_backward_cpu(self):
        self.check_double_backward(self.args, self.gy, self.ggargs)


@testing.parameterize(*testing.product({
    'use_cudnn': ['always', 'auto', 'never'],
    # TODO(bkvogel): Check float16 support again in next cuDNN version.
    'dtype': [numpy.float32, numpy.float64],
}))
@attr.cudnn
class TestBatchNormalizationCudnnCall(unittest.TestCase):

    def setUp(self):
        ndim = 0
        param_shape = (3,)
        self.gamma = cuda.cupy.random.uniform(.5, 1,
                                              param_shape).astype(self.dtype)
        self.beta = cuda.cupy.random.uniform(-1, 1,
                                             param_shape).astype(self.dtype)
        self.eps = 2e-5
        shape = (7,) + param_shape + (2,) * ndim
        self.x = cuda.cupy.random.uniform(-1, 1, shape).astype(self.dtype)
        self.gy = cuda.cupy.random.uniform(-1, 1, shape).astype(self.dtype)
        self.args = [self.x, self.gamma, self.beta]
        head_ndim = self.gamma.ndim + 1
        self.aggr_axes = (0,) + tuple(six.moves.range(head_ndim, self.x.ndim))
        self.mean = self.x.mean(axis=self.aggr_axes)
        self.var = self.x.var(axis=self.aggr_axes) + self.eps
        with chainer.using_config('use_cudnn', self.use_cudnn):
            self.expect = chainer.should_use_cudnn('>=auto', 5000)

    def forward(self):
        return functions.batch_normalization(
            *[chainer.Variable(i) for i in self.args], eps=self.eps,
            running_mean=self.mean, running_var=self.var)

    def test_call_cudnn_forward(self):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            with mock.patch(
                    'cupy.cudnn.cudnn.batchNormalizationForwardTraining'
            ) as func:
                self.forward()
                self.assertEqual(func.called, self.expect)

    def test_call_cudnn_backward(self):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            y = self.forward()
            y.grad = self.gy
            with mock.patch(
                    'cupy.cudnn.cudnn.batchNormalizationBackward'
            ) as func:
                y.backward()
                self.assertEqual(func.called, self.expect)


testing.run_module(__name__, __file__)
