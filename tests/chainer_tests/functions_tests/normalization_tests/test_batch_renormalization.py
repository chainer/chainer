import unittest

import numpy
import six

import chainer
from chainer import cuda
from chainer.functions.normalization import batch_renormalization
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


def _batch_renormalization(expander, gamma, beta, x, mean, var, r, d):
    mean = mean[expander]
    std = numpy.sqrt(var)[expander]
    y_expect = (gamma[expander] * ((x - mean) / std * r + d) + beta[expander])
    return y_expect


@testing.parameterize(*(testing.product({
    'param_shape': [(3, 4), (3, 2, 3)],
    'ndim': [0, 1, 2],
    'eps': [2e-5, 1e-1],
    'dtype': [numpy.float32],
}) + testing.product({
    'param_shape': [(3,)],
    'ndim': [1],
    'eps': [2e-5, 1e-1],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
})))
class TestBatchRenormalization(unittest.TestCase):

    def setUp(self):
        self.expander = (None, Ellipsis) + (None,) * self.ndim
        self.aggr_axes = (0,) + tuple(six.moves.range(2, self.ndim + 2))
        self.decay = 0.9

        self.rmax = self.dtype(3)
        self.dmax = self.dtype(5)

        self.gamma = numpy.random.uniform(.5, 1, (3,)).astype(self.dtype)
        self.beta = numpy.random.uniform(-1, 1, (3,)).astype(self.dtype)

        shape = (5, 3) + (2,) * self.ndim
        self.x = numpy.random.uniform(-1, 1, shape).astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1, shape).astype(self.dtype)

        self.args = [self.x, self.gamma, self.beta]
        self.mean = self.x.mean(axis=self.aggr_axes)
        self.var = self.x.var(axis=self.aggr_axes) + self.eps
        # Need to add some noise to running_mean and running_var,
        # otherwise we will always get r=1, d=0
        self.running_mean = self.mean + numpy.random.uniform(
            -1, 1, self.mean.shape).astype(self.dtype)
        self.running_var = numpy.abs(self.var + numpy.random.uniform(
            -1, 1, self.var.shape).astype(self.dtype))

        self.train = True
        self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-3}
        self.check_backward_options = {'dtype': numpy.float64}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-3, 'rtol': 1e-2}
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 1e-3, 'rtol': 1e-2}

    def check_forward(self, args):
        with chainer.using_config('train',  self.train):
            y = batch_renormalization.batch_renormalization(
                *[chainer.Variable(i) for i in args],
                rmax=self.rmax, dmax=self.dmax, running_mean=self.running_mean,
                running_var=self.running_var, decay=self.decay, eps=self.eps)
        self.assertEqual(y.data.dtype, self.dtype)

        sigma_batch = numpy.sqrt(self.var)
        running_sigma = numpy.sqrt(self.running_var + self.eps)
        r = numpy.clip(sigma_batch / running_sigma, 1.0 / self.rmax, self.rmax)
        d = numpy.clip((self.mean - self.running_mean) / running_sigma,
                       -self.dmax, self.dmax)
        y_expect = _batch_renormalization(
            self.expander, self.gamma, self.beta, self.x, self.mean, self.var,
            r[self.expander], d[self.expander])

        testing.assert_allclose(
            y_expect, y.data, **self.check_forward_options)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.args)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward([cuda.to_gpu(i) for i in self.args])

    def check_backward(self, args, y_grad):
        with chainer.using_config('train',  self.train):
            # Freezing the update of running statistics is needed in order to
            # make gradient check work, since the parameters r and d in batch
            # renormalization are calculated from the input, but should be
            # treated as constants during gradient computation, as stated in
            # the paper.
            gradient_check.check_backward(
                batch_renormalization.BatchRenormalizationFunction(
                    mean=self.running_mean, var=self.running_var,
                    decay=self.decay, eps=self.eps, rmax=self.rmax,
                    dmax=self.dmax,
                    freeze_running_statistics=True), args, y_grad,
                **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.args, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(
            [cuda.to_gpu(i) for i in self.args], cuda.to_gpu(self.gy))


@testing.parameterize(*testing.product({
    'ndim': [0, 1, 2, 3],
    'eps': [2e-5, 1e-1],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestFixedBatchRenormalization(unittest.TestCase):

    def setUp(self):
        self.gamma = numpy.random.uniform(.5, 1, (3,)).astype(self.dtype)
        self.beta = numpy.random.uniform(-1, 1, (3,)).astype(self.dtype)
        self.expander = (None, Ellipsis) + (None,) * self.ndim

        self.rmax = self.dtype(3)
        self.dmax = self.dtype(5)

        shape = (5, 3) + (2,) * self.ndim
        self.x = numpy.random.uniform(-1, 1, shape).astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1, shape).astype(self.dtype)
        self.decay = 0.0
        self.aggr_axes = (0,) + tuple(six.moves.range(2, self.ndim + 2))

        self.mean = numpy.random.uniform(-1, 1, (3,)).astype(self.dtype)
        self.var = numpy.random.uniform(
            0.5, 1, (3,)).astype(self.dtype)
        self.args = [self.x, self.gamma, self.beta, self.mean, self.var]
        self.train = False
        self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-3}
        self.check_backward_options = {'dtype': numpy.float64}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-3, 'rtol': 1e-2}
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 1e-3, 'rtol': 1e-2}

    def check_forward(self, args):
        with chainer.using_config('train',  self.train):
            y = batch_renormalization.fixed_batch_renormalization(
                *[chainer.Variable(i) for i in args],
                eps=self.eps)
        self.assertEqual(y.data.dtype, self.dtype)

        y_expect = _batch_renormalization(
            self.expander, self.gamma, self.beta, self.x, self.mean,
            self.var + self.eps,
            1, 0)

        testing.assert_allclose(
            y_expect, y.data, **self.check_forward_options)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.args)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward([cuda.to_gpu(i) for i in self.args])

    def check_backward(self, args, y_grad):
        with chainer.using_config('train',  self.train):
            gradient_check.check_backward(
                batch_renormalization.BatchRenormalizationFunction(
                    mean=None, var=None,
                    decay=self.decay, eps=self.eps,
                    rmax=self.rmax, dmax=self.dmax,
                    freeze_running_statistics=True),
                args, y_grad, **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.args, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(
            [cuda.to_gpu(i) for i in self.args], cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)
