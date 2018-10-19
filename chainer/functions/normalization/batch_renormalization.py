import warnings

import numpy

from chainer import backend
from chainer.backends import cuda
from chainer import configuration
from chainer import function
from chainer.functions.normalization import batch_normalization
from chainer.utils import type_check


def _as4darray(arr):
    if arr.ndim == 0:
        return arr.reshape(1, 1, 1, 1)
    elif arr.ndim == 4:
        return arr
    else:
        return arr.reshape(arr.shape[0], -1, 1, 1)


def _xhat(x, mean, std, expander):
    x_mu = x - mean[expander]
    x_mu /= std[expander]
    return x_mu


class BatchRenormalizationFunction(function.Function):

    def __init__(self, eps=2e-5, mean=None, var=None, decay=0.9,
                 rmax=1, dmax=0, update_statistics=True):
        self._running_mean = mean
        self._running_var = var
        self.rmax = rmax
        self.dmax = dmax
        self.r = None
        self.update_statistics = update_statistics

        self.eps = eps
        self.decay = decay

    def _warn_accessing_property(self):
        msg = '''\
batch_renormalization function updates running statistics by default. The \
properties of BatchRenormalizationFunction should not be accessed.

Set update_statistics=False to stop updating.'''
        if self.update_statistics:
            warnings.warn(msg, DeprecationWarning)

    @property
    def running_mean(self):
        self._warn_accessing_property()
        return self._running_mean

    @property
    def running_var(self):
        self._warn_accessing_property()
        return self._running_var

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        x_type, gamma_type, beta_type = in_types
        M = type_check.eval(gamma_type.ndim)
        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim >= gamma_type.ndim + 1,
            x_type.shape[1:1 + M] == gamma_type.shape,
            # TODO(tkerola): Check shape
            gamma_type.dtype == x_type.dtype,
            beta_type.dtype == x_type.dtype,
            gamma_type.shape == beta_type.shape,
        )

    def forward(self, inputs):
        xp = backend.get_array_module(*inputs)
        x, gamma, beta = inputs

        # Note: we must be in train mode.
        assert configuration.config.train

        if not self.update_statistics:
            self._running_mean = xp.array(self._running_mean)
            self._running_var = xp.array(self._running_var)

        head_ndim = gamma.ndim + 1
        expander = (None, Ellipsis) + (None,) * (x.ndim - head_ndim)

        # NOTE(tommi): cuDNN is not used since it does not support
        # batch renormalization
        axis = (0,) + tuple(range(head_ndim, x.ndim))
        mean = x.mean(axis=axis)
        var = x.var(axis=axis) + self.eps
        self.std = xp.sqrt(var, dtype=var.dtype)

        running_sigma = xp.sqrt(self._running_var + self.eps,
                                dtype=self._running_mean.dtype)
        self.r = xp.clip(self.std / running_sigma,
                         1.0 / self.rmax, self.rmax)
        d = xp.clip(
            (mean - self._running_mean) / running_sigma,
            -self.dmax, self.dmax)

        # Update running statistics:
        m = x.size // gamma[expander].size
        self._running_mean *= self.decay
        adjust = m / max(m - 1., 1.)  # unbiased estimation
        temp_ar = xp.array(mean)
        temp_ar *= (1 - self.decay)
        self._running_mean += temp_ar
        del temp_ar
        self._running_var *= self.decay
        temp_ar = xp.array(var)
        temp_ar *= (1 - self.decay) * adjust
        self._running_var += temp_ar
        del temp_ar

        gamma = gamma[expander]
        beta = beta[expander]

        if xp is numpy:
            self.x_hat = _xhat(x, mean, self.std, expander)
            self.x_hat_renorm = self.x_hat * self.r[expander] + d[expander]
            y = gamma * self.x_hat_renorm
            y += beta
        else:
            self.x_hat, self.x_hat_renorm, y = cuda.elementwise(
                'T x, T mean, T std, T gamma, T beta, T r, T d',
                'T x_hat, T x_hat_renorm, T y',
                '''
                x_hat = (x - mean) / std;
                x_hat_renorm = x_hat * r + d;
                y = gamma * x_hat_renorm + beta;
                ''',
                'bn_fwd')(x, mean[expander], self.std[expander], gamma,
                          beta, self.r[expander], d[expander])

        return y,

    def backward(self, inputs, grad_outputs):
        x, gamma, _ = inputs
        gy = grad_outputs[0]
        head_ndim = gamma.ndim + 1
        expander = (None, Ellipsis) + (None,) * (x.ndim - head_ndim)
        m = gamma.dtype.type(x.size // gamma.size)
        axis = (0,) + tuple(range(head_ndim, x.ndim))
        xp = backend.get_array_module(x)

        # Note: we must be in train mode.
        assert configuration.config.train
        # NOTE(tommi): cuDNN is not used since it does not support
        # batch renormalization
        gbeta = gy.sum(axis=axis)
        ggamma = (gy * self.x_hat_renorm).sum(axis=axis)
        gsigma_batch = (gy * self.x_hat).sum(axis=axis)
        if xp is numpy:
            scale = (self.r * gamma / self.std)[expander]
            gx = scale * (gy - (self.x_hat * gsigma_batch[expander] +
                                gbeta[expander]) / m)
        else:
            inv_m = numpy.float32(1) / m
            gx = cuda.elementwise(
                'T gy, T x_hat, T gamma, T std, T gsigma_batch, T gbeta, \
                T inv_m, T r',
                'T gx',
                'gx = (r * gamma / std) * (gy - (x_hat * gsigma_batch + gbeta) * \
                inv_m)',
                'bn_bwd')(gy, self.x_hat, gamma[expander],
                          self.std[expander], gsigma_batch[expander],
                          gbeta[expander], inv_m, self.r[expander])
        return gx, ggamma, gbeta


def batch_renormalization(x, gamma, beta, rmax, dmax, eps=2e-5,
                          running_mean=None, running_var=None, decay=0.9,
                          update_statistics=False):
    """Batch renormalization function.

    This is an extension of batch normalization, which ensures that the
    training and inference models generate the same outputs that depend on
    individual examples rather than the entire minibatch.

    Note: This function does not perform in-place update to
    ``running_mean`` and ``running_var``, contrary to
    :func:`~chainer.functions.batch_normalization`.
    If the function is called, it will not be possible to access the
    updated running mean and variance statistics, because they are members
    of the function object, which cannot be accessed by the caller.
    If it is desired to access the updated running statistics, it is necessary
    to get a new instance of the function object, call the object, and then
    access the ``running_mean`` and/or ``running_var`` attributes. See the
    corresponding Link class for an example of how to do this.

    See: `Batch Renormalization: Towards Reducing Minibatch Dependence in \
          Batch-Normalized Models <https://arxiv.org/abs/1702.03275>`_

    .. seealso:: :class:`links.BatchRenormalization`
    .. seealso:: :class:`functions.normalization.batch_normalization.BatchNormalization`  # NOQA

    """
    return BatchRenormalizationFunction(
        eps, running_mean, running_var, decay, rmax, dmax, update_statistics
    )(x, gamma, beta)


def fixed_batch_renormalization(x, gamma, beta, mean, var, eps=2e-5):
    warnings.warn(
        'fixed_batch_renormalization is deprecated. '
        'Use fixed_batch_normalization instead.',
        DeprecationWarning)
    with configuration.using_config('train', False):
        return batch_normalization.fixed_batch_normalization(
            x, gamma, beta, mean, var, eps
        )
