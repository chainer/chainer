import warnings

import numpy

from chainer import backend
from chainer.backends import cuda
from chainer import configuration
from chainer import function
from chainer.functions.normalization import batch_normalization
from chainer.utils import type_check


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
        warnings.warn(
            'The attributes of BatchRenormalizationFunction '
            'are deprecated. '
            'Consider setting update_statistics=True to '
            'batch_renormalization to update running statistics.',
            DeprecationWarning)

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
            gamma_type.dtype.kind == 'f',
            gamma_type.dtype == beta_type.dtype,
            gamma_type.shape == beta_type.shape,
        )

    def forward(self, inputs):
        xp = backend.get_array_module(*inputs)
        x, gamma, beta = inputs

        # Note: we must be in train mode.
        assert configuration.config.train

        head_ndim = gamma.ndim + 1
        expander = (None, Ellipsis) + (None,) * (x.ndim - head_ndim)

        # NOTE(tommi): cuDNN is not used since it does not support
        # batch renormalization
        axis = (0,) + tuple(range(head_ndim, x.ndim))
        mean = x.mean(axis=axis, dtype=gamma.dtype)
        var = x.var(axis=axis, dtype=gamma.dtype)
        self.std = xp.sqrt(var + self.eps, dtype=var.dtype)

        running_sigma = xp.sqrt(self._running_var + self.eps,
                                dtype=self._running_mean.dtype)
        self.r = xp.clip(self.std / running_sigma,
                         1.0 / self.rmax, self.rmax)
        d = xp.clip(
            (mean - self._running_mean) / running_sigma,
            -self.dmax, self.dmax)

        gamma = gamma[expander]
        beta = beta[expander]

        if xp is numpy:
            self.x_hat = _xhat(x, mean, self.std, expander)
            self.x_hat_renorm = self.x_hat * self.r[expander] + d[expander]
            y = gamma * self.x_hat_renorm
            y += beta
            y = y.astype(dtype=x.dtype)
        else:
            self.x_hat, self.x_hat_renorm, y = cuda.elementwise(
                'T x, U mean, U std, U gamma, U beta, U r, U d',
                'U x_hat, U x_hat_renorm, T y',
                '''
                x_hat = (x - mean) / std;
                x_hat_renorm = x_hat * r + d;
                y = gamma * x_hat_renorm + beta;
                ''',
                'brn_fwd')(
                    x, mean[expander], self.std[expander], gamma, beta,
                    self.r[expander], d[expander])

        if self.update_statistics:
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
        gbeta = gy.sum(axis=axis, dtype=gamma.dtype)
        ggamma = (gy * self.x_hat_renorm).sum(axis=axis)
        gsigma_batch = (gy * self.x_hat).sum(axis=axis)
        if xp is numpy:
            scale = (self.r * gamma / self.std)[expander]
            gx = scale * (gy - (self.x_hat * gsigma_batch[expander] +
                                gbeta[expander]) / m)
            gx = gx.astype(dtype=x.dtype)
        else:
            inv_m = numpy.float32(1) / m
            gx = cuda.elementwise(
                'T gy, U x_hat, U gamma, U std, U gsigma_batch, U gbeta, \
                U inv_m, U r',
                'T gx',
                'gx = (r * gamma / std) * (gy - (x_hat * gsigma_batch + gbeta) * \
                inv_m)',
                'brn_bwd')(
                    gy, self.x_hat, gamma[expander],
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

    .. note::

        This function does not perform in-place update to
        ``running_mean`` and ``running_var`` by default, contrary to
        :func:`~chainer.functions.batch_normalization`.
        If the function is called, it will not be possible to access the
        updated running mean and variance statistics, because they are members
        of the function object, which cannot be accessed by the caller.
        If it is desired to update the running statistics, call the function
        with ``update_statistics=True`` option.

    .. note::

        For the consistency with Batch Normalization, this function
        intentionally ignores some of the theoretical flaws in Algorithm 1 of
        the Batch Renormalization paper:

        - ``F.batch_renormalization`` maintains the moving average of variances
          :math:`\\sigma^2`, while the original paper maintains the moving
          average of standard deviations :math:`\\sigma`.
        - ``F.batch_renormalization`` applies Bessel's correction to update the
          moving average of variances.

    See: `Batch Renormalization: Towards Reducing Minibatch Dependence in
    Batch-Normalized Models <https://arxiv.org/abs/1702.03275>`_

    .. seealso::

        :class:`~chainer.links.BatchRenormalization` to manage the model
        parameters (``gamma``, ``beta``) and the statistics (``running_mean``,
        ``running_var``).

    """
    if running_mean is None:
        raise TypeError('running_mean is required')
    if running_var is None:
        raise TypeError('running_var is required')
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
