import numpy

from chainer.backends import cuda
from chainer import configuration
from chainer import function
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
                 rmax=1, dmax=0, freeze_running_statistics=False):
        self.running_mean = mean
        self.running_var = var
        self.rmax = rmax
        self.dmax = dmax
        self.r = None
        self.d = None
        self.freeze_running_statistics = freeze_running_statistics

        self.eps = eps
        self.mean_cache = None
        self.decay = decay

    def check_type_forward(self, in_types):
        n_in = type_check.eval(in_types.size())
        if n_in != 3 and n_in != 5:
            raise type_check.InvalidType(
                '%s or %s' % (in_types.size() == 3, in_types.size() == 5),
                '%s == %s' % (in_types.size(), n_in))
        x_type, gamma_type, beta_type = in_types[:3]
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
        if len(in_types) == 5:
            mean_type, var_type = in_types[3:]
            type_check.expect(
                mean_type.dtype == x_type.dtype,
                mean_type.shape == gamma_type.shape,
                var_type.dtype == x_type.dtype,
                var_type.shape == gamma_type.shape,
            )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x, gamma, beta = inputs[:3]

        # Note: If length of inputs is not 5, we must be in train mode.
        if len(inputs) != 5:
            assert configuration.config.train

        if configuration.config.train:
            if self.running_mean is None:
                self.running_mean = xp.zeros_like(gamma)
                self.running_var = xp.zeros_like(gamma)
            else:
                self.running_mean = xp.array(self.running_mean)
                self.running_var = xp.array(self.running_var)
        elif len(inputs) == 5:
            fixed_mean = inputs[3]
            fixed_var = inputs[4]

        head_ndim = gamma.ndim + 1
        expander = (None, Ellipsis) + (None,) * (x.ndim - head_ndim)

        # NOTE(tommi): cuDNN is not used since it does not support
        # batch renormalization
        if configuration.config.train:
            axis = (0,) + tuple(range(head_ndim, x.ndim))
            mean = x.mean(axis=axis)
            var = x.var(axis=axis) + self.eps
        else:
            mean = fixed_mean
            var = fixed_var + self.eps
        self.std = xp.sqrt(var, dtype=var.dtype)

        if not self.freeze_running_statistics or self.r is None:
            if configuration.config.train:
                running_sigma = xp.sqrt(self.running_var + self.eps,
                                        dtype=self.running_mean.dtype)
                self.r = xp.clip(self.std / running_sigma,
                                 1.0 / self.rmax, self.rmax)
                self.d = xp.clip((mean - self.running_mean) / running_sigma,
                                 -self.dmax, self.dmax)

                # Update running statistics:
                m = x.size // gamma[expander].size
                self.running_mean *= self.decay
                adjust = m / max(m - 1., 1.)  # unbiased estimation
                temp_ar = xp.array(mean)
                temp_ar *= (1 - self.decay)
                self.running_mean += temp_ar
                del temp_ar
                self.running_var *= self.decay
                temp_ar = xp.array(var)
                temp_ar *= (1 - self.decay) * adjust
                self.running_var += temp_ar
                del temp_ar
            else:
                self.r = xp.ones_like(gamma)
                self.d = xp.zeros_like(gamma)

        if self.freeze_running_statistics:
            # Need to explicitly cast during gradient check, as r and d are
            # not updated during finite differences
            self.r = self.r.astype(gamma.dtype)
            self.d = self.d.astype(gamma.dtype)

        gamma = gamma[expander]
        beta = beta[expander]

        if xp is numpy:
            self.x_hat = _xhat(x, mean, self.std, expander)
            self.x_hat_renorm = self.x_hat * self.r[expander] + \
                self.d[expander]
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
                          beta, self.r[expander], self.d[expander])

        return y,

    def backward(self, inputs, grad_outputs):
        x, gamma = inputs[:2]
        gy = grad_outputs[0]
        head_ndim = gamma.ndim + 1
        expander = (None, Ellipsis) + (None,) * (x.ndim - head_ndim)
        m = gamma.dtype.type(x.size // gamma.size)
        axis = (0,) + tuple(range(head_ndim, x.ndim))
        xp = cuda.get_array_module(x)
        if len(inputs) == 5:
            # This case is unlikely to be used in practice and so does not
            # need to be optimized for performance.
            mean = inputs[3]
            var = inputs[4] + self.eps
            std = xp.sqrt(var, dtype=var.dtype)
            gs = gamma / std
            gbeta = gy.sum(axis=axis)
            x_hat = _xhat(x, mean, std, expander)
            ggamma = (gy * x_hat).sum(axis=axis)
            gmean = -gs * gbeta
            gvar = -0.5 * gamma / var * ggamma
            gx = gs[expander] * gy
            return gx, ggamma, gbeta, gmean, gvar

        # Note: If length of inputs is not 5, we must be in train mode.
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
                          running_mean=None, running_var=None, decay=0.9):
    """Batch renormalization function.

    This is an extension of batch normalization, which ensures that the
    training and inference models generate the same outputs that depend on
    individual examples rather than the entire minibatch.

    See: `Batch Renormalization: Towards Reducing Minibatch Dependence in \
          Batch-Normalized Models <https://arxiv.org/abs/1702.03275>`_

    .. seealso:: :class:`links.BatchRenormalization`
    .. seealso:: :func:`functions.BatchNormalization`

    """
    return BatchRenormalizationFunction(eps, running_mean, running_var,
                                        decay, rmax, dmax)(x, gamma, beta)


def fixed_batch_renormalization(x, gamma, beta, mean, var, eps=2e-5):
    with configuration.using_config('train', False):
        return BatchRenormalizationFunction(eps, None, None, 0.0)(
            x, gamma, beta, mean, var)
