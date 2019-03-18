import warnings

import numpy

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer.backends import intel64
from chainer import configuration
from chainer import function_node
from chainer.utils import argument
from chainer.utils import collections_abc
from chainer.utils import type_check
import chainerx


if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cuda.libcudnn
    _cudnn_version = cuda.cuda.cudnn.getVersion()


def _compute_axis(x_ndim, gamma_ndim=1, axis=None):
    if axis is None:
        axis = (0,) + tuple(range(gamma_ndim + 1, x_ndim))
    return axis


# Computes a complementary set of axis
def _compute_key_axis(x_ndim, gamma_ndim=1, axis=None):
    axis = _compute_axis(x_ndim, gamma_ndim, axis)
    key_axis = tuple([i for i in range(x_ndim) if i not in axis])
    return key_axis


class BatchNormalization(function_node.FunctionNode):

    mean = None
    inv_std = None

    def __init__(self, eps=2e-5, mean=None, var=None, decay=0.9, axis=None):
        self.running_mean = mean
        self.running_var = var

        # Note: cuDNN requires that eps be greater than or equals to
        # CUDNN_BN_MIN_EPSILON. Otherwise, an error will occur.
        # See CUDNN_BN_MIN_EPSILON value in cudnn.h to verify minimum allowable
        # value.
        self.eps = eps
        if chainer.should_use_cudnn('>=auto'):
            if eps < libcudnn.CUDNN_BN_MIN_EPSILON:
                raise RuntimeError(
                    'cuDNN does not allow an eps value '
                    'less than {}.'.format(libcudnn.CUDNN_BN_MIN_EPSILON))
        self.decay = decay
        if isinstance(axis, collections_abc.Sequence):
            for i in range(1, len(axis)):
                if axis[i - 1] >= axis[i]:
                    msg = 'numbers in axis must be sorted in ascending order'
                    raise RuntimeError(msg)
        elif isinstance(axis, int):
            axis = axis,
        elif axis is not None:
            raise RuntimeError('axis must be int, tuple of int or None')
        self.axis = axis

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        x_type, gamma_type, beta_type = in_types
        type_check.expect(
            x_type.dtype.kind == 'f',
            gamma_type.dtype.kind == 'f',
            gamma_type.dtype == beta_type.dtype,
            gamma_type.shape == beta_type.shape,
        )
        _x_ndim = type_check.eval(x_type.ndim)
        _gamma_ndim = type_check.eval(gamma_type.ndim)
        _axis = _compute_axis(_x_ndim, _gamma_ndim, self.axis)
        type_check.expect(
            x_type.ndim >= len(_axis),
        )
        _key_axis = _compute_key_axis(_x_ndim, _gamma_ndim, _axis)
        type_check.expect(
            gamma_type.ndim == len(_key_axis),
        )
        for i in range(len(_key_axis)):
            type_check.expect(
                x_type.shape[_key_axis[i]] == gamma_type.shape[i],
            )

    def forward_chainerx(self, inputs):
        # TODO(niboshi): Support conditions implemented as fallback

        # Running statistics are required.
        if self.running_mean is None or self.running_var is None:
            return chainer.Fallback

        # Fall back if the running statistics are non-contiguous CUDA arrays
        # since they are not supported by cuDNN.
        # Assert that both running statistics belong to the same backend.
        if self.running_mean.device.backend.name == 'cuda' and not (
                self.running_mean.is_contiguous
                and self.running_var.is_contiguous):
            return chainer.Fallback

        x, gamma, beta = inputs
        axis_chx = _chainerx_compute_axis(x.ndim, gamma.ndim, self.axis)
        if not _chainerx_is_supported(x.device, axis_chx):
            return chainer.Fallback

        y = chainerx.batch_norm(
            x, gamma, beta, self.running_mean, self.running_var,
            self.eps, self.decay, axis_chx)
        return y,

    def forward(self, inputs):
        self.retain_inputs((0, 1))
        x, gamma, beta = inputs

        xp = backend.get_array_module(x)
        if self.running_mean is None:
            self.running_mean = xp.zeros_like(gamma, dtype=x.dtype)
            self.running_var = xp.zeros_like(gamma, dtype=x.dtype)

        self.axis = _compute_axis(x.ndim, gamma.ndim, self.axis)
        self.key_axis = _compute_key_axis(x.ndim, gamma.ndim, self.axis)

        if all(x.shape[i] == 1 for i in self.axis):
            if 0 in self.axis:
                warnings.warn(
                    'A batch with no more than one sample has been given'
                    ' to F.batch_normalization. F.batch_normalization'
                    ' will always output a zero tensor for such batches.'
                    ' This could be caused by incorrect configuration in'
                    ' your code (such as running evaluation while'
                    ' chainer.config.train=True),'
                    ' but could also happen in the last batch of training'
                    ' if non-repeating iterator is used.',
                    UserWarning)
            else:
                warnings.warn(
                    'F.batch_normalization received a batch with single'
                    ' dimensions along all axes that are used for aggregating'
                    ' statistics. F.batch_normalization'
                    ' will always output a zero tensor for such batches.',
                    UserWarning)

        # TODO(niboshi): Refactor calculation of expander and axis into a
        # function and call it just before they are used.

        # expander inserts singleton dimensions to gamma and beta so that they
        # can be broadcasted with x.
        expander = [None for _ in range(x.ndim)]
        for i in self.key_axis:
            expander[i] = slice(None)
        expander = tuple(expander)
        self.expander = expander

        self.mode = _BNMode(x, gamma, self.key_axis)
        self.use_cudnn = self.mode.can_use_cudnn(xp)
        self.use_ideep = self.mode.can_use_ideep()

        if self.use_ideep:
            # TODO(niboshi): Refactor iDeep part into a separate method
            expand_dim = False
            if x.ndim == 2:
                expand_dim = True
                x = x[:, :, None, None]

            y, self.mean, self.var, self.inv_std = (
                intel64.ideep.batchNormalization.Forward(
                    intel64.ideep.array(x.astype(gamma.dtype, copy=False)),
                    intel64.ideep.array(gamma),
                    intel64.ideep.array(beta),
                    None,
                    None,
                    self.eps
                ))
            y = y.astype(x.dtype, copy=False)

            m = x.size // gamma.size
            adjust = m / max(m - 1., 1.)

            # Update running_mean
            if isinstance(self.running_mean, intel64.ideep.mdarray):
                self.running_mean.inplace_axpby(
                    self.decay, (1 - self.decay), self.mean)
            else:
                self.running_mean *= self.decay
                self.running_mean += self.mean * (1 - self.decay)

            # Update running_var
            if isinstance(self.running_var, intel64.ideep.mdarray):
                self.running_var.inplace_axpby(
                    self.decay, (1 - self.decay), self.var * adjust)
            else:
                self.running_var *= self.decay
                self.running_var += self.var * adjust * (1 - self.decay)

            if expand_dim:
                y = numpy.squeeze(y, axis=(2, 3))

        elif self.use_cudnn:
            # self.mean and self.inv_std are used as buffers to save
            # intermediate results computed during forward pass. These buffers
            # are used to speed-up backward pass.
            y, self.mean, self.inv_std = (
                cudnn.batch_normalization_forward_training(
                    x, gamma, beta, self.running_mean, self.running_var,
                    None, None, self.eps, self.decay,
                    self.mode.is_for_conv2d, self.mode.get_cudnn_mode(),
                    chainer.is_debug()))
        else:
            # Generic CPU and GPU implementation

            gamma = gamma[expander]
            beta = beta[expander]
            self.mean = x.mean(axis=self.axis, dtype=gamma.dtype)
            var = x.var(axis=self.axis, dtype=gamma.dtype)
            if xp is numpy:
                self.inv_std = numpy.reciprocal(numpy.sqrt(
                    var + self.eps, dtype=gamma.dtype))
            else:
                self.inv_std = cuda.cupyx.rsqrt(var + self.eps,
                                                dtype=gamma.dtype)
            y = _apply_bn_fwd(xp, x, self.mean[expander],
                              self.inv_std[expander], gamma, beta)
            # Update running statistics
            m = x.size // gamma.size
            adjust = m / max(m - 1., 1.)  # unbiased estimation

            xp = backend.get_array_module(self.running_mean, self.running_var)
            if xp is chainerx:
                self.running_mean, self.running_var = backend.from_chx(
                    (self.running_mean, self.running_var))

            self.running_mean *= self.decay
            self.running_mean += (1 - self.decay) * self.mean
            self.running_var *= self.decay
            self.running_var += (1 - self.decay) * adjust * var

            if xp is chainerx:
                self.running_mean = backend.to_chx(self.running_mean)
                self.running_var = backend.to_chx(self.running_var)

        return y,

    def backward(self, indexes, grad_outputs):
        x, gamma = self.get_retained_inputs()
        gy, = grad_outputs

        if self.use_ideep:
            assert self.var is not None
            var = self.var
        else:
            var = None

        f = BatchNormalizationGrad(
            self.eps, self.use_cudnn, self.mode, self.expander, self.axis,
            self.mean, var, self.inv_std, self.key_axis)
        return f.apply((x, gamma, gy))


class BatchNormalizationGrad(function_node.FunctionNode):

    def __init__(self, eps, use_cudnn, mode, expander, axis, mean, var,
                 inv_std, key_axis):
        self.eps = eps
        self.use_cudnn = use_cudnn
        self.use_ideep = mode.can_use_ideep()
        self.mode = mode
        self.expander = expander
        self.axis = axis
        self.mean = mean
        self.var = var  # Only used in iDeep implementation
        self.inv_std = inv_std
        self.key_axis = key_axis

    def forward(self, inputs):
        self.retain_inputs((0, 1, 2))
        x, gamma, gy = inputs
        expander = self.expander
        xp = backend.get_array_module(x)

        if self.use_ideep:
            # TODO(niboshi): Refactor iDeep part into a separate method
            expand_dim = False
            if x.ndim == 2:
                expand_dim = True
                x = x[:, :, None, None]
                gy = gy[:, :, None, None]

            gx, gW = intel64.ideep.batchNormalization.Backward(
                intel64.ideep.array(x.astype(gamma.dtype, copy=False)),
                intel64.ideep.array(gy.astype(gamma.dtype, copy=False)),
                self.mean,
                self.var,
                intel64.ideep.array(gamma),
                self.eps)

            gx = gx.astype(x.dtype, copy=False)
            ggamma, gbeta = gW[:2]

            if expand_dim:
                gx = numpy.squeeze(gx, axis=(2, 3))

        elif self.use_cudnn:
            gx, ggamma, gbeta = cudnn.batch_normalization_backward(
                x, gamma, gy, self.mean, self.inv_std, self.eps,
                self.mode.is_for_conv2d, self.mode.get_cudnn_mode(),
                chainer.is_debug())
        else:
            # CPU and GPU implementation
            if isinstance(gy, intel64.mdarray):
                # intel64.mdarray does not support dtype option in sum, so we
                # convert it to numpy here.
                gy = numpy.asarray(gy)

            gbeta = gy.sum(axis=self.axis, dtype=gamma.dtype)
            x_hat = _x_hat(x, self.mean[expander], self.inv_std[expander])
            ggamma = (gy * x_hat).sum(axis=self.axis, dtype=gamma.dtype)

            inv_m = gamma.dtype.type(1. / (x.size // gamma.size))
            if xp is numpy:
                gx = (gamma * self.inv_std)[expander] * (
                    gy - (x_hat * ggamma[expander] + gbeta[expander]) * inv_m)
                gx = gx.astype(dtype=x.dtype, copy=False)
            else:
                gx = cuda.elementwise(
                    '''
                    T gy, U x_hat, U gamma, U inv_std, U ggamma, U gbeta,
                    U inv_m
                    ''',
                    'T gx',
                    '''
                    gx = (gamma * inv_std) * (
                        gy - (x_hat * ggamma + gbeta) * inv_m)
                    ''', 'bn_bwd')(gy, x_hat, gamma[expander],
                                   self.inv_std[expander], ggamma[expander],
                                   gbeta[expander], inv_m)
        self.retain_inputs((0, 1, 2))
        self.retain_outputs((0, 1))
        return gx, ggamma, gbeta

    def backward(self, indexes, grad_outputs):
        F = chainer.functions
        expander = self.expander

        x, gamma, gy = self.get_retained_inputs()
        gx1, ggamma1 = self.get_retained_outputs()
        ggx1, gggamma1, ggbeta1 = grad_outputs
        xp = backend.get_array_module(x)

        if gamma.dtype != x.dtype:
            gamma = F.cast(gamma, x.dtype)
            ggamma1 = F.cast(ggamma1, x.dtype)
            gggamma1 = F.cast(gggamma1, x.dtype)
            ggbeta1 = F.cast(ggbeta1, x.dtype)

        # auxiliary values
        inv_m = gamma.dtype.type(1. / (x.size // gamma.size))
        r = 0 if ggx1 is None else F.sum(gx1 * ggx1, axis=self.axis)
        coeff = gamma * self.inv_std
        coeff_m = coeff * inv_m
        x_hat = _x_hat(x, self.mean[expander], self.inv_std[expander])

        # handle None in output gradients
        ggx1 = _zero_if_none(xp, ggx1, x.shape, x.dtype)
        gggamma1 = _zero_if_none(xp, gggamma1, gamma.shape, gamma.dtype)
        ggbeta1 = _zero_if_none(xp, ggbeta1, gamma.shape, gamma.dtype)

        gggamma2 = gggamma1 - coeff_m * F.sum(x_hat * ggx1, axis=self.axis)
        ggbeta2 = ggbeta1 - coeff_m * F.sum(ggx1, axis=self.axis)

        ggamma2 = r / gamma

        gx_hat2 = (gggamma2[expander] * gy -
                   (coeff_m * ggamma1)[expander] * ggx1)
        gstd2 = -self.inv_std * (r + F.sum(x_hat * gx_hat2, axis=self.axis))
        gmean2 = -self.inv_std * F.sum(gx_hat2, axis=self.axis)
        gx2 = self.inv_std[expander] * gx_hat2 + inv_m * (
            gmean2[expander] + x_hat * gstd2[expander])
        ggy2 = (gggamma2[expander] * x_hat + ggbeta2[expander]
                + coeff[expander] * ggx1)

        gx2 = chainer.functions.cast(gx2, x.dtype)
        ggy2 = chainer.functions.cast(ggy2, gy.dtype)

        return gx2, ggamma2, ggy2


class FixedBatchNormalization(function_node.FunctionNode):

    inv_std = None
    inv_var = None

    def __init__(self, eps=2e-5, axis=None):
        # Note: cuDNN requires that eps be greater than or equals to
        # CUDNN_BN_MIN_EPSILON. Otherwise, an error will occur.
        # See CUDNN_BN_MIN_EPSILON value in cudnn.h to verify minimum allowable
        # value.
        self.eps = eps
        if chainer.should_use_cudnn('>=auto'):
            if eps < libcudnn.CUDNN_BN_MIN_EPSILON:
                raise RuntimeError(
                    'cuDNN does not allow an eps value '
                    'less than {}.'.format(libcudnn.CUDNN_BN_MIN_EPSILON))
        if isinstance(axis, collections_abc.Sequence):
            for i in range(1, len(axis)):
                if axis[i - 1] >= axis[i]:
                    msg = 'numbers in axis must be sorted in ascending order'
                    raise RuntimeError(msg)
        elif isinstance(axis, int):
            axis = axis,
        elif axis is not None:
            raise RuntimeError('axis must be int, tuple of int or None')
        self.axis = axis

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 5)
        x_type, gamma_type, beta_type, mean_type, var_type = in_types
        type_check.expect(
            x_type.dtype.kind == 'f',
            # TODO(beam2d): Check shape
            gamma_type.dtype.kind == 'f',
            beta_type.dtype == gamma_type.dtype,
            mean_type.dtype == gamma_type.dtype,
            var_type.dtype == gamma_type.dtype,
            beta_type.shape == gamma_type.shape,
            mean_type.shape == gamma_type.shape,
            var_type.shape == gamma_type.shape,
        )
        _x_ndim = type_check.eval(x_type.ndim)
        _gamma_ndim = type_check.eval(gamma_type.ndim)
        _axis = _compute_axis(_x_ndim, _gamma_ndim, self.axis)
        type_check.expect(
            x_type.ndim >= len(_axis),
        )
        _key_axis = _compute_key_axis(_x_ndim, _gamma_ndim, _axis)
        type_check.expect(
            gamma_type.ndim == len(_key_axis),
        )
        for i in range(len(_key_axis)):
            type_check.expect(
                x_type.shape[_key_axis[i]] == gamma_type.shape[i],
            )

    def forward_chainerx(self, inputs):
        # TODO(niboshi): Support conditions implemented as fallback

        # TODO(niboshi): chainerx.fixed_batch_norm does not support backward
        if chainer.config.enable_backprop:
            return chainer.Fallback

        x, gamma, beta, mean, var = inputs
        axis_chx = _chainerx_compute_axis(x.ndim, gamma.ndim, self.axis)
        if not _chainerx_is_supported(x.device, axis_chx):
            return chainer.Fallback

        y = chainerx.fixed_batch_norm(
            x, gamma, beta, mean, var, self.eps, axis_chx)
        return y,

    def forward(self, inputs):
        self.retain_inputs((0, 1, 3, 4))
        x, gamma, beta, mean, var = inputs
        xp = backend.get_array_module(x)

        self.axis = _compute_axis(x.ndim, gamma.ndim, self.axis)
        self.key_axis = _compute_key_axis(x.ndim, gamma.ndim, self.axis)

        # expander inserts singleton dimensions to gamma and beta so that they
        # can be broadcasted with x.
        expander = [None for _ in range(x.ndim)]
        for i in self.key_axis:
            expander[i] = slice(None)
        expander = tuple(expander)
        self.expander = expander

        mode = _BNMode(x, gamma, self.key_axis, inference=True)
        if mode.can_use_ideep():
            # TODO(niboshi): Refactor iDeep part into a separate method
            expand_dim = False
            if x.ndim == 2:
                expand_dim = True
                x = x[:, :, None, None]

            y, = intel64.ideep.batchNormalization.Forward(
                intel64.ideep.array(x.astype(gamma.dtype, copy=False)),
                intel64.ideep.array(gamma),
                intel64.ideep.array(beta),
                intel64.ideep.array(mean),
                intel64.ideep.array(var),
                self.eps
            )
            y = y.astype(x.dtype, copy=False)

            if expand_dim:
                y = numpy.squeeze(y, axis=(2, 3))

            # lazy
            self.inv_var = None
            self.inv_std = None

        elif mode.can_use_cudnn(xp):
            y = cudnn.batch_normalization_forward_inference(
                x, gamma, beta, mean, var, self.eps,
                mode.is_for_conv2d, mode.get_cudnn_mode())
        else:
            # Generic CPU and GPU implementation
            gamma = gamma[expander]
            beta = beta[expander]
            var = var + self.eps
            self.inv_var = xp.reciprocal(var)
            self.inv_std = xp.sqrt(self.inv_var, dtype=self.inv_var.dtype)
            y = _apply_bn_fwd(xp, x, mean[expander], self.inv_std[expander],
                              gamma, beta)

        return y,

    def backward(self, indexes, grad_outputs):
        x, gamma, mean, var = self.get_retained_inputs()
        gy, = grad_outputs
        f = FixedBatchNormalizationGrad(
            self.eps, self.expander, self.axis, self.inv_std, self.inv_var)
        return f.apply((x, gamma, mean, var, gy))


class FixedBatchNormalizationGrad(function_node.FunctionNode):

    def __init__(self, eps, expander, axis, inv_std, inv_var):
        self.eps = eps
        self.expander = expander
        self.axis = axis
        self.inv_std = inv_std  # may be None
        self.inv_var = inv_var  # may be None

    def forward(self, inputs):
        self.retain_inputs((0, 1, 2, 4))
        x, gamma, mean, var, gy = inputs
        expander = self.expander
        xp = backend.get_array_module(x)

        if self.inv_std is None or self.inv_var is None:
            self.inv_var = xp.reciprocal(var + self.eps)
            self.inv_std = xp.sqrt(self.inv_var, dtype=self.inv_var.dtype)

        self.gamma_over_std = gamma * self.inv_std
        x_hat = _x_hat(x, mean[expander], self.inv_std[expander])

        gx = self.gamma_over_std[expander] * gy
        gbeta = gy.sum(axis=self.axis, dtype=gamma.dtype)
        ggamma = (x_hat * gy).sum(axis=self.axis)
        gmean = -self.gamma_over_std * gbeta
        gvar = - 0.5 * self.inv_var * (
            gamma * ggamma).astype(var.dtype, copy=False)

        gx = gx.astype(dtype=x.dtype)

        self.retain_outputs((0, 1, 2, 3, 4))
        return gx, ggamma, gbeta, gmean, gvar

    def backward(self, indexes, grad_outputs):
        F = chainer.functions
        x, gamma, mean, gy = self.get_retained_inputs()
        ggx1, gggamma1, ggbeta1, ggmean1, ggvar1 = grad_outputs
        gx1, ggamma1, gbeta1, gmean1, gvar1 = self.get_retained_outputs()

        # Handle None in output gradients.
        xp = backend.get_array_module(x)
        ggx1 = _zero_if_none(xp, ggx1, x.shape, x.dtype)
        gggamma1 = _zero_if_none(xp, gggamma1, gamma.shape, gamma.dtype)
        ggbeta1 = _zero_if_none(xp, ggbeta1, gamma.shape, gamma.dtype)
        ggmean1 = _zero_if_none(xp, ggmean1, mean.shape, mean.dtype)
        ggvar1 = _zero_if_none(xp, ggvar1, mean.shape, mean.dtype)

        if x.dtype != gamma.dtype:
            gamma = F.cast(gamma, x.dtype)
            ggamma1 = F.cast(ggamma1, x.dtype)
            gggamma1 = F.cast(gggamma1, x.dtype)
            gbeta1 = F.cast(gbeta1, x.dtype)
            ggbeta1 = F.cast(ggbeta1, x.dtype)
            mean = F.cast(mean, x.dtype)
            gmean1 = F.cast(gmean1, x.dtype)
            ggmean1 = F.cast(ggmean1, x.dtype)
            gvar1 = F.cast(gvar1, x.dtype)
            ggvar1 = F.cast(ggvar1, x.dtype)

        expander = self.expander

        x_hat = _x_hat(x, mean[expander], self.inv_std[expander])
        tmp = -0.5 * ggvar1

        gamma_over_var = gamma * self.inv_var
        g_gamma_over_var = tmp * ggamma1

        gggamma2 = gggamma1 + tmp * gamma_over_var
        gx_hat = gy * gggamma2[expander]
        gx2 = self.inv_std[expander] * gx_hat
        gmean2 = -self.inv_std * F.sum(gx_hat, axis=self.axis)

        g_gamma_over_std = F.sum(ggx1 * gy, axis=self.axis) - ggmean1 * gbeta1
        ggbeta2 = ggbeta1 - ggmean1 * self.gamma_over_std
        ggy2 = (gggamma2[expander] * x_hat + ggbeta2[expander]
                + self.gamma_over_std[expander] * ggx1)

        ggamma2 = (self.inv_var * g_gamma_over_var
                   + self.inv_std * g_gamma_over_std)
        gvar2 = -(ggamma2 * gamma_over_var + 0.5 * self.inv_var * (
            F.sum(x_hat * gx_hat, axis=self.axis)
            - self.gamma_over_std * g_gamma_over_std))

        gx2 = chainer.functions.cast(gx2, x.dtype)
        ggy2 = chainer.functions.cast(ggy2, gy.dtype)

        return gx2, ggamma2, gmean2, gvar2, ggy2


class _BNMode(object):

    def __init__(self, x, gamma, key_axis, inference=False):
        is_gamma_1d = gamma.ndim == 1
        # cuDNN only supports these tensor dimensions because they are
        # the most commonly used. If there is a need to support other
        # dimensions with cuDNN, we could consider reshaping the input
        # into a 2-dim array with channels as second dim and m=<product
        # of all dimensions except the 2nd dimension> as the first
        # dimension.
        self.is_for_conv2d = is_gamma_1d and x.ndim == 4 and key_axis[0] == 1
        self.is_for_linear = is_gamma_1d and key_axis[0] == x.ndim - 1
        self.cudnn_dim_ok = self.is_for_conv2d or self.is_for_linear
        # self.cudnn_dtype_ok = x.dtype != numpy.float16
        self.cudnn_dtype_ok = self.is_for_conv2d or (x.dtype != numpy.float16)
        self.ideep_ok = is_gamma_1d and intel64.inputs_all_ready((x,))
        self.inference = inference

    def get_cudnn_mode(self):
        assert self.cudnn_dim_ok
        if self.is_for_linear:
            return libcudnn.CUDNN_BATCHNORM_PER_ACTIVATION

        if (not self.inference and _cudnn_version >= 7000 and
                configuration.config.cudnn_fast_batch_normalization):
            return libcudnn.CUDNN_BATCHNORM_SPATIAL_PERSISTENT
        return libcudnn.CUDNN_BATCHNORM_SPATIAL

    def can_use_ideep(self):
        return self.ideep_ok and intel64.should_use_ideep('>=auto')

    def can_use_cudnn(self, xp):
        # TODO(bkvogel): Check for float16 support again in next cuDNN version.
        # cuDNN v5 batch normalization does not seem to support float16.
        return (xp is cuda.cupy and
                chainer.should_use_cudnn('>=auto', 5000) and
                self.cudnn_dim_ok and
                self.cudnn_dtype_ok)


def _x_hat(x, mean, inv_std):
    x_mu = x - mean
    x_mu *= inv_std
    return x_mu


def _chainerx_compute_axis(x_ndim, gamma_ndim, axis):
    # Returns processed axis for ChainerX.
    axis_chx = (
        None if axis is None
        else axis if isinstance(axis, tuple)
        else (axis,))
    axis_chx = _compute_axis(x_ndim, gamma_ndim, axis_chx)
    return axis_chx


def _chainerx_is_supported(device, axis_chx):
    # Checks if the input configuration is supported in ChainerX
    axis_ndim_chx = len(axis_chx)
    if device.backend.name == 'cuda':
        # cuDNN batch norm restriction
        if not ((axis_ndim_chx == 3 and axis_chx[0] == 0
                 and axis_chx[1] == 2 and axis_chx[2] == 3)
                or (axis_ndim_chx == 4 and axis_chx[0] == 0
                    and axis_chx[1] == 2 and axis_chx[2] == 3
                    and axis_chx[3] == 4)):
            return False
    return True


def _apply_bn_fwd(xp, x, mean, inv_std, gamma, beta):
    # NOTE: all arguments should be broadcasted to x.shape
    # (mean, inv_std, gamma, and beta have to already be expanded)
    if xp is numpy:
        x_hat = _x_hat(x, mean, inv_std)
        y = x_hat * gamma
        y += beta
        y = y.astype(x.dtype)
    else:
        y = cuda.elementwise(
            'T x, U mean, U inv_std, U gamma, U beta', 'T y',
            'y = gamma * (x - mean) * inv_std + beta', 'bn_fwd'
        )(x, mean, inv_std, gamma, beta)
    return y


def _zero_if_none(xp, x, shape, dtype):
    # TODO(Tokui): Return broadcasted 0 instead of a zeroed array.
    if x is None:
        return xp.zeros(shape, dtype=dtype)
    return x


def batch_normalization(x, gamma, beta, **kwargs):
    """batch_normalization(x, gamma, beta, eps=2e-5, running_mean=None, \
running_var=None, decay=0.9, axis=None)

    Batch normalization function.

    It takes the input variable ``x`` and two parameter variables ``gamma`` and
    ``beta``. The parameter variables must both have the same dimensionality,
    which is referred to as the channel shape. This channel shape corresponds
    to the dimensions in the input which are not averaged over. Since the
    first dimension of the input corresponds to the batch size, the second
    dimension of ``x`` will correspond to the first dimension of the channel
    shape, the third dimension of ``x`` will correspond to the second channel
    dimension (if it exists) and so on. Therefore, the dimensionality of the
    input must be at least one plus the number of channel dimensions. The
    total effective "batch size" will then be considered to be the product of
    all dimensions in ``x`` except for the channel dimensions.

    As an example, if the input is four dimensional and the parameter
    variables are one dimensional, then it is assumed that the first
    dimension of the input is the batch size, the second dimension is the
    channel size, and the remaining two dimensions are considered
    to be spatial dimensions that will be averaged over along with the
    batch size in the batch normalization computations. That is,
    the total batch size will be considered to be the product of all
    input dimensions except the second dimension.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.
        gamma (:class:`~chainer.Variable` or :ref:`ndarray`): Scaling parameter
            of normalized data.
        beta (:class:`~chainer.Variable` or :ref:`ndarray`): Shifting parameter
            of scaled normalized data.
        eps (float): Epsilon value for numerical stability.
        running_mean (:ref:`ndarray`):
            Running average of the mean. This is a running average of
            the mean over several mini-batches using the decay parameter.
            The function takes a previous running average, and updates
            the array in-place by the new running average.
            If ``None``, the running average is not computed. If this is
            ``None``, then ``runnng_var`` must also be ``None``.
        running_var (:ref:`ndarray`):
            Running average of the variance. This is a running average of
            the variance over several mini-batches using the decay parameter.
            The function takes a previous running average, and updates
            the array in-place by the new running average.
            If ``None``, the running average is not computed. If this is
            ``None``, then ``running_mean`` must also be ``None``.
        decay (float): Decay rate of moving average. It is used during
            training.
        axis (int, tuple of int or None): Axis over which normalization is
            performed. When axis is ``None``, it is determined from input
            dimensions. For example, if ``x.ndim`` is 4, axis becomes (0, 2, 3)
            and normalization is performed over 0th, 2nd and 3rd axis of input.
            If it is 2, axis becomes (0) and normalization is performed
            over 0th axis of input. When a tuple of int is given to this
            option, numbers in the tuple must be being sorted in ascending
            order. For example, (0, 2) is OK, but (2, 0) is not.

    See: `Batch Normalization: Accelerating Deep Network Training by Reducing\
          Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_

    .. seealso:: :class:`~chainer.links.BatchNormalization`

    """

    eps, running_mean, running_var, decay, axis = argument.parse_kwargs(
        kwargs, ('eps', 2e-5), ('running_mean', None),
        ('running_var', None), ('decay', 0.9), ('axis', None),
        train='train argument is not supported anymore. '
        'Use chainer.using_config')

    return BatchNormalization(eps, running_mean, running_var, decay,
                              axis).apply((x, gamma, beta))[0]


def fixed_batch_normalization(x, gamma, beta, mean, var, eps=2e-5, axis=None):
    """Batch normalization function with fixed statistics.

    This is a variant of batch normalization, where the mean and variance
    statistics are given by the caller as fixed variables. This is
    used on testing mode of the batch normalization layer, where batch
    statistics cannot be used for prediction consistency.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.
        gamma (:class:`~chainer.Variable` or :ref:`ndarray`): Scaling parameter
            of normalized data.
        beta (:class:`~chainer.Variable` or :ref:`ndarray`): Shifting parameter
            of scaled normalized data.
        mean (:class:`~chainer.Variable` or :ref:`ndarray`): Shifting parameter
            of input.
        var (:class:`~chainer.Variable` or :ref:`ndarray`): Square of scaling
            parameter of input.
        eps (float): Epsilon value for numerical stability.
        axis (int, tuple of int or None): Axis over which normalization is
            performed. When axis is ``None``, it is determined from input
            dimensions. For example, if ``x.ndim is 4``, axis becomes (0, 2, 3)
            and normalization is performed over 0th, 2nd and 3rd axis of input.
            If it is 2, axis becomes (0) and normalization is performed
            over 0th axis of input. When a tuple of int is given to this
            option, numbers in the tuple must be being sorted in ascending
            order. For example, (0, 2) is OK, but (2, 0) is not.

    .. seealso::
       :func:`~chainer.functions.batch_normalization`,
       :class:`~chainer.links.BatchNormalization`

    """
    return FixedBatchNormalization(eps, axis).apply((x, gamma, beta, mean,
                                                     var))[0]
