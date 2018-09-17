import warnings

import numpy

import chainer
from chainer.backends import cuda
from chainer.backends import intel64
from chainer import configuration
from chainer import function
from chainer import function_node
from chainer.utils import argument
from chainer.utils import collections_abc
from chainer.utils import type_check


if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cuda.cuda.cudnn
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


def _compute_axis_keyaxis_expander(x_ndim, gamma_ndim, axis=None):
    axis = _compute_axis(x_ndim, gamma_ndim, axis)
    key_axis = _compute_key_axis(x_ndim, gamma_ndim, axis)
    expander = [None for _ in range(x_ndim)]
    for i in key_axis:
        expander[i] = slice(None)
    expander = tuple(expander)
    return axis, key_axis, expander


class InstanceNormalization(function_node.FunctionNode):

    mean = None
    inv_std = None

    def __init__(self, eps=2e-5, track_running_stats=False, mean=None, var=None,
                 decay=0.9, axis=None):
        self.track_running_stats = track_running_stats
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
            gamma_type.dtype == x_type.dtype,
            beta_type.dtype == x_type.dtype,
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

    def forward(self, inputs):
        self.retain_inputs((0, 1))
        # Shape
        # x_: (N, C, ...)
        # gamma_, beta_: (C,)
        x_, gamma_, beta_ = inputs
        self.org_shape = x_.shape
        batch_size, channels, *shape_res = self.org_shape
        self.new_shape = [1, batch_size * channels] + shape_res
        self.axis, self.key_axis, self.expander = _compute_axis_keyaxis_expander(
            x.ndim, gamma.ndim, self.axis
        )

        xp = cuda.get_array_module(x_)
        # Shape
        # x: (1, N * C, ...)
        # gamma, bet_: (N * C,)
        x = xp.reshape(x_, self.new_shape)
        gamma = xp.repeat(gamma_, batch_size, 0)
        beta = xp.repeat(beta_, batch_size, 0)

        self.mode = _INMode(x, gamma, self.key_axis)
        self.use_cudnn = self.mode.can_use_cudnn(xp)
        self.use_ideep = self.mode.can_use_ideep()

        if self.running_mean is None:
            self.running_mean = xp.zeros_like(gamma)
            self.running_var = xp.zeros_like(gamma)
        # Shape of `y` is the same as `x_`; (N, C, ...)
        if self.use_ideep:
            y = self._forward_ideep(xp, x, gamma, beta)
        elif self.use_cudnn:
            y = self._forward_cudnn(xp, x, gamma, beta)
        else:
            y = self._forward_generic(xp, x, gamma, beta)

        return y,

    def backward(self, indexes, grad_outputs):
        # Shape
        # x: (N, C, ...), gamma: (C,)
        # InstanceNormalizationGrad is responsible for shape handling.
        x, gamma = self.get_retained_inputs()
        gy, = grad_outputs
        if self.use_ideep:
            assert self.var is not None
            var = self.var
        else:
            var = None

        f = InstanceNormalizationGrad(
            self.eps, self.use_cudnn, self.mode, self.expander, self.axis,
            self.mean, var, self.inv_std, self.key_axis,
            self.org_shape, self.new_shape
        )
        return f(x, gamma, gy)

    # Below 3 methods are responsible for forward computations.
    # Arguments are xp, x, gamma, and beta for consistency and
    # x, gamma, and beta are already reshaped.
    # So they are ready for Batch Normalization forward
    # except for population statistics updates.
    def _forward_ideep(self, xp, x, gamma, beta):
        expand_dim = False
        if x.ndim == 2:
            expand_dim = True
            x = x[:, :, None, None]

        gamma, beta = gamma[self.expander], beta[self.expander]
        W = numpy.concatenate((gamma, beta), axis=0).reshape((2, -1))

        y, self.mean, self.var, self.inv_std = (
            intel64.ideep.batchNormalizationForwardTraining(
                intel64.ideep.array(x), intel64.ideep.array(W),
                None, None, self.eps
            )
        )

        m = x.size // (gamma.size / self.org_shape[0])
        adjus = m / max(m - 1., 1.)

        if self.track_running_stats:
            # Update running_mean
            if isinstance(self.running_mean, intel64.ideep.mdarray):
                mean = self.mean.reshape(
                    (self.org_shape[0], self.org_shape[1])
                ).sum(axis=0) / self.org_shape[0]
                self.running_mean.inplace_axpby(
                    self.decay, (1 - self.decay), mean)
            else:
                self.running_mean *= self.decay
                self.running_mean += self.mean.reshape(
                    (self.org_shape[0], -1)).mean(axis=0) * (1 - self.decay)

            # Update running_var
            if isinstance(self.running_var, intel64.ideep.mdarray):
                var = self.var.reshape(
                    (self.org_shape[0], self.org_shape[1])
                ).sum(axis=0) / self.org_shape[0]
                self.running_var.inplace_axpby(
                    self.decay, (1 - self.decay), var * adjust)
            else:
                moment = 1 - self.decay
                self.running_var *= self.decay
                self.running_var += self.var.reshape(
                    (self.org_shape[0], -1)).mean(axis=0) * adjust * moment

        if expand_dim:
            y = numpy.squeeze(y, axis=(2, 3))

        y = y.reshape(self.org_shape)
        return y

    def _forward_cudnn(self, xp, x, gamma, beta):
        # To handle different sizes of mini-batches,
        # dummy variables are used as running statistics intentionally.
        x = cuda.cupy.ascontiguousarray(x)

        gamma = cuda.cupy.ascontiguousarray(gamma)
        beta = cuda.cupy.ascontiguousarray(beta)
        # Shape; (N * C,)
        dummy_mean, dummy_var = xp.zeros_like(gamma), xp.zeros_like(beta)

        dtype = x.dtype
        handle = cudnn.get_handle()
        x_desc = cudnn.create_tensor_descriptor(_as4darray(x, self.mode))
        cudnn_mode = self.mode.get_cudnn_mode()
        derivedBnDesc = cudnn.create_uninitialized_tensor_descriptor()
        libcudnn.deriveBNTensorDescriptor(
            derivedBnDesc.value, x_desc.value, cudnn_mode
        )
        dtype_param = _get_dtype_of_tensor_descriptor(derivedBnDesc)
        if dtype_param is not dtype:
            gamma = gamma.astype(dtype_param)
            beta = beta.astype(dtype_param)
            dummy_mean = dummy_mean.astype(dtype_param)
            dummy_var = dummy_var.astype(dtype_param)

        oz_dtype = 'd' if x.dtype == 'd' else 'f'
        one = numpy.array(1, dtype=oz_dtype).ctypes
        zero = numpy.array(0, dtype=oz_dtype).ctypes
        y = cuda.cupy.empty_like(x)
        # Factor used in the moving average
        factor = 1 - self.decay

        if self.mean is None:
            # Output cache to speed up backward pass.
            self.mean = xp.empty_like(gamma)
            self.inv_std = xp.empty_like(gamma)

        libcudnn.batchNormalizationForwardTraining(
            handle, cudnn_mode, one.data, zero.data,
            x_desc.value, x.data.ptr, x_desc.value,
            y.data.ptr, derivedBnDesc.value, gamma.data.ptr,
            beta.data.ptr, factor, dummy_mean.data.ptr,
            dummy_var.data.ptr, self.eps,
            self.mean.data.ptr, self.inv_std.data.ptr)

        # Note: When the CUDNN_BATCHNORM_SPATIAL_PERSISTENT mode is used,
        # there is a possibility of numerical overflow. You can use
        # queryRuntimeError() to make sure whether the overflow actually
        # occured or not during the batch normalization.
        if (cudnn_mode is libcudnn.CUDNN_BATCHNORM_SPATIAL_PERSISTENT and
                configuration.config.debug):
            query_mode = libcudnn.CUDNN_ERRQUERY_BLOCKING
            rstatus = libcudnn.queryRuntimeError(handle, query_mode)
            if rstatus is not libcudnn.CUDNN_STATUS_SUCCESS:
                warnings.warn(
                    'A numerical overflow might have happend in cuDNN'
                    'batch normalization (status:{})'.format(rstatus))

        if self.track_running_stats:
            self._update_running_statistics(
                mean.astype(dtype), var.astype(dtype), x, gamma)

        y = y.reshape(self.org_shape)
        return y

    def _forward_generic(self, xp, x, gamma, beta):
        # Generic CPU and GPU implementation
        gamma = gamma[self.expander]
        beta = beta[self.expander]
        self.mean = x.mean(axis=self.axis)
        var = x.var(axis=self.axis)
        if xp is numpy:
            self.inv_std = numpy.reciprocal(
                numpy.sqrt(var + self.eps, dtype=self.dtype)
            )
        else:
            self.inv_std = cuda.cupyx.rsqrt(var + self.eps)

        y_ = _apply_in_fwd(
            xp, x, self.mean[self.expander], self.inv_std[self.expander],
            gamma, beta
        )
        if self.track_running_stats:
            self._update_running_statistics(mean, var, x, gamma)
        y = y_.reshape(self.org_shape)
        return y

    def _update_running_statistics(self, mean, var, x, gamma):
        m = x.size // (gamma.size / self.org_shape[0])
        adjust = m / max(m - 1., 1.)  # unbiased estimation
        mean = self.mean.reshape((self.org_shape[0], -1)).mean(axis=0)
        var = self.var.reshape((self.org_shape[0], -1)).mean(axis=0)
        self.running_mean *= self.decay
        self.running_mean += (1 - self.decay) * mean
        self.running_var *= self.decay
        self.running_var += (1 - self.decay) * adjust * var


class InstanceNormalizationGrad(function.Function):

    def __init__(self, eps, use_cudnn, mode, expander, axis, mean, var,
                 inv_std, key_axis, org_shape, new_shape):
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
        self.org_shape = org_shape
        self.new_shape = new_shape

    def forward(self, inputs):
        # This is responsible for shape handling.
        # Note that given mean and var have (N * C,).
        # While x_, gamma_, gy_ have N in the first axis.
        self.retain_inputs((0, 1, 2))
        x_, gamma_, gy_ = inputs
        expander = self.expander
        inv_m = gamma.dtype.type(1. / (x_.size // gamma_.size))
        xp = cuda.get_array_module(x)

        # Shape
        # x, gy: (1, N * C, ...)
        # gamma: (N * C,)
        x = x_.reshape(self.new_shape)
        gamma = xp.repeat(gamma_, self.org_shape[0], axis=0)
        gy = gy_.reshape(self.new_shape)

        # gx, ggamma, and gbeta are given off.
        # In each _forward_foo method, gx and ggamma are cached
        # and shapes are
        # (N, C, ...) and (C,)
        if self.use_ideep:
            return self._forward_ideep(xp, x, gy, gamma, inv_m)
        elif self.use_cudnn:
            return self._forward_cudnn(xp, x, gy, gamma, inv_m)
        else:
            return self._forward_generic(xp, x, gy, gamma, inv_m)

    def _forward_ideep(self, xp, x, gy, gamma, inv_m):
        expand_dim = False
        if x.ndim == 2:
            expand_dim = True
            x = x[:, :, None, None]
            gy = gy[:, :, None, None]

        gamma = gamma[expander]
        beta = beta[expander]
        W = numpy.concatenate((gamma, beta), axis=0).reshape((2, -1))

        gx, gW = intel64.ideep.batchNormalization.Backward(
            intel64.ideep.array(x),
            intel64.ideep.array(gy),
            self.mean,
            self.var,
            intel64.ideep.array(W),
            self.eps)

        ggamma, gbeta = gW[:2]

        if expand_dim:
            gx = numpy.squeeze(gx, axis=(2, 3))

        gx = gx.reshape(self.org_shape)
        ggamma = ggamma.reshape((self.org_shape[0], -1)).mean(axis=0)
        gbeta = gbeta.reshape((self.org_shape[0], -1)).mean(axis=0)

        self.retain_outputs((0, 1))
        return gx, ggamma, gbeta

    def _forward_cudnn(self, xp, x, gy, gamma, inv_m):
        x = cuda.cupy.ascontiguousarray(x)
        gamma = cuda.cupy.ascontiguousarray(gamma)
        gy = cuda.cupy.ascontiguousarray(gy)
        dtype = x.dtype
        handle = cudnn.get_handle()
        x_desc = cudnn.create_tensor_descriptor(_as4darray(x, self.mode))
        cudnn_mode = self.mode.get_cudnn_mode()
        derivedBnDesc = cudnn.create_uninitialized_tensor_descriptor()
        libcudnn.deriveBNTensorDescriptor(derivedBnDesc.value,
                                          x_desc.value, cudnn_mode)
        dtype_param = _get_dtype_of_tensor_descriptor(derivedBnDesc)
        if dtype_param is not dtype:
            gamma = gamma.astype(dtype_param)
        oz_dtype = 'd' if x.dtype == 'd' else 'f'
        one = numpy.array(1, dtype=oz_dtype).ctypes
        zero = numpy.array(0, dtype=oz_dtype).ctypes
        gx = cuda.cupy.empty_like(x)
        ggamma = cuda.cupy.empty_like(gamma)
        gbeta = cuda.cupy.empty_like(gamma)
        libcudnn.batchNormalizationBackward(
            handle, cudnn_mode, one.data, zero.data,
            one.data, zero.data, x_desc.value, x.data.ptr,
            x_desc.value, gy.data.ptr, x_desc.value, gx.data.ptr,
            derivedBnDesc.value, gamma.data.ptr,
            ggamma.data.ptr, gbeta.data.ptr,
            self.eps, self.mean.data.ptr, self.inv_std.data.ptr)

        # Note: When the CUDNN_BATCHNORM_SPATIAL_PERSISTENT mode is used,
        # there is a possibility of numerical overflow. You can use
        # queryRuntimeError() to make sure whether the overflow actually
        # occured or not during the batch normalization.
        if (cudnn_mode is libcudnn.CUDNN_BATCHNORM_SPATIAL_PERSISTENT and
                configuration.config.debug):
            query_mode = libcudnn.CUDNN_ERRQUERY_BLOCKING
            rstatus = libcudnn.queryRuntimeError(handle, query_mode)
            if rstatus is not libcudnn.CUDNN_STATUS_SUCCESS:
                warnings.warn(
                    'A numerical overflow might have happend in cuDNN'
                    'batch normalization (status:{})'.format(rstatus))

        if dtype_param is not dtype:
            ggamma = ggamma.astype(dtype)
            gbeta = gbeta.astype(dtype)
        gx = gx.reshape(self.org_shape)
        ggamma = ggamma.reshape((self.org_shape[0], -1)).mean(axis=0)
        gbeta = gbeta.reshape((self.org_shape[0], -1)).mean(axis=0)
        self.retain_outputs((0, 1))
        return gx, ggamma, gbeta

    def _forward_generic(self, xp, x, gy, gamma, inv_m):
        # Generic implementation
        gbeta = gy.sum(axis=self.axis)
        x_hat = _x_hat(x, self.mean[expander], self.inv_std[expander])
        ggamma = (gy * x_hat).sum(axis=self.axis)
        if xp is numpy:
            gy = (gamma * self.inv_std)[expander] * (
                gy - (x_hat * ggamma[expander] + gbeta[expander]) * inv_m)
            )
        else:
            gx = cuda.elementwise(
                '''
                T gy, T x_hat, T gamma, T inv_std, T ggamma, T gbeta,
                T inv_m
                ''',
                'T gx',
                '''
                gx = (gamma * inv_std) * (
                    gy - (x_hat * ggamma + gbeta) * inv_m)
            ''', 'in_bwd')(gy, x_hat, gamma[expander],
                           self.inv_std[expander], ggamma[expander],
                           gbeta[expander], inv_m
            )
        gx = gx.reshape(self.org_shape)
        ggamma = ggamma.reshape((self.org_shape[0], -1)).mean(axis=0)
        gbeta = gbeta.reshape((self.org_shape[0], -1)).mean(axis=0)
        self.retain_outputs((0, 1))
        return gx, ggamma, gbeta

    def backward(self, inputs, grad_outputs):
        expander = self.expander

        x, gamma, gy = inputs
        gx1, ggamma1, _ = self.output_data
        ggx1, ggamma1, ggbeta1 = grad_outputs
        xp = cuda.get_array_module(x)

        batch_size = len(x)
        x = x.reshape(self.new_shape)
        gamma = xp.repeat(gamma, batch_size, 0)
        gy = gy.reshape(self.new_shape)
        gx1 = gx1.reshape(self.new_shape)
        ggamma1 = xp.repeat(ggamma1, batch_size, 0)
        ggx1 = ggx1.reshape(self.new_shape)
        ggamma1 = xp.repeat(ggamma1, batch_size, 0)
        ggbeta1 = xp.repeat(ggbeta1, batch_size, 0)

        # auxiliary values
        inv_m = gamma.dtype.type(1. / (x.size * batch_size // gamma.size))
        r = 0 if ggx1 is None else (gx1 * ggx1).sum(axis=self.axis)
        coeff = gamma * self.inv_std
        coeff_m = coeff * inv_m
        x_hat = _x_hat(x, self.mean[expander], self.inv_std[expander])

        # handle None in output gradients
        ggx1 = _zero_if_none(xp, ggx1, x.shape, x.dtype)
        gggamma1 = _zero_if_none(xp, gggamma1, gamma.shape, gamma.dtype)
        ggbeta1 = _zero_if_none(xp, ggbeta1, gamma.shape, gamma.dtype)

        gggamma2 = gggamma1 - coeff_m * (x_hat * ggx1).sum(axis=self.axis)
        ggbeta2 = ggbeta1 - coeff_m * ggx1.sum(axis=self.axis)

        ggamma2 = r / gamma

        gx_hat2 = (gggamma2[expander] * gy -
                   (coeff_m * ggamma1)[expander] * ggx1)
        gstd2 = -self.inv_std * (r + (x_hat * gx_hat2).sum(axis=self.axis))
        gmean2 = -self.inv_std * gx_hat2.sum(axis=self.axis)
        gx2 = self.inv_std[expander] * gx_hat2 + inv_m * (
            gmean2[expander] + x_hat * gstd2[expander])
        ggy2 = (gggamma2[expander] * x_hat + ggbeta2[expander]
                + coeff[expander] * ggx1)

        return gx2.reshape(self.org_shape), ggamma2.reshape((batch_size, -1)).mean(axis=0), ggy2.reshape(self.org_shape)


class FixedInstanceNormalization(function_node.FunctionNode):

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
            axis = (axis,)
        elif axis is not None:
            raise RuntimeError('axis must be int, tuple of int or None')
        self.axis = axis

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 5)
        x_type, gamma_type, beta_type, mean_type, var_type = in_types
        type_check.expect(
            x_type.dtype.kind == 'f',
            gamma_type.dtype == x_type.dtype,
            beta_type.dtype == x_type.dtype,
            gamma_type.shape == beta_type.shape,
            mean_type.dtype == x_type.dtype,
            mean_type.shape == gamma_type.shape,
            var_type.dtype == x_type.dtype,
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
        # for i in range(len(_key_axis)):
        #     type_check.expect(
        #         x_type.shape[_key_axis[i]] == gamma_type.shape[i],
        #     )

    def forward(self, inputs):
        self.retain_inputs((0, 1, 3, 4))
        x_, gamma_, beta_, mean_, var_ = inputs
        xp = cuda.get_array_module(x_)
        self.org_shape = x_.shape
        batch_size, channels, *shape = self.org_shape
        self.new_shape = [1, batch_size * channels] + shape

        x = x_.reshape(self.new_shape)
        gamma = xp.repeat(gamma_, batch_size, 0)
        beta = xp.repeat(beta_, batch_size, 0)
        mean = xp.repeat(mean_, batch_size, 0)
        var = xp.repeat(var_, batch_size, 0)

        self.axis, self.key_axis, self.expander = _compute_axis_keyaxis_expander(
            x.ndim, gamma.ndim, axis
        )
        self.mode = _INMode(x, gamma, self.key_axis, inference=True)
        self.use_cudnn = self.mode.can_use_cudnn(xp)
        self.use_ideep = self.mode.can_use_ideep()

        if self.use_ideep():
            y = self._forward_ideep()
        elif self.use_cudnn:
            y = self._forward_cudnn()
        else:
            y = self._forward_generic()

    def _forward_ideep(self, xp, x, gamma, beta, mean, var):
        expand_dim = False
        if x.ndim == 2:
            expand_dim = True
            x = x[:, :, None, None]
        gamma = gamma[self.expander]
        beta = beta[self.expander]
        W = numpy.concatenate((gamma, beta), axis=0).reshape((2, -1))

        y, = intel64.ideep.batchNormalization.Forward(
            intel64.ideep.array(x),
            intel64.ideep.array(W),
            intel64.ideep.array(mean),
            intel64.ideep.array(var),
            self.eps
        )

        if expand_dim:
            y = numpy.squeeze(y, axis=(2, 3))

        # lazy
        self.inv_var = None
        self.inv_std = None

        y = y.reshape(self.org_shape)
        return y

    def _forward_cudnn(self, xp, x, gamma, beta, mean, var):
        x = cuda.cupy.ascontiguousarray(x)

        gamma = cuda.cupy.ascontiguousarray(gamma)
        beta = cuda.cupy.ascontiguousarray(beta)
        dtype = x.dtype
        handle = cudnn.get_handle()
        x_desc = cudnn.create_tensor_descriptor(_as4darray(x, mode))
        cudnn_mode = mode.get_cudnn_mode()
        derivedBnDesc = cudnn.create_uninitialized_tensor_descriptor()
        libcudnn.deriveBNTensorDescriptor(derivedBnDesc.value,
                                          x_desc.value, cudnn_mode)
        dtype_param = _get_dtype_of_tensor_descriptor(derivedBnDesc)
        if dtype_param is not dtype:
            gamma = gamma.astype(dtype_param)
            beta = beta.astype(dtype_param)
            mean = mean.astype(dtype_param)
            var = var.astype(dtype_param)
        oz_dtype = 'd' if x.dtype == 'd' else 'f'
        one = numpy.array(1, dtype=oz_dtype).ctypes
        zero = numpy.array(0, dtype=oz_dtype).ctypes
        y = cuda.cupy.empty_like(x)

        libcudnn.batchNormalizationForwardInference(
            handle, cudnn_mode, one.data, zero.data,
            x_desc.value, x.data.ptr, x_desc.value, y.data.ptr,
            derivedBnDesc.value, gamma.data.ptr, beta.data.ptr,
            mean.data.ptr, var.data.ptr, self.eps)

        y = y.reshape(self.org_shape)
        return y

    def _forward_generic(self, xp, x, gamma, mean, var):
        # Generic CPU and GPU implementation
        gamma = gamma[self.expander]
        beta = beta[self.expander]
        var = var + self.eps
        self.inv_var = xp.reciprocal(var)
        self.inv_std = xp.sqrt(self.inv_var, dtype=self.inv_var.dtype)
        y = _apply_in_fwd(xp, x, mean[self.expander], self.inv_std[self.expander],
                          gamma, beta)
        y = y.reshape(self.org_shape)
        return y

    def backward(self, indexes, grad_outputs):
        x, gamma, mean, var = self.get_retained_inputs()
        x = x.reshape(self.new_shape)
        xp = cuda.get_array_module(x)
        gamma = xp.repeat(gamma, self.org_shape[0], 0)
        mean = xp.repeat(mean, self.org_shape[0], 0)
        var = xp.repeat(var, self.org_shape[0], 0)
        gy, = grad_outputs
        f = FixedInstanceNormalizationGrad(
            self.eps, self.expander, self.axis, self.inv_std, self.inv_var,
            self.org_shape, self.new_shape
        )
        return f(x, gamma, mean, var, gy)


class FixedInstanceNormalizationGrad(function.Function):

    def __init__(self, eps, expander, axis, inv_std, inv_var,
                 org_shape, new_shape):
        self.eps = eps
        self.expander = expander
        self.axis = axis
        self.inv_std = inv_std  # may be None
        self.inv_var = inv_var  # may be None
        self.org_shape = org_shape
        self.new_shape = new_shape

    def forward(self, inputs):
        self.retain_inputs((0, 1, 2, 4))
        # Shape
        # (N, C, ...), (C,), (C,), (C,), (N, C, ...)
        x, gamma, mean, var, gy = inputs
        expander = self.expander
        xp = cuda.get_array_module(x)

        x = xp.reshape(x, self.new_shape)
        batch_size = self.org_shape[0]
        gamma = xp.repeat(gamma, batch_size, 0)
        mean = xp.repeat(mean, batch_size, 0)
        var = xp.repeat(var, batch_size, 0)
        gy = xp.reshape(gy, self.new_shape)

        if self.inv_std is None or self.inv_var is None:
            self.inv_var = xp.reciprocal(var + self.eps)
            self.inv_std = xp.sqrt(self.inv_var, dtype=self.inv_var.dtype)

        self.gamma_over_std = gamma * self.inv_std
        x_hat = _x_hat(x, mean[expander], self.inv_std[expander])

        gx = self.gamma_over_std[expander] * gy
        gbeta = gy.sum(axis=self.axis)
        ggamma = (x_hat * gy).sum(axis=self.axis)
        gmean = -self.gamma_over_std * gbeta
        gvar = - 0.5 * gamma * self.inv_var * ggamma

        gx = xp.reshape(self.org_shape)
        ggamma = ggamma.reshape(batch_size, -1).mean(axis=0)
        gbeta = gbeta.reshape(batch_size, -1).mean(axis=0)
        gmean = gmean.reshape(batch_size, -1).mean(axis=0)
        gvar = gvar.reshape(batch_size, -1).mean(axis=0)
        self.retain_outputs((0, 1, 2, 3, 4))
        return gx, ggamma, gbeta, gmean, gvar

    def backward(self, inputs, grad_outputs):
        x, gamma, mean, _, gy = inputs
        # Shape
        # (N, C, ...), (C,), (C,), (C,), (C,)
        ggx1, gggamma1, ggbeta1, ggmean1, ggvar1 = grad_outputs
        # Shape
        # (N, C, ...), (C,), (C,), (C,), (C,)
        gx1, ggamma1, gbeta1, gmean1, gvar1 = self.output_data

        # Handle None in output gradients.
        xp = cuda.get_array_module(x)
        ggx1 = _zero_if_none(xp, ggx1, x.shape, x.dtype)
        gggamma1 = _zero_if_none(xp, gggamma1, gamma.shape, gamma.dtype)
        ggbeta1 = _zero_if_none(xp, ggbeta1, gamma.shape, gamma.dtype)
        ggmean1 = _zero_if_none(xp, ggmean1, mean.shape, mean.dtype)
        ggvar1 = _zero_if_none(xp, ggvar1, mean.shape, mean.dtype)

        expander = self.expander

        x_hat = _x_hat(x, mean[expander], self.inv_std[expander])
        tmp = -0.5 * ggvar1

        gamma_over_var = gamma * self.inv_var
        g_gamma_over_var = tmp * ggamma1

        gggamma2 = gggamma1 + tmp * gamma_over_var
        gx_hat = gy * gggamma2[expander]
        gx2 = self.inv_std[expander] * gx_hat
        gmean2 = -self.inv_std * gx_hat.sum(axis=self.axis)

        g_gamma_over_std = (ggx1 * gy).sum(axis=self.axis) - ggmean1 * gbeta1
        ggbeta2 = ggbeta1 - ggmean1 * self.gamma_over_std
        ggy2 = (gggamma2[expander] * x_hat + ggbeta2[expander]
                + self.gamma_over_std[expander] * ggx1)

        ggamma2 = (self.inv_var * g_gamma_over_var
                   + self.inv_std * g_gamma_over_std)
        gvar2 = -(ggamma2 * gamma_over_var + 0.5 * self.inv_var * (
            (x_hat * gx_hat).sum(axis=self.axis)
            - self.gamma_over_std * g_gamma_over_std))

        return gx2, ggamma2, gmean2, gvar2, ggy2




class _INMode(object):

    def __init__(self, x, gamma, key_axis, inference=False):
        # NOTE(crcrpar): This class is wholly copied from
        # chainer/functions/normalization/batch_normalization.py
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
        return (xp is not numpy and
                chainer.should_use_cudnn('>=auto', 5000) and
                self.cudnn_dim_ok and
                self.cudnn_dtype_ok)


def _apply_in_fwd(xp, x, mean, inv_std, gamma, beta):
    # NOTE: all arguments should be broadcasted to x.shape
    # (mean, inv_std, gamma, and beta have to already be expanded)
    if xp is numpy:
        x_hat = _x_hat(x, mean, inv_std)
        y = gamma * x_hat
        y += beta
    else:
        y = cuda.elementwise(
            'T x, T mean, T inv_std, T gamma, T beta', 'T y',
            'y = gamma * (x - mean) * inv_std + beta', 'bn_fwd'
        )(x, mean, inv_std, gamma, beta)
    return y


def _x_hat(x, mean, inv_std):
    x_mu = x - mean
    x_mu *= inv_std
    return x_mu


def _get_dtype_of_tensor_descriptor(desc):
    cudnn_dtype, _, _, _, _, _, _, _, _ = libcudnn.getTensor4dDescriptor(
        desc.value)
    dtype = None
    if cudnn_dtype == libcudnn.CUDNN_DATA_DOUBLE:
        dtype = numpy.dtype(numpy.float64)
    elif cudnn_dtype == libcudnn.CUDNN_DATA_FLOAT:
        dtype = numpy.dtype(numpy.float32)
    elif cudnn_dtype == libcudnn.CUDNN_DATA_HALF:
        dtype = numpy.dtype(numpy.float16)
    else:
        msg = 'Unknow cudnn data type {} '.format(cudnn_dtype)
        raise RuntimeError(msg)
    return dtype


def instance_normalization(x, gamma, beta, **kwargs):
    """instance_normalization(x, gamma, beta, eps=2e-5, update_running_stats=False, running_mean=None, running_var=None, decay=0.9, axis=None)

    Instance normalization function.

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

    .. warning::

       ``train`` argument is not supported anymore since v2.
       Instead, use ``chainer.using_config('train', train)``.
       See :func:`chainer.using_config`.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable.
        gamma (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Scaling parameter of normalized data.
        beta (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Shifting parameter of scaled normalized data.
        eps (float): Epsilon value for numerical stability.
        track_running_stats (bool): Update running statistics or not.
            If ``True``, `running_mean` and `running_var` are updated
            using approximated mean and variance.
        running_mean (numpy.ndarray or cupy.ndarray):
            Running average of the mean. This is a running average of
            the mean over several mini-batches using the decay parameter.
            The function takes a previous running average, and updates
            the array in-place by the new running average.
            If ``None``, the running average is not computed. If this is
            ``None``, then ``runnng_var`` must also be ``None``.
        running_var (numpy.ndarray or cupy.ndarray):
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

    See: `Instance Normalization: The Missing Ingredient for Fast Stylization\
          <https://arxiv.org/abs/1607.08022>`_

    .. seealso:: :class:`~chainer.links.InstanceNormalization`

    """  # NOQA

    track_running_stats, eps, running_mean, running_var, decay, axis = argument.parse_kwargs(
        kwargs, ('eps', 2e-5), ('track_running_stats', False),
        ('running_mean', None), ('running_var', None),
        ('decay', 0.9), ('axis', None),
        train='train argument is not supported anymore. '
        'Use chainer.using_config')

    return InstanceNormalization(eps, track_running_stats,
                                 running_mean, running_var, decay,
                                 axis).apply((x, gamma, beta))[0]


def fixed_instance_normalization(x, gamma, beta, mean, var, eps=2e-5, axis=None):
    """Instance normalization function with fixed statistics.

    This is a variant of instance normalization, where the mean and variance
    statistics are given by the caller as fixed variables. This is
    used on testing mode of the batch normalization layer, where batch
    statistics cannot be used for prediction consistency.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable.
        gamma (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Scaling parameter of normalized data.
        beta (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Shifting parameter of scaled normalized data.
        mean (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Shifting parameter of input.
        var (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Square of scaling parameter of input.
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
       :func:`~chainer.functions.instance_normalization`,
       :class:`~chainer.links.InstanceNormalization`

    """
    return FixedInstanceNormalization(eps, axis).apply((x, gamma, beta, mean,
                                                        var))[0]
