# This file is heavily based on Chainer's batch normalization implementation.
# See: chainer/functions/normalization/batch_normalization.py (dbb650)

from abc import ABCMeta
from abc import abstractmethod
import chainer
from chainer import backend
from chainer import cuda
from chainer import function
import chainer.utils
from chainer.utils import type_check
import numpy
import six


def _xhat(x, mean, std, expander):
    x_mu = x - mean[expander]
    x_mu /= std[expander]
    return x_mu


class _MultiNodeBatchNormalizationBackend(six.with_metaclass(ABCMeta)):

    @abstractmethod
    def forward(self, axis, gamma, x, xp):
        raise NotImplementedError()

    @abstractmethod
    def backward(self, axis, gamma, gy, x_hat, x, xp):
        raise NotImplementedError()


class _MpiBackend(_MultiNodeBatchNormalizationBackend):

    def __init__(self, comm):
        self.comm = comm

    def forward(self, axis, gamma, x, xp):
        tmp = xp.empty(gamma.size * 2, dtype=gamma.dtype)
        x.mean(axis=axis, out=tmp[:gamma.size], dtype=gamma.dtype)
        xp.square(x).mean(axis=axis, out=tmp[gamma.size:], dtype=gamma.dtype)
        if xp is cuda.cupy:
            chainer.cuda.Stream.null.synchronize()
        self.comm.multi_node_mean(None, tmp)
        mean = tmp[:gamma.size]
        sqmean = tmp[gamma.size:]
        var = sqmean - xp.square(mean)
        return mean, var

    def backward(self, axis, gamma, gy, x_hat, x, xp):
        tmp = xp.empty(gamma.size * 2, dtype=gamma.dtype)
        gy.sum(axis=axis, out=tmp[:gamma.size], dtype=gamma.dtype)
        (gy * x_hat).sum(axis=axis, out=tmp[gamma.size:], dtype=gamma.dtype)
        if xp is cuda.cupy:
            chainer.cuda.Stream.null.synchronize()
        self.comm.multi_node_mean(None, tmp)
        gbeta = tmp[:gamma.size]
        ggamma = tmp[gamma.size:]
        return gbeta, ggamma


class _NcclBackend(_MultiNodeBatchNormalizationBackend):

    def __init__(self, comm):
        self.comm = comm

        # We need to delay importing MPI4py (and momdules that import MPI4py)
        import chainermn.communicators._memory_utility as memory_utility_module
        self.memory_utility_module = memory_utility_module

    def forward(self, axis, gamma, x, xp):
        gpu_buffer_n_elems = gamma.size * 2
        gpu_buffer_size = gamma.dtype.itemsize * gpu_buffer_n_elems
        gpu_buffer_a = self.memory_utility_module.DeviceMemory()
        gpu_buffer_b = self.memory_utility_module.DeviceMemory()
        gpu_buffer_a.assign(gpu_buffer_size)
        gpu_buffer_b.assign(gpu_buffer_size)
        gpu_buffer_a_array = gpu_buffer_a.array(
            gpu_buffer_n_elems, dtype=gamma.dtype)
        x.mean(axis=axis, out=gpu_buffer_a_array[:gamma.size],
               dtype=gamma.dtype)
        xp.square(x).mean(axis=axis, out=gpu_buffer_a_array[gamma.size:],
                          dtype=gamma.dtype)
        self.comm.multi_node_mean_nccl(gpu_buffer_a,
                                       gpu_buffer_b,
                                       gpu_buffer_n_elems,
                                       gamma.dtype)
        gpu_buffer_a_array = gpu_buffer_a.array(
            gpu_buffer_n_elems,
            dtype=gamma.dtype)

        mean = gpu_buffer_a_array[:gamma.size]
        sqmean = gpu_buffer_a_array[gamma.size:]
        var = sqmean - xp.square(mean)
        return mean, var

    def backward(self, axis, gamma, gy, x_hat, x, xp):
        gpu_buffer_n_elems = gamma.size * 2
        gpu_buffer_size = gamma.dtype.itemsize * gpu_buffer_n_elems
        gpu_buffer_a = self.memory_utility_module.DeviceMemory()
        gpu_buffer_b = self.memory_utility_module.DeviceMemory()
        gpu_buffer_a.assign(gpu_buffer_size)
        gpu_buffer_b.assign(gpu_buffer_size)
        gpu_buffer_a_array = gpu_buffer_a.array(
            gpu_buffer_n_elems, dtype=gamma.dtype)
        gy.sum(axis=axis, out=gpu_buffer_a_array[:gamma.size],
               dtype=gamma.dtype)
        (gy * x_hat).sum(axis=axis, out=gpu_buffer_a_array[gamma.size:],
                         dtype=gamma.dtype)
        self.comm.multi_node_mean_nccl(gpu_buffer_a,
                                       gpu_buffer_b,
                                       gpu_buffer_n_elems,
                                       gamma.dtype)
        gpu_buffer_a_array = gpu_buffer_a.array(
            gpu_buffer_n_elems,
            dtype=gamma.dtype)

        gbeta = gpu_buffer_a_array[:gamma.size]
        ggamma = gpu_buffer_a_array[gamma.size:]
        return gbeta, ggamma


def get_communication_backend(comm, communication_backend='auto'):
    if communication_backend not in ['mpi', 'nccl', 'auto']:
        raise ValueError('MultiNodeBatchNormalization does not support '
                         '{}.'.format(communication_backend))
    from chainermn.communicators.pure_nccl_communicator \
        import PureNcclCommunicator
    if communication_backend != 'auto':
        if 'nccl' == communication_backend:
            if not isinstance(comm, PureNcclCommunicator):
                raise ValueError('{} is not supported in '
                                 'MultiNodeBatchNormalization when using '
                                 '{}.'.format(communication_backend,
                                              type(comm)))
        selected_communication_backend = communication_backend
    else:
        if isinstance(comm, PureNcclCommunicator):
            selected_communication_backend = 'nccl'
        else:
            selected_communication_backend = 'mpi'
    return selected_communication_backend


class MultiNodeBatchNormalizationFunction(function.Function):

    def __init__(self, comm, eps=2e-5, mean=None, var=None, decay=0.9,
                 communication_backend='auto'):
        chainer.utils.experimental(
            'chainermn.functions.MultiNodeBatchNormalizationFunction')

        self.comm = comm
        self.running_mean = mean
        self.running_var = var

        # Note: cuDNN v5 requires that eps be greater than 1e-5. Otherwise, an
        # error will occur.
        # See CUDNN_BN_MIN_EPSILON value in cudnn.h to verify minimum allowable
        # value.
        self.eps = eps
        if chainer.should_use_cudnn('>=auto'):
            if eps < 1e-5:
                msg = 'cuDNN does not allow an eps value less than 1e-5.'
                raise RuntimeError(msg)
        self.mean_cache = None
        self.decay = decay

        selected_communication_backend = \
            get_communication_backend(comm, communication_backend)

        if selected_communication_backend == 'nccl':
            self._backend = _NcclBackend(comm)
        else:
            self._backend = _MpiBackend(comm)

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
            gamma_type.dtype.kind == 'f',
            gamma_type.dtype == beta_type.dtype,
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
        xp = backend.get_array_module(*inputs)
        x, gamma, beta = inputs[:3]
        if chainer.configuration.config.train:
            if self.running_mean is None:
                self.running_mean = xp.zeros_like(gamma, dtype=x.dtype)
                self.running_var = xp.zeros_like(gamma, dtype=x.dtype)
            else:
                self.running_mean = xp.array(self.running_mean)
                self.running_var = xp.array(self.running_var)
        elif len(inputs) == 5:
            self.fixed_mean = inputs[3]
            self.fixed_var = inputs[4]

        head_ndim = gamma.ndim + 1
        expander = (None, Ellipsis) + (None,) * (x.ndim - head_ndim)
        gamma = gamma[expander]
        beta = beta[expander]

        # cuDNN only supports these tensor dimensions because they are
        # the most commonly used. If there is a need to support other
        # dimensions with cuDNN, we could consider reshaping the input
        # into a 2-dim array with channels as second dim and m=<product
        # of all dimensions except the 2nd dimension> as the first
        # dimension.
        cudnn_dim_ok = x.ndim == 2 or (x.ndim == 4 and head_ndim == 2)
        # TODO(bkvogel): Check for float16 support again in next cuDNN version.
        # cuDNN v5 batch normalization does not seem to support float16.
        self._can_use_cudnn = cudnn_dim_ok and x[0].dtype != numpy.float16

        cudnn_updated_running_stats = False

        if chainer.configuration.config.train:
            axis = (0,) + tuple(range(head_ndim, x.ndim))
            mean, var = self._backend.forward(axis, gamma, x, xp)
            var += self.eps
        else:
            mean = self.fixed_mean
            var = self.fixed_var + self.eps
        self.std = xp.sqrt(var, dtype=var.dtype)
        if xp is numpy:
            self.x_hat = _xhat(x, mean, self.std, expander)
            y = gamma * self.x_hat
            y += beta
            y = y.astype(x.dtype)
        else:
            self.x_hat, y = cuda.elementwise(
                'T x, U mean, U std, U gamma, U beta', 'T x_hat, T y',
                '''
                x_hat = (x - mean) / std;
                y = gamma * x_hat + beta;
                ''',
                'bn_fwd')(x, mean[expander], self.std[expander], gamma,
                          beta)

        if chainer.configuration.config.train and \
                (not cudnn_updated_running_stats):
            # Note: If in training mode, the cuDNN forward training function
            # will do this for us, so
            # only run following code if cuDNN was not used.
            # Update running statistics:
            m = x.size // gamma.size
            adjust = m / max(m - 1., 1.)  # unbiased estimation
            if xp is numpy:
                self.running_mean *= self.decay
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
                cuda.elementwise(
                    'U mean, U var, U decay, U adjust',
                    'U r_mean, U r_var',
                    '''
                    r_mean = r_mean * decay + mean * (1 - decay);
                    r_var = r_var * decay + var * (1 - decay) * adjust;
                    ''',
                    'update_mean_var')(mean, var, self.decay, adjust,
                                       self.running_mean, self.running_var)

        return y,

    def backward(self, inputs, grad_outputs):
        x, gamma = inputs[:2]
        gy = grad_outputs[0]
        head_ndim = gamma.ndim + 1
        expander = (None, Ellipsis) + (None,) * (x.ndim - head_ndim)
        m = gamma.dtype.type(x.size // gamma.size)
        axis = (0,) + tuple(range(head_ndim, x.ndim))
        xp = backend.get_array_module(x)
        if len(inputs) == 5:
            # This case is unlikely to be used in practice and so does not
            # need to be optimized for performance.
            mean = inputs[3]
            var = inputs[4]
            std = xp.sqrt(var, dtype=var.dtype)
            gs = gamma / std
            gbeta = gy.sum(axis=axis)
            x_hat = _xhat(x, mean, std, expander)
            ggamma = (gy * x_hat).sum(axis=axis)
            gmean = -gs * gbeta
            gvar = -0.5 * gamma / var * ggamma
            gx = gs[expander] * gy
            gx = gx.astype(x.dtype, copy=False)
            return gx, ggamma, gbeta, gmean, gvar

        # Note: If length of inputs is not 5, we must be in train mode.
        assert chainer.configuration.config.train
        gbeta, ggamma = self._backend.backward(axis, gamma, gy,
                                               self.x_hat, x, xp)

        if xp is numpy:
            gx = (gamma / self.std)[expander] * (
                gy - (self.x_hat * ggamma[expander] + gbeta[expander]) / m)
            gx = gx.astype(x.dtype, copy=False)
        else:
            inv_m = numpy.float32(1) / m
            gx = cuda.elementwise(
                'T gy, T x_hat, U gamma, U std, U ggamma, U gbeta, \
                U inv_m',
                'T gx',
                'gx = (gamma / std) * (gy - (x_hat * ggamma + gbeta) * \
                inv_m)',
                'bn_bwd')(gy, self.x_hat, gamma[expander],
                          self.std[expander], ggamma[expander],
                          gbeta[expander], inv_m)

        return gx, ggamma, gbeta
