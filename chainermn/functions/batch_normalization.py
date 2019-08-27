# This file is heavily based on Chainer's batch normalization implementation.
# See: chainer/functions/normalization/batch_normalization.py (dbb650)

import chainer
from chainer import cuda
import chainer.utils
from chainer.functions.normalization.batch_normalization \
    import _GeneralBatchNormalizationBackend
from chainer.functions.normalization.batch_normalization \
    import BatchNormalization


class _MpiBackend(_GeneralBatchNormalizationBackend):
    def __init__(self, comm):
        self.comm = comm

    def _get_mean_and_var(self, axis, gamma, x, xp, interm_dtype):
        tmp = xp.empty(gamma.size * 2, dtype=gamma.dtype)
        x.mean(axis=axis, out=tmp[:gamma.size], dtype=gamma.dtype)
        xp.square(x).mean(axis=axis, out=tmp[gamma.size:], dtype=gamma.dtype)
        if xp is cuda.cupy:
            chainer.cuda.Stream.null.synchronize()
        self.comm._multi_node_mean(None, tmp)
        mean = tmp[:gamma.size]
        sqmean = tmp[gamma.size:]
        var = sqmean - xp.square(mean)
        return mean, var

    def _get_ggamma_and_gbeta(self, axis, gamma, gy, x_hat, xp):
        tmp = xp.empty(gamma.size * 2, dtype=gamma.dtype)
        gy.sum(axis=axis, out=tmp[:gamma.size], dtype=gamma.dtype)
        (gy * x_hat).sum(axis=axis, out=tmp[gamma.size:], dtype=gamma.dtype)
        if xp is cuda.cupy:
            chainer.cuda.Stream.null.synchronize()
        self.comm._multi_node_mean(None, tmp)
        gbeta = tmp[:gamma.size]
        ggamma = tmp[gamma.size:]
        return gbeta, ggamma


class _NcclBackend(_GeneralBatchNormalizationBackend):

    def __init__(self, comm):
        self.comm = comm

        # We need to delay importing MPI4py (and momdules that import MPI4py)
        import chainermn.communicators._memory_utility as memory_utility_module
        self.memory_utility_module = memory_utility_module

    def _get_mean_and_var(self, axis, gamma, x, xp, interm_dtype):
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
        self.comm._multi_node_mean_nccl(gpu_buffer_a,
                                        gpu_buffer_b,
                                        gpu_buffer_n_elems,
                                        gamma.dtype)
        gpu_buffer_a_array = gpu_buffer_b.array(
            gpu_buffer_n_elems,
            dtype=gamma.dtype)

        mean = gpu_buffer_a_array[:gamma.size]
        sqmean = gpu_buffer_a_array[gamma.size:]
        var = sqmean - xp.square(mean)
        return mean, var

    def _get_ggamma_and_gbeta(self, axis, gamma, gy, x_hat, xp):
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
        self.comm._multi_node_mean_nccl(gpu_buffer_a,
                                        gpu_buffer_b,
                                        gpu_buffer_n_elems,
                                        gamma.dtype)
        gpu_buffer_a_array = gpu_buffer_b.array(
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


class MultiNodeBatchNormalizationFunction(BatchNormalization):
    def __init__(self, comm, eps=2e-5, mean=None, var=None, decay=0.9,
                 communication_backend='auto'):
        super().__init__(eps, mean, var, decay)
        self.comm = comm
        self.communication_backend = communication_backend

    def _bn_backend_selector(self, xp):
        if self.communication_backend == 'nccl':
            return _NcclBackend(self.comm)
        else:
            return _MpiBackend(self.comm)

    def copy(self, mode='share'):
        to_be_preserved = ['bn_backend', 'comm']
        preserved = {}
        for name in to_be_preserved:
            preserved[name] = getattr(self, name)
            setattr(self, name, None)

        ret = super(BatchNormalization, self).copy(mode)

        for name in to_be_preserved:
            setattr(self, name, preserved[name])
            setattr(ret, name, preserved[name])

        return ret
