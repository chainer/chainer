import chainer
from chainer.backends import cuda
from chainer.functions.normalization import batch_normalization
import chainer.utils


class _MpiImpl(batch_normalization.GeneralBatchNormalizationImpl):
    def __init__(self, comm):
        self.comm = comm

    def get_mean_and_var(self, axis, gamma, x, xp, interm_dtype):
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

    def get_ggamma_and_gbeta(self, axis, gamma, gy, x_hat, xp):
        tmp = xp.empty(gamma.size * 2, dtype=gamma.dtype)
        gy.sum(axis=axis, out=tmp[:gamma.size], dtype=gamma.dtype)
        (gy * x_hat).sum(axis=axis, out=tmp[gamma.size:], dtype=gamma.dtype)
        if xp is cuda.cupy:
            chainer.cuda.Stream.null.synchronize()
        self.comm._multi_node_mean(None, tmp)
        gbeta = tmp[:gamma.size]
        ggamma = tmp[gamma.size:]
        return gbeta, ggamma


class _NcclImpl(batch_normalization.GeneralBatchNormalizationImpl):

    def __init__(self, comm):
        self.comm = comm

        # We need to delay importing MPI4py (and momdules that import MPI4py)
        import chainermn.communicators._memory_utility as memory_utility_module
        self.memory_utility_module = memory_utility_module

    def get_mean_and_var(self, axis, gamma, x, xp, interm_dtype):
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

    def get_ggamma_and_gbeta(self, axis, gamma, gy, x_hat, xp):
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


class MultiNodeBNImplSelector:
    def __init__(self, comm, communication_backend_name):
        self.comm = comm
        self.communication_backend_name = communication_backend_name

    def __call__(self, batch_norm_func, inputs):
        if self.communication_backend_name == 'nccl':
            return _NcclImpl(self.comm)
        else:
            return _MpiImpl(self.comm)
