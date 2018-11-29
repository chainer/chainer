import chainer
import copy


class _MultiNodeOptimizer(object):

    def __init__(self, actual_optimizer, communicator):
        super(_MultiNodeOptimizer, self).__setattr__(
            'communicator', communicator)
        super(_MultiNodeOptimizer, self).__setattr__(
            'actual_optimizer', actual_optimizer)
        super(_MultiNodeOptimizer, self).__setattr__(
            'target_params', [])

    def update(self, lossfun=None, *args, **kwds):
        target = self.target
        if lossfun is not None:
            loss = lossfun(*args, **kwds)
            target.cleargrads()
            loss.backward()
            del loss

        if self.is_changed(target):
            self.communicator.bcast_data(target)
        else:
            self.communicator.allreduce_grad(target)
            self.actual_optimizer.update(None, *args, **kwds)

    def is_changed(self, target):
        previous_params = self.target_params
        super(_MultiNodeOptimizer, self).__setattr__(
            'target_params', [(name, param.data is not None)
                              for name, param in sorted(target.namedparams())])
        if len(previous_params) != len(self.target_params):
            return True

        for param1, param2 in zip(self.target_params, previous_params):
            if (param1[0] != param2[0]) or param1[1] != param2[1]:
                return True
        return False

    def setup(self, link):
        self.actual_optimizer.setup(link)
        return self

    def __getattr__(self, attr_name):
        return getattr(self.actual_optimizer, attr_name)

    def __setattr__(self, attr_name, value):
        setattr(self.actual_optimizer, attr_name, value)


class _DoubleBufferingOptimizer(object):

    def __init__(self, actual_optimizer, communicator):
        super(_DoubleBufferingOptimizer, self).__setattr__(
            'communicator', communicator)
        super(_DoubleBufferingOptimizer, self).__setattr__(
            'actual_optimizer', actual_optimizer)
        super(_DoubleBufferingOptimizer, self).__setattr__(
            'needs_update', False)
        super(_DoubleBufferingOptimizer, self).__setattr__(
            'communicated_target', None)
        super(_DoubleBufferingOptimizer, self).__setattr__(
            'target_params_list', [[], []])
        super(_DoubleBufferingOptimizer, self).__setattr__(
            'allreduce_grad_stream', chainer.cuda.Stream(non_blocking=True))

    def update(self, lossfun=None, *args, **kwds):
        target = self.target
        if lossfun is not None:
            loss = lossfun(*args, **kwds)
            target.cleargrads()
            loss.backward()
            del loss

        if self.is_changed(target, self.target_params_list[0]):
            self.wait()
            self.communicator.bcast_data(target)
            super(_DoubleBufferingOptimizer, self).__setattr__(
                'communicated_target', copy.deepcopy(target))
            super(_DoubleBufferingOptimizer, self).__setattr__(
                'target_params_list', [
                    list(sorted(self.target.namedparams())),
                    list(sorted(self.communicated_target.namedparams()))])
            super(_DoubleBufferingOptimizer, self).__setattr__(
                'needs_update', False)
        else:
            self.wait()
            self.swap_grad(self.target_params_list[0],
                           self.target_params_list[1])
            self.allreduce_grad_async()
            if self.needs_update:
                self.actual_optimizer.update(None, *args, **kwds)
            else:
                super(_DoubleBufferingOptimizer, self).__setattr__(
                    'needs_update', True)

    def allreduce_grad_async(self):
        self.communicator._allreduce_grad_async(
            self.communicated_target, self.allreduce_grad_stream)

    def is_changed(self, target, previous_params):
        target_params = list(sorted(target.namedparams()))
        if len(previous_params) != len(target_params):
            return True

        for param1, param2 in zip(target_params, previous_params):
            name1, var1 = param1
            name2, var2 = param2
            if (name1 != name2) or (var1.data is None) != (var2.data is None):
                return True
        return False

    def swap_grad(self, target1_params, target2_params):
        for param1, param2 in zip(target1_params, target2_params):
            _, var1 = param1
            _, var2 = param2
            var1.grad, var2.grad = var2.grad, var1.grad

    def wait(self):
        self.allreduce_grad_stream.synchronize()
        chainer.cuda.Stream.null.synchronize()

    def setup(self, link):
        self.actual_optimizer.setup(link)
        return self

    def __getattr__(self, attr_name):
        return getattr(self.actual_optimizer, attr_name)

    def __setattr__(self, attr_name, value):
        setattr(self.actual_optimizer, attr_name, value)

from chainermn.communicators.pure_nccl_communicator import _get_converting_kernel
from chainermn.communicators.pure_nccl_communicator import _get_nccl_type_id
from chainermn import nccl
import numpy as np

class _MultiNodeOptimizerWithLayerWiseAllreduce(object):
    def __init__(self, actual_optimizer, communicator):
        super(_MultiNodeOptimizerWithLayerWiseAllreduce, self).__setattr__(
            'communicator', communicator)
        super(_MultiNodeOptimizerWithLayerWiseAllreduce, self).__setattr__(
            'actual_optimizer', actual_optimizer)
        super(_MultiNodeOptimizerWithLayerWiseAllreduce, self).__setattr__(
            'needs_bcast', True)
        super(_MultiNodeOptimizerWithLayerWiseAllreduce, self).__setattr__(
            'buffer_size', 4*1024*1024)
        super(_MultiNodeOptimizerWithLayerWiseAllreduce, self).__setattr__(
            'buffers', [])
        super(_MultiNodeOptimizerWithLayerWiseAllreduce, self).__setattr__(
            'buffers_i', 0)

    def update(self, lossfun=None, *args, **kwds):
        target = self.target
        if lossfun is not None:
            loss = lossfun(*args, **kwds)
            target.cleargrads()
            loss.backward()
            del loss

        if self.needs_bcast:
            self.communicator.bcast_data(target)
            super(_MultiNodeOptimizerWithLayerWiseAllreduce, self).__setattr__(
                'needs_bcast', False)
        else:
            buffers_i = self.buffers_i
            buffer = self.buffers[buffers_i]
            buffer.aggregate_grads(self.communicator)
            self.synchronize()
            self.clear_buffers()
            self.actual_optimizer.update(None, *args, **kwds)
        super(_MultiNodeOptimizerWithLayerWiseAllreduce, self).__setattr__(
            'target_params_i', 0)

    def setup(self, link):
        for param in link.params():
            param.add_hook(self.post_backward_hook, name='append_to_buffer')

        super(_MultiNodeOptimizerWithLayerWiseAllreduce, self).__setattr__(
            'needs_bcast', True)
        buffer = GradBuffer(max_memory_size=32*1024*1024, allreduce_grad_dtype=np.dtype('float32'))
        self.buffers.append(buffer)
        self.actual_optimizer.setup(link)
        return self


    def post_backward_hook(self, param):
        print('call post backward hook')
        if not self.needs_bcast:
            self.append_to_buffer(param)

    def append_to_buffer(self, param):
        buffers_i = self.buffers_i
        buffer = self.buffers[buffers_i]
        if not buffer.can_append_param(param):
            buffer.aggregate_grads(self.communicator)
            buffers_i += 1
            if len(self.buffers) == buffers_i:
                buffer = GradBuffer(max_memory_size=32*1024*1024, allreduce_grad_dtype=np.dtype('float32'))
                self.buffers.append(buffer)
            super(_MultiNodeOptimizerWithLayerWiseAllreduce, self).__setattr__(
                'buffers_i', buffers_i)
        buffer.append_param(param)


    def synchronize(self):
        for b in self.buffers:
            b.synchronize()

    def clear_buffers(self):
        for b in self.buffers:
            b.clear()

        super(_MultiNodeOptimizerWithLayerWiseAllreduce, self).__setattr__(
            'buffers_i', 0)

    def __getattr__(self, attr_name):
        return getattr(self.actual_optimizer, attr_name)

    def __setattr__(self, attr_name, value):
        setattr(self.actual_optimizer, attr_name, value)


from chainermn.communicators import _memory_utility


class GradBuffer(object):

    def __init__(self, max_memory_size, allreduce_grad_dtype):
        self.cuda_stream = chainer.cuda.Stream(non_blocking=True)
        self.length = 0
        self.memory_size = 0
        self.max_memory_size = max_memory_size
        self.params = []
        self.src_array = _memory_utility.DeviceMemory()
        self.dest_array = _memory_utility.DeviceMemory()
        self.allreduce_grad_dtype = allreduce_grad_dtype

    def can_append_param(self, param):
        data_dtype = self.allreduce_grad_dtype
        memory_size = param.data.size * data_dtype.itemsize
        print(self.memory_size + memory_size, self.max_memory_size)
        return (self.memory_size + memory_size) <= self.max_memory_size

    def append_param(self, param):
        self.length += param.data.size
        data_dtype = self.allreduce_grad_dtype
        self.memory_size = self.length * data_dtype.itemsize
        self.params.append(param)

    def get_memory_max_size(self):
        return self.max_memory_size

    def get_memory_size(self):
        return self.memory_size

    def get_length(self):
        return self.length

    def aggregate_grads(self, communicator):
        communicator._init_comms()
        assert len(self.params)
        grad_dtype = self.params[0].grad.dtype
        allreduce_grad_dtype = self.allreduce_grad_dtype
        assert grad_dtype == allreduce_grad_dtype
        n_elems = self.get_length()
        stream = self.cuda_stream
        needs_sync = self._assign_for_allreduce_grad(grad_dtype,
                                                     allreduce_grad_dtype,
                                                     n_elems)
        if stream != chainer.cuda.Stream.null and needs_sync:
            chainer.cuda.Stream.null.synchronize()

        self._pack_params_to_buffer(self.params, grad_dtype, allreduce_grad_dtype,
                                    n_elems, stream)
        self.nccl_comm.allReduce(self.gpu_buffer_a.ptr(),
                                 self.gpu_buffer_b.ptr(), n_elems,
                                 _get_nccl_type_id(allreduce_grad_dtype),
                                 nccl.NCCL_SUM,
                                 stream.ptr)
        if self.div_by_size is None:
            self.div_by_size = chainer.cuda.cupy.ElementwiseKernel(
                '{} x'.format(allreduce_grad_dtype.name),
                '{} y'.format(allreduce_grad_dtype.name),
                'y = x*(1.0/{})'.format(self.size), 'div_by_size')
        self.div_by_size(
            self.gpu_buffer_b.array(n_elems,
                                    dtype=allreduce_grad_dtype),
            self.gpu_buffer_a.array(n_elems,
                                    dtype=allreduce_grad_dtype),
            stream=stream)
        self._unpack_params_from_buffer(self.params, grad_dtype,
                                        allreduce_grad_dtype, n_elems, stream)


        
    def _pack_params_to_buffer(self, params, grad_dtype, allreduce_grad_dtype,
                               n_elems, stream):
        if grad_dtype == allreduce_grad_dtype:
            _memory_utility.pack_params(
                params, grad_dtype.itemsize, 'grad',
                self.gpu_buffer_a, stream=stream)
        else:
            if self.grad_dtype_to_allreduce_dtype_kernel is None:
                self.grad_dtype_to_allreduce_dtype_kernel = \
                    _get_converting_kernel(
                        grad_dtype, allreduce_grad_dtype,
                        'grad_dtype_to_allreduce_dtype_kernel')

            _memory_utility.pack_params(
                params, grad_dtype.itemsize, 'grad',
                self.gpu_tmp_buffer, stream=stream)

            self.grad_dtype_to_allreduce_dtype_kernel(
                self.gpu_tmp_buffer.array(n_elems, dtype=grad_dtype),
                self.gpu_buffer_a.array(n_elems,
                                        dtype=allreduce_grad_dtype),
                stream=stream)

    def _unpack_params_from_buffer(self, params, grad_dtype,
                                   allreduce_grad_dtype, n_elems, stream):
        if grad_dtype == allreduce_grad_dtype:
            _memory_utility.unpack_params(
                params, allreduce_grad_dtype.itemsize, 'grad',
                self.gpu_buffer_a, stream)

        else:
            if self.allreduce_dtype_to_grad_dtype_kernel is None:
                self.allreduce_dtype_to_grad_dtype_kernel = \
                    _get_converting_kernel(
                        allreduce_grad_dtype, grad_dtype,
                        'allreduce_dtype_to_grad_dtype_kernel')
            self.allreduce_dtype_to_grad_dtype_kernel(
                self.gpu_buffer_a.array(n_elems,
                                        dtype=allreduce_grad_dtype),
                self.gpu_tmp_buffer.array(n_elems, dtype=grad_dtype),
                stream=stream)

            _memory_utility.unpack_params(
                params, grad_dtype.itemsize, 'grad', self.gpu_tmp_buffer,
                stream=stream)

    def clear(self):
        self.params.clear()
        self.length = 0
        self.memory_size = 0
        
    def synchronize(self):
        self.cuda_stream.synchronize()
        

def create_multi_node_optimizer(actual_optimizer, communicator,
                                double_buffering=False):
    """Create a multi node optimizer from a Chainer optimizer.

    Args:
        actual_optimizer: Chainer optimizer
            (e.g., ``chainer.optimizers.Adam``).
        communicator: ChainerMN communicator.
        double_buffering: If ``True``, all-reduce and other
             processing (such as forward and backward) are
             overlapped using double buffering.
             There are cases where accuracy is affected because
             the gradients of the previous iteration are used
             for update. This flag is supported by
             ``PureNcclCommunicator`` only.
    Returns:
        The multi node optimizer based on ``actual_optimizer``.
    """
    if double_buffering:
        from chainermn.communicators.pure_nccl_communicator \
            import PureNcclCommunicator
        if not isinstance(communicator, PureNcclCommunicator):
            raise ValueError(
                'This communicator does not support double buffering.')
        return _DoubleBufferingOptimizer(actual_optimizer, communicator)
    #return _MultiNodeOptimizer(actual_optimizer, communicator)
    return _MultiNodeOptimizerWithLayerWiseAllreduce(actual_optimizer, communicator)
